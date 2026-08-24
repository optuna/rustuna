use pyo3::prelude::*;
use pyo3::types::{PyDict, PyFloat, PyInt, PyIterator, PyString, PyType};
use pyo3::{PyTypeInfo, Python};

use pyo3::exceptions::{PyRuntimeError, PyTypeError, PyUserWarning, PyValueError};
use rustuna_core::ErrorKind;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use rustuna_core::attr::AttrKey;
use rustuna_core::sampler::Sampler;
use rustuna_core::storage::Storage;
use rustuna_core::study::{
    create_study_with_arc, get_best_trial, get_pareto_front, Direction, PersistedStudy, Study,
};
use rustuna_core::trial::TrialStateValues;
use rustuna_sampler::tpe::TpeSampler;

use crate::attrs::pyobj_to_attrs;
use crate::attrs::{convert_pydict_to_fixed_params, pyobj_to_attrs_with_kind, AttrKind};
use crate::exception::err_to_exceptions;
use crate::sampler::cmaes::PyCmaEsSampler;
use crate::sampler::nsgaii::PyNSGAIISampler;
use crate::sampler::random::PyRandomSampler;
use crate::sampler::to_rust::ToRustSampler;
use crate::sampler::tpe::PyTpeSampler;
use crate::storage::cached::PyCachedStorage;
use crate::storage::in_memory::PyInMemoryStorage;
use crate::storage::journal::PyJournalFileStorage;
use crate::storage::sqlite3::PySQLite3Storage;
use crate::storage::to_rust::ToRustStorage;
use crate::trial::{PyPersistedTrial, PyTrial, PyTrialState};
use crate::trial_queue::directory::PyDirectoryTrialQueue;
use crate::trial_queue::inmemory::PyInMemoryTrialQueue;
use crate::trial_queue::sqlite3::PySQLite3TrialQueue;
use crate::trial_queue::to_rust::ToRustTrialQueue;

type SharedStorage = Arc<RwLock<dyn Storage>>;
type SharedSampler = Arc<dyn Sampler>;
type SharedTrialQueue = Arc<RwLock<dyn rustuna_core::trial_queue::TrialQueue>>;

mod py_exceptions {
    pyo3::import_exception!(rustuna.exceptions, TrialPruned);
}

fn normalize_catch(py: Python<'_>, catch: Option<&Bound<'_, PyAny>>) -> PyResult<Vec<Py<PyAny>>> {
    let Some(catch) = catch else {
        return Ok(Vec::new());
    };
    let base_exception = py.import("builtins")?.getattr("BaseException")?;

    if let Ok(catch_type) = catch.cast::<PyType>() {
        if catch_type.is_subclass(&base_exception)? {
            return Ok(vec![catch.clone().unbind()]);
        }
    }

    let iter = PyIterator::from_object(catch)?;
    let mut normalized = Vec::new();
    for item in iter {
        let item = item?;
        let item_type = item.cast::<PyType>().map_err(|_| {
            PyTypeError::new_err(
                "The catch argument must be an exception class or an iterable of exception classes.",
            )
        })?;
        if !item_type.is_subclass(&base_exception)? {
            return Err(PyTypeError::new_err(
                "The catch argument must be an exception class or an iterable of exception classes.",
            ));
        }
        normalized.push(item.unbind());
    }
    Ok(normalized)
}

fn objective_result_to_values(val: Bound<'_, PyAny>) -> PyResult<Vec<f64>> {
    if val.is_instance_of::<PyFloat>() || val.is_instance_of::<PyInt>() {
        let val = val
            .extract::<f64>()
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to extract f64: {e:?}")))?;
        Ok(vec![val])
    } else {
        let iter = PyIterator::from_object(&val).map_err(|e| {
            PyRuntimeError::new_err(format!(
                "Objective function must return either int, float or tuple[int | float]. error={e:?}"
            ))
        })?;
        let mut vals = Vec::new();
        for item in iter {
            let item = item.map_err(|e| {
                PyRuntimeError::new_err(format!(
                    "Objective function must return either int, float or tuple[int | float]. error={e:?}"
                ))
            })?;
            let v = if item.is_instance_of::<PyInt>() {
                item.extract::<i64>()? as f64
            } else {
                item.extract::<f64>()?
            };
            vals.push(v);
        }
        Ok(vals)
    }
}

/// Rust port of Optuna's `_check_values_are_feasible`: returns `Some(message)`
/// when the told values cannot be accepted as objective values of the study.
fn values_infeasible_reason(values: &[f64], n_objectives: usize) -> Option<String> {
    let errors: Vec<String> = values
        .iter()
        .filter(|v| v.is_nan())
        .map(|v| format!("The value {v} is not acceptable"))
        .collect();
    if !errors.is_empty() {
        return Some(errors.join("; "));
    }
    if values.len() != n_objectives {
        return Some(format!(
            "The number of the values {} did not match the number of the objectives {}",
            values.len(),
            n_objectives
        ));
    }
    None
}

/// Emits a `UserWarning`, matching Optuna's behavior of warning instead of
/// raising when `tell` receives infeasible values without an explicit state.
fn warn_user(py: Python<'_>, message: &str) -> PyResult<()> {
    let message = std::ffi::CString::new(message)
        .unwrap_or_else(|_| std::ffi::CString::new("invalid warning message").unwrap());
    PyErr::warn(py, &py.get_type::<PyUserWarning>(), &message, 1)
}

fn matches_any_exception(py: Python<'_>, err: &PyErr, catch: &[Py<PyAny>]) -> PyResult<bool> {
    for exc_type in catch {
        if err.matches(py, exc_type.bind(py))? {
            return Ok(true);
        }
    }
    Ok(false)
}

fn into_trial_queue_pyobj(
    py: Python<'_>,
    trial_queue: Option<Py<PyAny>>,
) -> PyResult<(SharedTrialQueue, Py<PyAny>)> {
    match trial_queue {
        Some(trial_queue) => {
            let trial_queue_ref = trial_queue.bind(py);
            if let Ok(py_inmemory_trial_queue) = trial_queue_ref.extract::<PyInMemoryTrialQueue>() {
                Ok((
                    py_inmemory_trial_queue.queue.clone() as SharedTrialQueue,
                    trial_queue.clone_ref(py),
                ))
            } else if let Ok(py_sqlite3_trial_queue) =
                trial_queue_ref.extract::<PySQLite3TrialQueue>()
            {
                Ok((
                    py_sqlite3_trial_queue.queue.clone() as SharedTrialQueue,
                    trial_queue.clone_ref(py),
                ))
            } else if let Ok(py_directory_trial_queue) =
                trial_queue_ref.extract::<PyDirectoryTrialQueue>()
            {
                Ok((
                    py_directory_trial_queue.queue.clone() as SharedTrialQueue,
                    trial_queue.clone_ref(py),
                ))
            } else {
                let queue: SharedTrialQueue = Arc::new(RwLock::new(ToRustTrialQueue::new(
                    trial_queue.clone_ref(py),
                )));
                Ok((queue, trial_queue.clone_ref(py)))
            }
        }
        None => {
            let trial_queue = PyInMemoryTrialQueue::new();
            let queue = trial_queue.queue.clone();
            let py_trial_queue = Py::new(py, trial_queue)?.into_any();
            Ok((queue, py_trial_queue))
        }
    }
}

fn resolve_storage_pyobj(
    py: Python<'_>,
    storage: Py<PyAny>,
) -> PyResult<(SharedStorage, Py<PyAny>)> {
    let storage_pyobj = storage.clone_ref(py);
    let storage_ref = storage.bind(py);
    if let Ok(py_inmemory_storage) = storage_ref.extract::<PyInMemoryStorage>() {
        Ok((py_inmemory_storage.storage(), storage_pyobj))
    } else if let Ok(py_cached_storage) = storage_ref.extract::<PyCachedStorage>() {
        Ok((py_cached_storage.storage(), storage_pyobj))
    } else if let Ok(py_journal_storage) = storage_ref.extract::<PyJournalFileStorage>() {
        Ok((py_journal_storage.storage(), storage_pyobj))
    } else if let Ok(py_sqlite3_storage) = storage_ref.extract::<PySQLite3Storage>() {
        Ok((py_sqlite3_storage.storage(), storage_pyobj))
    } else {
        let mut wrapped = ToRustStorage::new(storage);
        wrapped.sync_studies(true).map_err(err_to_exceptions)?;
        let wrapped: SharedStorage = Arc::new(RwLock::new(wrapped));
        Ok((wrapped, storage_pyobj))
    }
}

fn into_storage_pyobj(
    py: Python<'_>,
    storage: Option<Py<PyAny>>,
) -> PyResult<(SharedStorage, Py<PyAny>)> {
    match storage {
        Some(storage) => resolve_storage_pyobj(py, storage),
        None => {
            let storage = PyInMemoryStorage::default();
            let storage_arc = storage.storage();
            let storage_pyobj = Py::new(py, storage)?.into_any();
            Ok((storage_arc, storage_pyobj))
        }
    }
}

fn resolve_sampler_pyobj(
    py: Python<'_>,
    sampler: Py<PyAny>,
) -> PyResult<(SharedSampler, Py<PyAny>)> {
    let sampler_pyobj = sampler.clone_ref(py);
    let sampler_ref = sampler.bind(py);
    if let Ok(py_tpe_sampler) = sampler_ref.extract::<PyTpeSampler>() {
        Ok((py_tpe_sampler.sampler.clone(), sampler_pyobj))
    } else if let Ok(py_nsgaii_sampler) = sampler_ref.extract::<PyNSGAIISampler>() {
        Ok((py_nsgaii_sampler.sampler.clone(), sampler_pyobj))
    } else if let Ok(py_cmaes_sampler) = sampler_ref.extract::<PyCmaEsSampler>() {
        Ok((py_cmaes_sampler.sampler.clone(), sampler_pyobj))
    } else if let Ok(py_random_sampler) = sampler_ref.extract::<PyRandomSampler>() {
        Ok((py_random_sampler.sampler.clone(), sampler_pyobj))
    } else {
        let sampler: SharedSampler = Arc::new(ToRustSampler::new(sampler));
        Ok((sampler, sampler_pyobj))
    }
}

fn into_sampler_pyobj(
    py: Python<'_>,
    sampler: Option<Py<PyAny>>,
    _is_multi_objective: bool,
) -> PyResult<(SharedSampler, Py<PyAny>)> {
    match sampler {
        Some(sampler) => resolve_sampler_pyobj(py, sampler),
        None => {
            let sampler = Arc::new(TpeSampler::new());
            let py_sampler = PyTpeSampler {
                sampler: sampler.clone(),
            };
            let sampler_pyobj = Py::new(py, py_sampler)?.into_any();
            Ok((sampler, sampler_pyobj))
        }
    }
}

#[pyfunction]
#[pyo3(name = "create_study", signature = (*, study_name = None, storage = None, sampler = None, direction = None, directions = None, load_if_exists = false, trial_queue = None))]
pub fn py_create_study(
    study_name: Option<String>,
    storage: Option<Py<PyAny>>,
    sampler: Option<Py<PyAny>>,
    direction: Option<Py<PyAny>>,
    directions: Option<Vec<Py<PyAny>>>,
    load_if_exists: bool,
    trial_queue: Option<Py<PyAny>>,
) -> PyResult<PyStudy> {
    let study_name = match study_name {
        Some(s) => s,
        None => "default".to_string(), // TODO(c-bata): Generate random name with uuid.
    };
    let (storage_arc, storage_pyobj) = Python::attach(|py| into_storage_pyobj(py, storage))?;
    let directions = Python::attach(|py| convert_directions(py, direction, directions))?;
    let is_multi_objective = directions.len() > 1;
    let (sampler_arc, sampler_pyobj) =
        Python::attach(|py| into_sampler_pyobj(py, sampler, is_multi_objective))?;
    let (trial_queue_arc, trial_queue_pyobj) =
        Python::attach(|py| into_trial_queue_pyobj(py, trial_queue))?;
    let study = match create_study_with_arc(
        &study_name,
        storage_arc.clone(),
        sampler_arc.clone(),
        directions,
    ) {
        Ok(study) => study,
        Err(err) => {
            if !load_if_exists || !matches!(err.kind, ErrorKind::DuplicatedStudy) {
                return Err(err_to_exceptions(err));
            }
            let mut guard = storage_arc
                .write()
                .map_err(|_| PyRuntimeError::new_err("Failed to acquire the storage guard"))?;
            let (study_id, directions) = guard
                .get_studies()
                .map_err(err_to_exceptions)?
                .iter()
                .find(|s| s.name == study_name)
                .map(|s| (s.id, s.directions.clone()))
                .ok_or(PyRuntimeError::new_err(format!(
                    "Study {study_name} not found"
                )))?;
            drop(guard);
            Study::new(
                study_id,
                study_name.clone(),
                directions,
                storage_arc.clone(),
                sampler_arc.clone(),
                trial_queue_arc.clone(),
            )
        }
    };
    let study = Study::new(
        study.id,
        study.name,
        study.directions,
        study.storage,
        study.sampler,
        trial_queue_arc,
    );
    Ok(PyStudy {
        study,
        storage_pyobj,
        sampler_pyobj,
        trial_queue_pyobj,
    })
}

#[pyfunction]
#[pyo3(name = "load_study", signature = (study_name, storage, *, sampler = None, trial_queue = None))]
pub fn py_load_study(
    study_name: String,
    storage: Py<PyAny>,
    sampler: Option<Py<PyAny>>,
    trial_queue: Option<Py<PyAny>>,
) -> PyResult<PyStudy> {
    let (storage, storage_pyobj) = Python::attach(|py| resolve_storage_pyobj(py, storage))?;
    let mut guard = storage
        .write()
        .map_err(|_| PyRuntimeError::new_err("Failed to acquire the storage guard"))?;
    let (study_id, directions) = guard
        .get_studies()
        .map_err(|e| PyRuntimeError::new_err(format!("Failed to get the studies: {:?}", e.kind)))?
        .iter()
        .find(|s| s.name == study_name)
        .map(|s| (s.id, s.directions.clone()))
        .ok_or(PyRuntimeError::new_err(format!(
            "Study {study_name} not found"
        )))?;
    drop(guard);
    let is_multi_objective = directions.len() > 1;
    let (trial_queue_arc, trial_queue_pyobj) =
        Python::attach(|py| into_trial_queue_pyobj(py, trial_queue))?;
    let (sampler_arc, sampler_pyobj) =
        Python::attach(|py| into_sampler_pyobj(py, sampler, is_multi_objective))?;
    let study = Study::new(
        study_id,
        study_name,
        directions,
        storage,
        sampler_arc.clone(),
        trial_queue_arc,
    );
    Ok(PyStudy {
        study,
        storage_pyobj,
        sampler_pyobj,
        trial_queue_pyobj,
    })
}

#[pyclass(name = "Study")]
#[pyo3(module = "rustuna")]
pub struct PyStudy {
    pub study: Study,
    storage_pyobj: Py<PyAny>,
    sampler_pyobj: Py<PyAny>,
    trial_queue_pyobj: Py<PyAny>,
}
#[allow(non_local_definitions)]
#[pymethods]
impl PyStudy {
    #[new]
    #[pyo3(signature = (study_id, name, directions, storage, sampler))]
    fn py_new(
        study_id: u32,
        name: String,
        directions: Vec<PyDirection>,
        storage: Py<PyAny>,
        sampler: Py<PyAny>,
    ) -> PyResult<Self> {
        let directions: Vec<Direction> = directions.into_iter().map(|d| d.into()).collect();
        let (storage_arc, storage_pyobj) = Python::attach(|py| resolve_storage_pyobj(py, storage))?;
        let (sampler_arc, sampler_pyobj) = Python::attach(|py| resolve_sampler_pyobj(py, sampler))?;
        let trial_queue = PyInMemoryTrialQueue::new();
        let trial_queue_arc = trial_queue.queue.clone();
        let trial_queue_pyobj = Python::attach(|py| -> PyResult<Py<PyAny>> {
            Ok(Py::new(py, trial_queue)?.into_any())
        })?;
        let study = Study::new(
            study_id,
            name,
            directions,
            storage_arc,
            sampler_arc,
            trial_queue_arc,
        );
        Ok(PyStudy {
            study,
            storage_pyobj,
            sampler_pyobj,
            trial_queue_pyobj,
        })
    }

    #[pyo3(signature = (objective, n_trials, catch = None))]
    pub fn optimize(
        &self,
        py: Python<'_>,
        objective: Py<PyAny>,
        n_trials: usize,
        catch: Option<Py<PyAny>>,
    ) -> PyResult<()> {
        let catch = normalize_catch(py, catch.as_ref().map(|c| c.bind(py)))?;
        for _ in 0..n_trials {
            let rs_trial = py.detach(|| self.study.ask()).map_err(err_to_exceptions)?;
            let trial_number = rs_trial.number;

            let result: PyResult<Vec<f64>> = {
                let trial = PyTrial::new(rs_trial, self.storage_pyobj.clone_ref(py));
                objective
                    .call1(py, (trial,))
                    .and_then(|val| objective_result_to_values(val.bind(py).clone()))
            };

            match result {
                Ok(val) => {
                    let state_values =
                        match values_infeasible_reason(&val, self.study.directions.len()) {
                            Some(msg) => {
                                warn_user(py, &msg)?;
                                TrialStateValues::Fail
                            }
                            None => TrialStateValues::Complete(val),
                        };
                    py.detach(|| self.study.tell(trial_number, state_values))
                        .map_err(|e| PyRuntimeError::new_err(format!("Failed to tell: {e:?}")))?;
                }
                Err(e) => {
                    let (state, should_reraise) =
                        if e.matches(py, py_exceptions::TrialPruned::type_object(py))? {
                            (TrialStateValues::Pruned, false)
                        } else {
                            (
                                TrialStateValues::Fail,
                                !matches_any_exception(py, &e, &catch)?,
                            )
                        };
                    py.detach(|| self.study.tell(trial_number, state))
                        .map_err(|err| {
                            PyRuntimeError::new_err(format!("Failed to tell: {err:?}"))
                        })?;
                    if should_reraise {
                        return Err(e);
                    }
                }
            }
        }
        Ok(())
    }

    pub fn ask(&self, py: Python<'_>) -> PyResult<PyTrial> {
        // ask takes the storage/sampler locks inside rustuna_core: run it detached.
        let trial = py.detach(|| self.study.ask()).map_err(err_to_exceptions)?;
        Ok(PyTrial::new(trial, self.storage_pyobj.clone_ref(py)))
    }

    #[pyo3(signature = (number, values = None, state = None))]
    pub fn tell(
        &self,
        py: Python<'_>,
        number: u32,
        values: Option<Py<PyAny>>,
        state: Option<PyTrialState>,
    ) -> PyResult<PyPersistedTrial> {
        let state_values = match (state, values) {
            (None, None) => Err(PyValueError::new_err(
                "Either state or values must be specified",
            )),
            (Some(PyTrialState::RUNNING), _) => Err(PyValueError::new_err(
                "Cannot tell running trials with values",
            )),
            (Some(PyTrialState::WAITING), _) => Err(PyValueError::new_err(
                "Cannot tell waiting trials with values",
            )),
            (Some(PyTrialState::FAIL), _) => Ok(TrialStateValues::Fail),
            (Some(PyTrialState::PRUNED), _) => Ok(TrialStateValues::Pruned),
            (Some(PyTrialState::COMPLETE), None) => Err(PyValueError::new_err(
                "values must be specified when state is COMPLETE",
            )),
            (Some(PyTrialState::COMPLETE), Some(values)) => {
                objective_result_to_values(values.bind(py).clone()).and_then(|v| {
                    match values_infeasible_reason(&v, self.study.directions.len()) {
                        Some(msg) => Err(PyValueError::new_err(msg)),
                        None => Ok(TrialStateValues::Complete(v)),
                    }
                })
            }
            (None, Some(values)) => {
                objective_result_to_values(values.bind(py).clone()).and_then(|v| {
                    match values_infeasible_reason(&v, self.study.directions.len()) {
                        Some(msg) => {
                            warn_user(py, &msg)?;
                            Ok(TrialStateValues::Fail)
                        }
                        None => Ok(TrialStateValues::Complete(v)),
                    }
                })
            }
        };
        let state_values = state_values?;
        // tell and the trial fetch below take the storage lock: run them detached.
        py.detach(|| {
            self.study
                .tell(number, state_values)
                .map_err(|e| PyRuntimeError::new_err(format!("Failed to tell: {e:?}")))?;

            let mut guard = self.study.storage.write().map_err(|e| {
                PyRuntimeError::new_err(format!("Failed to acquire the storage guard: {e:?}"))
            })?;
            let trial_id = guard
                .get_trial_id_from_study_id_trial_number(self.study.id, number)
                .map_err(|e| {
                    PyRuntimeError::new_err(format!("Failed to get trial id: {:?}", e.kind))
                })?;
            let trial = guard.get_trial(trial_id).map_err(|e| {
                PyRuntimeError::new_err(format!("Failed to get trial: {:?}", e.kind))
            })?;
            Ok(PyPersistedTrial::from_storage(
                self.study.storage.clone(),
                trial,
            ))
        })
    }

    #[pyo3(signature = (params, user_attrs = None))]
    pub fn enqueue_trial(
        &self,
        params: &Bound<'_, PyDict>,
        user_attrs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<()> {
        let fixed_params = convert_pydict_to_fixed_params(params)?;
        let user_attrs_opt = user_attrs
            .map(|d| pyobj_to_attrs_with_kind(d.as_any(), AttrKind::User))
            .transpose()?;
        self.study
            .enqueue_trial(fixed_params, user_attrs_opt)
            .map_err(err_to_exceptions)?;
        Ok(())
    }

    pub fn add_trial(&self, trial: &Bound<'_, PyPersistedTrial>) -> PyResult<()> {
        // Extract the underlying PersistedTrial
        let persisted_trial = trial.borrow().with_trial(|t| Ok(t.clone()))?;

        // Call the core implementation
        self.study
            .add_trial(persisted_trial)
            .map_err(err_to_exceptions)?;

        Ok(())
    }

    #[pyo3(signature = (key, value))]
    pub fn set_user_attr(&self, key: String, value: String) -> PyResult<()> {
        let mut attrs = rustuna_core::attr::Attrs::new();
        attrs.insert(AttrKey::User(key.into()), value);
        let mut guard = self.study.storage.write().map_err(|e| {
            PyRuntimeError::new_err(format!("Failed to acquire the storage guard: {e:?}"))
        })?;
        guard
            .set_study_attrs(self.study.id, attrs, false)
            .map_err(err_to_exceptions)?;
        Ok(())
    }

    fn set_user_attrs(&mut self, attrs: Py<PyAny>) -> PyResult<()> {
        let user_attrs = Python::attach(|py| {
            let attrs = attrs.bind(py);
            pyobj_to_attrs_with_kind(attrs, AttrKind::User)
        })?;
        let mut guard = self.study.storage.write().map_err(|e| {
            PyRuntimeError::new_err(format!("Failed to acquire the storage guard: {e:?}"))
        })?;
        guard
            .set_study_attrs(self.study.id, user_attrs, false)
            .map_err(err_to_exceptions)?;
        Ok(())
    }

    #[pyo3(name = "get_user_attr", signature = (key, *, decoder = None, default = None))]
    pub fn get_user_attr<'py>(
        &self,
        py: Python<'py>,
        key: String,
        decoder: Option<Py<PyAny>>,
        default: Option<Py<PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        let result = {
            let mut guard = self.study.storage.write().map_err(|e| {
                PyRuntimeError::new_err(format!("Failed to acquire the storage guard: {e:?}"))
            })?;
            guard.get_study_attr(self.study.id, AttrKey::User(key.into()))
        };
        match result {
            Ok(value) => match decoder {
                Some(decoder) => {
                    let decoded = decoder.call1(py, (&value,))?;
                    Ok(decoded)
                }
                None => Ok(PyString::new(py, &value).into_any().unbind()),
            },
            Err(e) if matches!(e.kind, ErrorKind::AttrNotFound) => {
                Ok(default.unwrap_or_else(|| py.None()))
            }
            Err(e) => Err(err_to_exceptions(e)),
        }
    }

    #[getter]
    pub fn best_trial(&self) -> PyResult<PyPersistedTrial> {
        let trial_number = get_best_trial(&self.study).map_err(|e| {
            PyRuntimeError::new_err(format!("Failed to get the best trial: {:?}", e.kind))
        })?;

        let mut guard = self.study.storage.write().map_err(|e| {
            PyRuntimeError::new_err(format!("Failed to acquire the storage guard: {e:?}"))
        })?;
        let trial_id = guard
            .get_trial_id_from_study_id_trial_number(self.study.id, trial_number)
            .map_err(|e| {
                PyRuntimeError::new_err(format!("Failed to get trial id: {:?}", e.kind))
            })?;
        let trial = guard
            .get_trial(trial_id)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to get trial: {:?}", e.kind)))?;
        Ok(PyPersistedTrial::from_storage(
            self.study.storage.clone(),
            trial,
        ))
    }

    #[getter(trials)]
    pub fn py_trials(&self) -> PyResult<Vec<PyPersistedTrial>> {
        let mut guard = self
            .study
            .storage
            .write()
            .map_err(|_| PyRuntimeError::new_err("Failed to acquire the storage guard"))?;
        let trials_vec = guard
            .get_trials(self.study.id)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to get trials: {:?}", e.kind)))?;
        let trials: Vec<PyPersistedTrial> = trials_vec
            .iter()
            .flatten()
            .map(|t| PyPersistedTrial::from_storage(self.study.storage.clone(), t))
            .collect();
        Ok(trials)
    }

    #[pyo3(name = "get_trials", signature = (*, states = None))]
    pub fn py_get_trials(
        &self,
        states: Option<Vec<PyTrialState>>,
    ) -> PyResult<Vec<PyPersistedTrial>> {
        let mut guard = self
            .study
            .storage
            .write()
            .map_err(|_| PyRuntimeError::new_err("Failed to acquire the storage guard"))?;
        let trials_vec = guard
            .get_trials(self.study.id)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to get trials: {:?}", e.kind)))?;
        let trials: Vec<PyPersistedTrial> = match states {
            Some(states) => trials_vec
                .iter()
                .flatten()
                .filter(|trial| states.contains(&PyTrialState::from(trial.state_values.clone())))
                .map(|trial| PyPersistedTrial::from_storage(self.study.storage.clone(), trial))
                .collect(),
            None => trials_vec
                .iter()
                .flatten()
                .map(|trial| PyPersistedTrial::from_storage(self.study.storage.clone(), trial))
                .collect(),
        };
        Ok(trials)
    }

    #[getter]
    pub fn user_attrs(&self) -> PyResult<HashMap<String, String>> {
        let mut guard = self
            .study
            .storage
            .write()
            .map_err(|_| PyRuntimeError::new_err("Failed to acquire the storage guard"))?;
        let study = guard
            .get_study(self.study.id)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to get study: {:?}", e.kind)))?;
        let mut user_attrs = HashMap::new();
        for (key, value) in &study.attrs {
            if let AttrKey::User(k) = key {
                user_attrs.insert(k.to_string(), value.clone());
            }
        }
        Ok(user_attrs)
    }

    #[getter(_study_id)]
    pub fn id(&self) -> u32 {
        self.study.id
    }

    #[getter(study_name)]
    pub fn name(&self) -> &str {
        &self.study.name
    }

    #[getter(_storage)]
    pub fn storage<'py>(&self, py: Python<'py>) -> Py<PyAny> {
        self.storage_pyobj.clone_ref(py)
    }

    #[getter]
    pub fn sampler<'py>(&self, py: Python<'py>) -> Py<PyAny> {
        self.sampler_pyobj.clone_ref(py)
    }

    #[getter]
    pub fn trial_queue<'py>(&self, py: Python<'py>) -> Py<PyAny> {
        self.trial_queue_pyobj.clone_ref(py)
    }

    #[getter]
    pub fn directions(&self) -> Vec<PyDirection> {
        self.study
            .directions
            .iter()
            .map(|d| d.clone().into())
            .collect()
    }

    #[getter]
    pub fn best_trials(&self) -> PyResult<Vec<PyPersistedTrial>> {
        let pareto_front_numbers = get_pareto_front(&self.study).map_err(|e| {
            PyRuntimeError::new_err(format!("Failed to get the pareto front: {:?}", e.kind))
        })?;
        let mut guard = self
            .study
            .storage
            .write()
            .map_err(|_| PyRuntimeError::new_err("Failed to acquire the storage guard"))?;
        let trials_vec = guard
            .get_trials(self.study.id)
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to get trials: {:?}", e.kind)))?;
        let best_trials = pareto_front_numbers
            .iter()
            .map(|n| {
                trials_vec
                    .get(*n as usize)
                    .and_then(|t| t.as_ref())
                    .map(|t| PyPersistedTrial::from_storage(self.study.storage.clone(), t))
                    .ok_or_else(|| {
                        PyRuntimeError::new_err(format!("Pareto front trial {n} not found"))
                    })
            })
            .collect::<PyResult<Vec<_>>>()?;
        Ok(best_trials)
    }

    fn __repr__(slf: &Bound<'_, Self>) -> PyResult<String> {
        let type_obj = slf.get_type();
        let class_name = type_obj.name()?;
        Ok(format!("{}({})", class_name, slf.borrow().__str__()?))
    }

    fn __str__(&self) -> PyResult<String> {
        Ok(format!(
            "id={} name={} directions={:?}",
            self.study.id,
            self.study.name,
            self.directions()
        ))
    }
}

#[derive(Clone, Debug, PartialEq)]
#[pyclass(name = "StudyDirection", eq, eq_int, from_py_object)]
#[pyo3(module = "rustuna")]
pub enum PyDirection {
    #[pyo3(name = "MINIMIZE")]
    Minimize,
    #[pyo3(name = "MAXIMIZE")]
    Maximize,
}
impl From<Direction> for PyDirection {
    fn from(item: Direction) -> Self {
        match item {
            Direction::Minimize => PyDirection::Minimize,
            Direction::Maximize => PyDirection::Maximize,
        }
    }
}
impl From<PyDirection> for Direction {
    fn from(val: PyDirection) -> Self {
        match val {
            PyDirection::Minimize => Direction::Minimize,
            PyDirection::Maximize => Direction::Maximize,
        }
    }
}

fn convert_directions(
    py: Python<'_>,
    direction: Option<Py<PyAny>>,
    directions: Option<Vec<Py<PyAny>>>,
) -> PyResult<Vec<Direction>> {
    if direction.is_some() && directions.is_some() {
        Err(PyValueError::new_err(
            "Cannot specify both `direction` and `directions`",
        ))?;
    };
    let direction = direction
        .as_ref()
        .map(|d| convert_direction(d.bind(py)))
        .transpose()?
        .unwrap_or(Direction::Minimize);
    let directions = match directions {
        Some(ds) => ds
            .into_iter()
            .map(|d| convert_direction(d.bind(py)))
            .collect(),
        None => Ok(vec![direction]),
    }?;
    Ok(directions)
}

fn convert_direction(direction: &Bound<'_, PyAny>) -> PyResult<Direction> {
    if let Ok(direction) = direction.extract::<PyDirection>() {
        return Ok(direction.into());
    }

    match direction.extract::<String>() {
        Ok(direction) => match direction.as_str() {
            "minimize" => Ok(Direction::Minimize),
            "maximize" => Ok(Direction::Maximize),
            _ => Err(PyValueError::new_err(
                "Invalid direction. Please specify either `minimize` or `maximize`",
            )),
        },
        Err(_) => Err(PyValueError::new_err(
            "Invalid direction. Please specify either `minimize` or `maximize`",
        )),
    }
}

#[pyfunction]
#[pyo3(name = "copy_study", signature = (*, from_study_name, from_storage, to_storage, to_study_name = None))]
pub fn py_copy_study(
    from_study_name: String,
    from_storage: Py<PyAny>,
    to_storage: Py<PyAny>,
    to_study_name: Option<String>,
) -> PyResult<()> {
    let (from_storage_arc, _) = Python::attach(|py| resolve_storage_pyobj(py, from_storage))?;
    let (from_directions, from_attrs, trials) = {
        let mut guard = from_storage_arc
            .write()
            .map_err(|_| PyRuntimeError::new_err("Failed to acquire the storage guard"))?;
        let from_study_id = guard
            .get_studies()
            .map_err(err_to_exceptions)?
            .iter()
            .find(|s| s.name == from_study_name)
            .map(|s| s.id)
            .ok_or(PyRuntimeError::new_err(format!(
                "Study {from_study_name} not found"
            )))?;
        let study = guard
            .get_study(from_study_id)
            .map_err(err_to_exceptions)?
            .clone();
        let trials = guard
            .get_trials(from_study_id)
            .map_err(err_to_exceptions)?
            .clone();
        (study.directions, study.attrs, trials)
    };

    let copied_study_name = to_study_name.unwrap_or(from_study_name);
    let (to_storage_arc, _) = Python::attach(|py| resolve_storage_pyobj(py, to_storage))?;
    let mut guard = to_storage_arc
        .write()
        .map_err(|_| PyRuntimeError::new_err("Failed to acquire the storage guard"))?;
    let to_study_id = guard
        .create_new_study(&copied_study_name, from_directions)
        .map_err(err_to_exceptions)?
        .id;
    if !from_attrs.is_empty() {
        guard
            .set_study_attrs(to_study_id, from_attrs, false)
            .map_err(err_to_exceptions)?;
    }
    for trial in trials.into_iter().flatten() {
        guard
            .create_new_trial_from_template(to_study_id, &trial)
            .map_err(err_to_exceptions)?;
    }
    Ok(())
}

#[derive(Debug, Clone)]
#[pyclass(name = "PersistedStudy", skip_from_py_object)]
#[pyo3(module = "rustuna", get_all, set_all)]
pub struct PyPersistedStudy {
    pub id: u32,
    pub name: String,
    pub directions: Vec<PyDirection>,
    pub user_attrs: HashMap<String, String>,
    pub system_attrs: HashMap<String, String>,
}
impl From<PersistedStudy> for PyPersistedStudy {
    fn from(item: PersistedStudy) -> Self {
        let cap = std::cmp::min(item.attrs.len() / 2, 1);
        let mut user_attrs: HashMap<String, String> = HashMap::with_capacity(cap);
        let mut system_attrs: HashMap<String, String> = HashMap::with_capacity(cap);

        for (key, val) in item.attrs {
            match key {
                AttrKey::User(k) => {
                    user_attrs.insert(k.to_string(), val);
                }
                AttrKey::System(k) => {
                    system_attrs.insert(k.to_string(), val);
                }
            }
        }
        let directions = item.directions.into_iter().map(|d| d.into()).collect();
        PyPersistedStudy {
            id: item.id,
            name: item.name,
            directions,
            user_attrs,
            system_attrs,
        }
    }
}

#[allow(non_local_definitions)]
#[pymethods]
impl PyPersistedStudy {
    #[new]
    #[pyo3(signature = (id, name, directions, user_attrs=None, system_attrs=None))]
    pub fn py_new(
        id: u32,
        name: String,
        directions: Vec<PyDirection>,
        user_attrs: Option<HashMap<String, String>>,
        system_attrs: Option<HashMap<String, String>>,
    ) -> Self {
        PyPersistedStudy {
            id,
            name,
            directions,
            user_attrs: user_attrs.unwrap_or_default(),
            system_attrs: system_attrs.unwrap_or_default(),
        }
    }

    fn __repr__(slf: &Bound<'_, Self>) -> PyResult<String> {
        let type_obj = slf.get_type();
        let class_name = type_obj.name()?;
        Ok(format!("{}({:?})", class_name, slf.borrow().__str__()?))
    }

    fn __str__(&self) -> PyResult<String> {
        Ok(format!(
            "id={} name={} user_attrs={:?} system_attrs={:?}",
            self.id, self.name, self.user_attrs, self.system_attrs
        ))
    }
}

pub fn pyobject_to_persisted_study(study: &Bound<'_, PyAny>) -> PyResult<PersistedStudy> {
    let study_id = study.getattr("id")?.extract::<u32>()?;
    let name = study.getattr("name")?.extract::<String>()?;
    let directions = study.getattr("directions")?.extract::<Vec<PyDirection>>()?;
    let directions: Vec<Direction> = directions.iter().map(|d| d.clone().into()).collect();

    let user_attrs = study.getattr("user_attrs")?;
    let system_attrs = study.getattr("system_attrs")?;
    if !user_attrs.is_instance_of::<PyDict>() || !system_attrs.is_instance_of::<PyDict>() {
        return Err(PyRuntimeError::new_err(
            "user_attrs and system_attrs must be a dict",
        ));
    }
    let attrs = pyobj_to_attrs(&user_attrs, &system_attrs)?;
    Ok(PersistedStudy::new_with_attrs(
        study_id, name, directions, attrs,
    ))
}
