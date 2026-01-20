use std::sync::{Arc, RwLock};

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;

use pyo3::types::{PyList, PyType};
use rustuna_core::attr::CategoryLabel;
use rustuna_core::distribution::Distribution;
use rustuna_core::storage::{InMemoryStorage, Storage};
use rustuna_core::study::Direction;
use rustuna_core::trial::TrialStateValues;
use rustuna_storages::cache::CachedStorage;
use rustuna_storages::journal::file::{JournalFileBackend, JournalFileSymlinkLock};
use rustuna_storages::journal::storage::JournalStorage;
use rustuna_storages::sqlite3::SQLite3Storage;

use crate::attrs::{pyobj_to_system_attrs, pyobj_to_user_attrs};
use crate::distribution::{category_label_to_pyobject, pyobject_to_category_label, PyDistribution};
use crate::exception::err_to_exceptions;
use crate::study::{PyDirection, PyPersistedStudy};
use crate::trial::{PyPersistedTrial, PyTrialState};

#[derive(Clone)]
#[pyclass(name = "Storage")]
#[pyo3(module = "rustuna")]
pub struct PyStorage {
    pub storage: Arc<RwLock<dyn Storage>>,
    pub kind: &'static str,
}

#[pymethods]
impl PyStorage {
    #[classmethod]
    fn in_memory(_cls: &Bound<'_, PyType>) -> PyResult<Self> {
        Ok(PyStorage {
            storage: Arc::new(RwLock::new(InMemoryStorage::new())),
            kind: "in_memory",
        })
    }

    #[classmethod]
    #[pyo3(name = "sqlite3", signature = (file_path, *, create_database = false))]
    fn sqlite3(_cls: &Bound<'_, PyType>, file_path: &str, create_database: bool) -> PyResult<Self> {
        let backend = SQLite3Storage::new(file_path).map_err(|e| {
            PyRuntimeError::new_err(format!("Failed to open the SQLite3 file: {e:?}"))
        })?;
        if create_database {
            backend.create_database().map_err(|e| {
                PyRuntimeError::new_err(format!("Failed to create the database: {e:?}"))
            })?;
        }

        let arc_storage = Arc::new(RwLock::new(CachedStorage::new(Box::new(backend))));
        Ok(PyStorage {
            storage: arc_storage,
            kind: "sqlite3",
        })
    }

    #[classmethod]
    #[pyo3(name = "journal_file", signature = (file_path,))]
    fn journal_file(_cls: &Bound<'_, PyType>, file_path: &str) -> PyResult<Self> {
        // TODO(c-bata): Add lock_obj argument to use JournalFileOpenLock.
        let lock_obj = Box::new(JournalFileSymlinkLock::new(file_path));
        let backend = JournalFileBackend::new(file_path, Some(lock_obj)).map_err(|e| {
            PyRuntimeError::new_err(format!("Failed to create journal file: {e:?}"))
        })?;
        let storage = JournalStorage::new(Box::new(backend)).map_err(|e| {
            PyRuntimeError::new_err(format!("Failed to create journal storage: {e:?}"))
        })?;
        let arc_storage = Arc::new(RwLock::new(storage));
        Ok(PyStorage {
            storage: arc_storage,
            kind: "journal",
        })
    }

    fn create_new_study(
        &mut self,
        study_name: String,
        directions: Vec<PyDirection>,
    ) -> PyResult<PyPersistedStudy> {
        let mut guard = self
            .storage
            .write()
            .map_err(|_| PyRuntimeError::new_err("Failed to acquire the storage guard"))?;
        let directions: Vec<Direction> = directions.iter().map(|d| d.clone().into()).collect();
        let study = guard
            .create_new_study(&study_name, directions)
            .map_err(err_to_exceptions)?;
        Ok(study.clone().into())
    }

    fn delete_study(&mut self, study_id: u32) -> PyResult<()> {
        let mut guard = self
            .storage
            .write()
            .map_err(|_| PyRuntimeError::new_err("Failed to acquire the storage guard"))?;
        guard.delete_study(study_id).map_err(err_to_exceptions)?;
        Ok(())
    }

    fn create_new_trial(&mut self, study_id: u32) -> PyResult<PyPersistedTrial> {
        let mut guard = self
            .storage
            .write()
            .map_err(|_| PyRuntimeError::new_err("Failed to acquire the storage guard"))?;
        let trial = guard
            .create_new_trial(study_id)
            .map_err(err_to_exceptions)?;
        Ok(PyPersistedTrial::from_storage(self.storage.clone(), trial))
    }

    fn set_trial_param(
        &mut self,
        trial_id: u32,
        name: String,
        distribution: PyDistribution,
        value: f64,
    ) -> PyResult<()> {
        let category_labels = distribution.category_labels.clone();
        let distribution: Distribution = distribution.into();

        if let Some(labels) = category_labels {
            let study_id = {
                let mut guard = self
                    .storage
                    .write()
                    .map_err(|_| PyRuntimeError::new_err("Failed to acquire the storage guard"))?;
                guard
                    .get_trial(trial_id)
                    .map_err(err_to_exceptions)?
                    .study_id
            };
            self.set_category_labels_internal(study_id, name.clone(), labels)?;
        }

        let mut guard = self
            .storage
            .write()
            .map_err(|_| PyRuntimeError::new_err("Failed to acquire the storage guard"))?;
        guard
            .set_trial_param(trial_id, &name, &distribution, value)
            .map_err(err_to_exceptions)?;
        Ok(())
    }

    fn set_category_labels(
        &mut self,
        study_id: u32,
        param_name: String,
        choices: Vec<Py<PyAny>>,
    ) -> PyResult<()> {
        let category_labels = Python::attach(|py| -> PyResult<Vec<CategoryLabel>> {
            let mut labels: Vec<CategoryLabel> = Vec::with_capacity(choices.len());
            for choice in choices {
                let label = pyobject_to_category_label(choice.bind(py))?;
                labels.push(label);
            }
            Ok(labels)
        })?;
        self.set_category_labels_internal(study_id, param_name, category_labels)
    }

    fn get_category_labels(
        &mut self,
        study_id: u32,
        param_name: String,
        cardinality: usize,
    ) -> PyResult<Py<PyAny>> {
        let mut guard = self
            .storage
            .write()
            .map_err(|_| PyRuntimeError::new_err("Failed to acquire the storage guard"))?;
        Python::attach(|py| {
            match guard
                .get_category_labels(study_id, &param_name, cardinality)
                .map_err(err_to_exceptions)?
            {
                Some(labels) => {
                    let elements: PyResult<Vec<_>> = (0..cardinality)
                        .map(|i| {
                            let c = labels.get(i).ok_or(PyValueError::new_err(
                                "Internal representation of categorical value is out of range",
                            ))?;
                            category_label_to_pyobject(py, c)
                        })
                        .collect();
                    let choices = PyList::new(py, elements?)?;
                    Ok(choices.unbind().into_any())
                }
                None => Ok(py.None()),
            }
        })
    }

    #[pyo3(signature = (trial_id, state, values=None))]
    fn set_trial_state_values(
        &mut self,
        trial_id: u32,
        state: PyTrialState,
        values: Option<Vec<f64>>,
    ) -> PyResult<()> {
        let mut guard = self
            .storage
            .write()
            .map_err(|_| PyRuntimeError::new_err("Failed to acquire the storage guard"))?;

        let state_values = match state {
            PyTrialState::COMPLETE => {
                let values = values.ok_or(PyValueError::new_err(
                    "values must be specified when state is COMPLETE",
                ))?;
                TrialStateValues::Complete(values)
            }
            PyTrialState::RUNNING => TrialStateValues::Running,
            PyTrialState::PRUNED => TrialStateValues::Pruned,
            PyTrialState::WAITING => TrialStateValues::Waiting,
            PyTrialState::FAIL => TrialStateValues::Fail,
        };
        guard
            .set_trial_state_values(trial_id, state_values)
            .map_err(err_to_exceptions)?;
        Ok(())
    }

    fn get_studies(&mut self) -> PyResult<Vec<PyPersistedStudy>> {
        let mut guard = self
            .storage
            .write()
            .map_err(|_| PyRuntimeError::new_err("Failed to acquire the storage guard"))?;
        let studies = guard
            .get_studies()
            .map_err(|e| PyRuntimeError::new_err(format!("Failed to get studies: {:?}", e.kind)))?;
        Ok(studies.iter().map(|s| s.clone().into()).collect())
    }

    fn get_study(&mut self, study_id: u32) -> PyResult<PyPersistedStudy> {
        let mut guard = self
            .storage
            .write()
            .map_err(|_| PyRuntimeError::new_err("Failed to acquire the storage guard"))?;
        let study = guard.get_study(study_id).map_err(err_to_exceptions)?;
        Ok(study.clone().into())
    }

    fn get_trials(&mut self, study_id: u32) -> PyResult<Vec<PyPersistedTrial>> {
        let mut guard = self
            .storage
            .write()
            .map_err(|_| PyRuntimeError::new_err("Failed to acquire the storage guard"))?;
        let trials = guard.get_trials(study_id).map_err(err_to_exceptions)?;
        let py_trials: Vec<PyPersistedTrial> = trials
            .iter()
            .map(|t| PyPersistedTrial::from_storage(self.storage.clone(), t))
            .collect();
        Ok(py_trials)
    }

    fn get_trial(&mut self, trial_id: u32) -> PyResult<PyPersistedTrial> {
        let mut guard = self
            .storage
            .write()
            .map_err(|_| PyRuntimeError::new_err("Failed to acquire the storage guard"))?;
        let trial = guard
            .get_trial(trial_id)
            .map_err(err_to_exceptions)?
            .clone();
        let study_attrs = guard
            .get_study(trial.study_id)
            .map_err(err_to_exceptions)?
            .attrs
            .clone();
        Ok(PyPersistedTrial::new(trial, study_attrs))
    }

    fn get_trial_id_from_study_id_trial_number(
        &mut self,
        study_id: u32,
        trial_number: u32,
    ) -> PyResult<u32> {
        let mut guard = self
            .storage
            .write()
            .map_err(|_| PyRuntimeError::new_err("Failed to acquire the storage guard"))?;
        let trial_id = guard
            .get_trial_id_from_study_id_trial_number(study_id, trial_number)
            .map_err(err_to_exceptions)?;
        Ok(trial_id)
    }

    fn set_study_system_attrs(&mut self, study_id: u32, attrs: Py<PyAny>) -> PyResult<()> {
        let system_attrs = Python::attach(|py| {
            let attrs = attrs.bind(py);
            pyobj_to_system_attrs(attrs)
        })?;
        let mut guard = self
            .storage
            .write()
            .map_err(|_| PyRuntimeError::new_err("Failed to acquire the storage guard"))?;
        guard
            .set_study_attrs(study_id, system_attrs, false)
            .map_err(err_to_exceptions)?;
        Ok(())
    }

    fn set_study_user_attrs(&mut self, study_id: u32, attrs: Py<PyAny>) -> PyResult<()> {
        let user_attrs = Python::attach(|py| {
            let attrs = attrs.bind(py);
            pyobj_to_user_attrs(attrs)
        })?;
        let mut guard = self
            .storage
            .write()
            .map_err(|_| PyRuntimeError::new_err("Failed to acquire the storage guard"))?;
        guard
            .set_study_attrs(study_id, user_attrs, false)
            .map_err(err_to_exceptions)?;
        Ok(())
    }

    fn set_trial_system_attrs(&mut self, trial_id: u32, attrs: Py<PyAny>) -> PyResult<()> {
        let system_attrs = Python::attach(|py| {
            let attrs = attrs.bind(py);
            pyobj_to_system_attrs(attrs)
        })?;
        let mut guard = self
            .storage
            .write()
            .map_err(|_| PyRuntimeError::new_err("Failed to acquire the storage guard"))?;
        guard
            .set_trial_attrs(trial_id, system_attrs, false)
            .map_err(err_to_exceptions)?;
        Ok(())
    }

    fn set_trial_user_attrs(&mut self, trial_id: u32, attrs: Py<PyAny>) -> PyResult<()> {
        let user_attrs = Python::attach(|py| {
            let attrs = attrs.bind(py);
            pyobj_to_user_attrs(attrs)
        })?;
        let mut guard = self
            .storage
            .write()
            .map_err(|_| PyRuntimeError::new_err("Failed to acquire the storage guard"))?;
        guard
            .set_trial_attrs(trial_id, user_attrs, false)
            .map_err(err_to_exceptions)?;
        Ok(())
    }

    fn set_trial_intermediate_value(
        &mut self,
        trial_id: u32,
        step: u32,
        intermediate_value: f64,
    ) -> PyResult<()> {
        let mut guard = self
            .storage
            .write()
            .map_err(|_| PyRuntimeError::new_err("Failed to acquire the storage guard"))?;
        let mut intermediate_values = std::collections::HashMap::new();
        intermediate_values.insert(step, intermediate_value);
        guard
            .set_trial_intermediate_values(trial_id, intermediate_values)
            .map_err(err_to_exceptions)?;
        Ok(())
    }
}

impl PyStorage {
    fn set_category_labels_internal(
        &mut self,
        study_id: u32,
        param_name: String,
        category_labels: Vec<CategoryLabel>,
    ) -> PyResult<()> {
        let mut guard = self
            .storage
            .write()
            .map_err(|_| PyRuntimeError::new_err("Failed to acquire the storage guard"))?;
        match guard.set_category_labels(study_id, &param_name, category_labels.clone()) {
            Ok(_) => Ok(()),
            Err(e) => {
                if matches!(e.kind, rustuna_core::ErrorKind::AttrOverwriteNotAllowed) {
                    let existing_labels = guard
                        .get_category_labels(study_id, &param_name, category_labels.len())
                        .map_err(err_to_exceptions)?;
                    if let Some(existing) = existing_labels {
                        if existing == category_labels {
                            return Ok(());
                        }
                    }
                    return Err(PyValueError::new_err(format!(
                        "Cannot overwrite category labels for parameter '{param_name}'"
                    )));
                }
                Err(err_to_exceptions(e))
            }
        }
    }
}
