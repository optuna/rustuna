use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

use pyo3::types::{PyDict, PyList};
use rustuna_core::attr::{AttrKey, Attrs, CategoryLabel};
use rustuna_core::distribution::Distribution;
use rustuna_core::storage::{InMemoryStorage, Storage};
use rustuna_core::study::{Direction, PersistedStudy};
use rustuna_core::trial::{PersistedTrial, TrialState, TrialStateValues};
use rustuna_core::{Error, ErrorKind};

use crate::distribution::PyDistribution;
use crate::exception::err_to_exceptions;
use crate::study::{pyobject_to_persisted_study, PyDirection};
use crate::trial::{
    pyobject_to_persisted_trial_with_category_labels, PyPersistedTrial, PyTrialState,
};

// ToRustStorage wraps an Optuna storage object and maintains an in-memory cache.
//
// When using Optuna's API (e.g., `study.optimize()` with `ToOptunaSampler`), write operations
// (such as `set_trial_state_values`) are performed directly on the Optuna storage, bypassing
// ToRustStorage's write methods. Therefore, the cache is not updated during these operations.
//
// To ensure consistency, all `get_*` methods must synchronize with the backend storage before
// returning cached data.
pub struct ToRustStorage {
    obj: Py<PyAny>,
    cache: InMemoryStorage,
    cache_study_to_src_study: HashMap<u32, u32>,
    src_study_to_cache_study: HashMap<u32, u32>,
}
impl ToRustStorage {
    pub fn new(obj: Py<PyAny>) -> Self {
        ToRustStorage {
            obj,
            cache: InMemoryStorage::new(),
            cache_study_to_src_study: HashMap::new(),
            src_study_to_cache_study: HashMap::new(),
        }
    }

    fn register_src_study_to_cache(
        &mut self,
        src_study: PersistedStudy,
        sync_attrs: bool,
    ) -> rustuna_core::Result<()> {
        let cache_study_id = match self.src_study_to_cache_study.get(&src_study.id) {
            Some(cache_study_id) => Ok(*cache_study_id),
            None => {
                let cache_study = self.cache.insert_study_with_id(
                    src_study.id,
                    &src_study.name,
                    src_study.directions.clone(),
                )?;
                self.cache_study_to_src_study
                    .insert(cache_study.id, src_study.id);
                self.src_study_to_cache_study
                    .insert(src_study.id, cache_study.id);
                Ok(cache_study.id)
            }
        }?;
        if sync_attrs {
            self.cache
                .set_study_attrs(cache_study_id, src_study.attrs.clone(), false)?;
        }
        Ok(())
    }

    fn obj_create_new_study(
        &mut self,
        study_name: &str,
        directions: &[Direction],
    ) -> PyResult<u32> {
        Python::attach(|py| {
            let py_directions: Vec<PyDirection> =
                directions.iter().map(|d| d.clone().into()).collect();
            let py_study =
                self.obj
                    .call_method1(py, "create_new_study", (study_name, py_directions))?;
            let study_id = py_study.getattr(py, "id")?.extract::<u32>(py)?;
            Ok(study_id)
        })
    }

    fn obj_delete_study(&mut self, study_id: u32) -> PyResult<()> {
        Python::attach(|py| {
            self.obj.call_method1(py, "delete_study", (study_id,))?;
            Ok(())
        })
    }

    fn obj_create_new_trial(&mut self, study_id: u32) -> PyResult<(PersistedTrial, Attrs)> {
        Python::attach(|py| {
            let py_trial = self.obj.call_method1(py, "create_new_trial", (study_id,))?;
            let py_trial = py_trial.bind(py);
            pyobject_to_persisted_trial_with_category_labels(py_trial, study_id)
        })
    }

    fn obj_create_new_trial_from_template(
        &mut self,
        study_id: u32,
        template: &PersistedTrial,
    ) -> PyResult<(PersistedTrial, Attrs)> {
        Python::attach(|py| {
            let py_template = Py::new(py, PyPersistedTrial::new(template.clone(), Attrs::new()))?;
            let py_trial =
                self.obj
                    .call_method1(py, "create_new_trial", (study_id, py_template))?;
            let py_trial = py_trial.bind(py);
            pyobject_to_persisted_trial_with_category_labels(py_trial, study_id)
        })
    }

    fn obj_set_trial_param(
        &mut self,
        trial_id: u32,
        name: &str,
        distribution: &Distribution,
        value: f64,
    ) -> PyResult<()> {
        let category_labels = match distribution {
            Distribution::Categorical { cardinality } => {
                let study_id = self
                    .cache
                    .get_cached_trial(trial_id)
                    .map_err(|e| {
                        PyRuntimeError::new_err(format!(
                            "Failed to find trial while setting categorical parameter: {:?}",
                            e.kind
                        ))
                    })?
                    .study_id;
                let labels = self
                    .cache
                    .get_category_labels(study_id, name, *cardinality)
                    .map_err(|e| {
                        PyRuntimeError::new_err(format!(
                            "Failed to get categorical labels for parameter '{name}': {:?}",
                            e.kind
                        ))
                    })?
                    .ok_or_else(|| {
                        PyRuntimeError::new_err(format!(
                            "Categorical labels are not available for parameter '{name}'"
                        ))
                    })?;
                Some(labels)
            }
            _ => None,
        };
        Python::attach(|py| {
            let py_distribution =
                PyDistribution::new_with_category_labels(distribution.clone(), category_labels);
            self.obj.call_method1(
                py,
                "set_trial_param",
                (trial_id, name, py_distribution, value),
            )?;
            Ok(())
        })
    }

    fn obj_set_trial_state_values(
        &mut self,
        trial_id: u32,
        state: &TrialStateValues,
    ) -> PyResult<()> {
        Python::attach(|py| {
            let values = match state {
                TrialStateValues::Complete(ref values) => Some(values.clone()),
                _ => None,
            };
            let py_state = PyTrialState::from(state.clone());
            self.obj
                .call_method1(py, "set_trial_state_values", (trial_id, py_state, values))?;
            Ok(())
        })
    }

    fn obj_set_study_attrs(&mut self, study_id: u32, attrs: Attrs) -> PyResult<()> {
        Python::attach(|py| {
            let py_system_attrs = pyo3::types::PyDict::new(py);
            let py_user_attrs = pyo3::types::PyDict::new(py);
            for (k, v) in attrs.into_iter() {
                match k {
                    AttrKey::System(k) => {
                        py_system_attrs.set_item(k.as_str(), v)?;
                    }
                    AttrKey::User(k) => {
                        py_user_attrs.set_item(k.as_str(), v)?;
                    }
                }
            }
            self.obj
                .call_method1(py, "set_study_system_attrs", (study_id, py_system_attrs))?;
            self.obj
                .call_method1(py, "set_study_user_attrs", (study_id, py_user_attrs))?;
            Ok(())
        })
    }

    fn obj_set_trial_intermediate_values(
        &mut self,
        trial_id: u32,
        intermediate_values: &HashMap<u32, f64>,
    ) -> PyResult<()> {
        Python::attach(|py| {
            for (step, value) in intermediate_values {
                self.obj.call_method1(
                    py,
                    "set_trial_intermediate_value",
                    (trial_id, *step, *value),
                )?;
            }
            Ok(())
        })
    }

    fn obj_set_trial_constraints(
        &mut self,
        trial_id: u32,
        constraints: &HashMap<String, f64>,
    ) -> PyResult<()> {
        Python::attach(|py| {
            self.obj
                .call_method1(py, "set_trial_constraints", (trial_id, constraints.clone()))?;
            Ok(())
        })
    }

    fn obj_set_trial_attrs(&mut self, trial_id: u32, attrs: Attrs) -> PyResult<()> {
        Python::attach(|py| {
            let py_system_attrs = pyo3::types::PyDict::new(py);
            let py_user_attrs = pyo3::types::PyDict::new(py);
            for (k, v) in attrs.into_iter() {
                match k {
                    AttrKey::System(k) => {
                        py_system_attrs.set_item(k.as_str(), v)?;
                    }
                    AttrKey::User(k) => {
                        py_user_attrs.set_item(k.as_str(), v)?;
                    }
                }
            }
            self.obj
                .call_method1(py, "set_trial_system_attrs", (trial_id, py_system_attrs))?;
            self.obj
                .call_method1(py, "set_trial_user_attrs", (trial_id, py_user_attrs))?;
            Ok(())
        })
    }

    fn obj_get_study(&self, study_id: u32) -> PyResult<PersistedStudy> {
        Python::attach(|py| {
            let study = self.obj.call_method1(py, "get_study", (study_id,))?;
            let study = study.bind(py);
            pyobject_to_persisted_study(study)
        })
    }

    fn obj_get_studies(&self) -> PyResult<Vec<PersistedStudy>> {
        Python::attach(|py| {
            let studies = self.obj.call_method1(py, "get_studies", ())?;
            let studies_ref = studies.bind(py);
            if !studies_ref.is_instance_of::<PyList>() {
                return Err(PyRuntimeError::new_err("studies must be a list"));
            }
            let studies = studies_ref.cast::<PyList>()?;
            let mut persisted_studies: Vec<PersistedStudy> = Vec::with_capacity(studies.len());
            for study in studies.iter() {
                persisted_studies.push(pyobject_to_persisted_study(&study)?);
            }
            Ok(persisted_studies)
        })
    }

    fn obj_get_trials(&self, study_id: u32) -> PyResult<Vec<(PersistedTrial, Attrs)>> {
        Python::attach(|py| {
            let trials = self.obj.call_method1(py, "get_trials", (study_id,))?;
            let trials_ref = trials.bind(py);
            if !trials_ref.is_instance_of::<PyList>() {
                return Err(PyRuntimeError::new_err("studies must be a list"));
            }
            let trials = trials_ref.cast::<PyList>()?;
            let mut persisted_trials: Vec<(PersistedTrial, Attrs)> =
                Vec::with_capacity(trials.len());
            for trial in trials.iter() {
                persisted_trials.push(pyobject_to_persisted_trial_with_category_labels(
                    &trial, study_id,
                )?);
            }
            Ok(persisted_trials)
        })
    }

    fn obj_get_n_trials(&self, study_id: u32, states: Option<&[TrialState]>) -> PyResult<u32> {
        Python::attach(|py| {
            let kwargs = PyDict::new(py);
            if let Some(states) = states {
                let states: Vec<PyTrialState> =
                    states.iter().copied().map(PyTrialState::from).collect();
                kwargs.set_item("states", states)?;
            }
            self.obj
                .call_method(py, "get_n_trials", (study_id,), Some(&kwargs))?
                .extract(py)
        })
    }

    pub fn sync_studies(&mut self, sync_attrs: bool) -> rustuna_core::Result<()> {
        let studies = self.obj_get_studies().map_err(Self::map_pyerr)?;
        for src_study in studies {
            self.register_src_study_to_cache(src_study, sync_attrs)?
        }
        Ok(())
    }

    fn sync_study_from_id(
        &mut self,
        cache_study_id: u32,
        sync_attrs: bool,
    ) -> rustuna_core::Result<()> {
        match self.cache_study_to_src_study.get(&cache_study_id) {
            Some(src_study_id) => {
                if sync_attrs {
                    let src_study = self.obj_get_study(*src_study_id).map_err(Self::map_pyerr)?;
                    self.cache
                        .set_study_attrs(cache_study_id, src_study.attrs, false)?;
                }
                Ok(())
            }
            None => self.sync_studies(sync_attrs),
        }
    }

    fn sync_trial(
        &mut self,
        cache_study_id: u32,
        src_trial: PersistedTrial,
        category_attrs: Attrs,
        cache_n_trials: Option<u32>,
    ) -> rustuna_core::Result<()> {
        if !category_attrs.is_empty() {
            self.cache
                .set_study_attrs(cache_study_id, category_attrs, false)?;
        }
        let cache_n_trials = match cache_n_trials {
            Some(n) => n,
            None => self.cache.get_trials(cache_study_id)?.len() as u32,
        };
        match src_trial.number.cmp(&cache_n_trials) {
            std::cmp::Ordering::Equal => {
                self.cache
                    .insert_trial_with_id(cache_study_id, src_trial.id, src_trial.number)?;
            }
            std::cmp::Ordering::Greater => {
                return Err(rustuna_core::Error::new(
                    rustuna_core::ErrorKind::StorageError,
                ));
            }
            std::cmp::Ordering::Less => {
                let cache_trial_id = self
                    .cache
                    .get_trial_id_from_study_id_trial_number(cache_study_id, src_trial.number)?;
                if cache_trial_id != src_trial.id {
                    return Err(rustuna_core::Error::new(
                        rustuna_core::ErrorKind::StorageError,
                    ));
                }
            }
        }

        let cache_trial = self.cache.get_trial(src_trial.id)?.clone();
        if cache_trial.is_finished() {
            return Ok(());
        }
        self.cache
            .set_trial_attrs(src_trial.id, src_trial.attrs, false)?;
        if cache_trial.constraints != src_trial.constraints {
            self.cache
                .set_trial_constraints(src_trial.id, src_trial.constraints.clone())?;
        }

        for (name, distribution) in src_trial.distributions {
            let internal_repr =
                src_trial
                    .internal_params
                    .get(&name)
                    .ok_or(rustuna_core::Error::new(
                        rustuna_core::ErrorKind::StorageError,
                    ))?;
            if cache_trial.distributions.contains_key(&name) {
                continue;
            }
            self.cache
                .set_trial_param(src_trial.id, &name, &distribution, *internal_repr)?;
        }

        if cache_trial.state_values != src_trial.state_values {
            self.cache
                .set_trial_state_values(src_trial.id, src_trial.state_values.clone())?;
        }
        Ok(())
    }

    fn sync_trials(&mut self, cache_study_id: u32) -> rustuna_core::Result<()> {
        let src_study_id =
            self.cache_study_to_src_study
                .get(&cache_study_id)
                .ok_or(rustuna_core::Error::new(
                    rustuna_core::ErrorKind::StudyNotFound,
                ))?;

        let mut src_trials = self
            .obj_get_trials(*src_study_id)
            .map_err(Self::map_pyerr)?;
        src_trials.sort_by_key(|(trial, _)| trial.number);

        let mut cache_n_trials = self.cache.get_trials(cache_study_id)?.len() as u32;
        for (src_trial, category_attrs) in src_trials {
            self.sync_trial(
                cache_study_id,
                src_trial,
                category_attrs,
                Some(cache_n_trials),
            )?;
            cache_n_trials = self.cache.get_trials(cache_study_id)?.len() as u32;
        }
        Ok(())
    }

    fn sync_all_trials(&mut self) -> rustuna_core::Result<()> {
        self.sync_studies(false)?;
        let study_ids: Vec<u32> = self.cache.get_studies()?.iter().map(|s| s.id).collect();
        for cache_study_id in study_ids {
            self.sync_trials(cache_study_id)?;
        }
        Ok(())
    }

    fn map_pyerr(err: PyErr) -> Error {
        let reason = err.to_string();
        let class_name = match Python::attach(|py| {
            err.get_type(py)
                .name()
                .map(|name| name.to_string_lossy().into_owned())
        }) {
            Ok(class_name) => class_name,
            Err(type_error) => {
                return Error::with_reason(
                    ErrorKind::StorageError,
                    format!("{reason}; failed to inspect Python exception type: {type_error}"),
                )
            }
        };
        let kind = if class_name == "DuplicatedStudyError" {
            ErrorKind::DuplicatedStudy
        } else {
            ErrorKind::StorageError
        };
        Error::with_reason(kind, reason)
    }
}
impl Storage for ToRustStorage {
    fn create_new_study(
        &mut self,
        study_name: &str,
        directions: Vec<Direction>,
    ) -> rustuna_core::Result<&PersistedStudy> {
        let src_study_id = self
            .obj_create_new_study(study_name, &directions)
            .map_err(Self::map_pyerr)?;
        let cache_study = self
            .cache
            .insert_study_with_id(src_study_id, study_name, directions)?;
        self.cache_study_to_src_study
            .insert(cache_study.id, src_study_id);
        self.src_study_to_cache_study
            .insert(src_study_id, cache_study.id);
        Ok(cache_study)
    }

    fn delete_study(&mut self, study_id: u32) -> rustuna_core::Result<()> {
        self.sync_study_from_id(study_id, false)?;
        let src_study_id =
            *self
                .cache_study_to_src_study
                .get(&study_id)
                .ok_or(rustuna_core::Error::new(
                    rustuna_core::ErrorKind::StudyNotFound,
                ))?;
        self.obj_delete_study(src_study_id)
            .map_err(Self::map_pyerr)?;
        self.cache.delete_study(study_id)?;
        self.cache_study_to_src_study.remove(&study_id);
        self.src_study_to_cache_study.remove(&src_study_id);
        Ok(())
    }

    fn create_new_trial(
        &mut self,
        study_id: u32,
    ) -> rustuna_core::Result<&rustuna_core::trial::PersistedTrial> {
        self.sync_study_from_id(study_id, false)?;
        let src_study_id =
            self.cache_study_to_src_study
                .get(&study_id)
                .ok_or(rustuna_core::Error::new(
                    rustuna_core::ErrorKind::StudyNotFound,
                ))?;
        let (src_trial, category_attrs) = self
            .obj_create_new_trial(*src_study_id)
            .map_err(Self::map_pyerr)?;
        let src_trial_id = src_trial.id;
        let cached_n_trials = self.cache.get_trials(study_id)?.len() as u32;
        if src_trial.number != cached_n_trials {
            self.sync_trials(study_id)?;
            return self.cache.get_trial(src_trial_id);
        }
        self.sync_trial(study_id, src_trial, category_attrs, Some(cached_n_trials))?;
        self.cache.get_trial(src_trial_id)
    }

    fn create_new_trial_from_template(
        &mut self,
        study_id: u32,
        template: &PersistedTrial,
    ) -> rustuna_core::Result<&rustuna_core::trial::PersistedTrial> {
        self.sync_study_from_id(study_id, false)?;
        let src_study_id =
            *self
                .cache_study_to_src_study
                .get(&study_id)
                .ok_or(rustuna_core::Error::new(
                    rustuna_core::ErrorKind::StudyNotFound,
                ))?;
        let (src_trial, category_attrs) = self
            .obj_create_new_trial_from_template(src_study_id, template)
            .map_err(Self::map_pyerr)?;
        let src_trial_id = src_trial.id;
        let cached_n_trials = self.cache.get_trials(study_id)?.len() as u32;
        if src_trial.number != cached_n_trials {
            self.sync_trials(study_id)?;
            return self.cache.get_trial(src_trial_id);
        }
        self.sync_trial(study_id, src_trial, category_attrs, Some(cached_n_trials))?;
        self.cache.get_trial(src_trial_id)
    }

    fn set_trial_param(
        &mut self,
        trial_id: u32,
        name: &str,
        distribution: &Distribution,
        value: f64,
    ) -> rustuna_core::Result<()> {
        self.obj_set_trial_param(trial_id, name, distribution, value)
            .map_err(Self::map_pyerr)?;
        match self
            .cache
            .set_trial_param(trial_id, name, distribution, value)
        {
            Ok(_) => Ok(()),
            Err(e) => match e.kind {
                rustuna_core::ErrorKind::StudyNotFound | rustuna_core::ErrorKind::TrialNotFound => {
                    self.sync_all_trials()?;
                    self.cache
                        .set_trial_param(trial_id, name, distribution, value)?;
                    Ok(())
                }
                _ => Err(e),
            },
        }
    }

    fn set_trial_state_values(
        &mut self,
        trial_id: u32,
        state_values: TrialStateValues,
    ) -> rustuna_core::Result<()> {
        let state_values_clone = state_values.clone();
        self.obj_set_trial_state_values(trial_id, &state_values_clone)
            .map_err(Self::map_pyerr)?;

        match self.cache.set_trial_state_values(trial_id, state_values) {
            Ok(_) => Ok(()),
            Err(e) => match e.kind {
                rustuna_core::ErrorKind::StudyNotFound | rustuna_core::ErrorKind::TrialNotFound => {
                    self.sync_all_trials()?;
                    self.cache
                        .set_trial_state_values(trial_id, state_values_clone)?;
                    Ok(())
                }
                _ => Err(e),
            },
        }
    }

    fn set_trial_intermediate_values(
        &mut self,
        trial_id: u32,
        intermediate_values: HashMap<u32, f64>,
    ) -> rustuna_core::Result<()> {
        let intermediate_values_retry = intermediate_values.clone();
        self.obj_set_trial_intermediate_values(trial_id, &intermediate_values_retry)
            .map_err(Self::map_pyerr)?;

        match self
            .cache
            .set_trial_intermediate_values(trial_id, intermediate_values)
        {
            Ok(_) => Ok(()),
            Err(e) => match e.kind {
                rustuna_core::ErrorKind::StudyNotFound | rustuna_core::ErrorKind::TrialNotFound => {
                    self.sync_all_trials()?;
                    self.cache
                        .set_trial_intermediate_values(trial_id, intermediate_values_retry)?;
                    Ok(())
                }
                _ => Err(e),
            },
        }
    }

    fn set_trial_constraints(
        &mut self,
        trial_id: u32,
        constraints: HashMap<String, f64>,
    ) -> rustuna_core::Result<()> {
        let constraints_for_retry = constraints.clone();
        self.obj_set_trial_constraints(trial_id, &constraints_for_retry)
            .map_err(Self::map_pyerr)?;

        match self.cache.set_trial_constraints(trial_id, constraints) {
            Ok(_) => Ok(()),
            Err(e) => match e.kind {
                rustuna_core::ErrorKind::StudyNotFound | rustuna_core::ErrorKind::TrialNotFound => {
                    self.sync_all_trials()?;
                    self.cache
                        .set_trial_constraints(trial_id, constraints_for_retry)?;
                    Ok(())
                }
                _ => Err(e),
            },
        }
    }

    fn get_studies(&mut self) -> rustuna_core::Result<&Vec<rustuna_core::study::PersistedStudy>> {
        self.sync_studies(true)?;
        self.cache.get_studies()
    }

    fn get_study(
        &mut self,
        study_id: u32,
    ) -> rustuna_core::Result<&rustuna_core::study::PersistedStudy> {
        self.sync_study_from_id(study_id, true)?;
        self.cache.get_study(study_id)
    }

    fn get_trials(
        &mut self,
        study_id: u32,
    ) -> rustuna_core::Result<&Vec<Option<rustuna_core::trial::PersistedTrial>>> {
        // Ensure study mapping exists before sync_trials, which requires cache_study_to_src_study.
        self.sync_study_from_id(study_id, false)?;
        self.sync_trials(study_id)?;
        self.cache.get_trials(study_id)
    }

    fn get_n_trials(
        &mut self,
        study_id: u32,
        states: Option<&[TrialState]>,
    ) -> rustuna_core::Result<u32> {
        self.obj_get_n_trials(study_id, states)
            .map_err(Self::map_pyerr)
    }

    fn get_trial(
        &mut self,
        trial_id: u32,
    ) -> rustuna_core::Result<&rustuna_core::trial::PersistedTrial> {
        self.sync_all_trials()?;
        self.cache.get_trial(trial_id)
    }

    fn get_study_attr(&mut self, study_id: u32, key: AttrKey) -> rustuna_core::Result<String> {
        self.sync_study_from_id(study_id, true)?;
        self.cache.get_study_attr(study_id, key)
    }

    fn get_cached_trial(
        &self,
        trial_id: u32,
    ) -> rustuna_core::Result<&rustuna_core::trial::PersistedTrial> {
        self.cache.get_cached_trial(trial_id)
    }

    fn get_category_labels(
        &mut self,
        study_id: u32,
        param_name: &str,
        cardinality: usize,
    ) -> rustuna_core::Result<Option<Vec<CategoryLabel>>> {
        self.sync_study_from_id(study_id, true)?;
        self.cache
            .get_category_labels(study_id, param_name, cardinality)
    }

    fn set_category_labels(
        &mut self,
        study_id: u32,
        param_name: &str,
        labels: Vec<CategoryLabel>,
    ) -> rustuna_core::Result<()> {
        self.cache.set_category_labels(study_id, param_name, labels)
    }

    fn get_trial_id_from_study_id_trial_number(
        &mut self,
        study_id: u32,
        trial_number: u32,
    ) -> rustuna_core::Result<u32> {
        self.sync_study_from_id(study_id, false)?;
        match self
            .cache
            .get_trial_id_from_study_id_trial_number(study_id, trial_number)
        {
            Ok(trial_id) => Ok(trial_id),
            Err(e) => match e.kind {
                rustuna_core::ErrorKind::StudyNotFound | rustuna_core::ErrorKind::TrialNotFound => {
                    self.sync_trials(study_id)?;
                    self.cache
                        .get_trial_id_from_study_id_trial_number(study_id, trial_number)
                }
                _ => Err(e),
            },
        }
    }

    fn set_study_attrs(
        &mut self,
        study_id: u32,
        attrs: rustuna_core::attr::Attrs,
        _error_on_overwrite: bool,
    ) -> rustuna_core::Result<()> {
        // TODO(c-bata): Emit warnings if error_on_overwrite is true, since Optuna storage cannot support it.
        self.sync_study_from_id(study_id, false)?;
        let src_study_id =
            self.cache_study_to_src_study
                .get(&study_id)
                .ok_or(rustuna_core::Error::new(
                    rustuna_core::ErrorKind::StudyNotFound,
                ))?;
        self.obj_set_study_attrs(*src_study_id, attrs.clone())
            .map_err(Self::map_pyerr)?;
        match self.cache.set_study_attrs(study_id, attrs, false) {
            Ok(_) => Ok(()),
            Err(e) => match e.kind {
                rustuna_core::ErrorKind::StudyNotFound => {
                    self.sync_study_from_id(study_id, true)?;
                    Ok(())
                }
                _ => Err(e),
            },
        }
    }

    fn set_trial_attrs(
        &mut self,
        trial_id: u32,
        attrs: rustuna_core::attr::Attrs,
        _error_on_overwrite: bool,
    ) -> rustuna_core::Result<()> {
        // TODO(c-bata): Emit warnings if error_on_overwrite is true, since Optuna storage cannot support it.
        let attrs_for_obj = attrs.clone();
        let attrs_for_retry = attrs.clone();
        self.obj_set_trial_attrs(trial_id, attrs_for_obj)
            .map_err(Self::map_pyerr)?;
        match self.cache.set_trial_attrs(trial_id, attrs, false) {
            Ok(_) => Ok(()),
            Err(e) => match e.kind {
                rustuna_core::ErrorKind::StudyNotFound | rustuna_core::ErrorKind::TrialNotFound => {
                    self.sync_all_trials()?;
                    self.cache
                        .set_trial_attrs(trial_id, attrs_for_retry, false)?;
                    Ok(())
                }
                _ => Err(e),
            },
        }
    }

    fn get_joint_search_space(
        &mut self,
        study_id: u32,
    ) -> rustuna_core::Result<HashMap<String, Distribution>> {
        // Ensure study mapping exists before sync_trials, which requires cache_study_to_src_study.
        self.sync_study_from_id(study_id, false)?;
        self.sync_trials(study_id)?;
        self.cache.get_joint_search_space(study_id)
    }

    fn discard_trials(&mut self, trial_ids: &[u32]) -> rustuna_core::Result<()> {
        Python::attach(|py| {
            self.obj
                .call_method1(py, "discard_trials", (trial_ids.to_vec(),))
                .map_err(Self::map_pyerr)?;
            Ok(())
        })
    }

    fn may_omit_trials(&self) -> bool {
        Python::attach(|py| {
            self.obj
                .call_method0(py, "may_omit_trials")
                .and_then(|ret| ret.extract::<bool>(py))
                .unwrap_or(false)
        })
    }
}

#[derive(Clone)]
#[pyclass(name = "ToRustStorage")]
#[pyo3(module = "rustuna")]
pub struct PyToRustStorage {
    pub storage: Arc<RwLock<ToRustStorage>>,
}

#[pymethods]
impl PyToRustStorage {
    #[new]
    fn new(storage: Py<PyAny>) -> PyResult<Self> {
        Python::attach(|_py| {
            let mut inner = ToRustStorage::new(storage);
            inner.sync_studies(true).map_err(err_to_exceptions)?;
            Ok(PyToRustStorage {
                storage: Arc::new(RwLock::new(inner)),
            })
        })
    }

    #[pyo3(signature = (study_id, *, states = None))]
    fn get_trials(
        &mut self,
        study_id: u32,
        states: Option<Vec<PyTrialState>>,
    ) -> PyResult<Vec<PyPersistedTrial>> {
        let storage: Arc<RwLock<dyn Storage>> = self.storage.clone();
        let mut guard = storage.write().map_err(|e| {
            PyRuntimeError::new_err(format!("Failed to acquire the storage guard: {e:?}"))
        })?;
        let trials = guard.get_trials(study_id).map_err(err_to_exceptions)?;
        let py_trials: Vec<PyPersistedTrial> = trials
            .iter()
            .flatten()
            .filter(|trial| match &states {
                Some(s) => s.contains(&PyTrialState::from(trial.state_values.clone())),
                None => true,
            })
            .map(|t| PyPersistedTrial::from_storage(storage.clone(), t))
            .collect();
        Ok(py_trials)
    }

    #[pyo3(signature = (study_id, *, states = None))]
    fn get_n_trials(&mut self, study_id: u32, states: Option<Vec<PyTrialState>>) -> PyResult<u32> {
        let storage: Arc<RwLock<dyn Storage>> = self.storage.clone();
        let mut guard = storage.write().map_err(|e| {
            PyRuntimeError::new_err(format!("Failed to acquire the storage guard: {e:?}"))
        })?;
        let states =
            states.map(|states| states.into_iter().map(TrialState::from).collect::<Vec<_>>());
        guard
            .get_n_trials(study_id, states.as_deref())
            .map_err(err_to_exceptions)
    }
}
