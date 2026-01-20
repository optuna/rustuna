use std::collections::HashMap;

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;

use pyo3::types::PyList;
use rustuna_core::attr::{AttrKey, Attrs, CategoryLabel};
use rustuna_core::distribution::Distribution;
use rustuna_core::storage::{InMemoryStorage, Storage};
use rustuna_core::study::{Direction, PersistedStudy};
use rustuna_core::trial::{PersistedTrial, TrialStateValues};

use crate::distribution::PyDistribution;
use crate::study::{pyobject_to_persisted_study, PyDirection};
use crate::trial::{pyobject_to_persisted_trial, PyTrialState};

pub struct PyObjectStorage {
    obj: Py<PyAny>,
    is_distributed: bool,
    cache: InMemoryStorage,
    cache_study_to_src_study: HashMap<u32, u32>,
    src_study_to_cache_study: HashMap<u32, u32>,
}
impl PyObjectStorage {
    pub fn new(obj: Py<PyAny>, is_distributed: bool) -> Self {
        PyObjectStorage {
            obj,
            is_distributed,
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
                let cache_study = self
                    .cache
                    .create_new_study(&src_study.name.clone(), src_study.directions.clone())?;
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

    fn obj_create_new_trial(&mut self, study_id: u32) -> PyResult<PersistedTrial> {
        Python::attach(|py| {
            let py_trial = self.obj.call_method1(py, "create_new_trial", (study_id,))?;
            let py_trial = py_trial.bind(py);
            pyobject_to_persisted_trial(py_trial, study_id)
        })
    }

    fn obj_set_trial_param(
        &mut self,
        trial_id: u32,
        name: &str,
        distribution: &Distribution,
        value: f64,
    ) -> PyResult<()> {
        Python::attach(|py| {
            // TODO(c-bata): Consider how to set category_labels.
            let attrs = Attrs::new();
            let py_distribution = PyDistribution::new(distribution.clone(), name, &attrs);
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

    fn obj_set_trial_intermediate_value(
        &mut self,
        trial_id: u32,
        step: u32,
        intermediate_value: f64,
    ) -> PyResult<()> {
        Python::attach(|py| {
            self.obj.call_method1(
                py,
                "set_trial_intermediate_value",
                (trial_id, step, intermediate_value),
            )?;
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

    fn obj_get_trials(&self, study_id: u32) -> PyResult<Vec<PersistedTrial>> {
        Python::attach(|py| {
            let trials = self.obj.call_method1(py, "get_trials", (study_id,))?;
            let trials_ref = trials.bind(py);
            if !trials_ref.is_instance_of::<PyList>() {
                return Err(PyRuntimeError::new_err("studies must be a list"));
            }
            let trials = trials_ref.cast::<PyList>()?;
            let mut persisted_trials: Vec<PersistedTrial> = Vec::with_capacity(trials.len());
            for trial in trials.iter() {
                persisted_trials.push(pyobject_to_persisted_trial(&trial, study_id)?);
            }
            Ok(persisted_trials)
        })
    }

    pub fn sync_studies(&mut self, sync_attrs: bool) -> rustuna_core::Result<()> {
        let studies = self
            .obj_get_studies()
            .map_err(|_| rustuna_core::Error::new(rustuna_core::ErrorKind::StorageError))?;
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
                    let src_study = self.obj_get_study(*src_study_id).map_err(|_| {
                        rustuna_core::Error::new(rustuna_core::ErrorKind::StorageError)
                    })?;
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
        cache_n_trials: Option<u32>,
    ) -> rustuna_core::Result<()> {
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

        if !src_trial.intermediate_values.is_empty() {
            self.cache
                .set_trial_intermediate_values(src_trial.id, src_trial.intermediate_values)?;
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
            .map_err(|_| rustuna_core::Error::new(rustuna_core::ErrorKind::StorageError))?;
        src_trials.sort_by_key(|trial| trial.number);

        let mut cache_n_trials = self.cache.get_trials(cache_study_id)?.len() as u32;
        for src_trial in src_trials {
            self.sync_trial(cache_study_id, src_trial, Some(cache_n_trials))?;
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
}
impl Storage for PyObjectStorage {
    fn create_new_study(
        &mut self,
        study_name: &str,
        directions: Vec<Direction>,
    ) -> rustuna_core::Result<&PersistedStudy> {
        let src_study_id = self
            .obj_create_new_study(study_name, &directions)
            .map_err(|_| rustuna_core::Error::new(rustuna_core::ErrorKind::StorageError))?;
        let cache_study = self.cache.create_new_study(study_name, directions)?;
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
            .map_err(|_| rustuna_core::Error::new(rustuna_core::ErrorKind::StorageError))?;
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
        let src_trial = self
            .obj_create_new_trial(*src_study_id)
            .map_err(|_| rustuna_core::Error::new(rustuna_core::ErrorKind::StorageError))?;
        let src_trial_id = src_trial.id;
        if self.is_distributed {
            self.sync_trials(study_id)?;
            return self.cache.get_trial(src_trial_id);
        }
        let cached_n_trials = self.cache.get_trials(study_id)?.len() as u32;
        if src_trial.number != cached_n_trials {
            self.sync_trials(study_id)?;
            return self.cache.get_trial(src_trial_id);
        }
        self.sync_trial(study_id, src_trial, Some(cached_n_trials))?;
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
            .map_err(|_| rustuna_core::Error::new(rustuna_core::ErrorKind::StorageError))?;
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
            .map_err(|_| rustuna_core::Error::new(rustuna_core::ErrorKind::StorageError))?;

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
        if intermediate_values.is_empty() {
            return Ok(());
        }
        let mut steps: Vec<u32> = intermediate_values.keys().copied().collect();
        steps.sort_unstable();
        for step in steps {
            let value = intermediate_values
                .get(&step)
                .ok_or(rustuna_core::Error::new(
                    rustuna_core::ErrorKind::StorageError,
                ))?;
            self.obj_set_trial_intermediate_value(trial_id, step, *value)
                .map_err(|_| rustuna_core::Error::new(rustuna_core::ErrorKind::StorageError))?;
        }

        let retry_values = intermediate_values.clone();
        match self
            .cache
            .set_trial_intermediate_values(trial_id, intermediate_values)
        {
            Ok(_) => Ok(()),
            Err(e) => match e.kind {
                rustuna_core::ErrorKind::StudyNotFound | rustuna_core::ErrorKind::TrialNotFound => {
                    self.sync_all_trials()?;
                    self.cache
                        .set_trial_intermediate_values(trial_id, retry_values)?;
                    Ok(())
                }
                _ => Err(e),
            },
        }
    }

    fn get_studies(&mut self) -> rustuna_core::Result<&Vec<rustuna_core::study::PersistedStudy>> {
        self.cache.get_studies()
    }

    fn get_study(
        &mut self,
        study_id: u32,
    ) -> rustuna_core::Result<&rustuna_core::study::PersistedStudy> {
        self.cache.get_study(study_id)
    }

    fn get_trials(
        &mut self,
        study_id: u32,
    ) -> rustuna_core::Result<&Vec<rustuna_core::trial::PersistedTrial>> {
        self.cache.get_trials(study_id)
    }

    fn get_trial(
        &mut self,
        trial_id: u32,
    ) -> rustuna_core::Result<&rustuna_core::trial::PersistedTrial> {
        self.cache.get_trial(trial_id)
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
            .map_err(|_| rustuna_core::Error::new(rustuna_core::ErrorKind::StorageError))?;
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
            .map_err(|_| rustuna_core::Error::new(rustuna_core::ErrorKind::StorageError))?;
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
        self.sync_study_from_id(study_id, false)?;
        self.cache.get_joint_search_space(study_id)
    }
}
