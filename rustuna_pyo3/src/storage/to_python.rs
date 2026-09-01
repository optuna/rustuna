use std::sync::{Arc, RwLock};

use pyo3::prelude::*;

use rustuna_core::storage::{InMemoryStorage, Storage};

use crate::distribution::PyDistribution;
use crate::storage::binding::StorageBinding;
use crate::study::{PyDirection, PyPersistedStudy};
use crate::trial::{PyPersistedTrial, PyTrialState};

#[derive(Clone)]
#[pyclass(name = "ToPythonStorage", skip_from_py_object)]
#[pyo3(module = "rustuna")]
pub struct ToPythonStorage {
    pub(crate) binding: StorageBinding,
}

impl ToPythonStorage {
    pub fn new(storage: Arc<RwLock<dyn Storage>>) -> Self {
        Self {
            binding: StorageBinding { storage },
        }
    }

    pub fn storage(&self) -> Arc<RwLock<dyn Storage>> {
        self.binding.storage.clone()
    }
}

#[pymethods]
impl ToPythonStorage {
    #[new]
    fn py_new() -> PyResult<Self> {
        let binding = StorageBinding::new(Arc::new(RwLock::new(InMemoryStorage::new())));
        Ok(ToPythonStorage { binding })
    }

    fn create_new_study(
        &self,
        py: Python<'_>,
        study_name: String,
        directions: Vec<PyDirection>,
    ) -> PyResult<PyPersistedStudy> {
        self.binding.create_new_study(py, study_name, directions)
    }

    fn delete_study(&self, py: Python<'_>, study_id: u32) -> PyResult<()> {
        self.binding.delete_study(py, study_id)
    }

    #[pyo3(signature = (study_id, template_trial=None))]
    fn create_new_trial(
        &self,
        py: Python<'_>,
        study_id: u32,
        template_trial: Option<&Bound<'_, PyPersistedTrial>>,
    ) -> PyResult<PyPersistedTrial> {
        self.binding.create_new_trial(py, study_id, template_trial)
    }

    fn set_trial_param(
        &self,
        py: Python<'_>,
        trial_id: u32,
        name: String,
        distribution: PyDistribution,
        value: f64,
    ) -> PyResult<()> {
        self.binding
            .set_trial_param(py, trial_id, name, distribution, value)
    }

    fn set_category_labels(
        &self,
        py: Python<'_>,
        study_id: u32,
        param_name: String,
        choices: Vec<Py<PyAny>>,
    ) -> PyResult<()> {
        self.binding
            .set_category_labels(py, study_id, param_name, choices)
    }

    fn get_category_labels(
        &self,
        py: Python<'_>,
        study_id: u32,
        param_name: String,
        cardinality: usize,
    ) -> PyResult<Py<PyAny>> {
        self.binding
            .get_category_labels(py, study_id, param_name, cardinality)
    }

    #[pyo3(signature = (trial_id, state, values=None))]
    fn set_trial_state_values(
        &self,
        py: Python<'_>,
        trial_id: u32,
        state: PyTrialState,
        values: Option<Vec<f64>>,
    ) -> PyResult<()> {
        self.binding
            .set_trial_state_values(py, trial_id, state, values)
    }

    fn get_studies(&self, py: Python<'_>) -> PyResult<Vec<PyPersistedStudy>> {
        self.binding.get_studies(py)
    }

    fn get_study(&self, py: Python<'_>, study_id: u32) -> PyResult<PyPersistedStudy> {
        self.binding.get_study(py, study_id)
    }

    #[pyo3(signature = (study_id, *, states = None))]
    fn get_trials(
        &self,
        py: Python<'_>,
        study_id: u32,
        states: Option<Vec<PyTrialState>>,
    ) -> PyResult<Vec<PyPersistedTrial>> {
        self.binding.get_trials(py, study_id, states)
    }

    #[pyo3(signature = (study_id, *, states = None))]
    fn get_n_trials(
        &self,
        py: Python<'_>,
        study_id: u32,
        states: Option<Vec<PyTrialState>>,
    ) -> PyResult<u32> {
        self.binding.get_n_trials(py, study_id, states)
    }

    fn get_trial(&self, py: Python<'_>, trial_id: u32) -> PyResult<PyPersistedTrial> {
        self.binding.get_trial(py, trial_id)
    }

    fn get_cached_trial(&self, py: Python<'_>, trial_id: u32) -> PyResult<PyPersistedTrial> {
        self.binding.get_cached_trial(py, trial_id)
    }

    fn get_trial_number_from_id(&self, py: Python<'_>, trial_id: u32) -> PyResult<u32> {
        self.binding.get_trial_number_from_id(py, trial_id)
    }

    fn get_study_user_attr(&self, py: Python<'_>, study_id: u32, key: String) -> PyResult<String> {
        self.binding.get_study_user_attr(py, study_id, key)
    }

    fn get_study_system_attr(
        &self,
        py: Python<'_>,
        study_id: u32,
        key: String,
    ) -> PyResult<String> {
        self.binding.get_study_system_attr(py, study_id, key)
    }

    fn get_trial_id_from_study_id_trial_number(
        &self,
        py: Python<'_>,
        study_id: u32,
        trial_number: u32,
    ) -> PyResult<u32> {
        self.binding
            .get_trial_id_from_study_id_trial_number(py, study_id, trial_number)
    }

    fn set_study_system_attrs(
        &self,
        py: Python<'_>,
        study_id: u32,
        attrs: Py<PyAny>,
    ) -> PyResult<()> {
        self.binding.set_study_system_attrs(py, study_id, attrs)
    }

    fn set_study_user_attrs(
        &self,
        py: Python<'_>,
        study_id: u32,
        attrs: Py<PyAny>,
    ) -> PyResult<()> {
        self.binding.set_study_user_attrs(py, study_id, attrs)
    }

    fn set_trial_system_attrs(
        &self,
        py: Python<'_>,
        trial_id: u32,
        attrs: Py<PyAny>,
    ) -> PyResult<()> {
        self.binding.set_trial_system_attrs(py, trial_id, attrs)
    }

    fn set_trial_user_attrs(
        &self,
        py: Python<'_>,
        trial_id: u32,
        attrs: Py<PyAny>,
    ) -> PyResult<()> {
        self.binding.set_trial_user_attrs(py, trial_id, attrs)
    }

    fn set_trial_intermediate_value(
        &self,
        py: Python<'_>,
        trial_id: u32,
        step: u32,
        intermediate_value: f64,
    ) -> PyResult<()> {
        self.binding
            .set_trial_intermediate_value(py, trial_id, step, intermediate_value)
    }

    fn discard_trials(&self, py: Python<'_>, trial_ids: Vec<u32>) -> PyResult<()> {
        self.binding.discard_trials(py, trial_ids)
    }

    fn may_omit_trials(&self, py: Python<'_>) -> PyResult<bool> {
        self.binding.may_omit_trials(py)
    }
}
