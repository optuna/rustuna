use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

use rustuna_core::attr::{get_category_labels, AttrKey, Attrs, CategoryLabel};
use rustuna_core::distribution::Distribution;
use rustuna_core::storage::Storage;
use rustuna_core::study::{Direction, PersistedStudy};
use rustuna_core::trial::{PersistedTrial, TrialStateValues};
use rustuna_core::{Error, ErrorKind};
use rustuna_storage::cache::{CachedStorage, CachedStorageBackend, DiscardedTrialsDiff};

use crate::distribution::pyobject_to_category_label;
use crate::distribution::PyDistribution;
use crate::storage::binding::StorageBinding;
use crate::study::{pyobject_to_persisted_study, PyDirection, PyPersistedStudy};
use crate::trial::{
    pyobject_to_persisted_trial_with_category_labels, PyPersistedTrial, PyTrialState,
};

pub(crate) struct PyCachedStorageBackend {
    obj: Py<PyAny>,
}

impl PyCachedStorageBackend {
    pub(crate) fn new(obj: Py<PyAny>) -> Self {
        Self { obj }
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
        let kind = match class_name.as_str() {
            "DuplicatedStudyError" => ErrorKind::DuplicatedStudy,
            "StudyNotFoundError" => ErrorKind::StudyNotFound,
            "TrialNotFoundError" => ErrorKind::TrialNotFound,
            "UpdateFinishedTrialError" => ErrorKind::TrialAlreadyFinished,
            "KeyError" => ErrorKind::AttrNotFound,
            _ => ErrorKind::StorageError,
        };
        Error::with_reason(kind, reason)
    }

    fn py_trial(py: Python<'_>, trial: PersistedTrial) -> PyResult<Py<PyAny>> {
        Ok(Py::new(py, PyPersistedTrial::new(trial, Attrs::new()))?.into_any())
    }

    fn parse_trial(trial: &Bound<'_, PyAny>) -> PyResult<PersistedTrial> {
        pyobject_to_persisted_trial_with_category_labels(
            trial,
            trial.getattr("study_id")?.extract()?,
        )
        .map(|(trial, _)| trial)
    }

    fn parse_trial_for_study(trial: &Bound<'_, PyAny>, study_id: u32) -> PyResult<PersistedTrial> {
        pyobject_to_persisted_trial_with_category_labels(trial, study_id).map(|(trial, _)| trial)
    }

    fn parse_trials_for_study(
        trials: &Bound<'_, PyAny>,
        study_id: u32,
    ) -> PyResult<Vec<PersistedTrial>> {
        let trials = trials.cast::<PyList>()?;
        trials
            .iter()
            .map(|trial| Self::parse_trial_for_study(&trial, study_id))
            .collect()
    }

    fn call_get_trials(&self, py: Python<'_>, study_id: u32) -> PyResult<Vec<PersistedTrial>> {
        let trials = self.obj.call_method1(py, "get_trials", (study_id,))?;
        Self::parse_trials_for_study(trials.bind(py), study_id)
    }

    fn get_backend_category_labels(
        &self,
        py: Python<'_>,
        study_id: u32,
        param_name: &str,
        cardinality: usize,
    ) -> PyResult<Option<Vec<CategoryLabel>>> {
        let Ok(method) = self.obj.getattr(py, "get_category_labels") else {
            return Ok(None);
        };
        let labels = method.bind(py).call1((study_id, param_name, cardinality))?;
        if labels.is_none() {
            return Ok(None);
        }
        let labels = labels.cast::<PyList>()?;
        labels
            .iter()
            .map(|label| pyobject_to_category_label(&label))
            .collect::<PyResult<Vec<_>>>()
            .map(Some)
    }

    fn call_set_attrs(
        &self,
        py: Python<'_>,
        method_name: &str,
        id: u32,
        attrs: &Attrs,
    ) -> PyResult<()> {
        let (user_method, system_method) = if method_name == "set_study_attrs" {
            ("set_study_user_attrs", "set_study_system_attrs")
        } else {
            ("set_trial_user_attrs", "set_trial_system_attrs")
        };
        let user_attrs = PyDict::new(py);
        let system_attrs = PyDict::new(py);
        for (key, value) in attrs {
            match key {
                AttrKey::User(key) => user_attrs.set_item(key.as_str(), value)?,
                AttrKey::System(key) => system_attrs.set_item(key.as_str(), value)?,
            }
        }
        if !user_attrs.is_empty() {
            self.obj.call_method1(py, user_method, (id, user_attrs))?;
        }
        if !system_attrs.is_empty() {
            self.obj
                .call_method1(py, system_method, (id, system_attrs))?;
        }
        Ok(())
    }
}

impl CachedStorageBackend for PyCachedStorageBackend {
    fn create_new_study(
        &mut self,
        study_name: &str,
        directions: Vec<Direction>,
    ) -> rustuna_core::Result<PersistedStudy> {
        Python::attach(|py| {
            let py_directions: Vec<PyDirection> = directions.into_iter().map(Into::into).collect();
            let study =
                self.obj
                    .call_method1(py, "create_new_study", (study_name, py_directions))?;
            pyobject_to_persisted_study(study.bind(py))
        })
        .map_err(Self::map_pyerr)
    }

    fn delete_study(&mut self, study_id: u32) -> rustuna_core::Result<()> {
        Python::attach(|py| {
            self.obj.call_method1(py, "delete_study", (study_id,))?;
            Ok(())
        })
        .map_err(Self::map_pyerr)
    }

    fn create_new_trial(&mut self, study_id: u32) -> rustuna_core::Result<PersistedTrial> {
        Python::attach(|py| {
            let trial = self.obj.call_method1(py, "create_new_trial", (study_id,))?;
            Self::parse_trial_for_study(trial.bind(py), study_id)
        })
        .map_err(Self::map_pyerr)
    }

    fn create_new_trial_from_template(
        &mut self,
        study_id: u32,
        template: &PersistedTrial,
    ) -> rustuna_core::Result<PersistedTrial> {
        Python::attach(|py| {
            let py_template = Self::py_trial(py, template.clone())?;
            let trial = self
                .obj
                .call_method1(py, "create_new_trial", (study_id, py_template))?;
            Self::parse_trial_for_study(trial.bind(py), study_id)
        })
        .map_err(Self::map_pyerr)
    }

    fn set_trial_param(
        &mut self,
        trial_id: u32,
        name: &str,
        distribution: &Distribution,
        value: f64,
    ) -> rustuna_core::Result<()> {
        Python::attach(|py| {
            let category_labels = if let Distribution::Categorical { cardinality } = distribution {
                let trial = self.obj.call_method1(py, "get_trial", (trial_id,))?;
                let trial = Self::parse_trial(trial.bind(py))?;
                if let Some(labels) =
                    self.get_backend_category_labels(py, trial.study_id, name, *cardinality)?
                {
                    Some(labels)
                } else {
                    let study = self.obj.call_method1(py, "get_study", (trial.study_id,))?;
                    let study = pyobject_to_persisted_study(study.bind(py))?;
                    get_category_labels(&study.attrs, name, *cardinality)
                }
            } else {
                None
            };
            let py_distribution =
                PyDistribution::new_with_category_labels(distribution.clone(), category_labels);
            self.obj.call_method1(
                py,
                "set_trial_param",
                (trial_id, name, py_distribution, value),
            )?;
            Ok(())
        })
        .map_err(Self::map_pyerr)
    }

    fn set_trial_state_values(
        &mut self,
        trial_id: u32,
        state_values: TrialStateValues,
    ) -> rustuna_core::Result<()> {
        Python::attach(|py| {
            let values = match &state_values {
                TrialStateValues::Complete(values) => Some(values.clone()),
                _ => None,
            };
            let py_state = PyTrialState::from(state_values);
            self.obj
                .call_method1(py, "set_trial_state_values", (trial_id, py_state, values))?;
            Ok(())
        })
        .map_err(Self::map_pyerr)
    }

    fn set_trial_intermediate_values(
        &mut self,
        trial_id: u32,
        intermediate_values: HashMap<u32, f64>,
    ) -> rustuna_core::Result<()> {
        Python::attach(|py| {
            if let Ok(method) = self.obj.getattr(py, "set_trial_intermediate_values") {
                let values = PyDict::new(py);
                for (step, value) in intermediate_values {
                    values.set_item(step, value)?;
                }
                method.bind(py).call1((trial_id, values))?;
            } else {
                for (step, value) in intermediate_values {
                    self.obj.call_method1(
                        py,
                        "set_trial_intermediate_value",
                        (trial_id, step, value),
                    )?;
                }
            }
            Ok(())
        })
        .map_err(Self::map_pyerr)
    }

    fn get_studies(&mut self) -> rustuna_core::Result<Vec<PersistedStudy>> {
        Python::attach(|py| {
            let studies = self.obj.call_method0(py, "get_studies")?;
            let studies = studies.bind(py).cast::<PyList>()?;
            studies
                .iter()
                .map(|study| pyobject_to_persisted_study(&study))
                .collect::<PyResult<Vec<_>>>()
        })
        .map_err(Self::map_pyerr)
    }

    fn get_study(&mut self, study_id: u32) -> rustuna_core::Result<PersistedStudy> {
        Python::attach(|py| {
            let study = self.obj.call_method1(py, "get_study", (study_id,))?;
            pyobject_to_persisted_study(study.bind(py))
        })
        .map_err(Self::map_pyerr)
    }

    fn get_trial(&mut self, trial_id: u32) -> rustuna_core::Result<PersistedTrial> {
        Python::attach(|py| {
            let trial = self.obj.call_method1(py, "get_trial", (trial_id,))?;
            Self::parse_trial(trial.bind(py))
        })
        .map_err(Self::map_pyerr)
    }

    fn get_study_attr(&mut self, study_id: u32, key: AttrKey) -> rustuna_core::Result<String> {
        Python::attach(|py| {
            let (method_name, attrs_name, key) = match key {
                AttrKey::User(key) => ("get_study_user_attr", "user_attrs", key.to_string()),
                AttrKey::System(key) => ("get_study_system_attr", "system_attrs", key.to_string()),
            };
            if let Ok(method) = self.obj.getattr(py, method_name) {
                return method.bind(py).call1((study_id, key))?.extract();
            }
            let study = self.obj.call_method1(py, "get_study", (study_id,))?;
            let study = study.bind(py);
            study.getattr(attrs_name)?.get_item(key.as_str())?.extract()
        })
        .map_err(Self::map_pyerr)
    }

    fn set_study_attrs(
        &mut self,
        study_id: u32,
        attrs: Attrs,
        _error_on_overwrite: bool,
    ) -> rustuna_core::Result<()> {
        Python::attach(|py| self.call_set_attrs(py, "set_study_attrs", study_id, &attrs))
            .map_err(Self::map_pyerr)
    }

    fn set_trial_attrs(
        &mut self,
        trial_id: u32,
        attrs: Attrs,
        _error_on_overwrite: bool,
    ) -> rustuna_core::Result<()> {
        Python::attach(|py| self.call_set_attrs(py, "set_trial_attrs", trial_id, &attrs))
            .map_err(Self::map_pyerr)
    }

    fn apply_discard(&self) -> bool {
        Python::attach(|py| {
            self.obj
                .call_method0(py, "may_omit_trials")
                .and_then(|value| value.extract::<bool>(py))
                .unwrap_or(false)
        })
    }

    fn discard_trials(&mut self, trial_ids: &[u32]) -> rustuna_core::Result<()> {
        Python::attach(|py| {
            self.obj
                .call_method1(py, "discard_trials", (trial_ids.to_vec(),))?;
            Ok(())
        })
        .map_err(Self::map_pyerr)
    }

    fn get_discarded_trials_diff(
        &mut self,
        _study_id: u32,
        _cursor: Option<&str>,
    ) -> rustuna_core::Result<DiscardedTrialsDiff> {
        Ok(DiscardedTrialsDiff::default())
    }

    fn get_trials_diff(
        &mut self,
        study_id: u32,
        included_numbers: &[u32],
        trial_number_greater_than: i32,
    ) -> rustuna_core::Result<Vec<PersistedTrial>> {
        Python::attach(|py| {
            if let Ok(method) = self.obj.getattr(py, "get_trials_diff") {
                let trials = method.bind(py).call1((
                    study_id,
                    included_numbers.to_vec(),
                    trial_number_greater_than,
                ))?;
                return Self::parse_trials_for_study(&trials, study_id);
            }
            let trials = self.call_get_trials(py, study_id)?;
            Ok(trials
                .into_iter()
                .filter(|trial| {
                    included_numbers.contains(&trial.number)
                        || (trial.number as i32) > trial_number_greater_than
                })
                .collect())
        })
        .map_err(Self::map_pyerr)
    }

    fn get_n_trials(
        &mut self,
        study_id: u32,
        states: Option<&[rustuna_core::trial::TrialState]>,
    ) -> rustuna_core::Result<u32> {
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
        .map_err(Self::map_pyerr)
    }
}

#[derive(Clone)]
#[pyclass(name = "CachedStorage", from_py_object)]
#[pyo3(module = "rustuna")]
pub struct PyCachedStorage {
    pub(crate) binding: StorageBinding,
}

impl PyCachedStorage {
    pub(crate) fn new(backend: Py<PyAny>) -> Self {
        let storage = CachedStorage::new(Box::new(PyCachedStorageBackend::new(backend)));
        Self {
            binding: StorageBinding::new(Arc::new(RwLock::new(storage))),
        }
    }

    pub(crate) fn storage(&self) -> Arc<RwLock<dyn Storage>> {
        self.binding.storage.clone()
    }
}

#[pymethods]
impl PyCachedStorage {
    #[new]
    fn py_new(backend: Py<PyAny>) -> Self {
        Self::new(backend)
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
