use std::sync::{Arc, RwLock};

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyList;
use rustuna_core::attr::{AttrKey, CategoryLabel};
use rustuna_core::distribution::Distribution;
use rustuna_core::storage::Storage;
use rustuna_core::study::{Direction, PersistedStudy};
use rustuna_core::trial::{TrialState, TrialStateValues};

use crate::attrs::{pyobj_to_attrs_with_kind, AttrKind};
use crate::distribution::{category_label_to_pyobject, pyobject_to_category_label, PyDistribution};
use crate::exception::err_to_exceptions;
use crate::study::{PyDirection, PyPersistedStudy};
use crate::trial::{PyPersistedTrial, PyTrialState};

#[derive(Clone)]
pub(crate) struct StorageBinding {
    pub(crate) storage: Arc<RwLock<dyn Storage>>,
}

impl StorageBinding {
    pub(crate) fn new(storage: Arc<RwLock<dyn Storage>>) -> Self {
        Self { storage }
    }

    pub(crate) fn create_new_study(
        &self,
        py: Python<'_>,
        study_name: String,
        directions: Vec<PyDirection>,
    ) -> PyResult<PyPersistedStudy> {
        let directions: Vec<Direction> = directions.iter().map(|d| d.clone().into()).collect();
        let study = py.detach(|| -> PyResult<PersistedStudy> {
            let mut guard = self.storage.write().map_err(|e| {
                PyRuntimeError::new_err(format!("Failed to acquire the storage guard: {e:?}"))
            })?;
            let study = guard
                .create_new_study(&study_name, directions)
                .map_err(err_to_exceptions)?;
            Ok(study.clone())
        })?;
        Ok(study.into())
    }

    pub(crate) fn delete_study(&self, py: Python<'_>, study_id: u32) -> PyResult<()> {
        py.detach(|| -> PyResult<()> {
            let mut guard = self.storage.write().map_err(|e| {
                PyRuntimeError::new_err(format!("Failed to acquire the storage guard: {e:?}"))
            })?;
            guard.delete_study(study_id).map_err(err_to_exceptions)
        })
    }

    pub(crate) fn create_new_trial(
        &self,
        py: Python<'_>,
        study_id: u32,
        template_trial: Option<&Bound<'_, PyPersistedTrial>>,
    ) -> PyResult<PyPersistedTrial> {
        let template_trial = template_trial
            .map(|template_trial| template_trial.borrow().with_trial(|t| Ok(t.clone())))
            .transpose()?;

        py.detach(|| -> PyResult<PyPersistedTrial> {
            let mut guard = self.storage.write().map_err(|e| {
                PyRuntimeError::new_err(format!("Failed to acquire the storage guard: {e:?}"))
            })?;
            let trial = if let Some(template_trial) = template_trial {
                guard
                    .create_new_trial_from_template(study_id, &template_trial)
                    .map_err(err_to_exceptions)?
            } else {
                guard
                    .create_new_trial(study_id)
                    .map_err(err_to_exceptions)?
            };
            Ok(PyPersistedTrial::from_storage(self.storage.clone(), trial))
        })
    }

    pub(crate) fn set_trial_param(
        &self,
        py: Python<'_>,
        trial_id: u32,
        name: String,
        distribution: PyDistribution,
        value: f64,
    ) -> PyResult<()> {
        py.detach(|| -> PyResult<()> {
            let category_labels = distribution.category_labels.clone();
            let distribution: Distribution = distribution.into();

            if let Some(labels) = category_labels {
                let study_id = {
                    let mut guard = self.storage.write().map_err(|e| {
                        PyRuntimeError::new_err(format!(
                            "Failed to acquire the storage guard: {e:?}"
                        ))
                    })?;
                    guard
                        .get_trial(trial_id)
                        .map_err(err_to_exceptions)?
                        .study_id
                };
                self.set_category_labels_internal(study_id, name.clone(), labels)?;
            }

            let mut guard = self.storage.write().map_err(|e| {
                PyRuntimeError::new_err(format!("Failed to acquire the storage guard: {e:?}"))
            })?;
            guard
                .set_trial_param(trial_id, &name, &distribution, value)
                .map_err(err_to_exceptions)?;
            Ok(())
        })
    }

    pub(crate) fn set_category_labels(
        &self,
        py: Python<'_>,
        study_id: u32,
        param_name: String,
        choices: Vec<Py<PyAny>>,
    ) -> PyResult<()> {
        let mut labels: Vec<CategoryLabel> = Vec::with_capacity(choices.len());
        for choice in choices {
            let label = pyobject_to_category_label(choice.bind(py))?;
            labels.push(label);
        }
        py.detach(|| -> PyResult<()> {
            self.set_category_labels_internal(study_id, param_name, labels)
        })
    }

    pub(crate) fn get_category_labels(
        &self,
        py: Python<'_>,
        study_id: u32,
        param_name: String,
        cardinality: usize,
    ) -> PyResult<Py<PyAny>> {
        let category_labels = py.detach(|| -> PyResult<Option<Vec<CategoryLabel>>> {
            let mut guard = self.storage.write().map_err(|e| {
                PyRuntimeError::new_err(format!("Failed to acquire the storage guard: {e:?}"))
            })?;
            guard
                .get_category_labels(study_id, &param_name, cardinality)
                .map_err(err_to_exceptions)
        })?;
        match category_labels {
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
    }

    pub(crate) fn set_trial_state_values(
        &self,
        py: Python<'_>,
        trial_id: u32,
        state: PyTrialState,
        values: Option<Vec<f64>>,
    ) -> PyResult<()> {
        py.detach(|| -> PyResult<()> {
            let mut guard = self.storage.write().map_err(|e| {
                PyRuntimeError::new_err(format!("Failed to acquire the storage guard: {e:?}"))
            })?;

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
        })
    }

    pub(crate) fn get_studies(&self, py: Python<'_>) -> PyResult<Vec<PyPersistedStudy>> {
        py.detach(|| -> PyResult<Vec<PyPersistedStudy>> {
            let mut guard = self.storage.write().map_err(|e| {
                PyRuntimeError::new_err(format!("Failed to acquire the storage guard: {e:?}"))
            })?;
            let studies = guard.get_studies().map_err(|e| {
                PyRuntimeError::new_err(format!("Failed to get studies: {:?}", e.kind))
            })?;
            Ok(studies.iter().map(|s| s.clone().into()).collect())
        })
    }

    pub(crate) fn get_study(&self, py: Python<'_>, study_id: u32) -> PyResult<PyPersistedStudy> {
        py.detach(|| -> PyResult<PyPersistedStudy> {
            let mut guard = self.storage.write().map_err(|e| {
                PyRuntimeError::new_err(format!("Failed to acquire the storage guard: {e:?}"))
            })?;
            let study = guard.get_study(study_id).map_err(err_to_exceptions)?;
            Ok(study.clone().into())
        })
    }

    pub(crate) fn get_trials(
        &self,
        py: Python<'_>,
        study_id: u32,
        states: Option<Vec<PyTrialState>>,
    ) -> PyResult<Vec<PyPersistedTrial>> {
        py.detach(|| -> PyResult<Vec<PyPersistedTrial>> {
            let mut guard = self.storage.write().map_err(|e| {
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
                .map(|t| PyPersistedTrial::from_storage(self.storage.clone(), t))
                .collect();
            Ok(py_trials)
        })
    }

    pub(crate) fn get_n_trials(
        &self,
        py: Python<'_>,
        study_id: u32,
        states: Option<Vec<PyTrialState>>,
    ) -> PyResult<u32> {
        py.detach(|| -> PyResult<u32> {
            let states =
                states.map(|states| states.into_iter().map(TrialState::from).collect::<Vec<_>>());
            let mut guard = self.storage.write().map_err(|e| {
                PyRuntimeError::new_err(format!("Failed to acquire the storage guard: {e:?}"))
            })?;
            guard
                .get_n_trials(study_id, states.as_deref())
                .map_err(err_to_exceptions)
        })
    }

    pub(crate) fn get_trial(&self, py: Python<'_>, trial_id: u32) -> PyResult<PyPersistedTrial> {
        py.detach(|| -> PyResult<PyPersistedTrial> {
            let mut guard = self.storage.write().map_err(|e| {
                PyRuntimeError::new_err(format!("Failed to acquire the storage guard: {e:?}"))
            })?;
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
        })
    }

    pub(crate) fn get_cached_trial(
        &self,
        py: Python<'_>,
        trial_id: u32,
    ) -> PyResult<PyPersistedTrial> {
        py.detach(|| {
            let guard = self.storage.read().map_err(|e| {
                PyRuntimeError::new_err(format!("Failed to acquire the storage guard: {e:?}"))
            })?;
            let trial = guard
                .get_cached_trial(trial_id)
                .map_err(err_to_exceptions)?;
            Ok(PyPersistedTrial::from_storage(self.storage.clone(), trial))
        })
    }

    pub(crate) fn get_trial_number_from_id(&self, py: Python<'_>, trial_id: u32) -> PyResult<u32> {
        py.detach(|| {
            let mut guard = self.storage.write().map_err(|e| {
                PyRuntimeError::new_err(format!("Failed to acquire the storage guard: {e:?}"))
            })?;
            guard
                .get_trial_number_from_id(trial_id)
                .map_err(err_to_exceptions)
        })
    }

    pub(crate) fn get_study_user_attr(
        &self,
        py: Python<'_>,
        study_id: u32,
        key: String,
    ) -> PyResult<String> {
        py.detach(|| {
            let mut guard = self.storage.write().map_err(|e| {
                PyRuntimeError::new_err(format!("Failed to acquire the storage guard: {e:?}"))
            })?;
            guard
                .get_study_attr(study_id, AttrKey::User(key.into()))
                .map_err(err_to_exceptions)
        })
    }

    pub(crate) fn get_study_system_attr(
        &self,
        py: Python<'_>,
        study_id: u32,
        key: String,
    ) -> PyResult<String> {
        py.detach(|| {
            let mut guard = self.storage.write().map_err(|e| {
                PyRuntimeError::new_err(format!("Failed to acquire the storage guard: {e:?}"))
            })?;
            guard
                .get_study_attr(study_id, AttrKey::System(key.into()))
                .map_err(err_to_exceptions)
        })
    }

    pub(crate) fn get_trial_id_from_study_id_trial_number(
        &self,
        py: Python<'_>,
        study_id: u32,
        trial_number: u32,
    ) -> PyResult<u32> {
        py.detach(|| {
            let mut guard = self.storage.write().map_err(|e| {
                PyRuntimeError::new_err(format!("Failed to acquire the storage guard: {e:?}"))
            })?;
            let trial_id = guard
                .get_trial_id_from_study_id_trial_number(study_id, trial_number)
                .map_err(err_to_exceptions)?;
            Ok(trial_id)
        })
    }

    pub(crate) fn set_study_system_attrs(
        &self,
        py: Python<'_>,
        study_id: u32,
        attrs: Py<PyAny>,
    ) -> PyResult<()> {
        let attrs = attrs.bind(py);
        let system_attrs = pyobj_to_attrs_with_kind(attrs, AttrKind::System)?;
        py.detach(|| -> PyResult<()> {
            let mut guard = self.storage.write().map_err(|e| {
                PyRuntimeError::new_err(format!("Failed to acquire the storage guard: {e:?}"))
            })?;
            guard
                .set_study_attrs(study_id, system_attrs, false)
                .map_err(err_to_exceptions)?;
            Ok(())
        })
    }

    pub(crate) fn set_study_user_attrs(
        &self,
        py: Python<'_>,
        study_id: u32,
        attrs: Py<PyAny>,
    ) -> PyResult<()> {
        let attrs = attrs.bind(py);
        let user_attrs = pyobj_to_attrs_with_kind(attrs, AttrKind::User)?;
        py.detach(|| -> PyResult<()> {
            let mut guard = self.storage.write().map_err(|e| {
                PyRuntimeError::new_err(format!("Failed to acquire the storage guard: {e:?}"))
            })?;
            guard
                .set_study_attrs(study_id, user_attrs, false)
                .map_err(err_to_exceptions)?;
            Ok(())
        })
    }

    pub(crate) fn set_trial_system_attrs(
        &self,
        py: Python<'_>,
        trial_id: u32,
        attrs: Py<PyAny>,
    ) -> PyResult<()> {
        let attrs = attrs.bind(py);
        let system_attrs = pyobj_to_attrs_with_kind(attrs, AttrKind::System)?;
        py.detach(|| -> PyResult<()> {
            let mut guard = self.storage.write().map_err(|e| {
                PyRuntimeError::new_err(format!("Failed to acquire the storage guard: {e:?}"))
            })?;
            guard
                .set_trial_attrs(trial_id, system_attrs, false)
                .map_err(err_to_exceptions)?;
            Ok(())
        })
    }

    pub(crate) fn set_trial_user_attrs(
        &self,
        py: Python<'_>,
        trial_id: u32,
        attrs: Py<PyAny>,
    ) -> PyResult<()> {
        let attrs = attrs.bind(py);
        let user_attrs = pyobj_to_attrs_with_kind(attrs, AttrKind::User)?;
        py.detach(|| -> PyResult<()> {
            let mut guard = self.storage.write().map_err(|e| {
                PyRuntimeError::new_err(format!("Failed to acquire the storage guard: {e:?}"))
            })?;
            guard
                .set_trial_attrs(trial_id, user_attrs, false)
                .map_err(err_to_exceptions)?;
            Ok(())
        })
    }

    pub(crate) fn set_trial_intermediate_value(
        &self,
        py: Python<'_>,
        trial_id: u32,
        step: u32,
        intermediate_value: f64,
    ) -> PyResult<()> {
        py.detach(|| -> PyResult<()> {
            let mut intermediate_values = std::collections::HashMap::new();
            intermediate_values.insert(step, intermediate_value);

            let mut guard = self.storage.write().map_err(|e| {
                PyRuntimeError::new_err(format!("Failed to acquire the storage guard: {e:?}"))
            })?;
            guard
                .set_trial_intermediate_values(trial_id, intermediate_values)
                .map_err(err_to_exceptions)?;
            Ok(())
        })
    }

    pub(crate) fn discard_trials(&self, py: Python<'_>, trial_ids: Vec<u32>) -> PyResult<()> {
        py.detach(|| -> PyResult<()> {
            let mut guard = self.storage.write().map_err(|e| {
                PyRuntimeError::new_err(format!("Failed to acquire the storage guard: {e:?}"))
            })?;
            guard
                .discard_trials(&trial_ids)
                .map_err(err_to_exceptions)?;
            Ok(())
        })
    }

    pub(crate) fn may_omit_trials(&self, py: Python<'_>) -> PyResult<bool> {
        py.detach(|| -> PyResult<bool> {
            let guard = self.storage.read().map_err(|e| {
                PyRuntimeError::new_err(format!("Failed to acquire the storage guard: {e:?}"))
            })?;
            Ok(guard.may_omit_trials())
        })
    }
}

impl StorageBinding {
    fn set_category_labels_internal(
        &self,
        study_id: u32,
        param_name: String,
        category_labels: Vec<CategoryLabel>,
    ) -> PyResult<()> {
        let mut guard = self.storage.write().map_err(|e| {
            PyRuntimeError::new_err(format!("Failed to acquire the storage guard: {e:?}"))
        })?;
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
