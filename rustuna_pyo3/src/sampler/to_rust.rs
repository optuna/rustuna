use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use pyo3::prelude::*;
use pyo3::types::PyDict;
use pyo3::Py;

use rustuna_core::sampler::{Context as SamplerContext, Sampler};
use rustuna_core::storage::Storage;
use rustuna_core::trial::TrialStateValues;

use crate::distribution::PyDistribution;
use crate::sampler::PySamplerContext;
use crate::storage::to_python::ToPythonStorage;
use crate::trial::PyTrialState;

// ToRustSampler adapts a Python object implementing rustuna.SamplerProtocol to
// rustuna_core::sampler::Sampler. Rustuna can therefore use Python samplers through the same
// interface as native Rust samplers.
pub struct ToRustSampler {
    obj: Mutex<Py<PyAny>>,
}
impl ToRustSampler {
    pub fn new(obj: Py<PyAny>) -> Self {
        ToRustSampler {
            obj: Mutex::new(obj),
        }
    }
}
impl Sampler for ToRustSampler {
    fn before_trial(
        &self,
        ctx: &SamplerContext,
        storage: Arc<std::sync::RwLock<dyn rustuna_core::storage::Storage>>,
    ) -> rustuna_core::Result<()> {
        let obj = self.obj.lock().map_err(|e| {
            rustuna_core::Error::with_reason(
                rustuna_core::ErrorKind::SamplerError,
                format!("Failed to acquire sampler object guard: {e}"),
            )
        })?;
        Python::attach(|py| {
            let py_ctx = PySamplerContext::from(ctx.clone());
            let py_storage = ToPythonStorage::new(storage);
            obj.call_method1(py, "before_trial", (py_ctx, py_storage))
                .map_err(|e| {
                    rustuna_core::Error::with_reason(
                        rustuna_core::ErrorKind::SamplerError,
                        e.to_string(),
                    )
                })?;
            Ok(())
        })
    }

    fn sample_independent(
        &self,
        ctx: &SamplerContext,
        storage: Arc<std::sync::RwLock<dyn rustuna_core::storage::Storage>>,
        name: &str,
        distribution: &rustuna_core::distribution::Distribution,
    ) -> rustuna_core::Result<f64> {
        let mut guard = storage.write().map_err(|e| {
            rustuna_core::Error::with_reason(
                rustuna_core::ErrorKind::StorageError,
                format!("Failed to acquire storage guard: {e}"),
            )
        })?;
        let study = guard.get_study(ctx.study_id)?;
        let study_attrs = study.attrs.clone();
        drop(guard);

        let obj = self.obj.lock().map_err(|e| {
            rustuna_core::Error::with_reason(
                rustuna_core::ErrorKind::SamplerError,
                format!("Failed to acquire sampler object guard: {e}"),
            )
        })?;
        Python::attach(|py| {
            let py_ctx = PySamplerContext::from(ctx.clone());
            let py_storage = ToPythonStorage::new(storage.clone());
            let py_distribution = PyDistribution::new(distribution.clone(), name, &study_attrs);
            let py_result = obj
                .call_method1(
                    py,
                    "sample_independent",
                    (py_ctx, py_storage, name, py_distribution),
                )
                .map_err(|e| {
                    rustuna_core::Error::with_reason(
                        rustuna_core::ErrorKind::SamplerError,
                        e.to_string(),
                    )
                })?;
            let py_result_ref = py_result.bind(py);
            let ret = py_result_ref.extract::<f64>().map_err(|e| {
                rustuna_core::Error::with_reason(
                    rustuna_core::ErrorKind::SamplerError,
                    e.to_string(),
                )
            })?;
            Ok(ret)
        })
    }

    fn support_joint_sampling(&self) -> bool {
        let Ok(obj) = self.obj.lock() else {
            return false;
        };
        Python::attach(|py| {
            obj.getattr(py, "support_joint_sampling")
                .and_then(|x| x.extract::<bool>(py))
                .unwrap_or(false)
        })
    }

    fn sample_joint(
        &self,
        ctx: &SamplerContext,
        storage: Arc<std::sync::RwLock<dyn Storage>>,
        search_space: &HashMap<String, rustuna_core::distribution::Distribution>,
    ) -> rustuna_core::Result<HashMap<String, f64>> {
        let mut guard = storage.write().map_err(|e| {
            rustuna_core::Error::with_reason(rustuna_core::ErrorKind::StorageError, e.to_string())
        })?;
        let study = guard.get_study(ctx.study_id)?;
        let study_attrs = study.attrs.clone();
        drop(guard);

        let obj = self.obj.lock().map_err(|e| {
            rustuna_core::Error::with_reason(
                rustuna_core::ErrorKind::SamplerError,
                format!("Failed to acquire sampler object guard: {e}"),
            )
        })?;
        Python::attach(|py| {
            let py_ctx = PySamplerContext::from(ctx.clone());
            let py_storage = ToPythonStorage::new(storage.clone());
            let py_search_space = PyDict::new(py);
            for (k, v) in search_space {
                let py_distribution = Py::new(py, PyDistribution::new(v.clone(), k, &study_attrs))
                    .map_err(|e| {
                        rustuna_core::Error::with_reason(
                            rustuna_core::ErrorKind::SamplerError,
                            e.to_string(),
                        )
                    })?;
                py_search_space.set_item(k, py_distribution).map_err(|e| {
                    rustuna_core::Error::with_reason(
                        rustuna_core::ErrorKind::SamplerError,
                        e.to_string(),
                    )
                })?;
            }
            let py_result = obj
                .call_method1(py, "sample_joint", (py_ctx, py_storage, py_search_space))
                .map_err(|e| {
                    rustuna_core::Error::with_reason(
                        rustuna_core::ErrorKind::SamplerError,
                        e.to_string(),
                    )
                })?;
            let py_result = py_result.extract::<HashMap<String, f64>>(py).map_err(|e| {
                rustuna_core::Error::with_reason(
                    rustuna_core::ErrorKind::SamplerError,
                    e.to_string(),
                )
            })?;
            Ok(py_result)
        })
    }

    fn after_trial(
        &self,
        ctx: &SamplerContext,
        storage: Arc<std::sync::RwLock<dyn Storage>>,
        state_values: &TrialStateValues,
    ) -> rustuna_core::Result<()> {
        let py_state = PyTrialState::from(state_values.clone());
        let py_values = match state_values {
            TrialStateValues::Complete(values) => Some(values.clone()),
            _ => None,
        };

        let obj = self.obj.lock().map_err(|e| {
            rustuna_core::Error::with_reason(
                rustuna_core::ErrorKind::SamplerError,
                format!("Failed to acquire sampler object guard: {e}"),
            )
        })?;
        Python::attach(|py| {
            let py_ctx = PySamplerContext::from(ctx.clone());
            let py_storage = ToPythonStorage::new(storage.clone());
            obj.call_method1(py, "after_trial", (py_ctx, py_storage, py_state, py_values))
                .map_err(|e| {
                    rustuna_core::Error::with_reason(
                        rustuna_core::ErrorKind::SamplerError,
                        e.to_string(),
                    )
                })?;
            Ok(())
        })
    }
}
