use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::Py;

use rustuna_core::sampler::Sampler;
use rustuna_core::trial::TrialStateValues;
use rustuna_sampler::qmc::QmcSampler;

use crate::distribution::PyDistribution;
use crate::sampler::{extract_storage, PySamplerContext};
use crate::trial::PyTrialState;

#[derive(Clone)]
#[pyclass(name = "QMCSampler", from_py_object)]
#[pyo3(module = "rustuna")]
pub struct PyQmcSampler {
    pub sampler: Arc<Mutex<QmcSampler>>,
}
#[pymethods]
impl PyQmcSampler {
    #[new]
    #[pyo3(signature = (*, seed = None))]
    fn py_new(seed: Option<u64>) -> PyResult<Self> {
        let rs_sampler = match seed {
            Some(seed) => QmcSampler::seed_from_u64(seed),
            None => QmcSampler::new(),
        };
        Ok(PyQmcSampler {
            sampler: Arc::new(Mutex::new(rs_sampler)),
        })
    }

    #[getter]
    fn support_joint_sampling(&self, py: Python<'_>) -> PyResult<bool> {
        py.detach(|| {
            let guard = self.sampler.lock().map_err(|e| {
                PyRuntimeError::new_err(format!("Failed to acquire sampler lock: {e}"))
            })?;
            Ok(guard.support_joint_sampling())
        })
    }

    fn sample_independent(
        &self,
        py: Python<'_>,
        ctx: &PySamplerContext,
        storage: Py<PyAny>,
        name: &str,
        distribution: &PyDistribution,
    ) -> PyResult<f64> {
        let arc_storage = extract_storage(storage)?;
        let context = ctx.context.clone();
        let name = name.to_owned();
        let distribution = distribution.distribution.clone();
        py.detach(|| {
            self.sampler
                .lock()
                .map_err(|e| {
                    PyRuntimeError::new_err(format!("Failed to acquire the sampler guard: {e}"))
                })?
                .sample_independent(&context, arc_storage, &name, &distribution)
                .map_err(|e| {
                    PyRuntimeError::new_err(format!("Failed to sample independent: {e:?}"))
                })
        })
    }

    fn sample_joint(
        &self,
        py: Python<'_>,
        ctx: &PySamplerContext,
        storage: Py<PyAny>,
        search_space: HashMap<String, PyDistribution>,
    ) -> PyResult<HashMap<String, f64>> {
        let arc_storage = extract_storage(storage)?;
        let context = ctx.context.clone();
        let search_space = search_space
            .into_iter()
            .map(|(k, v)| (k, v.distribution))
            .collect();
        py.detach(|| {
            self.sampler
                .lock()
                .map_err(|e| {
                    PyRuntimeError::new_err(format!("Failed to acquire the sampler guard: {e}"))
                })?
                .sample_joint(&context, arc_storage, &search_space)
                .map_err(|e| PyRuntimeError::new_err(format!("Failed to sample joint: {e:?}")))
        })
    }

    #[pyo3(signature = (ctx, storage, state, values = None))]
    fn after_trial(
        &self,
        py: Python<'_>,
        ctx: &PySamplerContext,
        storage: Py<PyAny>,
        state: PyTrialState,
        values: Option<Vec<f64>>,
    ) -> PyResult<()> {
        let arc_storage = extract_storage(storage)?;
        let state_values = match state {
            PyTrialState::RUNNING => TrialStateValues::Running,
            PyTrialState::COMPLETE => TrialStateValues::Complete(values.ok_or(
                PyRuntimeError::new_err("values must be specified when state is COMPLETE"),
            )?),
            PyTrialState::PRUNED => TrialStateValues::Pruned,
            PyTrialState::WAITING => TrialStateValues::Waiting,
            PyTrialState::FAIL => TrialStateValues::Fail,
        };
        let context = ctx.context.clone();
        py.detach(|| {
            self.sampler
                .lock()
                .map_err(|e| {
                    PyRuntimeError::new_err(format!("Failed to acquire the sampler guard: {e}"))
                })?
                .after_trial(&context, arc_storage, &state_values)
                .map_err(|e| PyRuntimeError::new_err(format!("Failed to call after_trial: {e:?}")))
        })
    }
}
