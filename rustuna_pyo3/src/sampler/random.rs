use std::collections::HashMap;
use std::sync::Arc;

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::Py;
use rustuna_core::sampler::{RandomSampler, Sampler};

use crate::distribution::PyDistribution;
use crate::sampler::{extract_storage, PySamplerContext};
use crate::trial::PyTrialState;
use crate::trial_queue::extract_trial_queue;

#[derive(Clone)]
#[pyclass(name = "RandomSampler", from_py_object)]
#[pyo3(module = "rustuna")]
pub struct PyRandomSampler {
    pub sampler: Arc<RandomSampler>,
}
#[pymethods]
impl PyRandomSampler {
    #[new]
    #[pyo3(signature = (*, seed = None))]
    fn py_new(seed: Option<u64>) -> Self {
        let random_sampler = match seed {
            Some(seed) => RandomSampler::seed_from_u64(seed),
            None => RandomSampler::new(),
        };
        Self {
            sampler: Arc::new(random_sampler),
        }
    }

    #[getter]
    fn support_joint_sampling(&self) -> PyResult<bool> {
        Ok(false)
    }

    fn sample_independent(
        &self,
        py: Python<'_>,
        ctx: &PySamplerContext,
        storage: Py<PyAny>,
        trial_queue: Py<PyAny>,
        name: &str,
        distribution: &PyDistribution,
    ) -> PyResult<f64> {
        let arc_storage = extract_storage(storage)?;
        let arc_trial_queue = extract_trial_queue(trial_queue)?;
        let context = ctx.context.clone();
        let name = name.to_owned();
        let distribution = distribution.distribution.clone();
        py.detach(|| {
            self.sampler
                .sample_independent(&context, arc_storage, arc_trial_queue, &name, &distribution)
                .map_err(|e| {
                    PyRuntimeError::new_err(format!("Failed to sample independent: {e:?}"))
                })
        })
    }

    fn sample_joint(
        &self,
        ctx: &PySamplerContext,
        storage: Py<PyAny>,
        trial_queue: Py<PyAny>,
        search_space: HashMap<String, PyDistribution>,
    ) -> PyResult<HashMap<String, f64>> {
        let _ = (ctx, storage, trial_queue, search_space);
        Ok(HashMap::new())
    }

    fn before_trial(
        &self,
        ctx: &PySamplerContext,
        storage: Py<PyAny>,
        trial_queue: Py<PyAny>,
    ) -> PyResult<()> {
        let _ = (ctx, storage, trial_queue);
        Ok(())
    }

    #[pyo3(signature = (ctx, storage, trial_queue, state, values = None))]
    fn after_trial(
        &self,
        ctx: &PySamplerContext,
        storage: Py<PyAny>,
        trial_queue: Py<PyAny>,
        state: PyTrialState,
        values: Option<Vec<f64>>,
    ) -> PyResult<()> {
        let _ = (ctx, storage, trial_queue, state, values);
        Ok(())
    }
}
