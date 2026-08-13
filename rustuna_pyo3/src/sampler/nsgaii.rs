use std::collections::HashMap;
use std::sync::Arc;

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::Py;

use rustuna_core::sampler::Sampler;
use rustuna_core::trial::TrialStateValues;
use rustuna_sampler::nsgaii::NSGAIISampler;

use crate::distribution::PyDistribution;
use crate::sampler::{extract_storage, PySamplerContext};
use crate::trial::PyTrialState;

#[derive(Clone)]
#[pyclass(name = "NSGAIISampler")]
#[pyo3(module = "rustuna")]
pub struct PyNSGAIISampler {
    pub sampler: Arc<NSGAIISampler>,
}
#[pymethods]
impl PyNSGAIISampler {
    #[new]
    #[pyo3(signature = (*, seed = None, population_size = 50, mutation_prob = None, crossover_prob = 0.9, swapping_prob = 0.5))]
    fn py_new(
        seed: Option<u64>,
        population_size: usize,
        mutation_prob: Option<f64>,
        crossover_prob: f64,
        swapping_prob: f64,
    ) -> PyResult<Self> {
        let rs_sampler = match seed {
            Some(seed) => NSGAIISampler::seed_from_u64(
                seed,
                population_size,
                mutation_prob,
                crossover_prob,
                swapping_prob,
            ),
            None => NSGAIISampler::new(
                population_size,
                mutation_prob,
                crossover_prob,
                swapping_prob,
            ),
        };
        Ok(PyNSGAIISampler {
            sampler: Arc::new(rs_sampler),
        })
    }

    #[getter]
    fn support_joint_sampling(&self) -> bool {
        self.sampler.support_joint_sampling()
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
                .after_trial(&context, arc_storage, &state_values)
                .map_err(|e| PyRuntimeError::new_err(format!("Failed to call after_trial: {e:?}")))
        })
    }
}
