use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use rustuna_core::distribution::Distribution;
use rustuna_core::sampler::{Context, RandomSampler, Sampler};
use rustuna_core::storage::Storage;
use rustuna_core::study::Direction;
use rustuna_core::transform::SearchSpaceTransform;
use rustuna_core::trial::TrialStateValues;
use rustuna_core::{Error, ErrorKind, Result};

use super::{extract_storage, PySamplerContext};
use crate::distribution::PyDistribution;
use crate::exception::err_to_exceptions;
use crate::trial::PyTrialState;

/// A CMA-ES sampler backed by Python's `cmaes` package.
///
/// The optimizer state is kept in memory. It is not stored in the Rustuna storage, so each
/// sampler instance evolves independently. Like Optuna's `CmaEsSampler`, solutions of a
/// generation are reconstructed from completed trials in the storage when the next parameter
/// configuration is asked, so no bookkeeping is required when a trial finishes.
pub struct CmaEsSampler {
    state: Mutex<CmaEsSamplerState>,
    random_sampler: RandomSampler,
}

struct CmaEsSamplerState {
    seed: Option<u64>,
    popsize: Option<usize>,
    optimizer: Option<Py<PyAny>>,
    population_size: Option<usize>,
    search_space: Option<HashMap<String, Distribution>>,
    transform: Option<SearchSpaceTransform>,
    solution_trial_ids: Vec<u32>,
}

impl CmaEsSamplerState {
    fn new(seed: Option<u64>, popsize: Option<usize>) -> Self {
        Self {
            seed,
            popsize,
            optimizer: None,
            population_size: None,
            search_space: None,
            transform: None,
            solution_trial_ids: Vec::new(),
        }
    }

    fn initialize_optimizer(&mut self, transform: &SearchSpaceTransform) -> Result<()> {
        let dimension = transform.bounds().len();
        let (optimizer, population_size) = Python::attach(|py| -> Result<(Py<PyAny>, usize)> {
            let numpy = py.import("numpy").map_err(missing_dependency_err)?;
            let cmaes = py.import("cmaes").map_err(missing_dependency_err)?;
            let mean = numpy
                .call_method1("array", (vec![0.5; dimension],))
                .map_err(py_err)?;
            let bounds = numpy
                .call_method1("array", (transform.bounds().to_vec(),))
                .map_err(py_err)?;
            let kwargs = PyDict::new(py);
            kwargs.set_item("mean", mean).map_err(py_err)?;
            kwargs.set_item("sigma", 1.0 / 6.0).map_err(py_err)?;
            kwargs.set_item("bounds", bounds).map_err(py_err)?;
            kwargs
                .set_item("n_max_resampling", 10 * dimension)
                .map_err(py_err)?;
            if let Some(seed) = self.seed {
                kwargs.set_item("seed", seed).map_err(py_err)?;
            }
            if let Some(popsize) = self.popsize {
                kwargs
                    .set_item("population_size", popsize)
                    .map_err(py_err)?;
            }
            let optimizer = cmaes
                .getattr("CMA")
                .map_err(py_err)?
                .call((), Some(&kwargs))
                .map_err(py_err)?;
            let population_size = optimizer
                .getattr("population_size")
                .map_err(py_err)?
                .extract()
                .map_err(py_err)?;
            Ok((optimizer.unbind(), population_size))
        })?;
        self.optimizer = Some(optimizer);
        self.population_size = Some(population_size);
        Ok(())
    }

    fn ask(&self) -> Result<Vec<f64>> {
        let optimizer = self
            .optimizer
            .as_ref()
            .ok_or_else(|| Error::new(ErrorKind::Unexpected))?;
        Python::attach(|py| {
            optimizer
                .bind(py)
                .call_method0("ask")
                .map_err(py_err)?
                .call_method0("tolist")
                .map_err(py_err)?
                .extract::<Vec<f64>>()
                .map_err(py_err)
        })
    }

    fn tell(&self, solutions: Vec<(Vec<f64>, f64)>) -> Result<()> {
        let optimizer = self
            .optimizer
            .as_ref()
            .ok_or_else(|| Error::new(ErrorKind::Unexpected))?;
        Python::attach(|py| {
            optimizer
                .bind(py)
                .call_method1("tell", (solutions,))
                .map_err(py_err)?;
            Ok(())
        })
    }

    fn sample(
        &mut self,
        ctx: &Context,
        storage: Arc<RwLock<dyn Storage>>,
        search_space: &HashMap<String, Distribution>,
    ) -> Result<HashMap<String, f64>> {
        let search_space: HashMap<_, _> = search_space
            .iter()
            .filter_map(|(name, distribution)| match distribution {
                Distribution::Categorical { .. } => None,
                Distribution::Float { .. } | Distribution::Int { .. } => {
                    Some((name.clone(), distribution.clone()))
                }
            })
            .collect();
        if search_space.is_empty() {
            return Ok(HashMap::new());
        }
        if ctx.directions.len() != 1 {
            return Err(Error::new(ErrorKind::UnsupportedMultiObjective));
        }

        if self.optimizer.is_none() {
            let transform = match SearchSpaceTransform::new(&search_space) {
                Ok(transform) => transform,
                Err(_) => return Ok(HashMap::new()),
            };
            self.initialize_optimizer(&transform)?;
            self.search_space = Some(search_space.clone());
            self.transform = Some(transform);
        }

        if self.search_space.as_ref() != Some(&search_space) {
            return Ok(HashMap::new());
        }

        let population_size = self
            .population_size
            .ok_or_else(|| Error::new(ErrorKind::Unexpected))?;

        let solutions = {
            let transform = self
                .transform
                .as_ref()
                .ok_or_else(|| Error::new(ErrorKind::Unexpected))?;
            let guard = storage.read().map_err(|e| {
                Error::with_reason(
                    ErrorKind::Unexpected,
                    format!("Failed to acquire storage guard: {e}"),
                )
            })?;
            let mut solutions = Vec::with_capacity(population_size);
            for trial_id in self.solution_trial_ids.iter() {
                if solutions.len() == population_size {
                    break;
                }
                let trial = guard.get_cached_trial(*trial_id)?;
                let TrialStateValues::Complete(values) = &trial.state_values else {
                    continue;
                };
                if values.len() != 1 {
                    continue;
                }
                let Ok(x) = transform.transform(&trial.internal_params) else {
                    continue;
                };
                let y = match ctx.directions[0] {
                    Direction::Minimize => values[0],
                    Direction::Maximize => -values[0],
                };
                solutions.push((x, y));
            }
            solutions
        };
        if solutions.len() >= population_size {
            self.tell(solutions)?;
            // TODO(c-bata): Consider calling discard_trials here.
            // let mut guard = storage
            //     .write()
            //     .map_err(|_| Error::new(ErrorKind::Unexpected))?;
            // guard.discard_trials(&self.solution_trial_ids)?;
            self.solution_trial_ids.clear();
        }

        let x = self.ask()?;
        self.solution_trial_ids.push(ctx.trial_id);
        self.transform
            .as_ref()
            .ok_or_else(|| Error::new(ErrorKind::Unexpected))?
            .untransform(&x)
    }
}

impl CmaEsSampler {
    pub fn new(seed: Option<u64>, popsize: Option<usize>) -> Self {
        let random_sampler = match seed {
            Some(seed) => RandomSampler::seed_from_u64(seed),
            None => RandomSampler::new(),
        };
        Self {
            state: Mutex::new(CmaEsSamplerState::new(seed, popsize)),
            random_sampler,
        }
    }
}

impl Sampler for CmaEsSampler {
    fn sample_independent(
        &self,
        ctx: &Context,
        storage: Arc<RwLock<dyn Storage>>,
        name: &str,
        distribution: &Distribution,
    ) -> Result<f64> {
        self.random_sampler
            .sample_independent(ctx, storage, name, distribution)
    }

    fn support_joint_sampling(&self) -> bool {
        true
    }

    fn sample_joint(
        &self,
        ctx: &Context,
        storage: Arc<RwLock<dyn Storage>>,
        search_space: &HashMap<String, Distribution>,
    ) -> Result<HashMap<String, f64>> {
        self.state
            .lock()
            .map_err(|e| {
                Error::with_reason(
                    ErrorKind::SamplerError,
                    format!("Failed to acquire sampler state guard: {e}"),
                )
            })?
            .sample(ctx, storage, search_space)
    }
}

/// Sampler using CMA-ES (Covariance Matrix Adaptation Evolution Strategy) algorithm.
///
/// This sampler is backed by Python's `cmaes` package. The optimizer state is held only in
/// memory. Therefore, sampler instances in separate processes optimize independently.
///
/// Categorical parameters are not supported by CMA-ES. They are excluded from the joint
/// search space and sampled independently.
///
/// Args:
///     seed: Random seed for CMA-ES. If `None`, the backend chooses a random seed.
///     popsize: CMA-ES population size. If `None`, the backend default is used.
#[derive(Clone)]
#[pyclass(name = "CmaEsSampler", from_py_object)]
#[pyo3(module = "rustuna")]
pub struct PyCmaEsSampler {
    pub sampler: Arc<CmaEsSampler>,
}

// These methods release the GIL before acquiring the sampler's internal mutex because
// `CmaEsSampler` re-attaches to Python internally. Acquiring the mutex while holding the GIL
// could deadlock with another thread that holds the mutex and waits for the GIL.
#[pymethods]
impl PyCmaEsSampler {
    #[new]
    #[pyo3(signature = (*, seed = None, popsize = None))]
    fn py_new(seed: Option<u64>, popsize: Option<usize>) -> Self {
        PyCmaEsSampler {
            sampler: Arc::new(CmaEsSampler::new(seed, popsize)),
        }
    }

    #[getter]
    fn support_joint_sampling(&self) -> bool {
        true
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
                .map_err(err_to_exceptions)
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

fn py_err(err: PyErr) -> Error {
    Error::with_reason(ErrorKind::SamplerError, err.to_string())
}

fn missing_dependency_err(err: PyErr) -> Error {
    Error::with_reason(ErrorKind::MissingDependency, err.to_string())
}
