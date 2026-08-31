use rand::prelude::*;
use rand::rngs::StdRng;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};

use crate::distribution::Distribution;
use crate::storage::Storage;
use crate::study::Direction;
use crate::trial::TrialStateValues;
use crate::Result;

#[derive(Debug, Clone)]
/// Lightweight study and trial metadata passed to samplers.
///
/// Rustuna does not pass full `Study` or `FrozenTrial` objects to samplers. Instead, it provides
/// the identifiers and objective directions required to make sampling decisions, while richer
/// state can be retrieved through [`Storage`] when necessary.
pub struct Context {
    pub study_id: u32,
    pub directions: Vec<Direction>,
    pub trial_number: u32,
    pub trial_id: u32,
}

/// Interface for parameter suggestion algorithms.
///
/// Like Optuna, Rustuna supports two sampling modes:
/// - independent sampling, which suggests one parameter at a time without modeling relationships
///   between parameters, and
/// - joint sampling, which suggests multiple parameters at once from a shared search space.
///
/// In Optuna terminology, Rustuna's joint sampling is the counterpart of relative sampling.
/// If [`Sampler::support_joint_sampling`] returns `true`, [`Sampler::sample_joint`] is called once
/// at the beginning of a trial with the inferred joint search space. Parameters not returned from
/// that method, or all parameters when joint sampling is disabled, fall back to
/// [`Sampler::sample_independent`].
///
/// A sampler can be shared by multiple optimization threads. Implementations must synchronize
/// mutable state internally and keep immutable sampling work outside critical sections.
pub trait Sampler: Send + Sync {
    /// Hook called after a trial is created and before its search space is inferred.
    ///
    /// This corresponds to Optuna's `before_trial` hook. The trial can be retrieved from
    /// `storage` with `ctx.trial_id`, and the study can be retrieved with `ctx.study_id`.
    fn before_trial(&self, _ctx: &Context, _storage: Arc<RwLock<dyn Storage>>) -> Result<()> {
        Ok(())
    }

    /// Samples a single parameter independently.
    ///
    /// This method is used for parameters that are not covered by joint sampling. It is suitable
    /// for samplers such as random search or univariate TPE that decide each parameter
    /// separately.
    fn sample_independent(
        &self,
        ctx: &Context,
        storage: Arc<RwLock<dyn Storage>>,
        name: &str,
        distribution: &Distribution,
    ) -> Result<f64>;
    /// Returns whether the sampler supports joint sampling.
    ///
    /// When this returns `true`, [`Sampler::sample_joint`] is called once per trial before any
    /// parameter suggestions are requested from the objective function.
    fn support_joint_sampling(&self) -> bool;
    /// Samples multiple parameters at once from a joint search space.
    ///
    /// This is the Rustuna counterpart of Optuna's `sample_relative`. The input search space is
    /// inferred from previously observed compatible parameters in the study.
    fn sample_joint(
        &self,
        ctx: &Context,
        storage: Arc<RwLock<dyn Storage>>,
        search_space: &HashMap<String, Distribution>,
    ) -> Result<HashMap<String, f64>>;
    /// Hook called when a trial is about to be finalized with the given state.
    ///
    /// This corresponds to Optuna's `after_trial` hook. Rustuna calls it after the objective
    /// function returns and before the finalized state is written to storage.
    fn after_trial(
        &self,
        _ctx: &Context,
        _storage: Arc<RwLock<dyn Storage>>,
        _state_values: &TrialStateValues,
    ) -> Result<()> {
        Ok(())
    }
}

/// Uniform random sampler.
///
/// This sampler draws values independently from each parameter distribution and does not perform
/// joint sampling.
pub struct RandomSampler {
    rng: Mutex<StdRng>,
}
impl Default for RandomSampler {
    fn default() -> Self {
        RandomSampler::new()
    }
}
impl RandomSampler {
    /// Creates a sampler with a deterministic default seed.
    pub fn new() -> RandomSampler {
        RandomSampler {
            rng: Mutex::new(StdRng::from_seed(Default::default())),
        }
    }

    /// Creates a sampler seeded from a user-specified 64-bit seed.
    pub fn seed_from_u64(seed: u64) -> RandomSampler {
        RandomSampler {
            rng: Mutex::new(StdRng::seed_from_u64(seed)),
        }
    }
}

fn round_to_step(value: f64, low: f64, high: f64, step: f64) -> f64 {
    let mut stepped = low + ((value - low) / step).round() * step;
    if stepped < low {
        stepped = low;
    }
    if stepped > high {
        stepped = high;
    }
    stepped
}

fn sample_float_with_step(
    rng: &mut StdRng,
    low: f64,
    high: f64,
    step: Option<f64>,
    log: bool,
) -> f64 {
    match (step, log) {
        (None, false) => rng.gen_range(low..high),
        (None, true) => rng.gen_range(low.ln()..high.ln()).exp(),
        (Some(step), false) => {
            let max_index = ((high - low) / step).floor().max(0.0) as i64;
            let index = rng.gen_range(0..=max_index);
            low + (index as f64) * step
        }
        (Some(step), true) => {
            let value = rng.gen_range(low.ln()..high.ln()).exp();
            round_to_step(value, low, high, step)
        }
    }
}

fn sample_int_with_step(rng: &mut StdRng, low: i64, high: i64, step: i64, log: bool) -> f64 {
    let low_f = low as f64;
    let high_f = high as f64;
    let step_f = step as f64;
    if log {
        let value = rng.gen_range(low_f.ln()..high_f.ln()).exp();
        let max_index = ((high_f - low_f) / step_f).floor().max(0.0) as i64;
        let mut index = ((value - low_f) / step_f).round() as i64;
        if index < 0 {
            index = 0;
        }
        if index > max_index {
            index = max_index;
        }
        low_f + (index as f64) * step_f
    } else {
        let max_index = ((high - low) / step).max(0);
        let index = rng.gen_range(0..=max_index);
        (low + index * step) as f64
    }
}

impl Sampler for RandomSampler {
    fn sample_independent(
        &self,
        _ctx: &Context,
        _storage: Arc<RwLock<dyn Storage>>,
        _name: &str,
        distribution: &Distribution,
    ) -> Result<f64> {
        // Single-value distributions have only one possible value.
        if distribution.is_single() {
            return distribution.get_single_value();
        }

        let mut rng = self.rng.lock().map_err(|e| {
            crate::Error::with_reason(
                crate::ErrorKind::SamplerError,
                format!("Failed to acquire RNG guard: {e}"),
            )
        })?;
        match distribution {
            Distribution::Float {
                low,
                high,
                step,
                log,
            } => {
                let value = sample_float_with_step(&mut rng, *low, *high, *step, *log);
                Ok(value)
            }
            Distribution::Int {
                low,
                high,
                step,
                log,
            } => {
                let value = sample_int_with_step(&mut rng, *low, *high, *step, *log);
                Ok(value)
            }
            Distribution::Categorical { cardinality } => {
                let param_value = rng.gen_range(0..*cardinality);
                Ok(param_value as f64)
            }
        }
    }

    fn support_joint_sampling(&self) -> bool {
        false
    }

    fn sample_joint(
        &self,
        _ctx: &Context,
        _storage: Arc<RwLock<dyn Storage>>,
        _search_space: &HashMap<String, Distribution>,
    ) -> Result<HashMap<String, f64>> {
        unreachable!()
    }

    fn after_trial(
        &self,
        _ctx: &Context,
        _storage: Arc<RwLock<dyn Storage>>,
        _state_values: &TrialStateValues,
    ) -> Result<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    use crate::storage::InMemoryStorage;
    use crate::study::create_study;
    use crate::trial::Trial;

    pub struct DummyJointSampler {
        joint_params: HashMap<String, f64>,
    }
    impl Sampler for DummyJointSampler {
        fn sample_independent(
            &self,
            _ctx: &Context,
            _storage: Arc<RwLock<dyn Storage>>,
            _name: &str,
            _distribution: &Distribution,
        ) -> Result<f64> {
            Ok(0.0)
        }

        fn support_joint_sampling(&self) -> bool {
            true
        }

        fn sample_joint(
            &self,
            _ctx: &Context,
            _storage: Arc<RwLock<dyn Storage>>,
            _search_space: &HashMap<String, Distribution>,
        ) -> Result<HashMap<String, f64>> {
            Ok(self.joint_params.clone())
        }

        fn after_trial(
            &self,
            _ctx: &Context,
            _storage: Arc<RwLock<dyn Storage>>,
            _state_values: &TrialStateValues,
        ) -> Result<()> {
            Ok(())
        }
    }

    fn objective(mut t: Trial) -> Result<Vec<f64>> {
        let x = t.suggest_float("x", -10.0, 10.0)?;
        let y = t.suggest_float("y", -10.0, 10.0)?;
        Ok(vec![x * x + y * y])
    }

    #[test]
    fn random_sampler_can_be_shared_between_threads() -> Result<()> {
        let sampler: Arc<dyn Sampler> = Arc::new(RandomSampler::seed_from_u64(0));
        let storage: Arc<RwLock<dyn Storage>> = Arc::new(RwLock::new(InMemoryStorage::new()));
        let distribution = Distribution::new_float(-1.0, 1.0, None, false);
        let mut handles = Vec::new();

        for thread_id in 0..4 {
            let sampler = Arc::clone(&sampler);
            let storage = Arc::clone(&storage);
            let distribution = distribution.clone();
            handles.push(thread::spawn(move || -> Result<()> {
                for trial_number in 0..100 {
                    let ctx = Context {
                        study_id: 0,
                        directions: vec![Direction::Minimize],
                        trial_number,
                        trial_id: thread_id * 100 + trial_number,
                    };
                    let value = sampler.sample_independent(
                        &ctx,
                        Arc::clone(&storage),
                        "x",
                        &distribution,
                    )?;
                    assert!((-1.0..1.0).contains(&value));
                }
                Ok(())
            }));
        }

        for handle in handles {
            handle.join().expect("sampling thread must not panic")?;
        }
        Ok(())
    }

    #[test]
    fn test_joint_sampling_empty() -> Result<()> {
        let joint_params = HashMap::new();
        let study = create_study(
            "dummy",
            InMemoryStorage::new(),
            DummyJointSampler { joint_params },
            vec![Direction::Minimize],
        )?;
        study.optimize(objective, 2)?;
        Ok(())
    }

    #[test]
    fn test_joint_sampling_partially() -> Result<()> {
        let mut joint_params = HashMap::new();
        joint_params.insert(String::from("x"), 0.5);

        let study = create_study(
            "dummy",
            InMemoryStorage::new(),
            DummyJointSampler { joint_params },
            vec![Direction::Minimize],
        )?;
        study.optimize(objective, 2)?;

        let trials = study.get_trials()?;
        assert_eq!(trials.len(), 2);
        assert_eq!(trials[1].internal_params["x"], 0.5);
        assert!(trials[1].internal_params.contains_key("y"));
        Ok(())
    }

    #[test]
    fn test_joint_sampling_all() -> Result<()> {
        let mut joint_params = HashMap::new();
        joint_params.insert(String::from("x"), 1.0);
        joint_params.insert(String::from("y"), 1.0);

        let study = create_study(
            "dummy",
            InMemoryStorage::new(),
            DummyJointSampler { joint_params },
            vec![Direction::Minimize],
        )?;
        study.optimize(objective, 2)?;

        let trials = study.get_trials()?;
        assert_eq!(trials.len(), 2);
        assert_eq!(trials[1].internal_params["x"], 1.0);
        assert_eq!(trials[1].internal_params["y"], 1.0);
        Ok(())
    }
}
