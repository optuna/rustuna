use core::panic;
use std::cmp::Ordering;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rustuna_core::distribution::Distribution;
use rustuna_core::multi_objective;
use rustuna_core::parzen_estimator::ParzenEstimator;
use rustuna_core::sampler::{Context, RandomSampler, Sampler};
use rustuna_core::storage::Storage;
use rustuna_core::study::Direction;
use rustuna_core::trial::TrialStateValues;
use rustuna_core::Result;
use rustuna_core::{Error, ErrorKind};

/// Configuration for [`TpeSampler`].
pub struct TpeConfig {
    /// Whether to use multivariate TPE for joint suggestions over the inferred search space.
    ///
    /// `Some(true)` forces multivariate (joint) sampling and `Some(false)` forces independent
    /// (univariate) sampling. `None` selects automatically, matching Optuna: multivariate for
    /// single-objective studies and independent for multi-objective studies.
    pub multivariate: Option<bool>,
    /// Number of completed trials to collect before switching from random sampling to TPE.
    pub n_startup_trials: usize,
    /// Optional RNG seed.
    pub seed: Option<u64>,
}
impl Default for TpeConfig {
    fn default() -> Self {
        Self {
            multivariate: None,
            n_startup_trials: 10,
            seed: None,
        }
    }
}

type SplitKey = (Vec<u32>, usize);
type SplitValue = (Vec<u32>, Vec<u32>);
/// Tree-structured Parzen Estimator sampler.
///
/// This sampler is the Rustuna counterpart of Optuna's `TPESampler`.
///
/// For each parameter, TPE fits one Parzen estimator `l(x)` to parameter values observed in
/// promising trials and another Parzen estimator `g(x)` to the remaining trials, then chooses
/// the value that maximizes the ratio `l(x) / g(x)`.
///
/// Rustuna uses random sampling until `n_startup_trials` completed trials are available in the
/// same study, then switches to TPE-based suggestions. When `multivariate` is enabled, it uses
/// multivariate TPE to jointly sample parameters from the inferred search space instead of
/// sampling each parameter independently.
///
/// This sampler can also be used for multi-objective optimization. In that case, Rustuna splits
/// completed trials into promising and non-promising sets using the multi-objective variant of
/// TPE and a hypervolume-based weighting rule for promising trials.
///
/// For further information, see:
///
/// - [Algorithms for Hyper-Parameter Optimization](https://papers.nips.cc/paper/4443-algorithms-for-hyper-parameter-optimization.pdf)
/// - [Making a Science of Model Search: Hyperparameter Optimization in Hundreds of Dimensions for Vision Architectures](http://proceedings.mlr.press/v28/bergstra13.pdf)
/// - [Tree-Structured Parzen Estimator: Understanding Its Algorithm Components and Their Roles for Better Empirical Performance](https://arxiv.org/abs/2304.11127)
/// - [Multiobjective Tree-Structured Parzen Estimator for Computationally Expensive Optimization Problems](https://doi.org/10.1145/3377930.3389817)
/// - [Multiobjective Tree-Structured Parzen Estimator](https://doi.org/10.1613/jair.1.13188)
///
/// # Examples
///
/// ```no_run
/// use rustuna_core::storage::InMemoryStorage;
/// use rustuna_core::study::{create_study, Direction};
/// use rustuna_core::Result;
/// use rustuna_sampler::tpe::TpeSampler;
///
/// fn main() -> Result<()> {
///     let storage = InMemoryStorage::new();
///     let study = create_study(
///         "simple-quadratic",
///         storage,
///         TpeSampler::new(),
///         vec![Direction::Minimize],
///     )?;
///
///     study.optimize(
///         |mut trial| {
///             let x = trial.suggest_float("x", -10.0, 10.0)?;
///             Ok(vec![x * x])
///         },
///         100,
///     )?;
///     Ok(())
/// }
/// ```
pub struct TpeSampler {
    rng: StdRng,
    multivariate: Option<bool>,
    n_startup_trials: usize,
    random_sampler: RandomSampler,
    // TODO(y0z): Change to LruCache<(Vec<&PersistedTrial>, usize), (Vec<&PersistedTrial>, Vec<&PersistedTrial>)>
    split_cache: HashMap<SplitKey, SplitValue>,
}
impl Default for TpeSampler {
    fn default() -> Self {
        Self::new()
    }
}
impl TpeSampler {
    /// Creates a sampler from an explicit configuration.
    pub fn from_config(cfg: TpeConfig) -> TpeSampler {
        let mut rng = match cfg.seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_seed(Default::default()),
        };
        let seed_for_random_sampler = rng.gen();
        Self {
            rng,
            multivariate: cfg.multivariate,
            n_startup_trials: cfg.n_startup_trials,
            random_sampler: RandomSampler::seed_from_u64(seed_for_random_sampler),
            split_cache: HashMap::new(),
        }
    }

    /// Creates a sampler with the default configuration.
    ///
    /// The default configuration selects multivariate TPE automatically (multivariate for
    /// single-objective, independent for multi-objective, matching Optuna) and uses random
    /// sampling for the first 10 completed trials.
    pub fn new() -> TpeSampler {
        Self::from_config(TpeConfig::default())
    }

    /// Creates a reproducibly seeded sampler.
    ///
    /// This is equivalent to [`TpeSampler::new`] but initializes the internal random number
    /// generator from the provided seed.
    pub fn seed_from_u64(seed: u64) -> TpeSampler {
        Self::from_config(TpeConfig {
            multivariate: None,
            seed: Some(seed),
            n_startup_trials: 10,
        })
    }

    fn sample(
        &mut self,
        ctx: &Context,
        complete_trials: &[&rustuna_core::trial::PersistedTrial],
        search_space: &HashMap<String, Distribution>,
    ) -> Result<HashMap<String, f64>> {
        let is_multi_objective = ctx.directions.len() > 1;

        let (pe_good, pe_poor) = if !is_multi_objective {
            let gamma = Self::gamma_for_single_objective(complete_trials.len());
            let direction: &Direction = &ctx.directions[0];
            let (good_trials, poor_trials) =
                Self::split_trials_for_single_objective(complete_trials, direction, gamma);
            // Single-objective: recency ramp for both l(x) and g(x) (Optuna default_weights).
            (
                Self::build_parzen_estimator(&good_trials, search_space, true),
                Self::build_parzen_estimator(&poor_trials, search_space, true),
            )
        } else {
            let directions: &[Direction] = &ctx.directions;
            let gamma = Self::gamma_for_multi_objective(complete_trials.len());
            let complete_trial_numbers = complete_trials
                .iter()
                .map(|t| t.number)
                .collect::<Vec<u32>>();
            let split_cache_key = (complete_trial_numbers.clone(), gamma);
            let (good_trials, poor_trials): (
                Vec<&rustuna_core::trial::PersistedTrial>,
                Vec<&rustuna_core::trial::PersistedTrial>,
            ) = if self.split_cache.contains_key(&split_cache_key) {
                let (good_nums, poor_nums) = self.split_cache.get(&split_cache_key).unwrap();
                let good_trials = complete_trials
                    .iter()
                    .copied()
                    .filter(|t| good_nums.contains(&t.number))
                    .collect::<Vec<_>>();
                let poor_trials = complete_trials
                    .iter()
                    .copied()
                    .filter(|t| poor_nums.contains(&t.number))
                    .collect::<Vec<_>>();
                (good_trials, poor_trials)
            } else {
                let (good_trials, poor_trials) = multi_objective::split_trials_for_multi_objective(
                    complete_trials,
                    directions,
                    gamma,
                );
                let good_nums = good_trials.iter().map(|t| t.number).collect();
                let poor_nums = poor_trials.iter().map(|t| t.number).collect();
                // We only cache the most recent split
                self.split_cache.clear();
                self.split_cache
                    .insert(split_cache_key, (good_nums, poor_nums));
                (good_trials, poor_trials)
            };
            // Multi-objective: uniform weights for l(x) (below), recency ramp for g(x) (above),
            // matching Optuna's multi-objective TPE.
            (
                Self::build_parzen_estimator(&good_trials, search_space, false),
                Self::build_parzen_estimator(&poor_trials, search_space, true),
            )
        };

        let n_ei_candidates = 24;
        let samples_good = pe_good.sample(&mut self.rng, n_ei_candidates);
        let mut best_idx = 0usize;
        let mut best_val = f64::NEG_INFINITY;
        for (i, s) in samples_good.iter().enumerate() {
            let acquisition = pe_good.log_pdf(s) - pe_poor.log_pdf(s);
            if acquisition > best_val {
                best_val = acquisition;
                best_idx = i;
            }
        }
        Ok(samples_good[best_idx].clone())
    }

    fn split_trials_for_single_objective<'a>(
        trials: &[&'a rustuna_core::trial::PersistedTrial],
        direction: &Direction,
        gamma: usize,
    ) -> (
        Vec<&'a rustuna_core::trial::PersistedTrial>,
        Vec<&'a rustuna_core::trial::PersistedTrial>,
    ) {
        let n = trials.len();
        assert!(
            gamma <= n,
            "gamma must be less than or equal to the number of trials"
        );

        if n == 0 {
            return (Vec::new(), Vec::new());
        }
        if gamma == n {
            return (trials.to_vec(), Vec::new());
        }

        fn value_for(t: &rustuna_core::trial::PersistedTrial) -> f64 {
            match &t.state_values {
                TrialStateValues::Complete(v) => v[0],
                _ => panic!("Unexpected non-complete trial found during TPE sampling"),
            }
        }

        // NaN trials must always land in `poor_trials` regardless of `direction`:
        // a NaN observation is a failed evaluation, and feeding it into the Parzen
        // estimator would corrupt the `good_trials` model. Using
        // `partial_cmp(...).unwrap_or(Equal)` would let NaN keep its original
        // position in the partial sort and so non-deterministically slip into the
        // good half. Treat NaN as strictly worse than any finite value, in both
        // directions.
        let mut idx: Vec<usize> = (0..n).collect();
        idx.select_nth_unstable_by(gamma, |&i, &j| {
            let vi = value_for(trials[i]);
            let vj = value_for(trials[j]);
            match (vi.is_nan(), vj.is_nan()) {
                (true, true) => Ordering::Equal,
                (true, false) => Ordering::Greater,
                (false, true) => Ordering::Less,
                (false, false) => {
                    let ord = vi
                        .partial_cmp(&vj)
                        .expect("non-NaN partial_cmp must succeed");
                    match direction {
                        Direction::Minimize => ord,
                        Direction::Maximize => ord.reverse(),
                    }
                }
            }
        });

        let mut good_trials = Vec::with_capacity(gamma);
        let mut poor_trials = Vec::with_capacity(n - gamma);
        for &i in idx.iter().take(gamma) {
            good_trials.push(trials[i]);
        }
        for &i in idx.iter().skip(gamma) {
            poor_trials.push(trials[i]);
        }
        (good_trials, poor_trials)
    }

    fn gamma_for_single_objective(n: usize) -> usize {
        let threashold: usize = 25;

        std::cmp::min(((0.1 * n as f64).ceil()) as usize, threashold)
    }

    fn gamma_for_multi_objective(n: usize) -> usize {
        (0.1 * n as f64).ceil() as usize
    }

    /// Keep only trials whose `Complete` values are fully finite-or-±inf. Trials
    /// carrying NaN are dropped here, before the `n_startup_trials` gate, so they
    /// neither inflate `gamma` nor leak into `good_trials`. In the all-NaN /
    /// finite-count-below-startup case the resulting empty / short list naturally
    /// falls through to the random sampler via the existing startup gate.
    fn usable_complete_trials(
        trials: &[Option<rustuna_core::trial::PersistedTrial>],
    ) -> Vec<&rustuna_core::trial::PersistedTrial> {
        trials
            .iter()
            .flatten()
            .filter(|t| match &t.state_values {
                TrialStateValues::Complete(v) => !v.iter().any(|x| x.is_nan()),
                _ => false,
            })
            .collect()
    }

    fn weights_for_single_objective(x: usize) -> Vec<f64> {
        let threashold = 25;
        if x == 0 {
            vec![]
        } else if x < threashold {
            vec![1.0; x]
        } else {
            let n = x - threashold;
            let start = 1.0 / (x as f64);
            if n == 0 {
                vec![1.0; threashold]
            } else if n == 1 {
                let mut v = Vec::with_capacity(threashold + 1);
                v.push(start);
                v.extend(std::iter::repeat_n(1.0, threashold));
                v
            } else {
                let step = (1.0 - start) / ((n - 1) as f64);
                let mut v = Vec::with_capacity(n + threashold);
                for i in 0..n {
                    v.push(start + (i as f64) * step);
                }
                v.extend(std::iter::repeat_n(1.0, threashold));
                v
            }
        }
    }

    fn build_parzen_estimator(
        trials: &[&rustuna_core::trial::PersistedTrial],
        search_space: &HashMap<String, Distribution>,
        recency_ramp: bool,
    ) -> ParzenEstimator {
        let mut sorted_keys: Vec<&String> = search_space.keys().collect();
        sorted_keys.sort();

        let n_params = sorted_keys.len();
        let n_trials = trials.len();

        // Process trials in chronological (trial-number ascending) order so that the
        // recency ramp assigns the highest weights to the most recent trials, matching
        // Optuna's `default_weights`. (Uniform weights are order-invariant, so sorting is
        // harmless there; the split routines do not preserve chronological order.)
        let mut order: Vec<usize> = (0..n_trials).collect();
        order.sort_by_key(|&i| trials[i].number);
        let trials: Vec<&rustuna_core::trial::PersistedTrial> =
            order.iter().map(|&i| trials[i]).collect();

        let mut observations_vec: Vec<Vec<f64>> = (0..n_params)
            .map(|_| Vec::with_capacity(n_trials))
            .collect();
        let mut active_counts: Vec<u32> = vec![0; n_trials];

        for (param_idx, key) in sorted_keys.iter().enumerate() {
            for (trial_idx, t) in trials.iter().enumerate() {
                if let Some(&v) = t.internal_params.get(*key) {
                    observations_vec[param_idx].push(v);
                    active_counts[trial_idx] += 1;
                }
            }
        }
        let n_params_u32 = n_params as u32;
        let active_indices: Vec<usize> = (0..n_trials)
            .filter(|&idx| active_counts[idx] == n_params_u32)
            .collect();
        // Optuna: single-objective uses the recency ramp (`default_weights`) for both the
        // below (`l(x)`) and above (`g(x)`) estimators; multi-objective uses uniform weights
        // for the below estimator and the recency ramp for the above estimator.
        let weights = if recency_ramp {
            Self::weights_for_single_objective(trials.len())
        } else {
            vec![1.0; trials.len()]
        };
        let active_weights: Vec<f64> = active_indices.iter().map(|&i| weights[i]).collect();
        let prior_weight = 1.0;
        let observations: HashMap<String, Vec<f64>> = sorted_keys
            .iter()
            .zip(observations_vec)
            .map(|(k, v)| ((*k).clone(), v))
            .collect();
        ParzenEstimator::new(&observations, search_space, &active_weights, prior_weight)
    }
}

impl Sampler for TpeSampler {
    fn sample_independent(
        &mut self,
        ctx: &Context,
        storage: Arc<RwLock<dyn Storage>>,
        name: &str,
        distribution: &Distribution,
    ) -> Result<f64> {
        if distribution.is_single() {
            return distribution.get_single_value();
        }

        {
            let mut guard = storage.write().map_err(|e| {
                Error::with_reason(
                    ErrorKind::Unexpected,
                    format!("Failed to acquire storage guard: {e}"),
                )
            })?;
            let trials = guard.get_trials(ctx.study_id)?;
            let complete_trials: Vec<&rustuna_core::trial::PersistedTrial> =
                Self::usable_complete_trials(trials);
            if complete_trials.len() >= self.n_startup_trials {
                let search_space = HashMap::from([(name.to_string(), distribution.clone())]);
                let params = self.sample(ctx, &complete_trials, &search_space)?;
                return Ok(params[name]);
            }
        }
        self.random_sampler
            .sample_independent(ctx, storage, name, distribution)
    }

    fn support_joint_sampling(&self) -> bool {
        // `Some(false)` disables joint sampling outright. `Some(true)` and `None` both allow it;
        // the `None` (auto) case is resolved per objective-count inside `sample_joint`, which
        // returns an empty map for multi-objective studies so every parameter falls back to
        // independent sampling (matching Optuna's `multivariate=False` default for MO).
        self.multivariate != Some(false)
    }

    fn sample_joint(
        &mut self,
        ctx: &Context,
        storage: Arc<RwLock<dyn Storage>>,
        search_space: &HashMap<String, Distribution>,
    ) -> Result<HashMap<String, f64>> {
        // Resolve the effective multivariate flag. Default (`None`) follows Optuna: multivariate
        // for single-objective, independent for multi-objective. Returning an empty map routes
        // all parameters through independent sampling.
        let multivariate = self.multivariate.unwrap_or(ctx.directions.len() == 1);
        if !multivariate {
            return Ok(HashMap::new());
        }

        let mut guard = storage.write().map_err(|e| {
            Error::with_reason(
                ErrorKind::Unexpected,
                format!("Failed to acquire storage guard: {e}"),
            )
        })?;
        let trials = guard.get_trials(ctx.study_id)?;
        let complete_trials: Vec<&rustuna_core::trial::PersistedTrial> =
            Self::usable_complete_trials(trials);
        if complete_trials.len() < self.n_startup_trials {
            return Ok(HashMap::new());
        }

        self.sample(ctx, &complete_trials, search_space)
    }

    fn after_trial(
        &mut self,
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
    use rustuna_core::storage::InMemoryStorage;
    use rustuna_core::study::{create_study, Direction};
    use rustuna_core::study::{get_best_trial, get_pareto_front};
    #[test]
    fn test_optimize() {
        let storage = InMemoryStorage::new();
        let directions = vec![Direction::Minimize];
        let study =
            create_study("simple-quadratic", storage, TpeSampler::new(), directions).unwrap();
        study
            .optimize(
                |mut t| {
                    let x = t.suggest_float("x", 0.0, 10.0)?;
                    let y = t.suggest_int("y", 0, 10)?;
                    let z = *t.suggest_categorical("z", &[1, 2, 3, 4, 5])? as i64;
                    let value = (x - 3.0).powi(2) + (y - 5).pow(2) as f64 + (z - 2).pow(2) as f64;
                    println!(
                        "{:2} x: {}, y: {}, z: {}, value: {}",
                        t.number, x, y, z, value
                    );
                    Ok(vec![value])
                },
                50,
            )
            .unwrap();
        let best_trial_number = get_best_trial(&study);
        assert!(best_trial_number.is_ok());
    }

    #[test]
    fn test_optimize_conditional() {
        let storage = InMemoryStorage::new();
        let directions = vec![Direction::Minimize];
        let study =
            create_study("simple-quadratic", storage, TpeSampler::new(), directions).unwrap();
        study
            .optimize(
                |mut t| {
                    let x = t.suggest_float("x", 0.0, 10.0)?;
                    let y = t.suggest_float("y", 0.0, 10.0)?;

                    let mut value = (x - 3.0).powi(2) + (y - 5.0).powi(2);
                    if x < 5.0 {
                        println!("{:2} x: {}, y: {}, value: {}", t.number, x, y, value);
                    } else {
                        let z = t.suggest_float("z", -5.0, 5.0)?;
                        value += z;
                        println!(
                            "{:2} x: {}, y: {}, z: {}, value: {}",
                            t.number, x, y, z, value
                        );
                    }
                    Ok(vec![value])
                },
                50,
            )
            .unwrap();
        let best_trial_number = get_best_trial(&study);
        assert!(best_trial_number.is_ok());
    }

    #[test]
    fn test_multi_objective() {
        let storage = InMemoryStorage::new();
        let directions = vec![Direction::Minimize, Direction::Minimize];
        let study = create_study(
            "simple-bi-objective",
            storage,
            TpeSampler::new(),
            directions,
        )
        .unwrap();
        study
            .optimize(
                |mut t| {
                    let x = t.suggest_float("x", 0.0, 10.0)?;
                    let y = t.suggest_float("y", 0.0, 10.0)?;
                    let values = vec![
                        (x - 3.0).powi(2) + (y - 5.0).powi(2),
                        (x - 7.0).powi(2) + (y - 2.0).powi(2),
                    ];
                    println!("{:2} x: {}, y: {}, values: {:?}", t.number, x, y, values);
                    Ok(values)
                },
                50,
            )
            .unwrap();
        let best_trial_numbers = get_pareto_front(&study);
        assert!(best_trial_numbers.is_ok());
    }

    /// A `+inf` observation (failed evaluation) must not propagate into the reference
    /// point; the sampler builds the reference from the worst *finite* value per
    /// dimension and falls back to input-order selection if a dimension has no finite
    /// observation at all.
    #[test]
    fn multi_objective_handles_plus_inf_objective_values() {
        let storage = InMemoryStorage::new();
        let directions = vec![Direction::Minimize, Direction::Minimize];
        let study = create_study(
            "inf-bi-objective",
            storage,
            TpeSampler::seed_from_u64(0),
            directions,
        )
        .unwrap();
        study
            .optimize(
                |mut t| {
                    let x = t.suggest_float("x", 0.0, 10.0)?;
                    let y = t.suggest_float("y", 0.0, 10.0)?;
                    // Inject `+inf` for some trials to mimic an objective that
                    // occasionally fails / overflows.
                    let f0 = if (t.number as usize).is_multiple_of(7) {
                        f64::INFINITY
                    } else {
                        (x - 3.0).powi(2) + (y - 5.0).powi(2)
                    };
                    let f1 = (x - 7.0).powi(2) + (y - 2.0).powi(2);
                    Ok(vec![f0, f1])
                },
                80,
            )
            .unwrap();
        // Sampler must not panic / hang; the Pareto front should still come from the
        // finite-valued trials.
        let pareto = get_pareto_front(&study).unwrap();
        assert!(!pareto.is_empty());
    }

    /// When every completed trial is NaN the sampler must not feed NaN observations
    /// into the Parzen estimator; the entry-side `usable_complete_trials` filter
    /// drops NaN trials before the startup gate so the sampler falls back to random
    /// sampling cleanly.
    #[test]
    fn all_nan_observations_fall_back_to_random_sampling() {
        let storage = InMemoryStorage::new();
        let directions = vec![Direction::Minimize];
        let study = create_study(
            "all-nan-single-objective",
            storage,
            TpeSampler::seed_from_u64(0),
            directions,
        )
        .unwrap();
        study
            .optimize(
                |mut t| {
                    let _x = t.suggest_float("x", 0.0, 10.0)?;
                    Ok(vec![f64::NAN])
                },
                40,
            )
            .unwrap();
    }

    /// Multi-objective sibling of [`all_nan_observations_fall_back_to_random_sampling`].
    #[test]
    fn all_nan_observations_multi_objective_falls_back_cleanly() {
        let storage = InMemoryStorage::new();
        let directions = vec![Direction::Minimize, Direction::Minimize];
        let study = create_study(
            "all-nan-bi-objective",
            storage,
            TpeSampler::seed_from_u64(0),
            directions,
        )
        .unwrap();
        study
            .optimize(
                |mut t| {
                    let _x = t.suggest_float("x", 0.0, 10.0)?;
                    Ok(vec![f64::NAN, f64::NAN])
                },
                40,
            )
            .unwrap();
    }

    /// Single-objective NaN trials must always land in `poor_trials` so the good-half
    /// Parzen estimator never sees a failed evaluation, regardless of `direction`.
    #[test]
    fn single_objective_nan_is_treated_as_worst() {
        let storage = InMemoryStorage::new();
        let directions = vec![Direction::Minimize];
        let study = create_study(
            "nan-single-objective",
            storage,
            TpeSampler::seed_from_u64(0),
            directions,
        )
        .unwrap();
        study
            .optimize(
                |mut t| {
                    let x = t.suggest_float("x", 0.0, 10.0)?;
                    // Some trials report NaN (failed evaluation).
                    let v = if (t.number as usize) % 5 == 3 {
                        f64::NAN
                    } else {
                        (x - 4.0).powi(2)
                    };
                    Ok(vec![v])
                },
                60,
            )
            .unwrap();
        // Sampler must not panic; the best trial must come from the finite half.
        let best_number = get_best_trial(&study).unwrap();
        let trials = study.get_trials().unwrap();
        let best = trials.iter().find(|t| t.number == best_number).unwrap();
        let v = match &best.state_values {
            TrialStateValues::Complete(vs) => vs[0],
            _ => unreachable!("best trial must be complete"),
        };
        assert!(v.is_finite(), "best trial value should be finite, got {v}");
    }

    /// A `-inf` loss value must not poison HSSP via `ref - point = +inf`. The sampler
    /// classifies any row that contains `-inf` as "infinitely good" and picks those
    /// first, then runs HSSP on the finite remainder.
    #[test]
    fn multi_objective_handles_neg_inf_objective_values() {
        let storage = InMemoryStorage::new();
        let directions = vec![Direction::Minimize, Direction::Minimize];
        let study = create_study(
            "neg-inf-bi-objective",
            storage,
            TpeSampler::seed_from_u64(0),
            directions,
        )
        .unwrap();
        study
            .optimize(
                |mut t| {
                    let x = t.suggest_float("x", 0.0, 10.0)?;
                    let y = t.suggest_float("y", 0.0, 10.0)?;
                    // A handful of trials report `-inf` for the first objective.
                    let f0 = if (t.number as usize) % 11 == 5 {
                        f64::NEG_INFINITY
                    } else {
                        (x - 3.0).powi(2) + (y - 5.0).powi(2)
                    };
                    let f1 = (x - 7.0).powi(2) + (y - 2.0).powi(2);
                    Ok(vec![f0, f1])
                },
                80,
            )
            .unwrap();
        let pareto = get_pareto_front(&study).unwrap();
        assert!(!pareto.is_empty());
    }

    /// Even when every rank-i candidate carries a non-finite value the sampler must
    /// keep making progress: `+inf` reaches the "no finite candidate" branch and pads
    /// from the `+inf` group; `-inf` is taken first via the `-inf` group.
    #[test]
    fn multi_objective_all_non_finite_falls_back_cleanly() {
        let storage = InMemoryStorage::new();
        let directions = vec![Direction::Minimize, Direction::Minimize];
        let study = create_study(
            "all-non-finite-bi-objective",
            storage,
            TpeSampler::seed_from_u64(0),
            directions,
        )
        .unwrap();
        study
            .optimize(
                |mut t| {
                    let _x = t.suggest_float("x", 0.0, 10.0)?;
                    let _y = t.suggest_float("y", 0.0, 10.0)?;
                    // Half of the trials are `+inf` (failed) and half are `-inf`
                    // (impossibly good); no finite observations anywhere.
                    let v = if (t.number as usize).is_multiple_of(2) {
                        f64::INFINITY
                    } else {
                        f64::NEG_INFINITY
                    };
                    Ok(vec![v, v])
                },
                40,
            )
            .unwrap();
    }

    /// Test with single-value parameters to check if bandwidth=0 issue occurs.
    /// Single-value parameters should be excluded from the search space to avoid
    /// creating Parzen estimators with zero bandwidth.
    #[test]
    fn test_single_value_parameters() {
        let storage = InMemoryStorage::new();
        let directions = vec![Direction::Minimize];
        let study = create_study(
            "single-value-test",
            storage,
            TpeSampler::seed_from_u64(42),
            directions,
        )
        .unwrap();

        let result = study.optimize(
            |mut t| {
                // Normal parameter
                let x = t.suggest_float("x", -10.0, 10.0)?;

                // Single-value parameters (should be excluded from search space)
                let y = t.suggest_float("y", 5.0, 5.0)?; // single value: 5.0
                let z = t.suggest_int("z", 3, 3)?; // single value: 3

                let value = (x - 2.0).powi(2) + y + z as f64;
                println!(
                    "{:2} x: {}, y: {}, z: {}, value: {}",
                    t.number, x, y, z, value
                );
                Ok(vec![value])
            },
            30,
        );

        assert!(
            result.is_ok(),
            "Optimization should complete without panicking"
        );

        // Verify single-value params were always constant
        let trials = study.get_trials().unwrap();
        for trial in trials.iter() {
            if let Some(y_val) = trial.internal_params.get("y") {
                assert!(
                    (y_val - 5.0).abs() < 1e-10,
                    "y should always be 5.0, got {}",
                    y_val
                );
            }
            if let Some(z_val) = trial.internal_params.get("z") {
                assert!(
                    (z_val - 3.0).abs() < 1e-10,
                    "z should always be 3.0, got {}",
                    z_val
                );
            }
        }
    }
}
