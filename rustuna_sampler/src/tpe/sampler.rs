use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex, RwLock};

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
type SplitValue = (HashSet<u32>, HashSet<u32>);

/// Owned, columnar snapshot of the usable completed trials, copied out of the
/// storage so the storage guard can be dropped before the TPE model is built.
/// Row `i` describes one completed trial; rows are in storage order. The flat
/// layout keeps the copy under the storage guard to a handful of allocations
/// regardless of the trial count.
struct TpeObservations {
    trial_numbers: Vec<u32>,
    /// Objective values, `n_objectives` slots per row.
    values: Vec<f64>,
    n_objectives: usize,
    /// Parameter values in sorted-key order of the requested search space,
    /// `n_params` slots per row; `None` where the trial did not observe that
    /// parameter.
    params: Vec<Option<f64>>,
    n_params: usize,
    /// Constraint feasibility and total violation of row `i`.
    feasibles_violations: Vec<(bool, f64)>,
}

impl TpeObservations {
    fn len(&self) -> usize {
        self.trial_numbers.len()
    }

    fn values_row(&self, row: usize) -> &[f64] {
        &self.values[row * self.n_objectives..(row + 1) * self.n_objectives]
    }

    fn params_row(&self, row: usize) -> &[Option<f64>] {
        &self.params[row * self.n_params..(row + 1) * self.n_params]
    }
}

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
    rng: Mutex<StdRng>,
    multivariate: Option<bool>,
    n_startup_trials: usize,
    random_sampler: RandomSampler,
    // TODO(y0z): Change to LruCache<(Vec<&PersistedTrial>, usize), (Vec<&PersistedTrial>, Vec<&PersistedTrial>)>
    split_cache: RwLock<HashMap<SplitKey, SplitValue>>,
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
            rng: Mutex::new(rng),
            multivariate: cfg.multivariate,
            n_startup_trials: cfg.n_startup_trials,
            random_sampler: RandomSampler::seed_from_u64(seed_for_random_sampler),
            split_cache: RwLock::new(HashMap::new()),
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
        &self,
        ctx: &Context,
        observations: &TpeObservations,
        search_space: &HashMap<String, Distribution>,
    ) -> Result<HashMap<String, f64>> {
        let n = observations.len();
        let is_multi_objective = ctx.directions.len() > 1;

        let (pe_good, pe_poor) = if !is_multi_objective {
            let gamma = Self::gamma_for_single_objective(n);
            let direction: &Direction = &ctx.directions[0];
            let (good_rows, poor_rows) =
                Self::split_rows_for_single_objective(observations, direction, gamma);
            // Single-objective: recency ramp for both l(x) and g(x) (Optuna default_weights).
            (
                Self::build_parzen_estimator(observations, &good_rows, search_space, true),
                Self::build_parzen_estimator(observations, &poor_rows, search_space, true),
            )
        } else {
            let directions: &[Direction] = &ctx.directions;
            let gamma = Self::gamma_for_multi_objective(n);
            let split_cache_key = (observations.trial_numbers.clone(), gamma);
            let cached_split = self
                .split_cache
                .read()
                .map_err(|e| {
                    Error::with_reason(
                        ErrorKind::SamplerError,
                        format!("Failed to acquire split cache guard: {e}"),
                    )
                })?
                .get(&split_cache_key)
                .cloned();
            let (good_rows, poor_rows): (Vec<usize>, Vec<usize>) =
                if let Some((good_nums, poor_nums)) = cached_split {
                    let good_rows = (0..n)
                        .filter(|&row| good_nums.contains(&observations.trial_numbers[row]))
                        .collect();
                    let poor_rows = (0..n)
                        .filter(|&row| poor_nums.contains(&observations.trial_numbers[row]))
                        .collect();
                    (good_rows, poor_rows)
                } else {
                    let value_rows = (0..n)
                        .map(|row| observations.values_row(row))
                        .collect::<Vec<_>>();
                    let (good_rows, poor_rows) =
                        multi_objective::split_observation_indices_for_multi_objective(
                            &value_rows,
                            &observations.feasibles_violations,
                            directions,
                            gamma,
                        );
                    let good_nums = good_rows
                        .iter()
                        .map(|&row| observations.trial_numbers[row])
                        .collect();
                    let poor_nums = poor_rows
                        .iter()
                        .map(|&row| observations.trial_numbers[row])
                        .collect();
                    let mut split_cache = self.split_cache.write().map_err(|e| {
                        Error::with_reason(
                            ErrorKind::SamplerError,
                            format!("Failed to acquire split cache guard: {e}"),
                        )
                    })?;
                    split_cache.clear();
                    split_cache.insert(split_cache_key, (good_nums, poor_nums));
                    (good_rows, poor_rows)
                };
            // Multi-objective: uniform weights for l(x) (below), recency ramp for g(x) (above),
            // matching Optuna's multi-objective TPE.
            (
                Self::build_parzen_estimator(observations, &good_rows, search_space, false),
                Self::build_parzen_estimator(observations, &poor_rows, search_space, true),
            )
        };

        let n_ei_candidates = 24;
        let samples_good = {
            let mut rng = self.rng.lock().map_err(|e| {
                Error::with_reason(
                    ErrorKind::SamplerError,
                    format!("Failed to acquire RNG guard: {e}"),
                )
            })?;
            pe_good.sample(&mut rng, n_ei_candidates)
        };
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

    fn split_rows_for_single_objective(
        observations: &TpeObservations,
        direction: &Direction,
        gamma: usize,
    ) -> (Vec<usize>, Vec<usize>) {
        let n = observations.len();
        assert!(
            gamma <= n,
            "gamma must be less than or equal to the number of trials"
        );

        if n == 0 {
            return (Vec::new(), Vec::new());
        }
        if gamma == n {
            return ((0..n).collect(), Vec::new());
        }

        let value_for = |row: usize| observations.values_row(row)[0];

        // NaN trials must always land in `poor_rows` regardless of `direction`:
        // a NaN observation is a failed evaluation, and feeding it into the Parzen
        // estimator would corrupt the `good_rows` model. Using
        // `partial_cmp(...).unwrap_or(Equal)` would let NaN keep its original
        // position in the partial sort and so non-deterministically slip into the
        // good half. Treat NaN as strictly worse than any finite value, in both
        // directions.
        let mut idx: Vec<usize> = (0..n).collect();
        let compare_feasibles = |i: usize, j: usize| {
            let vi = value_for(i);
            let vj = value_for(j);
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
        };
        idx.select_nth_unstable_by(gamma, |&i, &j| {
            let (feasible_i, violation_i) = observations.feasibles_violations[i];
            let (feasible_j, violation_j) = observations.feasibles_violations[j];
            match (feasible_i, feasible_j) {
                (true, true) => compare_feasibles(i, j),
                (true, false) => Ordering::Less,
                (false, true) => Ordering::Greater,
                (false, false) => violation_i
                    .partial_cmp(&violation_j)
                    .expect("NaN is already filtered."),
            }
        });

        let good_rows = idx[..gamma].to_vec();
        let poor_rows = idx[gamma..].to_vec();
        (good_rows, poor_rows)
    }

    fn gamma_for_single_objective(n: usize) -> usize {
        let threashold: usize = 25;

        std::cmp::min(((0.1 * n as f64).ceil()) as usize, threashold)
    }

    fn gamma_for_multi_objective(n: usize) -> usize {
        (0.1 * n as f64).ceil() as usize
    }

    /// Counts the trials that [`Self::snapshot_usable_complete_trials`] would
    /// keep: only trials whose `Complete` values are fully finite-or-±inf.
    /// Trials carrying NaN are dropped here, before the `n_startup_trials`
    /// gate, so they neither inflate `gamma` nor leak into the good rows. In
    /// the all-NaN / finite-count-below-startup case the short count naturally
    /// falls through to the random sampler via the existing startup gate.
    fn count_usable_complete_trials(
        trials: &[Option<rustuna_core::trial::PersistedTrial>],
    ) -> usize {
        trials
            .iter()
            .flatten()
            .filter(|t| match &t.state_values {
                TrialStateValues::Complete(v) => !v.iter().any(|x| x.is_nan()),
                _ => false,
            })
            .count()
    }

    /// Copies the usable completed trials (the same selection as
    /// [`Self::count_usable_complete_trials`], in storage order) into owned
    /// observations so the storage guard can be dropped before the TPE model
    /// is built. `n_usable` is that count, used to size the buffers up front.
    /// Parameter values are captured in sorted-key order of `search_space`;
    /// constraint feasibility is evaluated per trial.
    fn snapshot_usable_complete_trials(
        trials: &[Option<rustuna_core::trial::PersistedTrial>],
        search_space: &HashMap<String, Distribution>,
        n_objectives: usize,
        n_usable: usize,
    ) -> Result<TpeObservations> {
        let mut sorted_keys = search_space.keys().collect::<Vec<_>>();
        sorted_keys.sort();
        let n_params = sorted_keys.len();
        let mut observations = TpeObservations {
            trial_numbers: Vec::with_capacity(n_usable),
            values: Vec::with_capacity(n_usable * n_objectives),
            n_objectives,
            params: Vec::with_capacity(n_usable * n_params),
            n_params,
            feasibles_violations: Vec::with_capacity(n_usable),
        };
        for trial in trials.iter().flatten() {
            let TrialStateValues::Complete(values) = &trial.state_values else {
                continue;
            };
            if values.iter().any(|x| x.is_nan()) {
                continue;
            }
            if values.len() != n_objectives {
                return Err(Error::with_reason(
                    ErrorKind::Unexpected,
                    format!(
                        "Trial {} has {} objective values but the study has {} directions",
                        trial.number,
                        values.len(),
                        n_objectives
                    ),
                ));
            }
            let constraints = trial.constraints()?;
            let feasible = constraints.values().all(|x| *x <= 0.0);
            let violation = constraints.values().filter(|&x| *x > 0.0).sum::<f64>();
            observations.trial_numbers.push(trial.number);
            observations.values.extend_from_slice(values);
            for name in &sorted_keys {
                observations
                    .params
                    .push(trial.internal_params.get(*name).copied());
            }
            observations
                .feasibles_violations
                .push((feasible, violation));
        }
        Ok(observations)
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
        observations: &TpeObservations,
        rows: &[usize],
        search_space: &HashMap<String, Distribution>,
        recency_ramp: bool,
    ) -> ParzenEstimator {
        let mut sorted_keys: Vec<&String> = search_space.keys().collect();
        sorted_keys.sort();

        let n_params = sorted_keys.len();
        let n_trials = rows.len();
        debug_assert_eq!(n_params, observations.n_params);

        // Process trials in chronological (trial-number ascending) order so that the
        // recency ramp assigns the highest weights to the most recent trials, matching
        // Optuna's `default_weights`. (Uniform weights are order-invariant, so sorting is
        // harmless there; the split routines do not preserve chronological order.)
        let mut order: Vec<usize> = rows.to_vec();
        order.sort_by_key(|&row| observations.trial_numbers[row]);

        let mut observations_vec: Vec<Vec<f64>> = (0..n_params)
            .map(|_| Vec::with_capacity(n_trials))
            .collect();
        let mut active_counts: Vec<u32> = vec![0; n_trials];

        for (trial_idx, &row) in order.iter().enumerate() {
            for (param_idx, param) in observations.params_row(row).iter().enumerate() {
                if let Some(v) = param {
                    observations_vec[param_idx].push(*v);
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
            Self::weights_for_single_objective(n_trials)
        } else {
            vec![1.0; n_trials]
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
        &self,
        ctx: &Context,
        storage: Arc<RwLock<dyn Storage>>,
        name: &str,
        distribution: &Distribution,
    ) -> Result<f64> {
        if distribution.is_single() {
            return distribution.get_single_value();
        }

        let search_space = HashMap::from([(name.to_string(), distribution.clone())]);
        // Hold the storage guard only while copying the observations; the TPE
        // model construction below runs without any storage lock so concurrent
        // trials can keep reading and writing the storage meanwhile.
        let observations = {
            let mut guard = storage.write().map_err(|e| {
                Error::with_reason(
                    ErrorKind::Unexpected,
                    format!("Failed to acquire storage guard: {e}"),
                )
            })?;
            let trials = guard.get_trials(ctx.study_id)?;
            let n_usable = Self::count_usable_complete_trials(trials);
            if n_usable >= self.n_startup_trials {
                Some(Self::snapshot_usable_complete_trials(
                    trials,
                    &search_space,
                    ctx.directions.len(),
                    n_usable,
                )?)
            } else {
                None
            }
        };
        if let Some(observations) = observations {
            let params = self.sample(ctx, &observations, &search_space)?;
            return Ok(params[name]);
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
        &self,
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
        // A dynamic search space may have no parameters in common across completed
        // trials. In that case, let each parameter fall back to independent sampling
        // instead of trying to build a Parzen estimator with no observations.
        if search_space.is_empty() {
            return Ok(HashMap::new());
        }

        // Hold the storage guard only while copying the observations; the TPE
        // model construction below runs without any storage lock so concurrent
        // trials can keep reading and writing the storage meanwhile.
        let observations = {
            let mut guard = storage.write().map_err(|e| {
                Error::with_reason(
                    ErrorKind::Unexpected,
                    format!("Failed to acquire storage guard: {e}"),
                )
            })?;
            let trials = guard.get_trials(ctx.study_id)?;
            let n_usable = Self::count_usable_complete_trials(trials);
            if n_usable < self.n_startup_trials {
                return Ok(HashMap::new());
            }
            Self::snapshot_usable_complete_trials(
                trials,
                search_space,
                ctx.directions.len(),
                n_usable,
            )?
        };

        self.sample(ctx, &observations, search_space)
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
    fn test_dynamic_float_range_falls_back_to_independent_sampling() {
        let storage = InMemoryStorage::new();
        let directions = vec![Direction::Minimize];
        let sampler = TpeSampler::from_config(TpeConfig {
            multivariate: None,
            n_startup_trials: 2,
            seed: Some(42),
        });
        let study = create_study("dynamic-float-range", storage, sampler, directions).unwrap();

        study
            .optimize(
                |mut t| {
                    let x = if t.number % 2 == 0 {
                        t.suggest_float("x", 0.0, 1.0)?
                    } else {
                        t.suggest_float("x", 0.5, 1.0)?
                    };
                    assert!((0.0..=1.0).contains(&x));
                    if t.number % 2 == 1 {
                        assert!(x >= 0.5);
                    }
                    Ok(vec![(x - 0.75).powi(2)])
                },
                6,
            )
            .unwrap();
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
    /// into the Parzen estimator; the entry-side snapshot filter
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

    #[test]
    fn test_single_objective_constraint() {
        let storage = InMemoryStorage::new();
        let directions = vec![Direction::Minimize];
        let study = create_study(
            "single-objective-constraint",
            storage,
            TpeSampler::seed_from_u64(0),
            directions,
        )
        .unwrap();
        let result = study.optimize(
            |mut t| {
                let x = t.suggest_float("x", -15.0, 15.0)?;
                let c0 = x.powi(2) - 8.0;
                t.set_constraints(HashMap::from([(String::from("c0"), c0)]))?;
                Ok(vec![x.powi(2)])
            },
            100,
        );
        assert!(
            result.is_ok(),
            "Optimization should complete without panicking."
        );
    }

    /// Trials that never call `set_constraints` have an empty constraint map and
    /// count as feasible; mixing them with constrained trials in one study must
    /// not panic or corrupt the split.
    #[test]
    fn test_mixed_constrained_and_unconstrained_trials() {
        let storage = InMemoryStorage::new();
        let directions = vec![Direction::Minimize];
        let study = create_study(
            "mixed-constraints",
            storage,
            TpeSampler::seed_from_u64(0),
            directions,
        )
        .unwrap();
        let result = study.optimize(
            |mut t| {
                let x = t.suggest_float("x", -10.0, 10.0)?;
                if t.number % 2 == 0 {
                    let c0 = x - 5.0;
                    t.set_constraints(HashMap::from([(String::from("c0"), c0)]))?;
                }
                Ok(vec![x.powi(2)])
            },
            60,
        );
        assert!(
            result.is_ok(),
            "Optimization should complete without panicking."
        );
    }

    /// One `TpeSampler` shared by several optimizing threads must stay
    /// consistent: every trial completes and the total count is exact.
    #[test]
    fn test_shared_sampler_across_threads() {
        let n_threads = 4;
        let n_trials_per_thread = 30;
        let storage = InMemoryStorage::new();
        let directions = vec![Direction::Minimize];
        let study = create_study(
            "threaded-quadratic",
            storage,
            TpeSampler::seed_from_u64(0),
            directions,
        )
        .unwrap();
        std::thread::scope(|scope| {
            let handles: Vec<_> = (0..n_threads)
                .map(|_| {
                    let study = study.clone();
                    scope.spawn(move || {
                        study.optimize(
                            |mut t| {
                                let x = t.suggest_float("x", -10.0, 10.0)?;
                                let y = t.suggest_float("y", -10.0, 10.0)?;
                                Ok(vec![(x - 3.0).powi(2) + (y + 1.0).powi(2)])
                            },
                            n_trials_per_thread,
                        )
                    })
                })
                .collect();
            for handle in handles {
                handle
                    .join()
                    .expect("optimization thread must not panic")
                    .unwrap();
            }
        });
        let trials = study.get_trials().unwrap();
        assert_eq!(trials.len(), n_threads * n_trials_per_thread);
        assert!(trials
            .iter()
            .all(|t| matches!(t.state_values, TrialStateValues::Complete(_))));
    }

    #[test]
    fn test_multi_objective_constraint_with_few_feasible_trials() {
        // Only a tiny slice of the search space is feasible, so the number of feasible
        // trials stays below gamma. The good half then has to be padded with the least
        // violating infeasible trials.
        let storage = InMemoryStorage::new();
        let directions = vec![Direction::Minimize, Direction::Minimize];
        let study = create_study(
            "multi-objective-constraint",
            storage,
            TpeSampler::seed_from_u64(0),
            directions,
        )
        .unwrap();
        let result = study.optimize(
            |mut t| {
                let x = t.suggest_float("x", 0.0, 1.0)?;
                let y = t.suggest_float("y", 0.0, 1.0)?;
                let c0 = if x < 0.02 { -1.0 } else { 1.0 };
                t.set_constraints(HashMap::from([(String::from("c0"), c0)]))?;
                Ok(vec![x, y])
            },
            200,
        );
        assert!(
            result.is_ok(),
            "Optimization should complete without panicking."
        );
    }
}
