use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::sync::{Arc, Mutex, RwLock, Weak};

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rustuna_core::distribution::Distribution;
use rustuna_core::internal::multi_objective;
use rustuna_core::internal::parzen_estimator::ParzenEstimator;
use rustuna_core::sampler::{Context, RandomSampler, Sampler};
use rustuna_core::storage::Storage;
use rustuna_core::study::Direction;
use rustuna_core::trial::{PersistedTrial, TrialState, TrialStateValues};
use rustuna_core::Result;
use rustuna_core::{Error, ErrorKind};

/// Builder for [`TpeSampler`], following the API style of [`std::thread::Builder`].
///
/// # Examples
///
/// ```
/// use rustuna_sampler::tpe::TpeBuilder;
///
/// let sampler = TpeBuilder::new()
///     .n_startup_trials(20)
///     .multivariate(true)
///     .seed(42)
///     .build();
/// ```
pub struct TpeBuilder {
    multivariate: Option<bool>,
    n_startup_trials: usize,
    seed: Option<u64>,
}
impl Default for TpeBuilder {
    fn default() -> Self {
        Self::new()
    }
}
impl TpeBuilder {
    /// Creates a builder with the default configuration.
    pub fn new() -> Self {
        Self {
            multivariate: None,
            n_startup_trials: 10,
            seed: None,
        }
    }

    /// Sets whether to force multivariate (joint) sampling. When unset, it is selected
    /// automatically (multivariate for single-objective, independent for multi-objective,
    /// matching Optuna).
    pub fn multivariate(self, multivariate: bool) -> Self {
        Self {
            multivariate: Some(multivariate),
            ..self
        }
    }

    /// Sets the number of completed trials before switching from random sampling to TPE.
    pub fn n_startup_trials(self, n_startup_trials: usize) -> Self {
        Self {
            n_startup_trials,
            ..self
        }
    }

    pub fn seed(self, seed: u64) -> Self {
        Self {
            seed: Some(seed),
            ..self
        }
    }

    pub fn build(self) -> TpeSampler {
        let mut rng = match self.seed {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::from_seed(Default::default()),
        };
        let seed_for_random_sampler = rng.gen();
        TpeSampler {
            rng: Mutex::new(rng),
            multivariate: self.multivariate,
            n_startup_trials: self.n_startup_trials,
            random_sampler: RandomSampler::seed_from_u64(seed_for_random_sampler),
            split_cache: RwLock::new(HashMap::new()),
            observations_cache: RwLock::new(ObservationsCache::default()),
        }
    }
}

type SplitKey = (Vec<u32>, usize);
type SplitValue = (HashSet<u32>, HashSet<u32>);
/// Identity of the storage allocation that owns a study's trial IDs.
#[derive(Clone)]
struct StorageKey(Weak<RwLock<dyn Storage>>);

impl PartialEq for StorageKey {
    fn eq(&self, other: &Self) -> bool {
        self.0.ptr_eq(&other.0)
    }
}
impl Eq for StorageKey {}
impl Hash for StorageKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        // The Weak owns the allocation identity until this key is removed.
        (self.0.as_ptr() as *const ()).hash(state);
    }
}

#[derive(Default)]
struct ObservationsCache {
    per_trial: HashMap<(StorageKey, u32, u32), Option<Arc<TpeObservationSet>>>,
    studies: HashMap<(StorageKey, u32), StudyObservations>,
    next_use: u64,
}

struct StudyObservations {
    seen: Vec<Option<(u32, u32, TrialState)>>,
    observations: Arc<TpeObservationSet>,
    last_used: u64,
}

impl StudyObservations {
    fn new(n_objectives: usize) -> Self {
        Self {
            seen: Vec::new(),
            observations: Arc::new(TpeObservationSet {
                trial_numbers: Vec::new(),
                values: Vec::new(),
                n_objectives,
                param_columns: HashMap::new(),
                feasibles_violations: Vec::new(),
            }),
            last_used: 0,
        }
    }

    fn update(&mut self, trials: &[Option<PersistedTrial>]) -> Result<()> {
        for old in self.seen.iter().skip(trials.len()).flatten() {
            Arc::make_mut(&mut self.observations).remove(old.1);
        }
        self.seen.resize(trials.len(), None);
        for (index, trial) in trials.iter().enumerate() {
            let stamp = trial
                .as_ref()
                .map(|t| (t.id, t.number, t.state_values.state()));
            if self.seen[index] == stamp {
                continue;
            }
            if let Some((_, number, _)) = self.seen[index] {
                if self
                    .observations
                    .trial_numbers
                    .binary_search(&number)
                    .is_ok()
                {
                    Arc::make_mut(&mut self.observations).remove(number);
                }
            }
            if let Some(trial) = trial {
                if matches!(trial.state_values, TrialStateValues::Complete(_)) {
                    Arc::make_mut(&mut self.observations).insert(trial)?;
                }
            }
            self.seen[index] = stamp;
        }
        Ok(())
    }
}

/// Upper bound on cached per-trial snapshots; trials returned from `ask` that
/// never reach `tell` would otherwise leak their entry.
const OBSERVATIONS_CACHE_CAP: usize = 64;

/// Owned, columnar snapshot of the usable completed trials, copied out of the
/// storage so the storage guard can be dropped before the TPE model is built.
/// Row `i` describes one completed trial; rows are in storage order. The
/// snapshot is search-space independent: it carries a column for every
/// parameter observed in a usable completed trial, so one copy per trial
/// serves every suggestion of that trial.
#[derive(Clone)]
struct TpeObservationSet {
    trial_numbers: Vec<u32>,
    /// Objective values, `n_objectives` slots per row.
    values: Vec<f64>,
    n_objectives: usize,
    /// Parameter columns keyed by name, one slot per row; `None` where the
    /// trial did not observe that parameter.
    param_columns: HashMap<String, Vec<Option<f64>>>,
    /// Constraint feasibility and total violation of row `i`.
    feasibles_violations: Vec<(bool, f64)>,
}

/// Borrowed projection of a [`TpeObservationSet`] onto one search space.
/// `param_columns` is in sorted-key order of the search space; `None` marks a
/// parameter that no trial in the snapshot observed.
struct TpeObservationsView<'a> {
    trial_numbers: &'a [u32],
    values: &'a [f64],
    n_objectives: usize,
    param_columns: Vec<TpeObservationColumn<'a>>,
    feasibles_violations: &'a [(bool, f64)],
}

struct TpeObservationColumn<'a> {
    name: &'a str,
    distribution: &'a Distribution,
    values: Option<&'a [Option<f64>]>,
}

impl TpeObservationColumn<'_> {
    fn at(&self, row: usize) -> Option<f64> {
        self.values.and_then(|values| values[row])
    }
}

impl TpeObservationSet {
    fn insert(&mut self, trial: &PersistedTrial) -> Result<()> {
        let TrialStateValues::Complete(values) = &trial.state_values else {
            return Ok(());
        };
        if values.iter().any(|x| x.is_nan()) {
            return Ok(());
        }
        if values.len() != self.n_objectives {
            return Err(Error::with_reason(
                ErrorKind::Unexpected,
                format!(
                    "Trial {} has {} objective values but the study has {} directions",
                    trial.number,
                    values.len(),
                    self.n_objectives
                ),
            ));
        }
        let constraints = trial.constraints()?;
        let feasible = constraints.values().all(|x| *x <= 0.0);
        let violation = constraints.values().filter(|&x| *x > 0.0).sum::<f64>();
        let row = self
            .trial_numbers
            .partition_point(|&number| number < trial.number);
        self.trial_numbers.insert(row, trial.number);
        self.values.splice(
            row * self.n_objectives..row * self.n_objectives,
            values.iter().copied(),
        );
        for column in self.param_columns.values_mut() {
            column.insert(row, None);
        }
        for (name, &value) in &trial.internal_params {
            if let Some(column) = self.param_columns.get_mut(name) {
                column[row] = Some(value);
            } else {
                let mut column = vec![None; self.trial_numbers.len()];
                column[row] = Some(value);
                self.param_columns.insert(name.clone(), column);
            }
        }
        self.feasibles_violations.insert(row, (feasible, violation));
        Ok(())
    }

    fn remove(&mut self, number: u32) {
        if let Ok(row) = self.trial_numbers.binary_search(&number) {
            self.trial_numbers.remove(row);
            self.values
                .drain(row * self.n_objectives..(row + 1) * self.n_objectives);
            self.feasibles_violations.remove(row);
            self.param_columns.retain(|_, column| {
                column.remove(row);
                column.iter().any(Option::is_some)
            });
        }
    }

    fn view<'a>(
        &'a self,
        search_space: &'a HashMap<String, Distribution>,
    ) -> TpeObservationsView<'a> {
        let mut sorted_keys: Vec<&String> = search_space.keys().collect();
        sorted_keys.sort();
        TpeObservationsView {
            trial_numbers: &self.trial_numbers,
            values: &self.values,
            n_objectives: self.n_objectives,
            param_columns: sorted_keys
                .into_iter()
                .map(|name| TpeObservationColumn {
                    name,
                    distribution: &search_space[name],
                    values: self.param_columns.get(name).map(Vec::as_slice),
                })
                .collect(),
            feasibles_violations: &self.feasibles_violations,
        }
    }
}

impl TpeObservationsView<'_> {
    fn len(&self) -> usize {
        self.trial_numbers.len()
    }

    fn values_row(&self, row: usize) -> &[f64] {
        &self.values[row * self.n_objectives..(row + 1) * self.n_objectives]
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
    /// Per-trial snapshot built in `before_trial` and dropped in `after_trial`.
    /// Study columns are retained between trials. Storage is locked before
    /// this cache during updates; both are released before model construction.
    observations_cache: RwLock<ObservationsCache>,
}
impl Default for TpeSampler {
    fn default() -> Self {
        Self::new()
    }
}
impl TpeSampler {
    /// Creates a sampler with the default configuration.
    ///
    /// The default configuration selects multivariate TPE automatically (multivariate for
    /// single-objective, independent for multi-objective, matching Optuna) and uses random
    /// sampling for the first 10 completed trials.
    pub fn new() -> TpeSampler {
        TpeBuilder::new().build()
    }

    /// Creates a reproducibly seeded sampler.
    ///
    /// This is equivalent to [`TpeSampler::new`] but initializes the internal random number
    /// generator from the provided seed.
    pub fn seed_from_u64(seed: u64) -> TpeSampler {
        TpeBuilder::new().seed(seed).build()
    }

    fn sample(
        &self,
        ctx: &Context,
        observations: &TpeObservationsView<'_>,
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
                Self::build_parzen_estimator(observations, &good_rows, true),
                Self::build_parzen_estimator(observations, &poor_rows, true),
            )
        } else {
            let directions: &[Direction] = &ctx.directions;
            let gamma = Self::gamma_for_multi_objective(n);
            let split_cache_key = (observations.trial_numbers.to_vec(), gamma);
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
                            observations.feasibles_violations,
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
                Self::build_parzen_estimator(observations, &good_rows, false),
                Self::build_parzen_estimator(observations, &poor_rows, true),
            )
        };

        assert_eq!(pe_good.parameter_names(), pe_poor.parameter_names());
        let n_ei_candidates = 24;
        let mut samples_good = {
            let mut rng = self.rng.lock().map_err(|e| {
                Error::with_reason(
                    ErrorKind::SamplerError,
                    format!("Failed to acquire RNG guard: {e}"),
                )
            })?;
            pe_good.sample_ordered(&mut rng, n_ei_candidates)
        };
        let mut best_idx = 0usize;
        let mut best_val = f64::NEG_INFINITY;
        let good_scores = pe_good.log_pdf_ordered_batch(&samples_good);
        let poor_scores = pe_poor.log_pdf_ordered_batch(&samples_good);
        for (i, (good, poor)) in good_scores.into_iter().zip(poor_scores).enumerate() {
            let acquisition = good - poor;
            if acquisition > best_val {
                best_val = acquisition;
                best_idx = i;
            }
        }
        Ok(pe_good
            .parameter_names()
            .iter()
            .cloned()
            .zip(samples_good.swap_remove(best_idx))
            .collect())
    }

    fn split_rows_for_single_objective(
        observations: &TpeObservationsView<'_>,
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

    /// Refreshes study-local columns from changed trial states. Finished trials
    /// are immutable under the Storage contract, so their parameter maps need
    /// not be copied again. Each trial retains its own immutable Arc snapshot.
    fn snapshot_from_storage(
        &self,
        ctx: &Context,
        storage: &Arc<RwLock<dyn Storage>>,
    ) -> Result<Option<Arc<TpeObservationSet>>> {
        let mut guard = storage.write().map_err(|e| {
            Error::with_reason(
                ErrorKind::Unexpected,
                format!("Failed to acquire storage guard: {e}"),
            )
        })?;
        let trials = guard.get_trials(ctx.study_id)?;
        // Startup uses random sampling without validating observations. Stop
        // counting at the threshold to avoid a second full scan of the history.
        let n_usable = trials
            .iter()
            .flatten()
            .filter(|trial| match &trial.state_values {
                TrialStateValues::Complete(values) => !values.iter().any(|x| x.is_nan()),
                _ => false,
            })
            .take(self.n_startup_trials)
            .count();
        if n_usable < self.n_startup_trials {
            return Ok(None);
        }
        // Locks always follow storage -> observations. Model construction and
        // cached snapshot reads hold neither lock while calling storage.
        let mut cache = self.observations_cache.write().map_err(|e| {
            Error::with_reason(
                ErrorKind::SamplerError,
                format!("Failed to acquire observations cache guard: {e}"),
            )
        })?;
        cache
            .studies
            .retain(|(owner, _), _| owner.0.strong_count() > 0);
        cache
            .per_trial
            .retain(|(owner, _, _), _| owner.0.strong_count() > 0);
        let key = (StorageKey(Arc::downgrade(storage)), ctx.study_id);
        cache.next_use = cache.next_use.wrapping_add(1);
        let last_used = cache.next_use;
        let history = cache
            .studies
            .entry(key.clone())
            .or_insert_with(|| StudyObservations::new(ctx.directions.len()));
        if let Err(error) = history.update(trials) {
            // Failed updates can leave partial observations. Drop the history so
            // errors cannot bypass eviction and accumulate unbounded entries.
            cache.studies.remove(&key);
            return Err(error);
        }
        history.last_used = last_used;
        let snapshot = (history.observations.trial_numbers.len() >= self.n_startup_trials)
            .then(|| Arc::clone(&history.observations));
        if cache.studies.len() > OBSERVATIONS_CACHE_CAP {
            let oldest = cache
                .studies
                .iter()
                .filter(|(candidate, _)| **candidate != key)
                .min_by_key(|(_, history)| history.last_used)
                .map(|(key, _)| key.clone());
            if let Some(oldest) = oldest {
                cache.studies.remove(&oldest);
            }
        }
        Ok(snapshot)
    }

    /// Returns the trial's cached snapshot, falling back to a direct storage
    /// snapshot when `before_trial` did not run for this trial. `None` means
    /// fewer than `n_startup_trials` usable completed trials were available.
    fn observations_for_trial(
        &self,
        ctx: &Context,
        storage: &Arc<RwLock<dyn Storage>>,
    ) -> Result<Option<Arc<TpeObservationSet>>> {
        let cached = self
            .observations_cache
            .read()
            .map_err(|e| {
                Error::with_reason(
                    ErrorKind::SamplerError,
                    format!("Failed to acquire observations cache guard: {e}"),
                )
            })?
            .per_trial
            .get(&(
                StorageKey(Arc::downgrade(storage)),
                ctx.study_id,
                ctx.trial_id,
            ))
            .cloned();
        match cached {
            Some(observations) => Ok(observations),
            None => self.snapshot_from_storage(ctx, storage),
        }
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
        observations: &TpeObservationsView<'_>,
        rows: &[usize],
        recency_ramp: bool,
    ) -> ParzenEstimator {
        let n_params = observations.param_columns.len();
        let n_trials = rows.len();

        // Process trials in chronological (trial-number ascending) order so that the
        // recency ramp assigns the highest weights to the most recent trials, matching
        // Optuna's `default_weights`. (Uniform weights are order-invariant, so sorting is
        // harmless there; the split routines do not preserve chronological order.)
        let mut order: Vec<usize> = rows.to_vec();
        order.sort_by_key(|&row| observations.trial_numbers[row]);

        let mut active_counts: Vec<u32> = vec![0; n_trials];

        for column in &observations.param_columns {
            for (trial_idx, &row) in order.iter().enumerate() {
                if column.at(row).is_some() {
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
        ParzenEstimator::from_ordered_observations(
            observations.param_columns.iter().map(|column| {
                (
                    column.name,
                    column.distribution,
                    order.iter().filter_map(|&row| column.at(row)),
                )
            }),
            &active_weights,
            prior_weight,
        )
    }
}

impl Sampler for TpeSampler {
    fn before_trial(&self, ctx: &Context, storage: Arc<RwLock<dyn Storage>>) -> Result<()> {
        let observations = self.snapshot_from_storage(ctx, &storage)?;
        let mut cache = self.observations_cache.write().map_err(|e| {
            Error::with_reason(
                ErrorKind::SamplerError,
                format!("Failed to acquire observations cache guard: {e}"),
            )
        })?;
        cache.per_trial.insert(
            (
                StorageKey(Arc::downgrade(&storage)),
                ctx.study_id,
                ctx.trial_id,
            ),
            observations,
        );
        // Trial ids grow monotonically, so the smallest id is the oldest entry.
        while cache.per_trial.len() > OBSERVATIONS_CACHE_CAP {
            let Some(oldest) = cache
                .per_trial
                .keys()
                .min_by_key(|(_, _, trial_id)| *trial_id)
                .cloned()
            else {
                break;
            };
            cache.per_trial.remove(&oldest);
        }
        Ok(())
    }

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
        let Some(observations) = self.observations_for_trial(ctx, &storage)? else {
            return self
                .random_sampler
                .sample_independent(ctx, storage, name, distribution);
        };
        let params = self.sample(ctx, &observations.view(&search_space))?;
        Ok(params[name])
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

        let Some(observations) = self.observations_for_trial(ctx, &storage)? else {
            return Ok(HashMap::new());
        };
        self.sample(ctx, &observations.view(search_space))
    }

    fn after_trial(
        &self,
        ctx: &Context,
        storage: Arc<RwLock<dyn Storage>>,
        _state_values: &TrialStateValues,
    ) -> Result<()> {
        self.observations_cache
            .write()
            .map_err(|e| {
                Error::with_reason(
                    ErrorKind::SamplerError,
                    format!("Failed to acquire observations cache guard: {e}"),
                )
            })?
            .per_trial
            .remove(&(
                StorageKey(Arc::downgrade(&storage)),
                ctx.study_id,
                ctx.trial_id,
            ));
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
        let sampler = TpeBuilder::new().n_startup_trials(2).seed(42).build();
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

    /// A trial served from the `before_trial` cache must sample exactly what
    /// a direct storage snapshot would, including for a parameter no trial
    /// has observed.
    #[test]
    fn test_before_trial_cache_matches_direct_sampling() {
        let storage = InMemoryStorage::new();
        let study = create_study(
            "cache-equality",
            storage,
            TpeSampler::seed_from_u64(0),
            vec![Direction::Minimize],
        )
        .unwrap();
        study
            .optimize(
                |mut t| {
                    let x = t.suggest_float("x", 0.0, 10.0)?;
                    let y = t.suggest_float("y", 0.0, 10.0)?;
                    Ok(vec![(x - 3.0).powi(2) + (y - 5.0).powi(2)])
                },
                15,
            )
            .unwrap();

        let ctx = Context {
            study_id: study.id,
            directions: vec![Direction::Minimize],
            trial_number: 15,
            trial_id: 999,
        };
        let distribution = Distribution::new_float(0.0, 10.0, None, false);
        let cached = TpeSampler::seed_from_u64(7);
        let direct = TpeSampler::seed_from_u64(7);
        cached.before_trial(&ctx, study.storage.clone()).unwrap();
        for name in ["x", "y", "unobserved"] {
            let a = cached
                .sample_independent(&ctx, study.storage.clone(), name, &distribution)
                .unwrap();
            let b = direct
                .sample_independent(&ctx, study.storage.clone(), name, &distribution)
                .unwrap();
            assert_eq!(a, b);
            assert!((0.0..=10.0).contains(&a));
        }
        let search_space = HashMap::from([
            ("x".to_string(), distribution.clone()),
            ("y".to_string(), distribution.clone()),
        ]);
        assert_eq!(
            cached
                .sample_joint(&ctx, study.storage.clone(), &search_space)
                .unwrap(),
            direct
                .sample_joint(&ctx, study.storage.clone(), &search_space)
                .unwrap()
        );
    }

    /// `before_trial` inserts the snapshot (or `None` when fewer than
    /// `n_startup_trials` trials are completed), `after_trial` removes it,
    /// and the cache never grows past its cap.
    #[test]
    fn test_observations_cache_lifecycle() {
        let storage = InMemoryStorage::new();
        let study = create_study(
            "cache-lifecycle",
            storage,
            TpeSampler::seed_from_u64(0),
            vec![Direction::Minimize],
        )
        .unwrap();
        let probe = TpeBuilder::new().n_startup_trials(1).seed(0).build();
        let ctx = |trial_id: u32| Context {
            study_id: study.id,
            directions: vec![Direction::Minimize],
            trial_number: trial_id,
            trial_id,
        };
        let key = |trial_id| {
            (
                StorageKey(Arc::downgrade(&study.storage)),
                study.id,
                trial_id,
            )
        };

        probe
            .before_trial(&ctx(100), study.storage.clone())
            .unwrap();
        assert!(matches!(
            probe
                .observations_cache
                .read()
                .unwrap()
                .per_trial
                .get(&key(100)),
            Some(None)
        ));

        study
            .optimize(|mut t| Ok(vec![t.suggest_float("x", 0.0, 1.0)?]), 2)
            .unwrap();
        probe
            .before_trial(&ctx(101), study.storage.clone())
            .unwrap();
        assert!(matches!(
            probe
                .observations_cache
                .read()
                .unwrap()
                .per_trial
                .get(&key(101)),
            Some(Some(_))
        ));

        probe
            .after_trial(
                &ctx(101),
                study.storage.clone(),
                &TrialStateValues::Complete(vec![0.0]),
            )
            .unwrap();
        assert!(!probe
            .observations_cache
            .read()
            .unwrap()
            .per_trial
            .contains_key(&key(101)));

        for trial_id in 200..(200 + OBSERVATIONS_CACHE_CAP as u32 + 5) {
            probe
                .before_trial(&ctx(trial_id), study.storage.clone())
                .unwrap();
        }
        let cache = probe.observations_cache.read().unwrap();
        assert_eq!(cache.per_trial.len(), OBSERVATIONS_CACHE_CAP);
        // The oldest entries (smallest trial ids) were evicted first.
        assert!(!cache.per_trial.contains_key(&key(100)));
        assert!(cache
            .per_trial
            .contains_key(&key(200 + OBSERVATIONS_CACHE_CAP as u32 + 4)));
    }

    #[test]
    fn malformed_observations_are_validated_only_after_startup() {
        use rustuna_core::attr::{AttrKey, Attrs};

        for values in [vec![0.0], vec![0.0, 1.0]] {
            let study = create_study(
                "startup-validation",
                InMemoryStorage::new(),
                TpeBuilder::new().n_startup_trials(2).build(),
                vec![Direction::Minimize],
            )
            .unwrap();
            let mut trial = PersistedTrial::new(0, study.id, 0);
            if values.len() == 1 {
                trial.attrs =
                    Attrs::from([(AttrKey::System("constraints:c".into()), "broken".into())]);
            }
            // Bypass add_trial's objective-count and NaN validation for malformed data.
            for values in [values, vec![f64::NAN]] {
                trial.state_values = TrialStateValues::Complete(values);
                study
                    .storage
                    .write()
                    .unwrap()
                    .create_new_trial_from_template(study.id, &trial)
                    .unwrap();
            }
            assert!(study.ask().is_ok()); // NaN must not count toward startup.
            trial.state_values = TrialStateValues::Complete(vec![1.0]);
            trial.attrs.clear();
            study.add_trial(trial).unwrap();
            assert!(study.ask().is_err()); // Two non-NaN completed trials.
        }
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

    #[test]
    fn study_columns_preserve_pending_trials_constraints_and_immutable_snapshots() {
        let mut storage =
            InMemoryStorage::new_with_option(rustuna_core::storage::InMemoryStorageOptions {
                apply_discard: true,
            });
        let study_id = storage
            .create_new_study("history", vec![Direction::Minimize; 2])
            .unwrap()
            .id;
        let earlier = storage.create_new_trial(study_id).unwrap().id;
        let later = storage.create_new_trial(study_id).unwrap().id;
        let nan = storage.create_new_trial(study_id).unwrap().id;
        let failed = storage.create_new_trial(study_id).unwrap().id;
        let distribution = Distribution::new_float(0.0, 1.0, None, false);
        for (trial, name, value) in [(earlier, "x", 0.1), (later, "x", 0.2), (later, "y", 0.8)] {
            storage
                .set_trial_param(trial, name, &distribution, value)
                .unwrap();
        }
        for (trial, state) in [
            (later, TrialStateValues::Complete(vec![3.0, 4.0])),
            (nan, TrialStateValues::Complete(vec![f64::NAN, 1.0])),
            (failed, TrialStateValues::Fail),
        ] {
            storage.set_trial_state_values(trial, state).unwrap();
        }
        let mut history = StudyObservations::new(2);
        history
            .update(storage.get_trials(study_id).unwrap())
            .unwrap();
        let old_snapshot = Arc::clone(&history.observations);
        assert_eq!(old_snapshot.trial_numbers, [1]);
        assert_eq!(old_snapshot.values, [3.0, 4.0]);
        history
            .update(storage.get_trials(study_id).unwrap())
            .unwrap();
        assert!(Arc::ptr_eq(&old_snapshot, &history.observations));

        storage
            .set_trial_attrs(
                earlier,
                rustuna_core::attr::Attrs::from([(
                    rustuna_core::attr::AttrKey::System("constraints:limit".into()),
                    "2".into(),
                )]),
                false,
            )
            .unwrap();
        storage
            .set_trial_state_values(earlier, TrialStateValues::Complete(vec![1.0, 2.0]))
            .unwrap();
        history
            .update(storage.get_trials(study_id).unwrap())
            .unwrap();
        assert_eq!(history.observations.trial_numbers, [0, 1]);
        assert_eq!(history.observations.values, [1.0, 2.0, 3.0, 4.0]);
        assert_eq!(
            history.observations.param_columns["x"],
            [Some(0.1), Some(0.2)]
        );
        assert_eq!(history.observations.param_columns["y"], [None, Some(0.8)]);
        assert_eq!(
            history.observations.feasibles_violations,
            [(false, 2.0), (true, 0.0)]
        );
        assert_eq!(old_snapshot.values, [3.0, 4.0]);
        assert!(!Arc::ptr_eq(&old_snapshot, &history.observations));
        assert!(storage
            .set_trial_param(earlier, "x", &distribution, 0.9)
            .is_err());
        assert!(storage
            .set_trial_state_values(earlier, TrialStateValues::Complete(vec![9.0, 9.0]))
            .is_err());
        assert!(storage
            .set_trial_attrs(earlier, rustuna_core::attr::Attrs::new(), false)
            .is_err());

        let before_discard = Arc::clone(&history.observations);
        storage.discard_trials(&[later]).unwrap();
        history
            .update(storage.get_trials(study_id).unwrap())
            .unwrap();
        assert_eq!(history.observations.trial_numbers, [0]);
        assert_eq!(history.observations.values, [1.0, 2.0]);
        assert!(!history.observations.param_columns.contains_key("y"));
        assert_eq!(before_discard.values, [1.0, 2.0, 3.0, 4.0]);

        let mut fresh = StudyObservations::new(2);
        fresh.update(storage.get_trials(study_id).unwrap()).unwrap();
        assert_eq!(
            history.observations.param_columns,
            fresh.observations.param_columns
        );
        assert_eq!(history.observations.values, fresh.observations.values);
    }

    #[test]
    fn observation_caches_distinguish_storages_with_identical_study_and_trial_ids() {
        let make_storage = |value| -> Arc<RwLock<dyn Storage>> {
            let mut storage = InMemoryStorage::new();
            let study_id = storage
                .create_new_study("independent", vec![Direction::Minimize])
                .unwrap()
                .id;
            let trial_id = storage.create_new_trial(study_id).unwrap().id;
            storage
                .set_trial_param(
                    trial_id,
                    "x",
                    &Distribution::new_float(0.0, 10.0, None, false),
                    value,
                )
                .unwrap();
            storage
                .set_trial_state_values(trial_id, TrialStateValues::Complete(vec![value]))
                .unwrap();
            Arc::new(RwLock::new(storage))
        };
        let first = make_storage(1.0);
        let second = make_storage(9.0);
        let sampler = TpeBuilder::new().n_startup_trials(1).seed(0).build();
        let ctx = Context {
            study_id: 0,
            trial_id: 100,
            trial_number: 1,
            directions: vec![Direction::Minimize],
        };
        sampler.before_trial(&ctx, first.clone()).unwrap();
        sampler.before_trial(&ctx, second.clone()).unwrap();
        let values = |storage: &Arc<RwLock<dyn Storage>>| {
            sampler
                .observations_for_trial(&ctx, storage)
                .unwrap()
                .unwrap()
                .values
                .clone()
        };
        assert_eq!(values(&first), [1.0]);
        assert_eq!(values(&second), [9.0]);
        sampler
            .after_trial(&ctx, first.clone(), &TrialStateValues::Fail)
            .unwrap();
        assert_eq!(values(&second), [9.0]);
        drop(first);
        sampler.before_trial(&ctx, second).unwrap();
        assert_eq!(sampler.observations_cache.read().unwrap().studies.len(), 1);
    }

    #[test]
    fn failed_study_updates_do_not_retain_partial_histories() {
        use rustuna_core::attr::{AttrKey, Attrs};

        for prime_cache in [false, true] {
            let mut storage = InMemoryStorage::new();
            let study_id = storage
                .create_new_study("malformed", vec![Direction::Minimize])
                .unwrap()
                .id;
            let mut trial = PersistedTrial::new(0, study_id, 0);
            trial.state_values = TrialStateValues::Complete(vec![1.0]);
            storage
                .create_new_trial_from_template(study_id, &trial)
                .unwrap();
            let storage: Arc<RwLock<dyn Storage>> = Arc::new(RwLock::new(storage));
            let sampler = TpeBuilder::new().n_startup_trials(1).build();
            let ctx = Context {
                study_id,
                trial_id: u32::MAX,
                trial_number: 2,
                directions: vec![Direction::Minimize],
            };
            if prime_cache {
                sampler.snapshot_from_storage(&ctx, &storage).unwrap();
            }
            trial.attrs = Attrs::from([(AttrKey::System("constraints:c".into()), "broken".into())]);
            storage
                .write()
                .unwrap()
                .create_new_trial_from_template(study_id, &trial)
                .unwrap();
            for _ in 0..2 {
                assert!(sampler.before_trial(&ctx, storage.clone()).is_err());
                let cache = sampler.observations_cache.read().unwrap();
                assert!(cache.studies.is_empty());
                assert!(cache.per_trial.is_empty());
            }
        }
    }

    #[test]
    fn retained_study_histories_have_a_bounded_capacity() {
        let storage: Arc<RwLock<dyn Storage>> = Arc::new(RwLock::new(InMemoryStorage::new()));
        let sampler = TpeBuilder::new().n_startup_trials(0).build();
        for index in 0..OBSERVATIONS_CACHE_CAP + 5 {
            let study_id = storage
                .write()
                .unwrap()
                .create_new_study(&format!("bounded-{index}"), vec![Direction::Minimize])
                .unwrap()
                .id;
            let ctx = Context {
                study_id,
                trial_id: study_id,
                trial_number: 0,
                directions: vec![Direction::Minimize],
            };
            sampler.before_trial(&ctx, storage.clone()).unwrap();
        }
        let cache = sampler.observations_cache.read().unwrap();
        assert_eq!(cache.studies.len(), OBSERVATIONS_CACHE_CAP);
        assert_eq!(cache.per_trial.len(), OBSERVATIONS_CACHE_CAP);
        assert!(!cache
            .studies
            .contains_key(&(StorageKey(Arc::downgrade(&storage)), 0)));
        assert!(cache.studies.contains_key(&(
            StorageKey(Arc::downgrade(&storage)),
            (OBSERVATIONS_CACHE_CAP + 4) as u32
        )));
    }
}
