use std::collections::HashMap;
use std::ops::DerefMut;
use std::sync::{Arc, Mutex, MutexGuard, RwLock, RwLockReadGuard, RwLockWriteGuard};

use rand::prelude::*;
use rand::rngs::StdRng;
use rustuna_core::attr::{AttrKey, Attrs};
use rustuna_core::distribution::Distribution;
use rustuna_core::sampler::{Context, Sampler};
use rustuna_core::storage::Storage;
use rustuna_core::study::dominates;
use rustuna_core::study::Direction;
use rustuna_core::trial::validate_trials;
use rustuna_core::trial::{PersistedTrial, TrialStateValues};
use rustuna_core::Result;
use rustuna_core::{Error, ErrorKind};

const PARENT_CACHE_KEY_PREFIX: &str = "NSGAIISampler:parent:";

/// NSGA-II sampler for multi-objective optimization.
///
/// NSGA-II stands for Nondominated Sorting Genetic Algorithm II, a fast and elitist
/// multi-objective genetic algorithm.
///
/// This sampler is the Rustuna counterpart of Optuna's `NSGAIISampler`. It tracks generations in
/// trial system attributes, performs elite population selection using non-dominated sorting and
/// crowding distance, and generates child solutions by crossover and mutation.
///
/// For further information, see
/// [A fast and elitist multiobjective genetic algorithm: NSGA-II](https://doi.org/10.1109/4235.996017).
///
/// # Examples
///
/// ```no_run
/// use rustuna_core::storage::InMemoryStorage;
/// use rustuna_core::study::{create_study, Direction};
/// use rustuna_core::Result;
/// use rustuna_sampler::nsgaii::NSGAIISampler;
///
/// fn main() -> Result<()> {
///     let storage = InMemoryStorage::new();
///     let study = create_study(
///         "bi-objective",
///         storage,
///         NSGAIISampler::new(50, None, 0.9, 0.5),
///         vec![Direction::Minimize, Direction::Maximize],
///     )?;
///
///     study.optimize(
///         |mut trial| {
///             let x = trial.suggest_float("x", -100.0, 100.0)?;
///             let y = trial.suggest_float("y", -100.0, 100.0)?;
///             Ok(vec![x * x + y * y, -((x - 2.0).powi(2) + y * y)])
///         },
///         100,
///     )?;
///     Ok(())
/// }
/// ```
pub struct NSGAIISampler {
    rng: Mutex<StdRng>,
    population_size: usize,
    mutation_prob: Option<f64>,
    crossover_prob: f64,
    swapping_prob: f64,
    /// Cache mapping generation number to completed trial numbers in that generation.
    /// Updated incrementally in `after_trial` so `sample_joint` does not scan all trials every time.
    generation_to_numbers: RwLock<HashMap<u32, Vec<u32>>>,
    generation_sync_state: Mutex<GenerationSyncState>,
}

#[derive(Default)]
struct GenerationSyncState {
    cached_study_id: Option<u32>,
    unfinished_trial_numbers: Vec<u32>,
    unseen_trial_start: usize,
}
impl Default for NSGAIISampler {
    fn default() -> Self {
        Self::new(50, None, 0.9, 0.5)
    }
}

impl NSGAIISampler {
    /// Creates an NSGA-II sampler.
    ///
    /// `population_size` is the number of individuals retained in each generation.
    /// `mutation_prob` is the probability of mutating each parameter when generating a child.
    /// `crossover_prob` is the probability of generating a child by crossover rather than
    /// cloning one parent. `swapping_prob` is the probability of taking each parameter from the
    /// second parent during crossover.
    pub fn new(
        population_size: usize,
        mutation_prob: Option<f64>,
        crossover_prob: f64,
        swapping_prob: f64,
    ) -> NSGAIISampler {
        NSGAIISampler {
            rng: Mutex::new(StdRng::from_seed(Default::default())),
            population_size,
            mutation_prob,
            crossover_prob,
            swapping_prob,
            generation_to_numbers: RwLock::new(HashMap::new()),
            generation_sync_state: Mutex::new(GenerationSyncState::default()),
        }
    }
    /// Creates a reproducibly seeded NSGA-II sampler.
    ///
    /// This is equivalent to [`NSGAIISampler::new`] but initializes the internal random number
    /// generator from the provided seed.
    pub fn seed_from_u64(
        seed: u64,
        population_size: usize,
        mutation_prob: Option<f64>,
        crossover_prob: f64,
        swapping_prob: f64,
    ) -> NSGAIISampler {
        NSGAIISampler {
            rng: Mutex::new(StdRng::seed_from_u64(seed)),
            population_size,
            mutation_prob,
            crossover_prob,
            swapping_prob,
            generation_to_numbers: RwLock::new(HashMap::new()),
            generation_sync_state: Mutex::new(GenerationSyncState::default()),
        }
    }
    fn get_rng_lock(&self) -> Result<MutexGuard<'_, StdRng>> {
        self.rng.lock().map_err(|e| {
            Error::with_reason(
                ErrorKind::SamplerError,
                format!("Failed to acquire RNG guard: {e}"),
            )
        })
    }
    fn get_generation_to_numbers_read_lock(
        &self,
    ) -> Result<RwLockReadGuard<'_, HashMap<u32, Vec<u32>>>> {
        self.generation_to_numbers.read().map_err(|e| {
            Error::with_reason(
                ErrorKind::SamplerError,
                format!("Failed to acquire generation_to_numbers read guard: {e}"),
            )
        })
    }
    fn get_generation_to_numbers_write_lock(
        &self,
    ) -> Result<RwLockWriteGuard<'_, HashMap<u32, Vec<u32>>>> {
        self.generation_to_numbers.write().map_err(|e| {
            Error::with_reason(
                ErrorKind::SamplerError,
                format!("Failed to acquire generation_to_numbers write guard: {e}"),
            )
        })
    }
    fn get_generation_sync_state_lock(&self) -> Result<MutexGuard<'_, GenerationSyncState>> {
        self.generation_sync_state.lock().map_err(|e| {
            Error::with_reason(
                ErrorKind::SamplerError,
                format!("Failed to acquire generation_sync_state guard: {e}"),
            )
        })
    }
    fn sync_generation_cache(
        &self,
        study_id: u32,
        trials: &[Option<PersistedTrial>],
    ) -> Result<()> {
        let mut sync_state = self.get_generation_sync_state_lock()?;
        let mut generation_to_numbers = self.get_generation_to_numbers_write_lock()?;

        if sync_state.cached_study_id != Some(study_id)
            || trials.len() < sync_state.unseen_trial_start
        {
            sync_state.cached_study_id = Some(study_id);
            generation_to_numbers.clear();
            sync_state.unfinished_trial_numbers.clear();
            sync_state.unseen_trial_start = 0;
        }

        let mut next_unfinished_trial_numbers = Vec::with_capacity(
            sync_state.unfinished_trial_numbers.len()
                + trials.len().saturating_sub(sync_state.unseen_trial_start),
        );
        for trial_number in sync_state.unfinished_trial_numbers.iter().copied() {
            let Some(trial) = trials.get(trial_number as usize).and_then(Option::as_ref) else {
                continue;
            };
            match &trial.state_values {
                TrialStateValues::Complete(_) => {
                    if let Some(generation) = trial
                        .attrs
                        .get(&AttrKey::System("generation".into()))
                        .and_then(|generation| generation.parse::<u32>().ok())
                    {
                        let numbers = generation_to_numbers.entry(generation).or_default();
                        if !numbers.contains(&trial.number) {
                            numbers.push(trial.number);
                        }
                    }
                }
                TrialStateValues::Pruned | TrialStateValues::Fail => {}
                TrialStateValues::Running | TrialStateValues::Waiting => {
                    next_unfinished_trial_numbers.push(trial.number);
                }
            }
        }

        for trial in trials.iter().skip(sync_state.unseen_trial_start).flatten() {
            match &trial.state_values {
                TrialStateValues::Complete(_) => {
                    if let Some(generation) = trial
                        .attrs
                        .get(&AttrKey::System("generation".into()))
                        .and_then(|generation| generation.parse::<u32>().ok())
                    {
                        let numbers = generation_to_numbers.entry(generation).or_default();
                        if !numbers.contains(&trial.number) {
                            numbers.push(trial.number);
                        }
                    }
                }
                TrialStateValues::Pruned | TrialStateValues::Fail => {}
                TrialStateValues::Running | TrialStateValues::Waiting => {
                    if !next_unfinished_trial_numbers.contains(&trial.number) {
                        next_unfinished_trial_numbers.push(trial.number);
                    }
                }
            }
        }
        sync_state.unfinished_trial_numbers = next_unfinished_trial_numbers;
        sync_state.unseen_trial_start = trials.len();
        Ok(())
    }
    /// Builds the study-system-attribute key under which parent trial IDs for `generation`
    /// are persisted.
    fn parent_cache_key(generation: u32) -> AttrKey {
        AttrKey::System(format!("{PARENT_CACHE_KEY_PREFIX}{generation}").into())
    }

    /// Encodes a list of trial numbers as a JSON array of trial IDs for persistence.
    fn encode_parent_trial_ids(
        trials: &[Option<PersistedTrial>],
        population_numbers: &[u32],
    ) -> Result<String> {
        let trial_ids = population_numbers
            .iter()
            .map(|number| {
                trials
                    .get(*number as usize)
                    .and_then(Option::as_ref)
                    .map(|trial| trial.id)
                    .ok_or_else(|| Error::new(ErrorKind::TrialDiscarded))
            })
            .collect::<Result<Vec<_>>>()?;
        serde_json::to_string(&trial_ids).map_err(|error| {
            Error::with_reason(
                ErrorKind::Unexpected,
                format!("Failed to encode NSGA-II parent cache: {error}"),
            )
        })
    }

    /// Decodes a persisted JSON array of trial IDs back into trial numbers.
    ///
    /// Returns `None` when the cache is stale or incompatible with the current sampler,
    /// allowing the caller to fall back to full recomputation.
    fn decode_parent_population_numbers(
        mut get_trial_number: impl FnMut(u32) -> Result<Option<u32>>,
        encoded: &str,
        population_size: usize,
    ) -> Result<Option<Vec<u32>>> {
        let trial_ids: Vec<u32> = serde_json::from_str(encoded).map_err(|error| {
            Error::with_reason(
                ErrorKind::StorageError,
                format!("Invalid NSGA-II parent cache: {error}"),
            )
        })?;
        if trial_ids.len() != population_size {
            return Ok(None);
        }
        let mut unique_trial_ids = trial_ids.clone();
        unique_trial_ids.sort_unstable();
        unique_trial_ids.dedup();
        if unique_trial_ids.len() != trial_ids.len() {
            return Ok(None);
        }

        let mut population_numbers = Vec::with_capacity(trial_ids.len());
        for trial_id in trial_ids {
            let Some(trial_number) = get_trial_number(trial_id)? else {
                return Ok(None);
            };
            population_numbers.push(trial_number);
        }
        Ok(Some(population_numbers))
    }

    fn select_elite_population_numbers(
        &self,
        ctx: &Context,
        trials: &[Option<PersistedTrial>],
        population_numbers: &[u32],
    ) -> Result<Vec<u32>> {
        let population_numbers_per_rank = fast_non_dominated_sort(ctx, trials, population_numbers)?;

        let mut elite_population_numbers = vec![];
        for population_numbers in population_numbers_per_rank {
            if elite_population_numbers.len() + population_numbers.len() <= self.population_size {
                elite_population_numbers.extend(population_numbers);
            } else {
                let n = self.population_size - elite_population_numbers.len();
                let crowding_sorted_population_numbers =
                    crowding_distance_sort(ctx, trials, population_numbers)?;
                elite_population_numbers.extend(&crowding_sorted_population_numbers[..n]);
                break;
            }
        }
        Ok(elite_population_numbers)
    }
    fn get_child_generation(
        &self,
        study_id: u32,
        trials: &[Option<PersistedTrial>],
    ) -> Result<u32> {
        self.sync_generation_cache(study_id, trials)?;
        let mut child_generation = 0u32;
        loop {
            let full = self
                .get_generation_to_numbers_read_lock()?
                .get(&child_generation)
                .is_some_and(|numbers| numbers.len() >= self.population_size);
            if !full {
                break;
            }
            child_generation = child_generation.checked_add(1).ok_or_else(|| {
                Error::with_reason(ErrorKind::Unexpected, "NSGA-II generation overflow")
            })?;
        }
        Ok(child_generation)
    }

    fn get_parent_population_numbers(
        &self,
        ctx: &Context,
        trials: &[Option<PersistedTrial>],
        child_generation: u32,
        cached_parent: Option<(u32, Vec<u32>)>,
    ) -> Result<(u32, Vec<u32>, Attrs)> {
        if child_generation == 0 {
            return Ok((0, Vec::new(), Attrs::new()));
        }

        // Try to restore the parent population from persisted cache.
        let (first_missing_generation, mut parent_population_numbers) =
            if let Some((generation, numbers)) = cached_parent {
                if generation == child_generation {
                    return Ok((child_generation, numbers, Attrs::new()));
                }
                (generation + 1, numbers)
            } else {
                (1, Vec::new())
            };

        // Recompute elite selection for each missing generation.
        let mut new_attrs = Attrs::new();
        for generation in first_missing_generation..=child_generation {
            let population_numbers = match self
                .get_generation_to_numbers_read_lock()?
                .get(&(generation - 1))
            {
                Some(numbers) if numbers.len() >= self.population_size => numbers.clone(),
                _ => break,
            };

            let mut candidates = population_numbers;
            candidates.append(&mut parent_population_numbers);
            parent_population_numbers =
                self.select_elite_population_numbers(ctx, trials, &candidates)?;

            let encoded = Self::encode_parent_trial_ids(trials, &parent_population_numbers)?;
            new_attrs.insert(Self::parent_cache_key(generation), encoded);
        }

        Ok((child_generation, parent_population_numbers, new_attrs))
    }
    fn crossover(
        &self,
        parent0: &HashMap<String, f64>,
        parent1: &HashMap<String, f64>,
        sorted_names: &[&str],
    ) -> Result<HashMap<String, f64>> {
        let mut child = HashMap::new();
        for name in sorted_names {
            let param_value0 = *parent0.get(*name).unwrap();
            let param_value1 = *parent1.get(*name).unwrap();
            let param_value = if self.get_rng_lock()?.gen_bool(self.swapping_prob) {
                param_value1
            } else {
                param_value0
            };
            child.insert((*name).to_string(), param_value);
        }
        Ok(child)
    }
}
impl Sampler for NSGAIISampler {
    fn sample_independent(
        &self,
        _ctx: &Context,
        _storage: Arc<RwLock<dyn Storage>>,
        _name: &str,
        distribution: &Distribution,
    ) -> Result<f64> {
        if distribution.is_single() {
            return distribution.get_single_value();
        }

        let mut rng = self.get_rng_lock()?;
        match distribution {
            Distribution::Float {
                low,
                high,
                step,
                log,
            } => {
                let param_value = match (step, log) {
                    (None, false) => rng.gen_range(*low..*high),
                    (None, true) => rng.gen_range(low.ln()..high.ln()).exp(),
                    (Some(step), false) => {
                        let max_index = ((high - low) / step).floor().max(0.0) as i64;
                        let index = rng.gen_range(0..=max_index);
                        low + (index as f64) * step
                    }
                    (Some(step), true) => {
                        let value = rng.gen_range(low.ln()..high.ln()).exp();
                        let mut stepped = low + ((value - low) / step).round() * step;
                        if stepped < *low {
                            stepped = *low;
                        }
                        if stepped > *high {
                            stepped = *high;
                        }
                        stepped
                    }
                };
                Ok(param_value)
            }
            Distribution::Int {
                low,
                high,
                step,
                log,
            } => {
                let low_f = *low as f64;
                let high_f = *high as f64;
                let step_f = *step as f64;
                let param_value = if *log {
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
                };
                Ok(param_value)
            }
            Distribution::Categorical { cardinality } => {
                let param_value = rng.gen_range(0..*cardinality);
                Ok(param_value as f64)
            }
        }
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
        let mut guard = storage
            .write()
            .map_err(|_e| Error::new(ErrorKind::Unexpected))?;
        let (child_generation, parent_population_numbers, parent_cache_attrs) = {
            let child_generation = {
                let trials = guard.get_trials(ctx.study_id)?;
                self.get_child_generation(ctx.study_id, trials)?
            };

            let study_attrs = guard.get_study(ctx.study_id)?.attrs.clone();

            let cached_parent = if child_generation == 0 {
                None
            } else {
                // get_child_generation calls get_trials above, which synchronizes this study's
                // trials into the storage cache. Resolve only the persisted parent IDs from that
                // cache instead of rebuilding an ID-to-number map from every completed trial.
                let mut cached_parent = None;
                for generation in (1..=child_generation).rev() {
                    let Some(encoded) = study_attrs.get(&Self::parent_cache_key(generation)) else {
                        continue;
                    };
                    let numbers = Self::decode_parent_population_numbers(
                        |trial_id| match guard.get_cached_trial(trial_id) {
                            Ok(trial)
                                if trial.study_id == ctx.study_id
                                    && matches!(
                                        trial.state_values,
                                        TrialStateValues::Complete(_)
                                    ) =>
                            {
                                Ok(Some(trial.number))
                            }
                            Ok(_) => Ok(None),
                            Err(error)
                                if matches!(
                                    &error.kind,
                                    ErrorKind::TrialNotFound | ErrorKind::TrialDiscarded
                                ) =>
                            {
                                Ok(None)
                            }
                            Err(error) => Err(error),
                        },
                        encoded,
                        self.population_size,
                    )?;
                    if let Some(numbers) = numbers {
                        cached_parent = Some((generation, numbers));
                        break;
                    }
                }
                cached_parent
            };

            let trials = guard.get_trials(ctx.study_id)?;
            self.get_parent_population_numbers(ctx, trials, child_generation, cached_parent)?
        };
        let mut attrs = Attrs::with_capacity(1);
        attrs.insert(
            AttrKey::System("generation".into()),
            (child_generation as f64).to_string(),
        );
        guard.set_trial_attrs(ctx.trial_id, attrs, false)?;
        if !parent_cache_attrs.is_empty() {
            guard.set_study_attrs(ctx.study_id, parent_cache_attrs, false)?;
        }

        if child_generation == 0 {
            drop(guard);
            return Ok(HashMap::new());
        }

        let (parent0_number, parent1_number) = {
            let mut selected = parent_population_numbers
                .choose_multiple(self.get_rng_lock()?.deref_mut(), 2)
                .copied();
            let parent0_number = selected.next().ok_or_else(|| {
                Error::with_reason(
                    ErrorKind::SamplerError,
                    "NSGA-II requires at least two parent trials",
                )
            })?;
            let parent1_number = selected.next().ok_or_else(|| {
                Error::with_reason(
                    ErrorKind::SamplerError,
                    "NSGA-II requires at least two parent trials",
                )
            })?;
            (parent0_number, parent1_number)
        };

        let trials = guard.get_trials(ctx.study_id)?;
        let sorted_names = sorted_parameter_names(search_space);
        let build_parent_params = |number: u32| -> Result<HashMap<String, f64>> {
            let trial = trials
                .get(number as usize)
                .and_then(Option::as_ref)
                .ok_or_else(|| Error::new(ErrorKind::TrialDiscarded))?;
            let mut params = HashMap::with_capacity(search_space.len());
            for name in &sorted_names {
                let param_value = *trial.internal_params.get(*name).unwrap();
                params.insert((*name).to_string(), param_value);
            }
            Ok(params)
        };
        let parent0 = build_parent_params(parent0_number)?;
        let parent1 = build_parent_params(parent1_number)?;
        drop(guard);

        let child = if self.get_rng_lock()?.gen_bool(self.crossover_prob) {
            self.crossover(&parent0, &parent1, &sorted_names)?
        } else {
            parent0
        };

        let mutation_prob = self
            .mutation_prob
            .unwrap_or(1.0 / 1.0_f64.max(child.len() as f64));
        let mut params = HashMap::new();

        for name in &sorted_names {
            if !self.get_rng_lock()?.gen_bool(mutation_prob) {
                let param_value = *child.get(*name).unwrap();
                params.insert((*name).to_string(), param_value);
            }
        }
        Ok(params)
    }

    fn after_trial(
        &self,
        ctx: &Context,
        storage: Arc<RwLock<dyn Storage>>,
        state_values: &TrialStateValues,
    ) -> Result<()> {
        if let TrialStateValues::Complete(_) = state_values {
            let mut guard = storage
                .write()
                .map_err(|_e| Error::new(ErrorKind::Unexpected))?;
            let trial = guard.get_trial(ctx.trial_id)?;
            if let Some(generation) = trial
                .attrs
                .get(&AttrKey::System("generation".into()))
                .and_then(|generation| generation.parse::<u32>().ok())
            {
                let mut sync_state = self.get_generation_sync_state_lock()?;
                if sync_state.cached_study_id == Some(ctx.study_id) {
                    sync_state
                        .unfinished_trial_numbers
                        .retain(|&number| number != trial.number);
                    let mut generation_to_numbers = self.get_generation_to_numbers_write_lock()?;
                    let numbers = generation_to_numbers.entry(generation).or_default();
                    if !numbers.contains(&trial.number) {
                        numbers.push(trial.number);
                    }
                }
            }
        }
        Ok(())
    }
}

/// Returns parameter names sorted lexicographically.
///
/// Iterating over a `HashMap` keys yields a non-deterministic order, which breaks
/// reproducibility when `self.rng` is consumed inside the loop. Sorting once at the
/// call site ensures stable RNG draw ordering across runs.
fn sorted_parameter_names(search_space: &HashMap<String, Distribution>) -> Vec<&str> {
    let mut names = search_space.keys().map(String::as_str).collect::<Vec<_>>();
    names.sort_unstable();
    names
}

/// Return whether `trial0` constrained-dominates `trial1`.
///
/// A trial x is said to constrained-dominate a trial y, if any of the following conditions is
/// true:
/// 1) Trial x is feasible and trial y is not.
/// 2) Trial x and y are both infeasible, but solution x has a smaller overall constraint
///    violation.
/// 3) Trial x and y are feasible and trial x dominates trial y.
///
fn constrained_dominates(
    values_feasible_violation_0: &Option<(&[f64], bool, f64)>,
    values_feasible_violation_1: &Option<(&[f64], bool, f64)>,
    directions: &[Direction],
) -> bool {
    let Some((values0, feasible0, violation0)) = values_feasible_violation_0 else {
        return false;
    };
    let Some((values1, feasible1, violation1)) = values_feasible_violation_1 else {
        return true;
    };

    if *feasible0 && *feasible1 {
        dominates(values0, values1, directions)
    } else if *feasible0 {
        true
    } else if *feasible1 {
        false
    } else {
        *violation0 < *violation1
    }
}

fn fast_non_dominated_sort(
    ctx: &Context,
    trials: &[Option<PersistedTrial>],
    population_numbers: &[u32],
) -> Result<Vec<Vec<u32>>> {
    let n = population_numbers.len();

    let population_trials = population_numbers
        .iter()
        .map(|i| {
            trials
                .get(*i as usize)
                .and_then(|trial| trial.as_ref())
                .ok_or_else(|| Error::new(ErrorKind::TrialDiscarded))
        })
        .collect::<Result<Vec<_>>>()?;

    validate_trials(&population_trials, &ctx.directions)?;

    let population_infos = population_trials
        .iter()
        .map(|t| match &t.state_values {
            TrialStateValues::Complete(values) => {
                let constraints = t.constraints()?;
                let is_feasible = constraints.values().all(|x| *x <= 0.0);
                let violation = constraints.values().filter(|&x| *x > 0.0).sum();
                Ok(Some((values.as_slice(), is_feasible, violation)))
            }
            _ => Ok(None),
        })
        .collect::<Result<Vec<_>>>()?;

    let mut dominated_count = vec![0u32; n];
    let mut dominates_list: Vec<Vec<usize>> = vec![vec![]; n];

    for i in 0..n {
        for j in (i + 1)..n {
            if constrained_dominates(&population_infos[i], &population_infos[j], &ctx.directions) {
                dominates_list[i].push(j);
                dominated_count[j] += 1;
            } else if constrained_dominates(
                &population_infos[j],
                &population_infos[i],
                &ctx.directions,
            ) {
                dominates_list[j].push(i);
                dominated_count[i] += 1;
            }
        }
    }

    let mut population_numbers_per_rank = vec![];
    let mut mutable_population_indices: Vec<usize> = (0..n).collect();
    while !mutable_population_indices.is_empty() {
        let mut non_dominated_indices = vec![];
        let mut i = 0;
        while i < mutable_population_indices.len() {
            let idx = mutable_population_indices[i];
            if dominated_count[idx] == 0 {
                if i == mutable_population_indices.len() - 1 {
                    mutable_population_indices.pop();
                } else {
                    mutable_population_indices[i] = mutable_population_indices.pop().unwrap();
                }
                non_dominated_indices.push(idx);
            } else {
                i += 1;
            }
        }
        for &x in &non_dominated_indices {
            for &y in &dominates_list[x] {
                dominated_count[y] -= 1;
            }
        }
        population_numbers_per_rank.push(
            non_dominated_indices
                .into_iter()
                .map(|idx| population_numbers[idx])
                .collect(),
        );
    }
    Ok(population_numbers_per_rank)
}

fn calc_crowding_distance(
    ctx: &Context,
    trials: &[Option<PersistedTrial>],
    population_numbers: &[u32],
) -> Result<Vec<(u32, f64)>> {
    let population_values = population_numbers
        .iter()
        .map(|n| {
            let trial = trials
                .get(*n as usize)
                .and_then(|trial| trial.as_ref())
                .ok_or_else(|| Error::new(ErrorKind::TrialDiscarded))?;
            match &trial.state_values {
                TrialStateValues::Complete(values) => Ok(values.clone()),
                _ => Ok(vec![f64::NAN; ctx.directions.len()]),
            }
        })
        .collect::<Result<Vec<_>>>()?;

    let mut crowding_distance = vec![0.0; population_numbers.len()];

    for i in 0..ctx.directions.len() {
        let mut indices_and_values = (0..population_numbers.len())
            .zip(population_values.iter().map(|v| v[i]))
            .collect::<Vec<_>>();
        indices_and_values.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        if indices_and_values[0].1 == indices_and_values[population_numbers.len() - 1].1 {
            continue;
        }

        let values = [-f64::INFINITY]
            .into_iter()
            .chain(indices_and_values.iter().map(|t| t.1))
            .chain([f64::INFINITY])
            .collect::<Vec<_>>();

        let min_value = values.iter().cloned().find(|v| v.is_finite()).unwrap();
        let max_value = values.iter().cloned().rfind(|v| v.is_finite()).unwrap();

        let mut width = max_value - min_value;
        if width <= 0.0 {
            width = 1.0;
        }

        for j in 0..indices_and_values.len() {
            let gap = if values[j] == values[j + 2] {
                0.0
            } else {
                values[j + 2] - values[j]
            };
            crowding_distance[indices_and_values[j].0] += gap / width;
        }
    }
    Ok(population_numbers
        .iter()
        .copied()
        .zip(crowding_distance)
        .collect())
}

fn crowding_distance_sort(
    ctx: &Context,
    trials: &[Option<PersistedTrial>],
    population_numbers: Vec<u32>,
) -> Result<Vec<u32>> {
    let mut population_and_distance = calc_crowding_distance(ctx, trials, &population_numbers)?;
    population_and_distance.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    Ok(population_and_distance
        .into_iter()
        .map(|(number, _)| number)
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustuna_core::storage::InMemoryStorage;
    use rustuna_core::study::{create_study, Direction};

    fn persisted_trial(
        number: u32,
        state_values: TrialStateValues,
        generation: Option<u32>,
    ) -> PersistedTrial {
        let mut trial = PersistedTrial::new(number, 0, number);
        trial.state_values = state_values;
        if let Some(generation) = generation {
            trial
                .attrs
                .insert(AttrKey::System("generation".into()), generation.to_string());
        }
        trial
    }
    #[test]
    fn test_optimize() {
        let storage = InMemoryStorage::new();
        let directions = vec![Direction::Minimize, Direction::Minimize];
        let study = create_study(
            "simple-quadratic",
            storage,
            NSGAIISampler::new(2, None, 1.0, 1.0),
            directions,
        )
        .unwrap();
        let n_trials = 10;
        study
            .optimize(
                |mut t| {
                    let x = t.suggest_float("x", 0.0, 10.0)?;
                    let y = t.suggest_float("y", 0.0, 10.0)?;
                    let value0 = (x - 3.0).powi(2) + (y - 5.0).powi(2);
                    let value1 = (x - 5.0).powi(2) + (y - 3.0).powi(2);
                    Ok(vec![value0, value1])
                },
                n_trials,
            )
            .unwrap();
        assert!(study.get_trials().unwrap().len() == n_trials);
    }

    #[test]
    fn test_single_value_parameters() {
        let storage = InMemoryStorage::new();
        let directions = vec![Direction::Minimize, Direction::Minimize];
        let study = create_study(
            "single-value-test",
            storage,
            NSGAIISampler::new(3, None, 0.9, 0.5),
            directions,
        )
        .unwrap();

        let result = study.optimize(
            |mut t| {
                // Normal parameters
                let x = t.suggest_float("x", 0.0, 10.0)?;
                let y = t.suggest_float("y", 0.0, 10.0)?;

                // Single-value parameters (should be excluded from genetic algorithm)
                let z = t.suggest_float("z", 5.0, 5.0)?; // single value: 5.0
                let w = t.suggest_int("w", 3, 3)?; // single value: 3

                let value0 = (x - 3.0).powi(2) + (y - 5.0).powi(2) + z + w as f64;
                let value1 = (x - 5.0).powi(2) + (y - 3.0).powi(2) + z - w as f64;

                println!(
                    "{:2} x: {}, y: {}, z: {}, w: {}, v0: {}, v1: {}",
                    t.number, x, y, z, w, value0, value1
                );
                Ok(vec![value0, value1])
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
            if let Some(z_val) = trial.internal_params.get("z") {
                assert!(
                    (z_val - 5.0).abs() < 1e-10,
                    "z should always be 5.0, got {}",
                    z_val
                );
            }
            if let Some(w_val) = trial.internal_params.get("w") {
                assert!(
                    (w_val - 3.0).abs() < 1e-10,
                    "w should always be 3.0, got {}",
                    w_val
                );
            }
        }
    }

    #[test]
    fn test_constraints() {
        let storage = InMemoryStorage::new();
        let directions = vec![Direction::Minimize, Direction::Minimize];
        let study = create_study(
            "constraints",
            storage,
            NSGAIISampler::new(2, None, 1.0, 1.0),
            directions,
        )
        .unwrap();
        let n_trials = 10;
        let result = study.optimize(
            |mut t| {
                let x = t.suggest_float("x", -15.0, 30.0)?;
                let y = t.suggest_float("y", -15.0, 30.0)?;
                let v0 = 4.0 * x.powi(2) + 4.0 * y.powi(2);
                let v1 = (x - 5.0).powi(2) + (y - 5.0).powi(2);
                t.set_constraints(HashMap::from([(String::from("c0"), 1000.0 - v0)]))?;
                Ok(vec![v0, v1])
            },
            n_trials,
        );
        assert!(
            result.is_ok(),
            "Optimization with constraints should complete without panicking"
        );
    }

    #[test]
    fn test_uncompleted_trial() -> Result<()> {
        let storage = InMemoryStorage::new();
        let directions = vec![Direction::Minimize, Direction::Minimize];
        let study = create_study(
            "uncompleted_trial",
            storage,
            NSGAIISampler::new(2, None, 1.0, 1.0),
            directions,
        )
        .unwrap();

        let n_trials = 50;
        for _ in 0..n_trials {
            let mut trial = study.ask()?;
            let x = trial.suggest_int("x", 1, 2)?;
            let y = trial.suggest_int("y", 1, 2)?;
            let v0 = (x as f64 - 1.5).powi(2);
            let v1 = (y as f64 - 1.5).powi(2);
            if x == 1 {
                study.tell(trial.number, TrialStateValues::Complete(vec![v0, v1]))?;
            } else {
                study.tell(trial.number, TrialStateValues::Fail)?;
            }
        }

        assert!(study.get_trials()?.len() == n_trials);
        Ok(())
    }

    #[test]
    fn test_reproducibility_with_seeded_rng() {
        // Run the same optimization twice with an identical seed and assert that
        // every trial produces exactly the same parameter values.  This catches
        // non-determinism caused by iterating over a HashMap whose order is not
        // guaranteed, such as the search-space keys inside crossover / mutation.
        let run = || {
            let storage = InMemoryStorage::new();
            let directions = vec![Direction::Minimize, Direction::Minimize];
            let study = create_study(
                "reproducibility-test",
                storage,
                NSGAIISampler::seed_from_u64(42, 10, None, 0.9, 0.5),
                directions,
            )
            .unwrap();
            study
                .optimize(
                    |mut t| {
                        // Use many parameters so that crossover and mutation
                        // consume RNG draws in the key-iteration order.
                        let mut value0 = 0.0;
                        let mut value1 = 0.0;
                        for i in 0..10 {
                            let name = format!("x{i}");
                            let xi = t.suggest_float(&name, -10.0, 10.0)?;
                            value0 += (xi - 5.0).powi(2);
                            value1 += (xi + 5.0).powi(2);
                        }
                        Ok(vec![value0, value1])
                    },
                    50,
                )
                .unwrap();
            study.get_trials().unwrap()
        };

        let trials_a = run();
        let trials_b = run();

        assert_eq!(
            trials_a.len(),
            trials_b.len(),
            "both runs should produce the same number of trials"
        );
        for (ta, tb) in trials_a.iter().zip(trials_b.iter()) {
            assert_eq!(
                ta.internal_params, tb.internal_params,
                "trial {} params differ between runs -- non-deterministic key ordering",
                ta.number
            );
        }
    }

    #[test]
    fn test_get_child_generation_indexes_newly_appended_completed_trials() {
        let sampler = NSGAIISampler::new(2, None, 1.0, 1.0);
        let mut trials = vec![Some(persisted_trial(
            0,
            TrialStateValues::Complete(vec![0.0, 0.0]),
            Some(0),
        ))];

        assert_eq!(sampler.get_child_generation(0, &trials).unwrap(), 0);

        trials.push(Some(persisted_trial(
            1,
            TrialStateValues::Complete(vec![1.0, 1.0]),
            Some(0),
        )));

        assert_eq!(sampler.get_child_generation(0, &trials).unwrap(), 1);
        assert_eq!(
            sampler
                .get_generation_to_numbers_read_lock()
                .unwrap()
                .get(&0),
            Some(&vec![0, 1])
        );
    }

    #[test]
    fn test_get_child_generation_rechecks_unfinished_trials() {
        let sampler = NSGAIISampler::new(1, None, 1.0, 1.0);
        let mut trials = vec![Some(persisted_trial(0, TrialStateValues::Running, Some(0)))];

        assert_eq!(sampler.get_child_generation(0, &trials).unwrap(), 0);
        assert_eq!(
            sampler
                .get_generation_sync_state_lock()
                .unwrap()
                .unfinished_trial_numbers
                .as_slice(),
            &[0]
        );

        trials[0] = Some(persisted_trial(
            0,
            TrialStateValues::Complete(vec![0.0, 0.0]),
            Some(0),
        ));

        assert_eq!(sampler.get_child_generation(0, &trials).unwrap(), 1);
        assert!(sampler
            .get_generation_sync_state_lock()
            .unwrap()
            .unfinished_trial_numbers
            .is_empty());
        assert_eq!(
            sampler
                .get_generation_to_numbers_read_lock()
                .unwrap()
                .get(&0),
            Some(&vec![0])
        );
    }

    #[test]
    fn test_get_child_generation_resets_cache_when_trial_list_shrinks() {
        let sampler = NSGAIISampler::new(2, None, 1.0, 1.0);
        let full_trials = vec![
            Some(persisted_trial(
                0,
                TrialStateValues::Complete(vec![0.0, 0.0]),
                Some(0),
            )),
            Some(persisted_trial(
                1,
                TrialStateValues::Complete(vec![1.0, 1.0]),
                Some(0),
            )),
        ];

        assert_eq!(sampler.get_child_generation(0, &full_trials).unwrap(), 1);

        let shortened_trials = vec![Some(persisted_trial(
            0,
            TrialStateValues::Complete(vec![0.0, 0.0]),
            Some(0),
        ))];

        assert_eq!(
            sampler.get_child_generation(0, &shortened_trials).unwrap(),
            0
        );
        assert_eq!(
            sampler
                .get_generation_sync_state_lock()
                .unwrap()
                .unseen_trial_start,
            1
        );
        assert_eq!(
            sampler
                .get_generation_to_numbers_read_lock()
                .unwrap()
                .get(&0),
            Some(&vec![0])
        );
    }

    #[test]
    fn test_generation_cache_picks_up_external_completion_before_assigning_child_generation() {
        let study = create_study(
            "generation-cache-stale-workers",
            InMemoryStorage::new(),
            NSGAIISampler::new(2, None, 1.0, 1.0),
            vec![Direction::Minimize, Direction::Minimize],
        )
        .unwrap();
        let worker = rustuna_core::study::Study::from_id(
            study.id,
            Arc::clone(&study.storage),
            Arc::new(NSGAIISampler::new(2, None, 1.0, 1.0)),
        )
        .unwrap();

        let trial0 = study.ask().unwrap();
        let trial1 = worker.ask().unwrap();
        study
            .tell(trial0.number, TrialStateValues::Complete(vec![0.0, 1.0]))
            .unwrap();
        worker
            .tell(trial1.number, TrialStateValues::Complete(vec![1.0, 0.0]))
            .unwrap();

        let next_trial = study.ask().unwrap();
        let mut storage = study.storage.write().unwrap();
        assert_eq!(
            storage
                .get_trial(next_trial.id)
                .unwrap()
                .attrs
                .get(&AttrKey::System("generation".into())),
            Some(&"1".to_string())
        );
    }

    #[test]
    fn test_parent_population_cache_persists() {
        let storage = InMemoryStorage::new();
        let directions = vec![Direction::Minimize, Direction::Minimize];
        let study = create_study(
            "parent-cache-test",
            storage,
            NSGAIISampler::new(2, None, 1.0, 1.0),
            directions,
        )
        .unwrap();
        study
            .optimize(
                |mut trial| {
                    let x = trial.suggest_float("x", 0.0, 10.0)?;
                    Ok(vec![x, -x])
                },
                24,
            )
            .unwrap();

        // After 24 trials with population_size=2, there should be cached parent
        // populations persisted as study system attributes.
        let mut guard = study.storage.write().unwrap();
        let persisted_study = guard.get_study(study.id).unwrap();
        let has_cache = persisted_study
            .attrs
            .iter()
            .any(|(key, _)| {
                matches!(key, AttrKey::System(ref s) if s.as_str().starts_with(PARENT_CACHE_KEY_PREFIX))
            });
        assert!(has_cache, "parent population cache should be persisted");
    }

    #[test]
    fn test_parent_population_cache_restored_after_restart() {
        let storage = InMemoryStorage::new();
        let directions = vec![Direction::Minimize, Direction::Minimize];
        let study = create_study(
            "parent-cache-restart",
            storage,
            NSGAIISampler::new(2, None, 1.0, 1.0),
            directions,
        )
        .unwrap();
        study
            .optimize(
                |mut trial| {
                    let x = trial.suggest_float("x", 0.0, 10.0)?;
                    Ok(vec![x, -x])
                },
                20,
            )
            .unwrap();

        // Record the cached parent IDs before restart.
        let cached_ids: Vec<String> = {
            let mut guard = study.storage.write().unwrap();
            let persisted_study = guard.get_study(study.id).unwrap();
            persisted_study
                .attrs
                .iter()
                .filter(|(key, _)| {
                    matches!(key, AttrKey::System(ref s) if s.as_str().starts_with(PARENT_CACHE_KEY_PREFIX))
                })
                .map(|(_, v)| v.clone())
                .collect()
        };
        assert!(!cached_ids.is_empty(), "cache should exist before restart");

        // Create a new sampler instance (simulating restart) and run more trials.
        let new_sampler = NSGAIISampler::new(2, None, 1.0, 1.0);
        let resumed = rustuna_core::study::Study::from_id(
            study.id,
            std::sync::Arc::clone(&study.storage),
            std::sync::Arc::new(new_sampler),
        )
        .unwrap();

        // The resumed study should produce more trials without errors.
        resumed
            .optimize(
                |mut trial| {
                    let x = trial.suggest_float("x", 0.0, 10.0)?;
                    Ok(vec![x, -x])
                },
                10,
            )
            .unwrap();

        let total = resumed.get_trials().unwrap().len();
        assert_eq!(total, 30, "should have 30 trials after restart");
    }

    #[test]
    fn test_invalid_parent_population_cache_is_recomputed() {
        let study = create_study(
            "parent-cache-invalid",
            InMemoryStorage::new(),
            NSGAIISampler::new(2, None, 1.0, 1.0),
            vec![Direction::Minimize, Direction::Minimize],
        )
        .unwrap();
        study
            .optimize(
                |mut trial| {
                    let x = trial.suggest_float("x", 0.0, 10.0)?;
                    Ok(vec![x, -x])
                },
                3,
            )
            .unwrap();

        let cache_key = NSGAIISampler::parent_cache_key(1);
        {
            let mut attrs = Attrs::new();
            attrs.insert(cache_key.clone(), "[]".to_string());
            study
                .storage
                .write()
                .unwrap()
                .set_study_attrs(study.id, attrs, false)
                .unwrap();
        }

        let resumed = rustuna_core::study::Study::from_id(
            study.id,
            Arc::clone(&study.storage),
            Arc::new(NSGAIISampler::new(2, None, 1.0, 1.0)),
        )
        .unwrap();
        resumed.ask().unwrap();

        let encoded = resumed
            .storage
            .write()
            .unwrap()
            .get_study_attr(resumed.id, cache_key)
            .unwrap();
        let parent_ids: Vec<u32> = serde_json::from_str(&encoded).unwrap();
        assert_eq!(parent_ids.len(), 2);
    }

    #[test]
    fn test_parent_population_cache_preserves_parent_order() {
        let completed_trial_numbers_by_id = HashMap::from([(10, 2), (20, 0), (30, 1)]);
        let population_numbers = NSGAIISampler::decode_parent_population_numbers(
            |trial_id| Ok(completed_trial_numbers_by_id.get(&trial_id).copied()),
            "[10,20,30]",
            3,
        )
        .unwrap()
        .unwrap();

        assert_eq!(population_numbers, vec![2, 0, 1]);
    }

    #[test]
    fn test_parent_population_cache_is_recomputed_when_population_size_changes() {
        let study = create_study(
            "parent-cache-population-size",
            InMemoryStorage::new(),
            NSGAIISampler::new(2, None, 1.0, 1.0),
            vec![Direction::Minimize, Direction::Minimize],
        )
        .unwrap();
        study
            .optimize(
                |mut trial| {
                    let x = trial.suggest_float("x", 0.0, 10.0)?;
                    Ok(vec![x, -x])
                },
                3,
            )
            .unwrap();

        let resumed = rustuna_core::study::Study::from_id(
            study.id,
            Arc::clone(&study.storage),
            Arc::new(NSGAIISampler::new(3, None, 1.0, 1.0)),
        )
        .unwrap();
        let generation_zero_trial = resumed.ask().unwrap();
        resumed
            .tell(
                generation_zero_trial.number,
                TrialStateValues::Complete(vec![5.0, -5.0]),
            )
            .unwrap();
        resumed.ask().unwrap();

        let encoded = resumed
            .storage
            .write()
            .unwrap()
            .get_study_attr(resumed.id, NSGAIISampler::parent_cache_key(1))
            .unwrap();
        let parent_ids: Vec<u32> = serde_json::from_str(&encoded).unwrap();
        assert_eq!(parent_ids.len(), 3);
    }
}
