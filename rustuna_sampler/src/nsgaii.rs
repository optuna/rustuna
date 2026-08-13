use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use rand::prelude::*;
use rand::rngs::StdRng;
use rustuna_core::attr::{AttrKey, Attrs};
use rustuna_core::distribution::Distribution;
use rustuna_core::sampler::{Context, Sampler};
use rustuna_core::storage::Storage;
use rustuna_core::study::dominates;
use rustuna_core::study::Direction;
use rustuna_core::trial::{PersistedTrial, TrialStateValues};
use rustuna_core::Result;
use rustuna_core::{Error, ErrorKind};

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
    rng: StdRng,
    population_size: usize,
    mutation_prob: Option<f64>,
    crossover_prob: f64,
    swapping_prob: f64,
    /// Cache mapping generation number to completed trial numbers in that generation.
    /// Updated incrementally in `after_trial` so `sample_joint` does not scan all trials every time.
    generation_to_numbers: HashMap<u32, Vec<u32>>,
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
            rng: StdRng::from_seed(Default::default()),
            population_size,
            mutation_prob,
            crossover_prob,
            swapping_prob,
            generation_to_numbers: HashMap::new(),
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
            rng: StdRng::seed_from_u64(seed),
            population_size,
            mutation_prob,
            crossover_prob,
            swapping_prob,
            generation_to_numbers: HashMap::new(),
        }
    }
    fn rebuild_generation_cache(&mut self, trials: &[Option<PersistedTrial>]) {
        self.generation_to_numbers.clear();
        let generation_key = AttrKey::System("generation".into());
        for trial in trials.iter().flatten() {
            if !matches!(trial.state_values, TrialStateValues::Complete(_)) {
                continue;
            }
            if let Some(gen_str) = trial.attrs.get(&generation_key) {
                if let Ok(generation) = gen_str.parse::<u32>() {
                    self.generation_to_numbers
                        .entry(generation)
                        .or_default()
                        .push(trial.number);
                }
            }
        }
    }
    fn select_elite_population_numbers(
        &mut self,
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
    fn get_parent_population_numbers(
        &mut self,
        ctx: &Context,
        trials: &[Option<PersistedTrial>],
    ) -> Result<(i32, Vec<u32>)> {
        if self.generation_to_numbers.is_empty() {
            self.rebuild_generation_cache(trials);
        }

        let mut parent_generation = -1;
        let mut parent_population_numbers = Vec::with_capacity(10);
        for generation in 0..10 {
            let population_numbers = match self.generation_to_numbers.get(&generation) {
                Some(numbers) if numbers.len() >= self.population_size => numbers.clone(),
                _ => break,
            };

            let mut population_numbers = population_numbers;
            population_numbers.append(&mut parent_population_numbers);
            let selected_population_numbers =
                self.select_elite_population_numbers(ctx, trials, &population_numbers)?;
            parent_generation = generation as i32;
            parent_population_numbers = selected_population_numbers;
        }
        Ok((parent_generation, parent_population_numbers))
    }
    fn crossover(
        &mut self,
        parent0: HashMap<String, f64>,
        parent1: HashMap<String, f64>,
        search_space: &HashMap<String, Distribution>,
    ) -> HashMap<String, f64> {
        let mut child = HashMap::new();
        for name in search_space.keys() {
            let param_value0 = *parent0.get(name).unwrap();
            let param_value1 = *parent1.get(name).unwrap();
            let param_value = if self.rng.gen_bool(self.swapping_prob) {
                param_value1
            } else {
                param_value0
            };
            child.insert(name.clone(), param_value);
        }
        child
    }
}
impl Sampler for NSGAIISampler {
    fn sample_independent(
        &mut self,
        _ctx: &Context,
        _storage: Arc<RwLock<dyn Storage>>,
        _name: &str,
        distribution: &Distribution,
    ) -> Result<f64> {
        if distribution.is_single() {
            return distribution.get_single_value();
        }

        match distribution {
            Distribution::Float {
                low,
                high,
                step,
                log,
            } => {
                let param_value = match (step, log) {
                    (None, false) => self.rng.gen_range(*low..*high),
                    (None, true) => self.rng.gen_range(low.ln()..high.ln()).exp(),
                    (Some(step), false) => {
                        let max_index = ((high - low) / step).floor().max(0.0) as i64;
                        let index = self.rng.gen_range(0..=max_index);
                        low + (index as f64) * step
                    }
                    (Some(step), true) => {
                        let value = self.rng.gen_range(low.ln()..high.ln()).exp();
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
                    let value = self.rng.gen_range(low_f.ln()..high_f.ln()).exp();
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
                    let index = self.rng.gen_range(0..=max_index);
                    (low + index * step) as f64
                };
                Ok(param_value)
            }
            Distribution::Categorical { cardinality } => {
                let param_value = self.rng.gen_range(0..*cardinality);
                Ok(param_value as f64)
            }
        }
    }

    fn support_joint_sampling(&self) -> bool {
        true
    }

    fn sample_joint(
        &mut self,
        ctx: &Context,
        storage: Arc<RwLock<dyn Storage>>,
        search_space: &HashMap<String, Distribution>,
    ) -> Result<HashMap<String, f64>> {
        let mut guard = storage
            .write()
            .map_err(|_e| Error::new(ErrorKind::Unexpected))?;
        let (parent_generation, parent_population_numbers) = {
            let trials = guard.get_trials(ctx.study_id)?;
            self.get_parent_population_numbers(ctx, trials)?
        };
        let child_generation = u32::try_from(parent_generation + 1).unwrap();
        let mut attrs = Attrs::with_capacity(1);
        attrs.insert(
            AttrKey::System("generation".into()),
            (child_generation as f64).to_string(),
        );
        guard.set_trial_attrs(ctx.trial_id, attrs, false)?;

        if parent_generation < 0 {
            drop(guard);
            let params = HashMap::new();
            return Ok(params);
        }

        let (parent0_number, parent1_number) = {
            let mut selected = parent_population_numbers
                .choose_multiple(&mut self.rng, 2)
                .copied();
            (selected.next().unwrap(), selected.next().unwrap())
        };

        let trials = guard.get_trials(ctx.study_id)?;
        let build_parent_params = |number: u32| -> Result<HashMap<String, f64>> {
            let trial = trials
                .get(number as usize)
                .and_then(Option::as_ref)
                .ok_or_else(|| Error::new(ErrorKind::TrialDiscarded))?;
            let mut params = HashMap::with_capacity(search_space.len());
            for name in search_space.keys() {
                let param_value = *trial.internal_params.get(name).unwrap();
                params.insert(name.clone(), param_value);
            }
            Ok(params)
        };
        let parent0 = build_parent_params(parent0_number)?;
        let parent1 = build_parent_params(parent1_number)?;
        drop(guard);

        let child = if self.rng.gen_bool(self.crossover_prob) {
            self.crossover(parent0, parent1, search_space)
        } else {
            parent0
        };

        let mutation_prob = self
            .mutation_prob
            .unwrap_or(1.0 / 1.0_f64.max(child.len() as f64));
        let mut params = HashMap::new();
        for name in search_space.keys() {
            if !self.rng.gen_bool(mutation_prob) {
                let param_value = *child.get(name).unwrap();
                params.insert(name.clone(), param_value);
            }
        }
        Ok(params)
    }

    fn after_trial(
        &mut self,
        ctx: &Context,
        storage: Arc<RwLock<dyn Storage>>,
        state_values: &TrialStateValues,
    ) -> Result<()> {
        if let TrialStateValues::Complete(_) = state_values {
            let mut guard = storage
                .write()
                .map_err(|_e| Error::new(ErrorKind::Unexpected))?;
            let trial = guard.get_trial(ctx.trial_id)?;
            let generation_key = AttrKey::System("generation".into());
            if let Some(gen_str) = trial.attrs.get(&generation_key) {
                if let Ok(generation) = gen_str.parse::<u32>() {
                    self.generation_to_numbers
                        .entry(generation)
                        .or_default()
                        .push(trial.number);
                }
            }
        }
        Ok(())
    }
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
    trial0: &PersistedTrial,
    trial1: &PersistedTrial,
    directions: &[Direction],
) -> Result<bool> {
    let values0 = match &trial0.state_values {
        TrialStateValues::Complete(values) => values,
        _ => return Ok(false),
    };
    let values1 = match &trial1.state_values {
        TrialStateValues::Complete(values) => values,
        _ => return Ok(true),
    };

    let constraints0 = trial0.constraints()?;
    let satisfy_constraints0 = constraints0.values().all(|x| *x <= 0.0);
    let constraints1 = trial1.constraints()?;
    let satisfy_constraints1 = constraints1.values().all(|x| *x <= 0.0);

    if satisfy_constraints0 && satisfy_constraints1 {
        return dominates(values0, values1, directions);
    }
    if satisfy_constraints0 {
        return Ok(true);
    }
    if satisfy_constraints1 {
        return Ok(false);
    }

    let violation0: f64 = constraints0.values().filter(|&x| *x > 0.0).sum();
    let violation1: f64 = constraints1.values().filter(|&x| *x > 0.0).sum();
    Ok(violation0 < violation1)
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

    let mut dominated_count = vec![0u32; n];
    let mut dominates_list: Vec<Vec<usize>> = vec![vec![]; n];

    for (i, _) in population_numbers.iter().enumerate() {
        for (j, _) in population_numbers.iter().enumerate() {
            if i >= j {
                continue;
            }
            if constrained_dominates(population_trials[i], population_trials[j], &ctx.directions)? {
                dominates_list[i].push(j);
                dominated_count[j] += 1;
            } else if constrained_dominates(
                population_trials[j],
                population_trials[i],
                &ctx.directions,
            )? {
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
}
