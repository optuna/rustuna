use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use rand::prelude::*;
use rand::rngs::StdRng;
use rustuna_core::attr::{AttrKey, Attrs};
use rustuna_core::distribution::Distribution;
use rustuna_core::sampler::{Context, Sampler};
use rustuna_core::storage::Storage;
use rustuna_core::study::dominates;
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
        }
    }
    fn select_elite_population_numbers(
        &mut self,
        ctx: &Context,
        trials: &[PersistedTrial],
        population_numbers: &Vec<u32>,
    ) -> Vec<u32> {
        let population_numbers_per_rank = fast_non_dominated_sort(ctx, trials, population_numbers);

        let mut elite_population_numbers = vec![];
        for population_numbers in population_numbers_per_rank {
            if elite_population_numbers.len() + population_numbers.len() <= self.population_size {
                elite_population_numbers.extend(population_numbers);
            } else {
                let n = self.population_size - elite_population_numbers.len();
                let crowding_sorted_population_numbers =
                    crowding_distance_sort(ctx, trials, population_numbers);
                elite_population_numbers.extend(&crowding_sorted_population_numbers[..n]);
                break;
            }
        }
        elite_population_numbers
    }
    fn get_parent_population_numbers(
        &mut self,
        ctx: &Context,
        trials: &[PersistedTrial],
    ) -> Result<(i32, Vec<u32>)> {
        let mut generation_to_population_numbers =
            HashMap::<usize, Vec<u32>>::with_capacity(trials.len());
        for trial in trials {
            let generation_or_none = trial.attrs.get(&AttrKey::System("generation".into()));
            if generation_or_none.is_none() {
                continue;
            }
            let generation = generation_or_none.unwrap().parse::<usize>().unwrap();
            if let TrialStateValues::Complete(ref _values) = trial.state_values {
                generation_to_population_numbers
                    .entry(generation)
                    .or_default()
                    .push(trial.number);
            }
        }

        let mut parent_generation = -1;
        let mut parent_population_numbers = Vec::with_capacity(10);
        for _ in 0..10 {
            let generation = parent_generation + 1;
            let population_numbers_or_none =
                generation_to_population_numbers.get(&(generation as usize));

            if population_numbers_or_none.is_none() {
                break;
            }
            let mut population_numbers = population_numbers_or_none.unwrap().clone();
            if population_numbers.len() < self.population_size {
                break;
            }

            population_numbers.append(&mut parent_population_numbers);
            let selected_population_numbers =
                self.select_elite_population_numbers(ctx, trials, &population_numbers);
            parent_generation = generation;
            parent_population_numbers = selected_population_numbers;
        }
        Ok((parent_generation, parent_population_numbers))
    }
    fn select_parents(
        &mut self,
        ctx: &Context,
        storage: Arc<RwLock<dyn Storage>>,
        population_numbers: Vec<u32>,
        search_space: &HashMap<String, Distribution>,
    ) -> Result<(HashMap<String, f64>, HashMap<String, f64>)> {
        let mut guard = storage
            .write()
            .map_err(|_e| Error::new(ErrorKind::Unexpected))?;
        let trials = guard.get_trials(ctx.study_id)?.clone();
        let population_params = trials
            .iter()
            .flatten()
            .filter(|trial| population_numbers.contains(&trial.number))
            .map(|trial| {
                let mut params = HashMap::new();
                for name in sorted_parameter_names(search_space) {
                    let param_value = *trial.internal_params.get(name).unwrap();
                    params.insert(name.to_string(), param_value);
                }
                params
            })
            .collect::<Vec<_>>();
        let [parent0, parent1] = population_params
            .choose_multiple(&mut self.rng, 2)
            .cloned()
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();
        Ok((parent0.clone(), parent1.clone()))
    }
    fn crossover(
        &mut self,
        parent0: HashMap<String, f64>,
        parent1: HashMap<String, f64>,
        search_space: &HashMap<String, Distribution>,
    ) -> HashMap<String, f64> {
        let mut child = HashMap::new();
        for name in sorted_parameter_names(search_space) {
            let param_value0 = *parent0.get(name).unwrap();
            let param_value1 = *parent1.get(name).unwrap();
            let param_value = if self.rng.gen_bool(self.swapping_prob) {
                param_value1
            } else {
                param_value0
            };
            child.insert(name.to_string(), param_value);
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
        let trials = guard
            .get_trials(ctx.study_id)?
            .iter()
            .flatten()
            .cloned()
            .collect::<Vec<_>>();
        let (parent_generation, parent_population_numbers) =
            self.get_parent_population_numbers(ctx, &trials)?;
        let child_generation = u32::try_from(parent_generation + 1).unwrap();
        let mut attrs = Attrs::with_capacity(1);
        attrs.insert(
            AttrKey::System("generation".into()),
            (child_generation as f64).to_string(),
        );
        guard.set_trial_attrs(ctx.trial_id, attrs, false)?;
        drop(guard);

        if parent_generation < 0 {
            let params = HashMap::new();
            return Ok(params);
        }

        let (parent0, parent1) =
            self.select_parents(ctx, storage, parent_population_numbers, search_space)?;

        let child = if self.rng.gen_bool(self.crossover_prob) {
            self.crossover(parent0, parent1, search_space)
        } else {
            parent0
        };

        let mutation_prob = self
            .mutation_prob
            .unwrap_or(1.0 / 1.0_f64.max(child.len() as f64));
        let mut params = HashMap::new();
        for name in sorted_parameter_names(search_space) {
            if !self.rng.gen_bool(mutation_prob) {
                let param_value = *child.get(name).unwrap();
                params.insert(name.to_string(), param_value);
            }
        }
        Ok(params)
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

fn sorted_parameter_names(search_space: &HashMap<String, Distribution>) -> Vec<&str> {
    let mut names = search_space.keys().map(String::as_str).collect::<Vec<_>>();
    names.sort_unstable();
    names
}

fn fast_non_dominated_sort(
    ctx: &Context,
    trials: &[PersistedTrial],
    population_numbers: &Vec<u32>,
) -> Vec<Vec<u32>> {
    let population_values = population_numbers
        .iter()
        .map(|n| match &trials[*n as usize].state_values {
            TrialStateValues::Complete(values) => values.clone(),
            _ => vec![f64::NAN; ctx.directions.len()],
        })
        .collect::<Vec<_>>();

    let mut dominated_count = HashMap::<u32, u32>::with_capacity(population_numbers.len());
    let mut dominates_list = HashMap::<u32, Vec<u32>>::with_capacity(population_numbers.len());
    for p in population_numbers {
        dominated_count.insert(*p, 0);
        dominates_list.insert(*p, vec![]);
    }

    for (i, p) in population_numbers.iter().enumerate() {
        for (j, q) in population_numbers.iter().enumerate() {
            if p >= q {
                continue;
            }
            if dominates(
                &population_values[i],
                &population_values[j],
                &ctx.directions,
            ) {
                dominates_list.entry(*p).or_default().push(*q);
                *dominated_count.entry(*q).or_insert(0) += 1;
            } else if dominates(
                &population_values[j],
                &population_values[i],
                &ctx.directions,
            ) {
                dominates_list.entry(*q).or_default().push(*p);
                *dominated_count.entry(*p).or_insert(0) += 1;
            }
        }
    }

    let mut population_numbers_per_rank = vec![];
    let mut mutable_population_numbers = population_numbers.clone();
    while !mutable_population_numbers.is_empty() {
        let mut non_dominated_population_numbers = vec![];
        let mut i = 0;
        while i < mutable_population_numbers.len() {
            if dominated_count[&mutable_population_numbers[i]] == 0 {
                let individual = mutable_population_numbers[i];
                if i == mutable_population_numbers.len() - 1 {
                    mutable_population_numbers.pop();
                } else {
                    mutable_population_numbers[i] = mutable_population_numbers.pop().unwrap();
                }
                non_dominated_population_numbers.push(individual);
            } else {
                i += 1;
            }
        }
        for x in &non_dominated_population_numbers {
            for y in &dominates_list[x] {
                *dominated_count.get_mut(y).unwrap() -= 1;
            }
        }
        population_numbers_per_rank.push(non_dominated_population_numbers);
    }
    population_numbers_per_rank
}

fn calc_crowding_distance(
    ctx: &Context,
    trials: &[PersistedTrial],
    population_numbers: &Vec<u32>,
) -> HashMap<u32, f64> {
    let population_values = population_numbers
        .iter()
        .map(|n| match &trials[*n as usize].state_values {
            TrialStateValues::Complete(values) => values.clone(),
            _ => vec![f64::NAN; ctx.directions.len()],
        })
        .collect::<Vec<_>>();

    let mut crowding_distance = HashMap::<u32, f64>::with_capacity(population_numbers.len());
    for p in population_numbers {
        crowding_distance.insert(*p, 0.0);
    }

    for i in 0..ctx.directions.len() {
        let mut population_numbers_and_values = population_numbers
            .iter()
            .zip(population_values.iter().map(|v| v[i]))
            .collect::<Vec<_>>();
        population_numbers_and_values.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        if population_numbers_and_values[0].1
            == population_numbers_and_values[population_numbers.len() - 1].1
        {
            continue;
        }

        let values = [-f64::INFINITY]
            .into_iter()
            .chain(population_numbers_and_values.iter().map(|t| t.1))
            .chain([f64::INFINITY])
            .collect::<Vec<_>>();

        let min_value = values.iter().cloned().find(|v| v.is_finite()).unwrap();
        let max_value = values.iter().cloned().rfind(|v| v.is_finite()).unwrap();

        let mut width = max_value - min_value;
        if width <= 0.0 {
            width = 1.0;
        }

        for j in 0..population_numbers_and_values.len() {
            let gap = if values[j] == values[j + 2] {
                0.0
            } else {
                values[j + 2] - values[j]
            };
            crowding_distance
                .entry(*population_numbers_and_values[j].0)
                .and_modify(|e| *e += gap / width);
        }
    }
    crowding_distance
}

fn crowding_distance_sort(
    ctx: &Context,
    trials: &[PersistedTrial],
    population_numbers: Vec<u32>,
) -> Vec<u32> {
    let manhattan_distance = calc_crowding_distance(ctx, trials, &population_numbers);
    let mut mutable_population_numbers = population_numbers.clone();
    mutable_population_numbers.sort_by(|a, b| {
        manhattan_distance[b]
            .partial_cmp(&manhattan_distance[a])
            .unwrap()
    });
    mutable_population_numbers
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
}
