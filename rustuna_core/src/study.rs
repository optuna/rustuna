use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use crate::attr::{extract_fixed_params, fixed_params_to_attrs, AttrKey, Attrs, CategoryLabel};
use crate::distribution::Distribution;
use crate::sampler::{Context as SamplerContext, Sampler};
use crate::storage::Storage;
use crate::trial::{validate_trials, PersistedTrial, Trial, TrialStateValues};
use crate::trial_queue::{InMemoryTrialQueue, TrialQueue};
use crate::{Error, ErrorKind, Result};

/// Creates a study backed by the given storage and sampler.
pub fn create_study<S: Storage + Send + Sync + 'static, T: Sampler + 'static>(
    study_name: &str,
    mut storage: S,
    sampler: T,
    directions: Vec<Direction>,
) -> Result<Study> {
    let study_id = storage.create_new_study(study_name, directions.clone())?.id;
    let storage = Arc::new(RwLock::new(storage));
    let sampler: Arc<dyn Sampler> = Arc::new(sampler);
    let queue = Arc::new(RwLock::new(InMemoryTrialQueue::new()));
    Ok(Study::new(
        study_id,
        study_name.to_string(),
        directions,
        storage,
        sampler,
        queue,
    ))
}

/// Creates a study from shared storage and sampler handles.
pub fn create_study_with_arc(
    study_name: &str,
    storage: Arc<RwLock<dyn Storage>>,
    sampler: Arc<dyn Sampler>,
    directions: Vec<Direction>,
) -> Result<Study> {
    let mut guard = storage.write().map_err(|e| {
        Error::with_reason(
            ErrorKind::Unexpected,
            format!("Failed to acquire a storage guard: {e}"),
        )
    })?;
    let study_id = guard.create_new_study(study_name, directions.clone())?.id;
    drop(guard);
    let queue = Arc::new(RwLock::new(InMemoryTrialQueue::new()));
    Ok(Study::new(
        study_id,
        study_name.to_string(),
        directions,
        storage,
        sampler,
        queue,
    ))
}

#[derive(Clone)]
/// A study corresponds to an optimization task, that is, a set of trials.
///
/// This is the central object for running optimization in Rustuna. It provides interfaces to
/// create new trials, evaluate objective functions, access trial history, enqueue fixed trials,
/// and set or get study-level user attributes.
///
/// A `Study` holds shared handles to a storage backend, a sampler, and a trial queue.
pub struct Study {
    pub id: u32,
    pub name: String,
    pub directions: Vec<Direction>,
    pub storage: Arc<RwLock<dyn Storage>>,
    pub sampler: Arc<dyn Sampler>,
    pub queue: Arc<RwLock<dyn TrialQueue>>,
}
impl Study {
    /// Constructs a study from fully initialized components.
    pub fn new(
        id: u32,
        name: String,
        directions: Vec<Direction>,
        storage: Arc<RwLock<dyn Storage>>,
        sampler: Arc<dyn Sampler>,
        queue: Arc<RwLock<dyn TrialQueue>>,
    ) -> Self {
        Study {
            id,
            name,
            directions,
            storage,
            sampler,
            queue,
        }
    }

    /// Loads a study by ID from storage.
    pub fn from_id(
        id: u32,
        storage: Arc<RwLock<dyn Storage>>,
        sampler: Arc<dyn Sampler>,
    ) -> Result<Self> {
        let mut guard = storage.write().map_err(|e| {
            Error::with_reason(
                ErrorKind::Unexpected,
                format!("Failed to acquire a storage guard: {e}"),
            )
        })?;
        let study = guard.get_study(id)?;
        let name = study.name.clone();
        let directions = study.directions.clone();
        drop(guard);
        let queue = Arc::new(RwLock::new(InMemoryTrialQueue::new()));
        Ok(Study::new(id, name, directions, storage, sampler, queue))
    }

    /// Loads a study by name from storage.
    pub fn from_name(
        name: String,
        storage: Arc<RwLock<dyn Storage>>,
        sampler: Arc<dyn Sampler>,
    ) -> Result<Self> {
        let mut guard = storage.write().map_err(|e| {
            Error::with_reason(
                ErrorKind::Unexpected,
                format!("Failed to acquire a storage guard: {e}"),
            )
        })?;
        let studies = guard.get_studies()?;
        let study = studies
            .iter()
            .find(|s| s.name == name)
            .ok_or(Error::new(ErrorKind::StudyNotFound))?;
        let study_id = study.id;
        let directions = study.directions.clone();
        drop(guard);
        let queue = Arc::new(RwLock::new(InMemoryTrialQueue::new()));
        Ok(Study::new(
            study_id, name, directions, storage, sampler, queue,
        ))
    }

    /// Creates or dequeues the next trial to evaluate.
    pub fn ask(&self) -> Result<Trial> {
        let queued_trial_id = {
            let mut queue_guard = self.queue.write().map_err(|e| {
                Error::with_reason(
                    ErrorKind::Unexpected,
                    format!("Failed to acquire a queue guard: {e}"),
                )
            })?;
            match queue_guard.dequeue() {
                Ok(trial_id) => Some(trial_id),
                Err(error) if matches!(error.kind, ErrorKind::TrialQueueEmpty) => None,
                Err(error) => return Err(error),
            }
        };

        let (trial_id, trial_number, datetime_start, datetime_complete, fixed_params) =
            if let Some(trial_id) = queued_trial_id {
                // Try to get trial from storage and transition to Running state.
                // If any storage operation fails, enqueue the trial_id back to the queue.
                let result = (|| {
                    let mut guard = self.storage.write().map_err(|e| {
                        Error::with_reason(
                            ErrorKind::Unexpected,
                            format!("Failed to acquire a storage guard: {e}"),
                        )
                    })?;
                    let trial = guard.get_trial(trial_id)?;

                    let trial_number = trial.number;
                    let datetime_start = trial.datetime_start.clone();
                    let datetime_complete = trial.datetime_complete.clone();
                    let fixed_params = extract_fixed_params(&trial.attrs);

                    guard.set_trial_state_values(trial_id, TrialStateValues::Running)?;

                    Ok((
                        trial_id,
                        trial_number,
                        datetime_start,
                        datetime_complete,
                        fixed_params,
                    ))
                })();

                match result {
                    Ok(values) => values,
                    Err(e) => {
                        // Push the trial_id back to the queue on storage error
                        let mut queue_guard = self.queue.write().map_err(|queue_err| {
                            Error::with_reason(
                                ErrorKind::Unexpected,
                                format!("Failed to acquire queue guard for recovery: {queue_err}"),
                            )
                        })?;
                        if let Err(queue_err) = queue_guard.enqueue(trial_id) {
                            return Err(Error::with_reason(
                                ErrorKind::Unexpected,
                                format!(
                                    "Failed to restore queued trial {trial_id} after ask error: {queue_err}; original error: {e}"
                                ),
                            ));
                        }
                        return Err(e);
                    }
                }
            } else {
                let mut guard = self.storage.write().map_err(|e| {
                    Error::with_reason(
                        ErrorKind::Unexpected,
                        format!("Failed to acquire a storage guard: {e}"),
                    )
                })?;
                let trial = guard.create_new_trial(self.id)?;
                (
                    trial.id,
                    trial.number,
                    trial.datetime_start.clone(),
                    trial.datetime_complete.clone(),
                    HashMap::new(),
                )
            };

        let joint_params: HashMap<String, (Distribution, f64)> =
            if self.sampler.support_joint_sampling() {
                let joint_search_space = self
                    .storage
                    .write()
                    .map_err(|e| {
                        Error::with_reason(
                            ErrorKind::Unexpected,
                            format!("Failed to acquire a storage guard: {e}"),
                        )
                    })?
                    .get_joint_search_space(self.id)?;

                let ctx = SamplerContext {
                    study_id: self.id,
                    trial_number,
                    trial_id,
                    directions: self.directions.clone(),
                };
                let params =
                    self.sampler
                        .sample_joint(&ctx, self.storage.clone(), &joint_search_space)?;
                let mut joint_params = HashMap::new();
                for (name, param_value) in params {
                    if !joint_search_space.contains_key(&name) {
                        continue;
                    }
                    let distribution = joint_search_space[&name].clone();
                    joint_params.insert(name, (distribution, param_value));
                }
                joint_params
            } else {
                HashMap::new()
            };

        let trial = Trial::new(
            trial_id,
            self.id,
            trial_number,
            datetime_start,
            datetime_complete,
            self.directions.clone(),
            Arc::clone(&self.storage),
            Arc::clone(&self.sampler),
            joint_params,
            fixed_params,
        );
        Ok(trial)
    }

    /// Finalizes a trial with the given state and objective values.
    pub fn tell(&self, trial_number: u32, state_values: TrialStateValues) -> Result<()> {
        let mut storage_guard = self.storage.write().map_err(|e| {
            Error::with_reason(
                ErrorKind::Unexpected,
                format!("Failed to acquire a storage guard: {e}"),
            )
        })?;
        let trial_id =
            storage_guard.get_trial_id_from_study_id_trial_number(self.id, trial_number)?;
        drop(storage_guard);

        let ctx = SamplerContext {
            study_id: self.id,
            directions: self.directions.clone(),
            trial_number,
            trial_id,
        };
        let after_trial_result =
            self.sampler
                .after_trial(&ctx, self.storage.clone(), &state_values);

        let mut storage_guard = self.storage.write().map_err(|e| {
            Error::with_reason(
                ErrorKind::Unexpected,
                format!("Failed to acquire a storage guard: {e}"),
            )
        })?;
        storage_guard.set_trial_state_values(trial_id, state_values)?;
        drop(storage_guard);

        after_trial_result?;
        Ok(())
    }

    /// Runs the objective function repeatedly for `n_trials`.
    ///
    /// The objective returns one value per study direction. Returning an error marks the current
    /// trial as failed and aborts the optimization loop.
    pub fn optimize<F>(&self, mut objective: F, n_trials: usize) -> Result<()>
    where
        F: FnMut(Trial) -> Result<Vec<f64>>,
    {
        for _ in 0..n_trials {
            let trial = self.ask()?;
            let trial_number = trial.number;

            // Call an objective function.
            let values = objective(trial);
            match values {
                Ok(values) => {
                    if self.directions.len() != values.len() {
                        return Err(Error::new(ErrorKind::InvalidObjectiveValues));
                    }
                    self.tell(trial_number, TrialStateValues::Complete(values))?;
                }
                Err(e) => {
                    self.tell(trial_number, TrialStateValues::Fail)?;
                    return Err(e);
                }
            }
        }
        Ok(())
    }

    /// Returns all trials that belong to this study.
    pub fn get_trials(&self) -> Result<Vec<PersistedTrial>> {
        let mut guard = self.storage.write().map_err(|e| {
            Error::with_reason(
                ErrorKind::StorageError,
                format!("Failed to acquire a storage guard: {e}"),
            )
        })?;
        let trials = guard.get_trials(self.id)?;
        Ok(trials.iter().flatten().cloned().collect())
    }

    /// Returns a user attribute stored on the study.
    pub fn get_user_attr(&self, key: String) -> Result<Option<String>> {
        let mut guard = self.storage.write().map_err(|e| {
            Error::with_reason(
                ErrorKind::Unexpected,
                format!("Failed to acquire a storage guard: {e}"),
            )
        })?;
        let study = guard.get_study(self.id)?;

        let key = AttrKey::User(key.into());
        match study.attrs.get(&key) {
            Some(value) => Ok(Some(value.clone())),
            _ => Ok(None),
        }
    }

    /// Sets one or more user attributes on the study.
    pub fn set_user_attr(&self, attrs: HashMap<String, String>) -> Result<()> {
        let mut guard = self.storage.write().map_err(|e| {
            Error::with_reason(
                ErrorKind::Unexpected,
                format!("Failed to acquire a storage guard: {e}"),
            )
        })?;
        let mut a = Attrs::new();
        for (key, value) in attrs {
            a.insert(AttrKey::User(key.into()), value);
        }
        guard.set_study_attrs(self.id, a, false)?;
        Ok(())
    }

    /// Inserts an existing persisted trial into the study.
    ///
    /// This is the Rustuna counterpart of Optuna's `study.add_trial`.
    pub fn add_trial(&self, mut trial: PersistedTrial) -> Result<()> {
        for distribution in trial.distributions.values_mut() {
            *distribution = distribution.adjusted();
        }
        trial.validate()?;
        if let TrialStateValues::Complete(ref values) = trial.state_values {
            if values.len() != self.directions.len() {
                return Err(Error::with_reason(
                    ErrorKind::InvalidObjectiveValues,
                    format!(
                        "The added trial has {} values, which is different from the number of objectives {} in the study.",
                        values.len(),
                        self.directions.len()
                    ),
                ));
            }
        }
        let mut guard = self.storage.write().map_err(|e| {
            Error::with_reason(
                ErrorKind::Unexpected,
                format!("Failed to acquire a storage guard: {e}"),
            )
        })?;
        guard.create_new_trial_from_template(self.id, &trial)?;
        Ok(())
    }

    /// Enqueues a trial with fixed parameters to be evaluated later.
    ///
    /// Queued parameters are stored as trial attributes and popped through the study's
    /// [`TrialQueue`] implementation.
    pub fn enqueue_trial(
        &self,
        params: HashMap<String, CategoryLabel>,
        user_attrs: Option<Attrs>,
    ) -> Result<()> {
        let mut guard = self.storage.write().map_err(|e| {
            Error::with_reason(
                ErrorKind::Unexpected,
                format!("Failed to acquire a storage guard: {e}"),
            )
        })?;
        let mut template = PersistedTrial::new(0, self.id, 0);
        template.state_values = TrialStateValues::Waiting;
        let fixed_attrs = fixed_params_to_attrs(&params);
        template.attrs.extend(fixed_attrs);

        if let Some(attrs) = user_attrs {
            template.attrs.extend(attrs);
        }

        let trial = guard.create_new_trial_from_template(self.id, &template)?;
        let trial_id = trial.id;
        drop(guard);

        let mut queue_guard = self.queue.write().map_err(|e| {
            Error::with_reason(
                ErrorKind::Unexpected,
                format!("Failed to acquire a queue guard: {e}"),
            )
        })?;
        queue_guard.enqueue(trial_id)?;

        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq)]
/// Optimization direction for an objective value.
pub enum Direction {
    Minimize,
    Maximize,
}

#[derive(Clone)]
/// Storage-side representation of a study.
///
/// This corresponds to Optuna's `FrozenStudy`.
pub struct PersistedStudy {
    pub id: u32,
    pub name: String,
    pub directions: Vec<Direction>,
    pub attrs: Attrs,
}
impl PersistedStudy {
    /// Creates a persisted study with empty attributes.
    pub fn new(id: u32, study_name: String, directions: Vec<Direction>) -> PersistedStudy {
        PersistedStudy {
            id,
            name: study_name,
            directions,
            attrs: Attrs::new(),
        }
    }
    // TODO(knshnb): Consider a builder pattern:
    // https://github.com/optuna/rustuna/pull/37#discussion_r1503510194
    /// Creates a persisted study with pre-populated attributes.
    pub fn new_with_attrs(
        id: u32,
        study_name: String,
        directions: Vec<Direction>,
        attrs: Attrs,
    ) -> PersistedStudy {
        PersistedStudy {
            id,
            name: study_name,
            directions,
            attrs,
        }
    }
}

/// Returns the best completed trial number for a single-objective study.
pub fn get_best_trial(study: &Study) -> Result<u32> {
    let mut guard = study.storage.write().map_err(|e| {
        Error::with_reason(
            ErrorKind::StorageError,
            format!("Failed to acquire a storage guard: {e}"),
        )
    })?;
    let trials = guard.get_trials(study.id)?;

    let best_trial = trials
        .iter()
        .flatten()
        .filter(|trial| matches!(trial.state_values, TrialStateValues::Complete(_)))
        .min_by(|a, b| {
            let a_value = match a.state_values {
                TrialStateValues::Complete(ref v) => {
                    assert!(v.len() == 1);
                    v[0]
                }
                _ => unreachable!("Unexpected state"),
            };
            let b_value = match b.state_values {
                TrialStateValues::Complete(ref v) => {
                    assert!(v.len() == 1);
                    v[0]
                }
                _ => unreachable!("Unexpected state"),
            };
            a_value
                .partial_cmp(&b_value)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .ok_or(Error::new(ErrorKind::NoCompletedTrial))?;
    Ok(best_trial.number)
}

// TODO(HideakiImamura): Support the faster algorithm for `len(directions) == 2`.
/// Returns the trial numbers on the Pareto front of a multi-objective study.
pub fn get_pareto_front(study: &Study) -> Result<Vec<u32>> {
    let mut guard = study.storage.write().map_err(|e| {
        Error::with_reason(
            ErrorKind::StorageError,
            format!("Failed to acquire a storage guard: {e}"),
        )
    })?;
    let trials = guard
        .get_trials(study.id)?
        .iter()
        .flatten()
        .filter(|t| matches!(t.state_values, TrialStateValues::Complete(ref _v)))
        .collect::<Vec<_>>();

    validate_trials(&trials, &study.directions)?;

    // TODO(HideakiImamura): Use Vec::with_capacity() to reduce the number of memory allocations.
    let mut pareto_front_numbers = vec![];
    trials.iter().try_for_each(|trial| {
        let mut dominated = false;
        let TrialStateValues::Complete(ref trial_values) = trial.state_values else {
            return Ok(());
        };
        for other in trials.iter() {
            let TrialStateValues::Complete(ref other_values) = other.state_values else {
                continue;
            };
            if dominates(other_values, trial_values, &study.directions) {
                dominated = true;
                break;
            }
        }
        if !dominated {
            pareto_front_numbers.push(trial.number);
        }
        Ok(())
    })?;

    Ok(pareto_front_numbers)
}

/// Returns whether `values0` dominates `values1` under the given directions.
/// Validate `values0` and `values1` have the same length and neither of them contains f64::NAN
/// by `rustuna_core::trial::validate_trials` before calling this function.
pub fn dominates(values0: &[f64], values1: &[f64], directions: &[Direction]) -> bool {
    debug_assert_eq!(values0.len(), values1.len());
    debug_assert_eq!(values0.len(), directions.len());
    debug_assert!(values0.iter().all(|x| !x.is_nan()));
    debug_assert!(values1.iter().all(|x| !x.is_nan()));

    let mut equal = true;
    for ((v0, v1), d) in values0.iter().zip(values1).zip(directions) {
        if *v0 != *v1 {
            equal = false;
        }
        let v1_dominate_v0 = match d {
            Direction::Minimize => *v0 > *v1,
            Direction::Maximize => *v0 < *v1,
        };
        if v1_dominate_v0 {
            return false; // Early return
        }
    }
    !equal
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attr::AttrKey;
    use crate::distribution::Distribution;
    use crate::sampler::{Context as SamplerContext, Sampler};
    use std::sync::Mutex;
    use std::thread;

    use crate::sampler::RandomSampler;
    use crate::storage::InMemoryStorage;
    use crate::study::create_study;
    use crate::study::create_study_with_arc;
    use crate::study::get_best_trial;

    struct RecordingSampler {
        calls: Arc<Mutex<Vec<(SamplerContext, TrialStateValues)>>>,
        fail_after_trial: bool,
    }

    impl Sampler for RecordingSampler {
        fn sample_independent(
            &self,
            _ctx: &SamplerContext,
            _storage: Arc<RwLock<dyn Storage>>,
            _name: &str,
            _distribution: &Distribution,
        ) -> Result<f64> {
            Ok(0.0)
        }

        fn support_joint_sampling(&self) -> bool {
            false
        }

        fn sample_joint(
            &self,
            _ctx: &SamplerContext,
            _storage: Arc<RwLock<dyn Storage>>,
            _search_space: &HashMap<String, Distribution>,
        ) -> Result<HashMap<String, f64>> {
            unreachable!()
        }

        fn after_trial(
            &self,
            ctx: &SamplerContext,
            storage: Arc<RwLock<dyn Storage>>,
            state_values: &TrialStateValues,
        ) -> Result<()> {
            let mut storage_guard = storage.write().map_err(|e| {
                Error::with_reason(
                    ErrorKind::Unexpected,
                    format!("Failed to acquire a storage guard: {e}"),
                )
            })?;
            let trial = storage_guard.get_trial(ctx.trial_id)?;
            assert_eq!(trial.state_values, TrialStateValues::Running);
            drop(storage_guard);

            self.calls
                .lock()
                .map_err(|e| {
                    Error::with_reason(
                        ErrorKind::Unexpected,
                        format!("Failed to acquire a calls guard: {e}"),
                    )
                })?
                .push((ctx.clone(), state_values.clone()));

            if self.fail_after_trial {
                return Err(Error::with_reason(
                    ErrorKind::SamplerError,
                    "after_trial failed",
                ));
            }
            Ok(())
        }
    }

    #[test]
    fn test_optimize() -> Result<()> {
        let storage = InMemoryStorage::new();
        let directions = vec![Direction::Minimize];
        let study = create_study("dummy", storage, RandomSampler::new(), directions)?;

        study.optimize(
            |mut t| {
                let x = t.suggest_float("x", 0.0, 10.0)?;
                let y = t.suggest_float("y", 0.0, 10.0)?;
                let z = t.suggest_int("z", 0, 10)?;

                let value = (x - 3.0).powi(2) + (y - 5.0).powi(2) + (z as f64);
                Ok(vec![value])
            },
            100,
        )?;
        assert!(get_best_trial(&study).is_ok());
        Ok(())
    }

    #[test]
    fn test_optimize_parallel() -> Result<()> {
        let storage = InMemoryStorage::new();
        let directions = vec![Direction::Minimize];
        let study = create_study("dummy-study", storage, RandomSampler::new(), directions)?;

        thread::scope(|s| {
            for _i in 0..4 {
                let study = study.clone();
                let choices = vec![String::from("foo"), String::from("bar")];
                s.spawn(move || {
                    study
                        .optimize(
                            |mut t| {
                                let x = t.suggest_float("x", 0.0, 10.0)?;
                                let y = t.suggest_float("y", 0.0, 10.0)?;
                                let z = t.suggest_int("z", 0, 10)?;
                                let _c = t.suggest_categorical("cat", &choices)?;
                                let value = (x - 3.0).powi(2) + (y - 5.0).powi(2) + (z as f64);
                                Ok(vec![value])
                            },
                            100,
                        )
                        .expect("Optimization failed");
                });
            }
        });
        assert!(get_best_trial(&study).is_ok());
        assert_eq!(study.get_trials()?.len(), 400);
        Ok(())
    }

    #[test]
    fn test_user_attr() -> Result<()> {
        let storage = InMemoryStorage::new();
        let directions = vec![Direction::Minimize];
        let study = create_study("dummy", storage, RandomSampler::new(), directions)?;

        let mut trial = study.ask()?;
        trial.set_user_attr("key", String::from("bar"))?;
        let user_attr = trial
            .get_user_attr("key")
            .ok_or_else(|| Error::new(ErrorKind::StorageError))?;
        assert_eq!(user_attr, "bar");
        Ok(())
    }

    #[test]
    fn test_get_best_trial() -> Result<()> {
        let storage = InMemoryStorage::new();
        let directions = vec![Direction::Minimize];
        let study = create_study("dummy", storage, RandomSampler::new(), directions)?;

        study.optimize(
            |mut t| {
                let x = t.suggest_float("x", 0.0, 10.0)?;
                let y = t.suggest_float("y", 0.0, 10.0)?;
                let z = t.suggest_int("z", 0, 10)?;

                let value = (x - 3.0).powi(2) + (y - 5.0).powi(2) + (z as f64);
                Ok(vec![value])
            },
            100,
        )?;

        let best_trial_number = get_best_trial(&study)?;
        assert!(best_trial_number < 100);
        Ok(())
    }

    #[test]
    fn test_get_pareto_front_trials() -> Result<()> {
        let storage = InMemoryStorage::new();
        let directions = vec![Direction::Minimize, Direction::Maximize];
        let study = create_study("dummy", storage, RandomSampler::new(), directions)?;

        study.optimize(
            |mut t| {
                let x = t.suggest_float("x", 0.0, 10.0)?;
                let y = t.suggest_float("y", 0.0, 10.0)?;
                let z = t.suggest_int("z", 0, 10)?;

                let value0 = (x - 3.0).powi(2) + (y - 5.0).powi(2) + (z as f64);
                let value1 = (x - 3.0).powi(2) + (y - 5.0).powi(2) + (z as f64) * 2.0;
                Ok(vec![value0, value1])
            },
            100,
        )?;

        let pareto_front_numbers = get_pareto_front(&study)?;
        assert!(!pareto_front_numbers.is_empty());
        assert!(pareto_front_numbers.len() <= 100);
        Ok(())
    }

    #[test]
    fn test_dominates() {
        let directions = vec![Direction::Minimize, Direction::Maximize];
        assert!(dominates(&[1.0, 2.0], &[2.0, 1.0], &directions));
        assert!(!dominates(&[2.0, 1.0], &[1.0, 2.0], &directions));
        assert!(!dominates(&[1.0, 2.0], &[1.0, 2.0], &directions));
        assert!(!dominates(&[2.0, 1.0], &[2.0, 1.0], &directions));
    }

    #[test]
    fn test_dynamic_search_space() -> Result<()> {
        let storage = InMemoryStorage::new();
        let directions = vec![Direction::Minimize];
        let study = create_study("dummy", storage, RandomSampler::new(), directions)?;

        study.optimize(
            |mut t| {
                t.suggest_float("x", 0.0, 10.0)?;
                Ok(vec![0.0])
            },
            5,
        )?;
        study.optimize(
            |mut t| {
                t.suggest_float("x", 1.0, 10.0)?;
                Ok(vec![0.0])
            },
            5,
        )?;
        Ok(())
    }

    #[test]
    fn test_optimize_adjusts_discrete_distribution_high() -> Result<()> {
        let storage = InMemoryStorage::new();
        let directions = vec![Direction::Minimize];
        let study = create_study(
            "adjust-distribution-high",
            storage,
            RandomSampler::seed_from_u64(42),
            directions,
        )?;

        study.optimize(
            |mut trial| {
                // Bypass the constructors to verify that `Trial::suggest` adjusts distributions.
                trial.suggest(
                    "float",
                    &Distribution::Float {
                        low: -5.0,
                        high: 10.0,
                        step: Some(2.0),
                        log: false,
                    },
                )?;
                trial.suggest(
                    "int",
                    &Distribution::Int {
                        low: -5,
                        high: 10,
                        step: 2,
                        log: false,
                    },
                )?;
                Ok(vec![0.0])
            },
            1,
        )?;

        let trials = study.get_trials()?;
        assert_eq!(
            trials[0].distributions["float"],
            Distribution::new_float(-5.0, 9.0, Some(2.0), false)
        );
        assert_eq!(
            trials[0].distributions["int"],
            Distribution::new_int(-5, 9, 2, false)
        );
        Ok(())
    }

    #[test]
    fn test_invalid_dynamic_search_space() -> Result<()> {
        let storage = InMemoryStorage::new();
        let directions = vec![Direction::Minimize];
        let study = create_study("dummy", storage, RandomSampler::new(), directions)?;

        study.optimize(
            |mut t| {
                t.suggest_float("x", 0.0, 10.0)?;
                Ok(vec![0.0])
            },
            5,
        )?;
        let error = study
            .optimize(
                |mut t| {
                    t.suggest_int("x", 0, 10)?;
                    Ok(vec![0.0])
                },
                5,
            )
            .unwrap_err();
        assert!(matches!(error.kind, ErrorKind::IncompatibleDistribution));
        Ok(())
    }

    #[test]
    fn test_add_trial_preserves_template_fields() -> Result<()> {
        let storage = InMemoryStorage::new();
        let directions = vec![Direction::Minimize];
        let study = create_study("dummy-add-trial", storage, RandomSampler::new(), directions)?;

        let mut trial = PersistedTrial::new(999, 888, 777);
        trial.state_values = TrialStateValues::Complete(vec![1.23]);
        trial.datetime_start = Some("2026-04-02 03:04:05.678".to_string());
        trial.datetime_complete = Some("2026-04-02 03:14:15.678".to_string());
        trial.internal_params.insert("x".to_string(), 0.5);
        trial.distributions.insert(
            "x".to_string(),
            Distribution::new_float(0.0, 1.0, None, false),
        );
        trial
            .attrs
            .insert(AttrKey::User("memo".into()), "\"imported\"".to_string());

        study.add_trial(trial)?;
        let trials = study.get_trials()?;
        assert_eq!(trials.len(), 1);
        assert_eq!(trials[0].number, 0);
        assert_eq!(trials[0].study_id, study.id);
        assert_eq!(
            trials[0].datetime_start.as_deref(),
            Some("2026-04-02 03:04:05.678")
        );
        assert_eq!(
            trials[0].datetime_complete.as_deref(),
            Some("2026-04-02 03:14:15.678")
        );
        assert_eq!(trials[0].internal_params.get("x"), Some(&0.5));
        assert_eq!(
            trials[0].attrs.get(&AttrKey::User("memo".into())),
            Some(&"\"imported\"".to_string())
        );
        Ok(())
    }

    #[test]
    fn test_add_trial_adjusts_distribution_high() -> Result<()> {
        let storage = InMemoryStorage::new();
        let directions = vec![Direction::Minimize];
        let study = create_study(
            "adjust-added-trial-high",
            storage,
            RandomSampler::new(),
            directions,
        )?;

        let mut trial = PersistedTrial::new(999, 888, 777);
        trial.state_values = TrialStateValues::Complete(vec![1.0]);
        trial.internal_params.insert("x".to_string(), 9.0);
        // Bypass the constructor to verify that `Study::add_trial` adjusts distributions.
        trial.distributions.insert(
            "x".to_string(),
            Distribution::Float {
                low: -5.0,
                high: 10.0,
                step: Some(2.0),
                log: false,
            },
        );

        study.add_trial(trial)?;

        assert_eq!(
            study.get_trials()?[0].distributions["x"],
            Distribution::new_float(-5.0, 9.0, Some(2.0), false)
        );
        Ok(())
    }

    #[test]
    fn test_tell_calls_after_trial_before_storage_update() -> Result<()> {
        let storage = Arc::new(RwLock::new(InMemoryStorage::new()));
        let calls = Arc::new(Mutex::new(Vec::new()));
        let sampler = Arc::new(RecordingSampler {
            calls: calls.clone(),
            fail_after_trial: false,
        });
        let study = create_study_with_arc(
            "dummy-after-trial",
            storage.clone(),
            sampler,
            vec![Direction::Minimize],
        )?;

        let trial = study.ask()?;
        study.tell(trial.number, TrialStateValues::Complete(vec![1.5]))?;

        let calls_guard = calls.lock().map_err(|e| {
            Error::with_reason(
                ErrorKind::Unexpected,
                format!("Failed to acquire a calls guard: {e}"),
            )
        })?;
        assert_eq!(calls_guard.len(), 1);
        let (ctx, state_values) = &calls_guard[0];
        assert_eq!(ctx.study_id, study.id);
        assert_eq!(ctx.trial_id, trial.id);
        assert_eq!(ctx.trial_number, trial.number);
        assert_eq!(state_values, &TrialStateValues::Complete(vec![1.5]));
        drop(calls_guard);

        let mut storage_guard = storage.write().map_err(|e| {
            Error::with_reason(
                ErrorKind::Unexpected,
                format!("Failed to acquire a storage guard: {e}"),
            )
        })?;
        let persisted_trial = storage_guard.get_trial(trial.id)?;
        assert_eq!(
            persisted_trial.state_values,
            TrialStateValues::Complete(vec![1.5])
        );
        Ok(())
    }

    #[test]
    fn test_tell_persists_trial_even_if_after_trial_fails() -> Result<()> {
        let storage = Arc::new(RwLock::new(InMemoryStorage::new()));
        let calls = Arc::new(Mutex::new(Vec::new()));
        let sampler = Arc::new(RecordingSampler {
            calls,
            fail_after_trial: true,
        });
        let study = create_study_with_arc(
            "dummy-after-trial-failure",
            storage.clone(),
            sampler,
            vec![Direction::Minimize],
        )?;

        let trial = study.ask()?;
        // Optuna persists the final state in a finally block even if after_trial raises.
        let err = study
            .tell(trial.number, TrialStateValues::Complete(vec![2.5]))
            .expect_err("tell must propagate after_trial error");
        assert!(matches!(err.kind, ErrorKind::SamplerError));

        let mut storage_guard = storage.write().map_err(|e| {
            Error::with_reason(
                ErrorKind::Unexpected,
                format!("Failed to acquire a storage guard: {e}"),
            )
        })?;
        let persisted_trial = storage_guard.get_trial(trial.id)?;
        assert_eq!(
            persisted_trial.state_values,
            TrialStateValues::Complete(vec![2.5])
        );
        Ok(())
    }
}
