use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};

use crate::attr::{category_labels_to_attrs, AttrKey, Attrs, CategoryLabel};
use crate::distribution::Distribution;
use crate::sampler::{Context as SamplerContext, Sampler};
use crate::storage::Storage;
use crate::study::Direction;
use crate::{Error, ErrorKind, Result};

/// A trial object used while evaluating an objective function.
///
/// This is the Rustuna counterpart of `optuna.trial.Trial`. It provides parameter suggestion
/// APIs and access to user-defined trial attributes.
pub struct Trial {
    pub id: u32,
    pub study_id: u32,
    pub number: u32,
    pub datetime_start: Option<String>,
    pub datetime_complete: Option<String>,
    directions: Vec<Direction>,
    storage: Arc<RwLock<dyn Storage>>,
    sampler: Arc<Mutex<dyn Sampler>>,
    joint_params: HashMap<String, (Distribution, f64)>,
    fixed_params: HashMap<String, CategoryLabel>,
    cached_trial: PersistedTrial,
}
impl Trial {
    /// Constructs a trial from storage and sampler state.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        trial_id: u32,
        study_id: u32,
        number: u32,
        datetime_start: Option<String>,
        datetime_complete: Option<String>,
        directions: Vec<Direction>,
        storage: Arc<RwLock<dyn Storage>>,
        sampler: Arc<Mutex<dyn Sampler>>,
        joint_params: HashMap<String, (Distribution, f64)>,
        fixed_params: HashMap<String, CategoryLabel>,
    ) -> Self {
        let mut cached_trial = PersistedTrial::new(trial_id, study_id, number);
        cached_trial.datetime_start = datetime_start.clone();
        cached_trial.datetime_complete = datetime_complete.clone();
        Trial {
            id: trial_id,
            study_id,
            number,
            datetime_start,
            datetime_complete,
            directions,
            storage,
            sampler,
            joint_params,
            fixed_params,
            cached_trial,
        }
    }

    /// Suggests a parameter value from the given distribution.
    ///
    /// The returned value is in Rustuna's internal representation. Categorical parameters are
    /// represented by zero-based indices.
    pub fn suggest(&mut self, name: &str, distribution: &Distribution) -> Result<f64> {
        let distribution = &distribution.adjusted();
        if let Some(fixed_value) = self.fixed_params.get(name) {
            let result = if let Distribution::Categorical { cardinality } = distribution {
                // CategoricalDistribution
                let mut storage_guard = self.storage.write().map_err(|e| {
                    Error::with_reason(
                        ErrorKind::StorageError,
                        format!("Failed to acquire storage guard: {e}"),
                    )
                })?;
                let labels =
                    storage_guard.get_category_labels(self.study_id, name, *cardinality)?;
                drop(storage_guard);

                labels.and_then(|labels| {
                    labels
                        .iter()
                        .position(|l| l == fixed_value)
                        .map(|index| index as f64)
                })
            } else {
                // IntDistribution and FloatDistribution
                // Note: step and log validation are intentionally omitted, matching Optuna's behavior.
                match (fixed_value, distribution) {
                    (CategoryLabel::Float(f), Distribution::Float { .. }) => Some(*f),
                    (CategoryLabel::Int(i), Distribution::Float { .. }) => Some(*i as f64),
                    (CategoryLabel::Int(i), Distribution::Int { .. }) => Some(*i as f64),
                    (CategoryLabel::Float(f), Distribution::Int { .. }) => Some(*f as i64 as f64),
                    _ => None,
                }
                .filter(|&v| distribution.contains(v))
            };

            if let Some(internal_value) = result {
                let mut storage_guard = self.storage.write().map_err(|e| {
                    Error::with_reason(
                        ErrorKind::StorageError,
                        format!("Failed to acquire storage guard: {e}"),
                    )
                })?;
                storage_guard.set_trial_param(self.id, name, distribution, internal_value)?;
                drop(storage_guard);

                self.cached_trial
                    .internal_params
                    .insert(name.to_string(), internal_value);
                self.cached_trial
                    .distributions
                    .insert(name.to_string(), distribution.clone());

                return Ok(internal_value);
            }
        }

        if let Some((d, val)) = self.joint_params.get(name) {
            if *d == *distribution {
                let param_value = *val;
                let mut storage_guard = self.storage.write().map_err(|e| {
                    Error::with_reason(
                        ErrorKind::StorageError,
                        format!("Failed to acquire storage guard: {e}"),
                    )
                })?;
                storage_guard.set_trial_param(self.id, name, distribution, param_value)?;
                drop(storage_guard);
                return Ok(param_value);
            }
        }

        let context = SamplerContext {
            study_id: self.study_id,
            trial_number: self.number,
            trial_id: self.id,
            directions: self.directions.clone(),
        };
        let mut sampler_guard = self.sampler.lock().map_err(|e| {
            Error::with_reason(
                ErrorKind::SamplerError,
                format!("Failed to acquire sampler guard: {e}"),
            )
        })?;
        let param_value =
            sampler_guard.sample_independent(&context, self.storage.clone(), name, distribution)?;
        drop(sampler_guard);

        let mut storage_guard = self.storage.write().map_err(|e| {
            Error::with_reason(
                ErrorKind::StorageError,
                format!("Failed to acquire storage guard: {e}"),
            )
        })?;
        storage_guard.set_trial_param(self.id, name, distribution, param_value)?;
        drop(storage_guard);

        self.cached_trial
            .internal_params
            .insert(name.to_string(), param_value);
        self.cached_trial
            .distributions
            .insert(name.to_string(), distribution.clone());

        Ok(param_value)
    }

    // Design Note:
    // suggest_float and suggest_int do not support `step` and `log` arguments to keep the API easy to use.
    // Users who want to use them need to call `suggest()` function directly.
    /// Suggests a floating-point parameter from a linear range.
    pub fn suggest_float(&mut self, name: &str, low: f64, high: f64) -> Result<f64> {
        let distribution = Distribution::new_float(low, high, None, false);
        let param_value = self.suggest(name, &distribution)?;
        Ok(param_value)
    }

    /// Suggests an integer parameter from a linear range with step `1`.
    pub fn suggest_int(&mut self, name: &str, low: i64, high: i64) -> Result<i64> {
        let distribution = Distribution::new_int(low, high, 1, false);
        let param_value = self.suggest(name, &distribution)?;
        Ok(param_value as i64)
    }

    /// Suggests a categorical parameter and returns a borrowed [`CategoryLabel`].
    pub fn suggest_categorical_enum<'a>(
        &'a mut self,
        name: &str,
        choices: &'a [CategoryLabel],
    ) -> Result<&'a CategoryLabel> {
        let study_id = self.study_id;
        let storage = self.storage.clone();

        // Save labels in the study system attr before calling suggest(),
        // so that fixed_params can look up category labels during suggest().
        let category_labels = category_labels_to_attrs(name, choices);
        let mut guard = storage.write().map_err(|e| {
            Error::with_reason(
                ErrorKind::Unexpected,
                format!("Failed to acquire storage guard: {e}"),
            )
        })?;
        guard.set_study_attrs(study_id, category_labels, false)?;
        drop(guard);

        let c = self.suggest_categorical(name, choices)?;
        Ok(c)
    }

    /// Suggests a categorical parameter and returns a borrowed element from `choices`.
    pub fn suggest_categorical<'a, T>(&'a mut self, name: &str, choices: &'a [T]) -> Result<&'a T> {
        let distribution = Distribution::new_categorical(choices.len());
        let param_value = self.suggest(name, &distribution)?;
        if choices.len() <= (param_value as usize) {
            return Err(Error::new(ErrorKind::Unexpected));
        }
        let choice = &choices[param_value as usize];
        Ok(choice)
    }

    // Design Note:
    // Note that `AttrKind::System` should not be exposed to users,
    // as it may be updated without going through the `Trial` object,
    // making it difficult to cache the value.
    /// Returns a user attribute stored on the trial.
    pub fn get_user_attr(&mut self, key: &str) -> Option<&String> {
        let key = AttrKey::User(key.into());
        self.cached_trial.attrs.get(&key)
    }
    /// Returns user attributes stored on the trial.
    pub fn get_user_attrs(&self) -> HashMap<String, String> {
        let mut user_attrs = HashMap::with_capacity(self.cached_trial.attrs.len());
        for (key, value) in &self.cached_trial.attrs {
            if let AttrKey::User(key) = key {
                user_attrs.insert(key.to_string(), value.clone());
            }
        }
        user_attrs
    }

    /// Sets a single user attribute on the trial.
    pub fn set_user_attr(&mut self, key: &str, value: String) -> Result<()> {
        let mut guard = self.storage.write().map_err(|e| {
            Error::with_reason(
                ErrorKind::StorageError,
                format!("Failed to acquire storage guard: {e}"),
            )
        })?;
        let mut attrs = Attrs::new();

        let key = AttrKey::User(key.into());
        attrs.insert(key.clone(), value.clone());
        guard.set_trial_attrs(self.id, attrs, false)?;
        self.cached_trial.attrs.insert(key, value);
        Ok(())
    }

    /// Sets multiple user attributes on the trial.
    pub fn set_user_attrs(&mut self, user_attrs: HashMap<String, String>) -> Result<()> {
        let mut attrs = Attrs::with_capacity(user_attrs.len());
        for (key, value) in &user_attrs {
            attrs.insert(AttrKey::User(key.as_str().into()), value.clone());
        }
        let mut guard = self.storage.write().map_err(|e| {
            Error::with_reason(
                ErrorKind::StorageError,
                format!("Failed to acquire storage guard: {e}"),
            )
        })?;
        guard.set_trial_attrs(self.id, attrs, false)?;
        drop(guard);

        for (key, value) in user_attrs {
            self.cached_trial
                .attrs
                .insert(AttrKey::User(key.into()), value);
        }
        Ok(())
    }

    /// Sets multiple constraints on the trial.
    pub fn set_constraints(&mut self, constraints: HashMap<String, f64>) -> Result<()> {
        let mut guard = self.storage.write().map_err(|e| {
            Error::with_reason(
                ErrorKind::StorageError,
                format!("Failed to acquire storage guard: {e}"),
            )
        })?;
        guard.set_trial_constraints(self.id, constraints.clone())?;
        drop(guard);
        self.cached_trial.constraints = constraints;
        Ok(())
    }
}

/// Storage-side representation of a trial.
///
/// This corresponds to Optuna's `FrozenTrial`, but it stores user and system attributes as
/// strings in a unified attribute map.
#[derive(Clone, Debug)]
pub struct PersistedTrial {
    pub id: u32,
    pub study_id: u32,
    pub number: u32,
    pub state_values: TrialStateValues,
    pub internal_params: HashMap<String, f64>,
    pub distributions: HashMap<String, Distribution>,
    /// Intermediate values reported for Optuna-compatible storage migration.
    ///
    /// Rustuna does not currently provide pruner support. This field exists to preserve Optuna
    /// trial data and may be used by future pruning APIs.
    pub intermediate_values: HashMap<u32, f64>,
    /// Named constraint values used by constrained samplers.
    pub constraints: HashMap<String, f64>,
    pub attrs: Attrs,
    /// When the trial started, as timezone-naive **UTC** (`%Y-%m-%d %H:%M:%S%.f`).
    ///
    /// UTC needs a clock but no timezone database, which is why it is the representation shared by
    /// every backend: see [`crate::datetime`]. Storage backends encode it slightly differently on
    /// disk (SQLite keeps naive UTC, journal logs carry an explicit `+00:00`), and the conversion
    /// to the local time that `FrozenTrial` exposes happens in the Python bindings.
    pub datetime_start: Option<String>,
    /// When the trial finished, in the same representation as [`Self::datetime_start`].
    pub datetime_complete: Option<String>,
}
impl PersistedTrial {
    /// Creates a new running trial with no parameters or attributes.
    pub fn new(id: u32, study_id: u32, number: u32) -> PersistedTrial {
        PersistedTrial {
            id,
            study_id,
            number,
            state_values: TrialStateValues::Running,
            internal_params: HashMap::new(),
            distributions: HashMap::new(),
            intermediate_values: HashMap::new(),
            constraints: HashMap::new(),
            attrs: Attrs::new(),
            datetime_start: None,
            datetime_complete: None,
        }
    }

    /// Returns whether the trial is already in a finished state.
    pub fn is_finished(&self) -> bool {
        matches!(
            self.state_values,
            TrialStateValues::Complete(_) | TrialStateValues::Pruned | TrialStateValues::Fail
        )
    }

    /// Validates the internal consistency of the persisted trial.
    pub fn validate(&self) -> Result<()> {
        // TODO(c-bata): Consider introducing ErrorKind::TrialInvalid.
        if let TrialStateValues::Complete(values) = &self.state_values {
            if values.iter().any(|v| v.is_nan()) {
                return Err(Error::with_reason(
                    ErrorKind::StorageError,
                    "values should not contain NaN.".to_string(),
                ));
            }
        }
        if self.internal_params.len() != self.distributions.len() {
            return Err(Error::with_reason(
                ErrorKind::StorageError,
                format!(
                    "The number of parameters {} and distributions {} don't match.",
                    self.internal_params.len(),
                    self.distributions.len()
                ),
            ));
        }
        for (param_name, &internal_value) in &self.internal_params {
            let distribution = self.distributions.get(param_name).ok_or_else(|| {
                Error::with_reason(
                    ErrorKind::StorageError,
                    format!("Parameter '{param_name}' is not found in distributions."),
                )
            })?;
            if !distribution.contains(internal_value) {
                return Err(Error::with_reason(
                    ErrorKind::StorageError,
                    format!(
                        "The value {internal_value} of parameter '{param_name}' isn't contained in the distribution {distribution:?}."
                    ),
                ));
            }
        }
        Ok(())
    }

    /// Gets constraints on the trial.
    pub fn constraints(&self) -> &HashMap<String, f64> {
        &self.constraints
    }
}

/// Trial state together with objective values when available.
#[derive(PartialEq, Clone, Debug)]
pub enum TrialStateValues {
    /// Trial is running.
    Running,
    /// Trial was pruned.
    Pruned,
    /// Trial finished successfully with one or more objective values.
    Complete(Vec<f64>),
    /// Trial failed with an error.
    Fail,
    /// Trial is waiting in a queue.
    Waiting,
}

/// State of a trial without its objective values.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum TrialState {
    /// Trial is running.
    Running,
    /// Trial finished successfully.
    Complete,
    /// Trial was pruned.
    Pruned,
    /// Trial is waiting in a queue.
    Waiting,
    /// Trial failed.
    Fail,
}

impl TrialStateValues {
    /// Returns the state without objective values.
    pub fn state(&self) -> TrialState {
        match self {
            TrialStateValues::Running => TrialState::Running,
            TrialStateValues::Complete(_) => TrialState::Complete,
            TrialStateValues::Pruned => TrialState::Pruned,
            TrialStateValues::Waiting => TrialState::Waiting,
            TrialStateValues::Fail => TrialState::Fail,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::sampler::RandomSampler;
    use crate::storage::InMemoryStorage;
    use crate::study::create_study_with_arc;

    #[test]
    fn test_enqueue_and_suggest_float() -> Result<()> {
        let storage = Arc::new(RwLock::new(InMemoryStorage::new()));
        let sampler = Arc::new(Mutex::new(RandomSampler::new()));
        let directions = vec![Direction::Minimize];
        let study = create_study_with_arc("dummy", storage.clone(), sampler, directions)?;

        let mut params = HashMap::new();
        params.insert("x".to_string(), CategoryLabel::Float(5.0));
        study.enqueue_trial(params, None)?;

        let mut trial = study.ask()?;
        let value = trial.suggest_float("x", 0.0, 10.0)?;
        assert_eq!(value, 5.0);
        Ok(())
    }

    #[test]
    fn test_enqueue_and_suggest_int() -> Result<()> {
        let storage = Arc::new(RwLock::new(InMemoryStorage::new()));
        let sampler = Arc::new(Mutex::new(RandomSampler::new()));
        let directions = vec![Direction::Minimize];
        let study = create_study_with_arc("dummy", storage.clone(), sampler, directions)?;

        let mut params = HashMap::new();
        params.insert("x".to_string(), CategoryLabel::Int(7));
        study.enqueue_trial(params, None)?;

        let mut trial = study.ask()?;
        let value = trial.suggest_int("x", 0, 10)?;
        assert_eq!(value, 7);
        Ok(())
    }

    #[test]
    fn test_enqueue_fallback_on_out_of_range() -> Result<()> {
        let storage = Arc::new(RwLock::new(InMemoryStorage::new()));
        let sampler = Arc::new(Mutex::new(RandomSampler::new()));
        let directions = vec![Direction::Minimize];
        let study = create_study_with_arc("dummy", storage.clone(), sampler, directions)?;

        let mut params = HashMap::new();
        params.insert("x".to_string(), CategoryLabel::Float(100.0));
        study.enqueue_trial(params, None)?;

        let mut trial = study.ask()?;
        let value = trial.suggest_float("x", 0.0, 10.0)?;
        assert!((0.0..=10.0).contains(&value));
        Ok(())
    }

    #[test]
    fn test_enqueue_mixed_with_normal_ask() -> Result<()> {
        let storage = Arc::new(RwLock::new(InMemoryStorage::new()));
        let sampler = Arc::new(Mutex::new(RandomSampler::new()));
        let directions = vec![Direction::Minimize];
        let study = create_study_with_arc("dummy", storage.clone(), sampler, directions)?;

        let mut params = HashMap::new();
        params.insert("x".to_string(), CategoryLabel::Float(5.0));
        study.enqueue_trial(params, None)?;

        // First ask should use enqueued trial
        let mut trial1 = study.ask()?;
        assert_eq!(trial1.suggest_float("x", 0.0, 10.0)?, 5.0);

        // Second ask should create a new trial (sampled)
        let mut trial2 = study.ask()?;
        let value = trial2.suggest_float("x", 0.0, 10.0)?;
        assert!((0.0..=10.0).contains(&value));

        Ok(())
    }

    #[test]
    fn test_enqueue_unspecified_param_sampled() -> Result<()> {
        let storage = Arc::new(RwLock::new(InMemoryStorage::new()));
        let sampler = Arc::new(Mutex::new(RandomSampler::new()));
        let directions = vec![Direction::Minimize];
        let study = create_study_with_arc("dummy", storage.clone(), sampler, directions)?;

        let mut params = HashMap::new();
        params.insert("x".to_string(), CategoryLabel::Float(5.0));
        study.enqueue_trial(params, None)?;

        let mut trial = study.ask()?;
        assert_eq!(trial.suggest_float("x", 0.0, 10.0)?, 5.0);
        // "y" is not in fixed_params, so it should be sampled
        let y = trial.suggest_float("y", 0.0, 10.0)?;
        assert!((0.0..=10.0).contains(&y));

        Ok(())
    }

    #[test]
    fn test_trial_user_attr() -> Result<()> {
        let storage = Arc::new(RwLock::new(InMemoryStorage::new()));
        let sampler = Arc::new(Mutex::new(RandomSampler::new()));
        let directions = vec![Direction::Minimize];
        let study = create_study_with_arc("dummy", storage.clone(), sampler, directions)?;

        // Set user attributes
        let mut trial = study.ask()?;
        trial.set_user_attr("key", "user".to_string())?;

        // Set system attributes
        let mut guard = storage.write().map_err(|e| {
            Error::with_reason(
                ErrorKind::StorageError,
                format!("Failed to acquire storage guard: {e}"),
            )
        })?;
        let mut attrs = Attrs::new();
        attrs.insert(AttrKey::System("key".into()), "system".to_string());
        guard.set_trial_attrs(trial.id, attrs, false)?;

        // Check the attributes
        assert_eq!(
            trial.get_user_attrs(),
            HashMap::from([(String::from("key"), String::from("user"))])
        );
        let trial = guard.get_trial(trial.id)?;
        assert_eq!(
            trial.attrs.get(&AttrKey::User("key".into())),
            Some(&"user".to_string())
        );
        assert_eq!(
            trial.attrs.get(&AttrKey::System("key".into())),
            Some(&"system".to_string())
        );
        Ok(())
    }

    #[test]
    fn test_set_constraints() -> Result<()> {
        let storage = Arc::new(RwLock::new(InMemoryStorage::new()));
        let sampler = Arc::new(Mutex::new(RandomSampler::new()));
        let directions = vec![Direction::Minimize];
        let study = create_study_with_arc("dummy", storage.clone(), sampler, directions)?;

        let mut trial = study.ask()?;
        let _ = trial.suggest_float("x", -10.0, 10.0)?;
        let constraints = HashMap::from([(String::from("c0"), 10.0)]);
        trial.set_constraints(constraints)?;

        let _ = study.tell(trial.number, TrialStateValues::Complete(vec![0.0]));
        let trials = study.get_trials()?;

        assert_eq!(
            trials[0].constraints(),
            &HashMap::from([(String::from("c0"), 10.0)])
        );
        assert!(trials[0].attrs.keys().all(|key| match key {
            AttrKey::System(key) => !key.as_str().starts_with("constraints"),
            AttrKey::User(_) => true,
        }));
        Ok(())
    }
}
