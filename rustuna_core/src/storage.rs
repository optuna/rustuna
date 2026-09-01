use std::collections::{HashMap, HashSet};

use crate::attr::{category_labels_to_attrs, get_category_labels, AttrKey, Attrs, CategoryLabel};
use crate::datetime::now_naive_utc;
use crate::distribution::Distribution;
use crate::study::{Direction, PersistedStudy};
use crate::study_cache::StudyCache;
use crate::trial::{PersistedTrial, TrialState, TrialStateValues};
use crate::{Error, ErrorKind, Result};

/// Abstraction over study and trial persistence.
///
/// This trait is the Rustuna counterpart of Optuna's `BaseStorage`. It abstracts the backend
/// that stores studies, trials, parameter values, and metadata for hyperparameter optimization.
///
/// Rustuna differs from Optuna in a few storage-facing areas:
/// - user and system attributes are merged into a single [`Attrs`] map keyed by [`AttrKey`],
/// - attribute writes are exposed as bulk operations, and
/// - categorical choice labels are stored separately from [`Distribution::Categorical`] and are
///   retrieved through dedicated category-label APIs.
pub trait Storage: Send + Sync {
    /// Creates a new study.
    ///
    /// The returned study must have a unique ID even if previously created studies were deleted.
    fn create_new_study(
        &mut self,
        study_name: &str,
        directions: Vec<Direction>,
    ) -> Result<&PersistedStudy>;
    /// Deletes an existing study and its trials.
    fn delete_study(&mut self, study_id: u32) -> Result<()>;
    /// Creates a new running trial.
    ///
    /// Newly created trials are expected to start in [`TrialStateValues::Running`] unless they
    /// are created from a template.
    fn create_new_trial(&mut self, study_id: u32) -> Result<&PersistedTrial>;
    /// Creates a new trial by copying a template.
    ///
    /// This is used by features such as queued trials and `Study::add_trial`.
    fn create_new_trial_from_template(
        &mut self,
        study_id: u32,
        template: &PersistedTrial,
    ) -> Result<&PersistedTrial>;
    /// Records a suggested parameter in internal representation.
    ///
    /// Implementations should reject incompatible distributions for the same parameter name
    /// within a study.
    fn set_trial_param(
        &mut self,
        trial_id: u32,
        name: &str,
        distribution: &Distribution,
        value: f64,
    ) -> Result<()>;
    /// Updates the state and objective values of a trial.
    ///
    /// Finished trials must not become mutable again.
    fn set_trial_state_values(
        &mut self,
        trial_id: u32,
        state_values: TrialStateValues,
    ) -> Result<()>;
    /// Stores all intermediate values for a trial.
    ///
    /// Existing values for the same steps should be overwritten.
    fn set_trial_intermediate_values(
        &mut self,
        trial_id: u32,
        intermediate_values: HashMap<u32, f64>,
    ) -> Result<()>;
    // Design Note:
    // get_* methods take &mut self to allow in-place cache refresh in wrapper implementations
    // (e.g., CachedStorage). With &self it is impossible to safely update caches and return
    // references without relying on unsafe patterns.
    /// Returns all studies.
    fn get_studies(&mut self) -> Result<&Vec<PersistedStudy>>;
    /// Returns a study by ID.
    fn get_study(&mut self, study_id: u32) -> Result<&PersistedStudy>;
    /// Returns a study attribute by key.
    fn get_study_attr(&mut self, study_id: u32, key: AttrKey) -> Result<String>;
    /// Returns all trials that belong to a study.
    ///
    /// Trials are expected to be ordered by their trial number.
    fn get_trials(&mut self, study_id: u32) -> Result<&Vec<Option<PersistedTrial>>>;
    /// Returns a trial by ID.
    fn get_trial(&mut self, trial_id: u32) -> Result<&PersistedTrial>;
    // Design Note:
    // get_cached_* methods always return references from the in-memory cache without
    // synchronizing with backends. These methods are separate from get_* to allow
    // callers to use read locks for cache-only reads.
    /// Returns a cached trial reference without synchronizing external backends.
    ///
    /// This method is intended for read-only cache hits under a shared lock.
    fn get_cached_trial(&self, trial_id: u32) -> Result<&PersistedTrial>;
    /// Returns the trial number for a trial ID.
    fn get_trial_number_from_id(&mut self, trial_id: u32) -> Result<u32>;
    // Design Note:
    // Category labels are stored in study system attrs internally, but exposed via dedicated
    // APIs for caching efficiency. Since category labels cannot be overwritten once set for
    // a given (study_id, param_name), implementations can safely cache them without
    // invalidation concerns.
    /// Stores categorical labels for a study parameter.
    ///
    /// Rustuna keeps categorical labels outside [`Distribution::Categorical`] to avoid storing
    /// heap-allocated choice lists repeatedly in each trial.
    fn set_category_labels(
        &mut self,
        study_id: u32,
        param_name: &str,
        labels: Vec<CategoryLabel>,
    ) -> Result<()>;
    /// Returns categorical labels for a study parameter if present.
    fn get_category_labels(
        &mut self,
        study_id: u32,
        param_name: &str,
        cardinality: usize,
    ) -> Result<Option<Vec<CategoryLabel>>>;
    /// Resolves a trial ID from a study ID and trial number.
    fn get_trial_id_from_study_id_trial_number(
        &mut self,
        study_id: u32,
        trial_number: u32,
    ) -> Result<u32>;
    // Design Note:
    // Unlike the storage APIs in Optuna, the `set_study_attrs` and `set_trial_attrs` methods
    // are designed to receive multiple attributes for bulk insert operations.
    // Furthermore, the `user_attrs` and `system_attrs` are merged into a single HashMap,
    // which simplifies the implementation process for third-party storages.
    // Note: Some backend implementations (e.g., SQLite) may partially apply attributes across
    // user/system tables when error_on_overwrite is true.
    /// Sets study attributes in bulk.
    ///
    /// If `error_on_overwrite` is `true`, implementations should reject keys that already exist.
    fn set_study_attrs(
        &mut self,
        study_id: u32,
        attrs: Attrs,
        error_on_overwrite: bool,
    ) -> Result<()>;
    /// Sets trial attributes in bulk.
    ///
    /// If `error_on_overwrite` is `true`, implementations should reject keys that already exist.
    fn set_trial_attrs(
        &mut self,
        trial_id: u32,
        attrs: Attrs,
        error_on_overwrite: bool,
    ) -> Result<()>;
    /// Returns the joint search space inferred from stored trials.
    ///
    /// This corresponds to the search space used for joint sampling.
    fn get_joint_search_space(&mut self, study_id: u32) -> Result<HashMap<String, Distribution>>;
    /// Returns the number of trials in a study, optionally filtered by state.
    ///
    /// Discarded trials are included in the count because their trial state is unchanged.
    fn get_n_trials(&mut self, study_id: u32, states: Option<&[TrialState]>) -> Result<u32>;
    fn discard_trials(&mut self, trial_ids: &[u32]) -> Result<()>;
    fn may_omit_trials(&self) -> bool;
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
/// Options for [`InMemoryStorage`].
pub struct InMemoryStorageOptions {
    pub apply_discard: bool,
}

/// In-memory storage implementation used by default in Rust code and tests.
///
/// This implementation keeps all studies, trials, and caches in process memory.
#[derive(Default)]
pub struct InMemoryStorage {
    studies: Vec<PersistedStudy>,
    trials: HashMap<u32, Vec<Option<PersistedTrial>>>,
    trial_id_number_map: TrialIdNumberHashMap,
    study_caches: HashMap<u32, StudyCache>,
    next_study_id: u32,
    next_trial_id: u32,
    // Supports the state-counting API required when `discard_trials` removes trials.
    // Storing only discarded trials' states avoids tracking state transitions and keeps this simple.
    discarded_state_counts: HashMap<(u32, TrialState), u32>,
    option: InMemoryStorageOptions,
}
impl InMemoryStorage {
    /// Creates an empty in-memory storage.
    pub fn new() -> InMemoryStorage {
        InMemoryStorage {
            studies: vec![],
            trials: HashMap::new(),
            trial_id_number_map: TrialIdNumberHashMap::new(),
            study_caches: HashMap::new(),
            next_study_id: 0,
            next_trial_id: 0,
            discarded_state_counts: HashMap::new(),
            option: InMemoryStorageOptions::default(),
        }
    }

    /// Creates an empty in-memory storage.
    pub fn new_with_option(option: InMemoryStorageOptions) -> InMemoryStorage {
        InMemoryStorage {
            studies: vec![],
            trials: HashMap::new(),
            trial_id_number_map: TrialIdNumberHashMap::new(),
            study_caches: HashMap::new(),
            next_study_id: 0,
            next_trial_id: 0,
            discarded_state_counts: HashMap::new(),
            option,
        }
    }

    pub fn insert_study_with_id(
        &mut self,
        study_id: u32,
        study_name: &str,
        directions: Vec<Direction>,
    ) -> Result<&PersistedStudy> {
        if let Some(pos) = self.studies.iter().position(|s| s.id == study_id) {
            if self.studies[pos].name != study_name {
                return Err(Error::new(ErrorKind::StorageError));
            }
            return Ok(&self.studies[pos]);
        }
        if self.studies.iter().any(|s| s.name == study_name) {
            return Err(Error::new(ErrorKind::DuplicatedStudy));
        }
        self.studies.push(PersistedStudy::new(
            study_id,
            study_name.to_string(),
            directions,
        ));
        self.trials.insert(study_id, vec![]);
        if study_id >= self.next_study_id {
            self.next_study_id = study_id + 1;
        }
        self.studies
            .last()
            .ok_or_else(|| Error::new(ErrorKind::StorageError))
    }

    pub fn insert_trial_with_id(
        &mut self,
        study_id: u32,
        trial_id: u32,
        number: u32,
    ) -> Result<&PersistedTrial> {
        let trials = get_mut_trials_by_study_id(&mut self.trials, study_id)?;
        let trials_len = trials.len() as u32;
        if number > trials_len {
            return Err(Error::new(ErrorKind::StorageError));
        }
        if number < trials_len {
            let trial = discarded_if_none(trials.get(number as usize).and_then(Option::as_ref))?;
            if trial.id != trial_id {
                return Err(Error::new(ErrorKind::StorageError));
            }
            return Ok(trial);
        }
        if self.trial_id_number_map.contains_trial_id(trial_id) {
            return Err(Error::new(ErrorKind::StorageError));
        }
        let trial = PersistedTrial::new(trial_id, study_id, number);
        trials.push(Some(trial));
        self.trial_id_number_map.insert(study_id, trial_id, number);
        if trial_id >= self.next_trial_id {
            self.next_trial_id = trial_id + 1;
        }
        discarded_if_none(trials[number as usize].as_ref())
    }

    /// Updates trial timestamps when synchronizing an external storage into the cache.
    pub fn set_trial_datetimes(
        &mut self,
        trial_id: u32,
        datetime_start: Option<String>,
        datetime_complete: Option<String>,
    ) -> Result<()> {
        let (study_id, trial_number) =
            get_study_id_trial_number_by_trial_id(&self.trial_id_number_map, trial_id)?;
        let trial = discarded_if_none(
            get_mut_trials_by_study_id(&mut self.trials, study_id)?
                .get_mut(trial_number as usize)
                .and_then(Option::as_mut),
        )?;
        trial.datetime_start = datetime_start;
        trial.datetime_complete = datetime_complete;
        Ok(())
    }
}
fn get_trials_by_study_id(
    all_trials: &HashMap<u32, Vec<Option<PersistedTrial>>>,
    study_id: u32,
) -> Result<&Vec<Option<PersistedTrial>>> {
    let trials = all_trials
        .get(&study_id)
        .ok_or(Error::new(ErrorKind::StudyNotFound))?;
    Ok(trials)
}
fn get_mut_trials_by_study_id(
    all_trials: &mut HashMap<u32, Vec<Option<PersistedTrial>>>,
    study_id: u32,
) -> Result<&mut Vec<Option<PersistedTrial>>> {
    let trials = all_trials
        .get_mut(&study_id)
        .ok_or(Error::new(ErrorKind::StudyNotFound))?;
    Ok(trials)
}
fn get_study_id_trial_number_by_trial_id(
    trial_id_number_map: &TrialIdNumberHashMap,
    trial_id: u32,
) -> Result<(u32, u32)> {
    trial_id_number_map
        .get_study_id_trial_number(trial_id)
        .ok_or(Error::new(ErrorKind::TrialNotFound))
}

fn discarded_if_none<T>(value: Option<T>) -> Result<T> {
    value.ok_or(Error::new(ErrorKind::TrialDiscarded))
}
impl Storage for InMemoryStorage {
    fn create_new_study(
        &mut self,
        study_name: &str,
        directions: Vec<Direction>,
    ) -> Result<&PersistedStudy> {
        if self.studies.iter().any(|s| s.name == study_name) {
            return Err(Error::new(ErrorKind::DuplicatedStudy));
        }

        let study_id = self.next_study_id;
        self.next_study_id += 1;
        self.studies.push(PersistedStudy::new(
            study_id,
            study_name.to_string(),
            directions,
        ));
        self.trials.insert(study_id, vec![]);
        self.studies
            .last()
            .ok_or_else(|| Error::new(ErrorKind::StorageError))
    }

    fn delete_study(&mut self, study_id: u32) -> Result<()> {
        if !self.studies.iter().any(|s| s.id == study_id) {
            return Err(Error::new(ErrorKind::StudyNotFound));
        }

        self.studies.retain(|s| s.id != study_id);
        self.trials.remove(&study_id);
        self.trial_id_number_map.remove_study(study_id);
        self.discarded_state_counts
            .retain(|(id, _), _| *id != study_id);
        self.study_caches.remove(&study_id);

        Ok(())
    }

    fn create_new_trial(&mut self, study_id: u32) -> Result<&PersistedTrial> {
        let trials = get_mut_trials_by_study_id(&mut self.trials, study_id)?;
        let trial_id = self.next_trial_id;
        self.next_trial_id += 1;
        let number = trials.len() as u32;
        let mut trial = PersistedTrial::new(trial_id, study_id, number);
        trial.datetime_start = Some(now_naive_utc());
        trials.push(Some(trial));
        self.trial_id_number_map.insert(study_id, trial_id, number);
        discarded_if_none(trials[number as usize].as_ref())
    }

    fn create_new_trial_from_template(
        &mut self,
        study_id: u32,
        template: &PersistedTrial,
    ) -> Result<&PersistedTrial> {
        let trials = get_mut_trials_by_study_id(&mut self.trials, study_id)?;
        let trial_id = self.next_trial_id;
        self.next_trial_id += 1;
        let number = trials.len() as u32;

        let mut trial = template.clone();
        trial.id = trial_id;
        trial.study_id = study_id;
        trial.number = number;
        trials.push(Some(trial));
        self.trial_id_number_map.insert(study_id, trial_id, number);
        discarded_if_none(trials[number as usize].as_ref())
    }

    fn set_trial_param(
        &mut self,
        trial_id: u32,
        name: &str,
        distribution: &Distribution,
        value: f64,
    ) -> Result<()> {
        let (study_id, trial_number) =
            get_study_id_trial_number_by_trial_id(&self.trial_id_number_map, trial_id)?;
        let trial = discarded_if_none(
            get_mut_trials_by_study_id(&mut self.trials, study_id)?
                .get_mut(trial_number as usize)
                .and_then(Option::as_mut),
        )?;
        check_trial_is_updatable(trial)?;

        // Check param distribution compatibility with previous trial(s).
        let study_distributions = &mut self
            .study_caches
            .entry(study_id)
            .or_default()
            .param_distribution;
        if let Some(study_distribution) = study_distributions.get(name) {
            study_distribution.check_compatibility(distribution)?;
        }
        study_distributions.insert(name.to_string(), distribution.clone());

        trial
            .distributions
            .insert(name.to_string(), distribution.clone());
        trial.internal_params.insert(name.to_string(), value);
        Ok(())
    }

    fn set_trial_state_values(
        &mut self,
        trial_id: u32,
        state_values: TrialStateValues,
    ) -> Result<()> {
        let (study_id, trial_number) =
            get_study_id_trial_number_by_trial_id(&self.trial_id_number_map, trial_id)?;
        let trial = discarded_if_none(
            get_mut_trials_by_study_id(&mut self.trials, study_id)?
                .get_mut(trial_number as usize)
                .and_then(Option::as_mut),
        )?;
        check_trial_is_updatable(trial)?;
        trial.datetime_complete = if matches!(
            state_values,
            TrialStateValues::Complete(_) | TrialStateValues::Pruned | TrialStateValues::Fail
        ) {
            Some(now_naive_utc())
        } else {
            None
        };
        trial.state_values = state_values;
        Ok(())
    }

    fn set_trial_intermediate_values(
        &mut self,
        trial_id: u32,
        intermediate_values: HashMap<u32, f64>,
    ) -> Result<()> {
        let (study_id, trial_number) =
            get_study_id_trial_number_by_trial_id(&self.trial_id_number_map, trial_id)?;
        let trial = discarded_if_none(
            get_mut_trials_by_study_id(&mut self.trials, study_id)?
                .get_mut(trial_number as usize)
                .and_then(Option::as_mut),
        )?;
        check_trial_is_updatable(trial)?;
        trial.intermediate_values.extend(intermediate_values);
        Ok(())
    }

    fn get_studies(&mut self) -> Result<&Vec<PersistedStudy>> {
        Ok(&self.studies)
    }

    fn get_study(&mut self, study_id: u32) -> Result<&PersistedStudy> {
        let study = self
            .studies
            .iter()
            .find(|s| s.id == study_id)
            .ok_or(Error::new(ErrorKind::StudyNotFound))?;
        Ok(study)
    }

    fn get_study_attr(&mut self, study_id: u32, key: AttrKey) -> Result<String> {
        let study = self.get_study(study_id)?;
        study
            .attrs
            .get(&key)
            .cloned()
            .ok_or(Error::new(ErrorKind::AttrNotFound))
    }

    fn get_trials(&mut self, study_id: u32) -> Result<&Vec<Option<PersistedTrial>>> {
        get_trials_by_study_id(&self.trials, study_id)
    }

    fn get_trial(&mut self, trial_id: u32) -> Result<&PersistedTrial> {
        self.get_cached_trial(trial_id)
    }

    fn get_cached_trial(&self, trial_id: u32) -> Result<&PersistedTrial> {
        let (study_id, trial_number) =
            get_study_id_trial_number_by_trial_id(&self.trial_id_number_map, trial_id)?;
        let trial = discarded_if_none(
            get_trials_by_study_id(&self.trials, study_id)?
                .get(trial_number as usize)
                .and_then(Option::as_ref),
        )?;
        Ok(trial)
    }

    fn get_trial_number_from_id(&mut self, trial_id: u32) -> Result<u32> {
        let (_, trial_number) =
            get_study_id_trial_number_by_trial_id(&self.trial_id_number_map, trial_id)?;
        Ok(trial_number)
    }

    fn set_category_labels(
        &mut self,
        study_id: u32,
        param_name: &str,
        labels: Vec<CategoryLabel>,
    ) -> Result<()> {
        let attrs = category_labels_to_attrs(param_name, &labels);
        self.set_study_attrs(study_id, attrs, true)
    }

    fn get_category_labels(
        &mut self,
        study_id: u32,
        param_name: &str,
        cardinality: usize,
    ) -> Result<Option<Vec<CategoryLabel>>> {
        let study = self.get_study(study_id)?;
        Ok(get_category_labels(&study.attrs, param_name, cardinality))
    }

    fn get_trial_id_from_study_id_trial_number(
        &mut self,
        study_id: u32,
        trial_number: u32,
    ) -> Result<u32> {
        self.trial_id_number_map
            .get_trial_id(study_id, trial_number)
            .ok_or(Error::new(ErrorKind::TrialNotFound))
    }

    fn set_study_attrs(
        &mut self,
        study_id: u32,
        attrs: Attrs,
        error_on_overwrite: bool,
    ) -> Result<()> {
        let study = self
            .studies
            .iter_mut()
            .find(|s| s.id == study_id)
            .ok_or(Error::new(ErrorKind::StudyNotFound))?;
        for (key, value) in attrs {
            if error_on_overwrite && study.attrs.contains_key(&key) {
                return Err(Error::new(ErrorKind::AttrOverwriteNotAllowed));
            }
            study.attrs.insert(key, value);
        }
        Ok(())
    }

    fn set_trial_attrs(
        &mut self,
        trial_id: u32,
        attrs: Attrs,
        error_on_overwrite: bool,
    ) -> Result<()> {
        let (study_id, trial_number) =
            get_study_id_trial_number_by_trial_id(&self.trial_id_number_map, trial_id)?;
        let trial = discarded_if_none(
            get_mut_trials_by_study_id(&mut self.trials, study_id)?
                .get_mut(trial_number as usize)
                .and_then(Option::as_mut),
        )?;
        check_trial_is_updatable(trial)?;
        for (key, value) in attrs {
            if error_on_overwrite && trial.attrs.contains_key(&key) {
                return Err(Error::new(ErrorKind::AttrOverwriteNotAllowed));
            }
            trial.attrs.insert(key, value);
        }
        Ok(())
    }

    fn get_joint_search_space(&mut self, study_id: u32) -> Result<HashMap<String, Distribution>> {
        let study_cache = self.study_caches.entry(study_id).or_default();
        let trials = get_trials_by_study_id(&self.trials, study_id)?;
        study_cache.update(trials);
        Ok(study_cache.get_joint_search_space())
    }

    fn get_n_trials(&mut self, study_id: u32, states: Option<&[TrialState]>) -> Result<u32> {
        if !self.studies.iter().any(|study| study.id == study_id) {
            return Err(Error::new(ErrorKind::StudyNotFound));
        }

        let states: Option<HashSet<TrialState>> =
            states.map(|states| states.iter().copied().collect());
        let live_count = self
            .trials
            .get(&study_id)
            .ok_or(Error::new(ErrorKind::StudyNotFound))?
            .iter()
            .flatten()
            .filter(|trial| {
                states
                    .as_ref()
                    .is_none_or(|states| states.contains(&trial.state_values.state()))
            })
            .count() as u32;
        let discarded_count = self
            .discarded_state_counts
            .iter()
            .filter(|((id, state), _)| {
                *id == study_id && states.as_ref().is_none_or(|states| states.contains(state))
            })
            .map(|(_, count)| count)
            .sum::<u32>();
        Ok(live_count + discarded_count)
    }

    fn discard_trials(&mut self, trial_ids: &[u32]) -> Result<()> {
        if !self.option.apply_discard {
            return Ok(());
        }
        for trial_id in trial_ids {
            let (study_id, trial_number) =
                get_study_id_trial_number_by_trial_id(&self.trial_id_number_map, *trial_id)?;

            let state = {
                let trials = get_mut_trials_by_study_id(&mut self.trials, study_id)?;
                let slot = trials
                    .get_mut(trial_number as usize)
                    .ok_or(Error::new(ErrorKind::TrialNotFound))?;
                slot.take().map(|trial| trial.state_values.state())
            };
            if let Some(state) = state {
                *self
                    .discarded_state_counts
                    .entry((study_id, state))
                    .or_default() += 1;
            }
        }
        Ok(())
    }

    fn may_omit_trials(&self) -> bool {
        self.option.apply_discard
    }
}

#[derive(Default)]
struct TrialIdNumberHashMap {
    trial_id_to_study_number: HashMap<u32, (u32, u32)>,
    study_number_to_trial_id: HashMap<(u32, u32), u32>,
}
impl TrialIdNumberHashMap {
    fn new() -> Self {
        Self {
            trial_id_to_study_number: HashMap::new(),
            study_number_to_trial_id: HashMap::new(),
        }
    }
    fn insert(&mut self, study_id: u32, trial_id: u32, trial_number: u32) {
        self.trial_id_to_study_number
            .insert(trial_id, (study_id, trial_number));
        self.study_number_to_trial_id
            .insert((study_id, trial_number), trial_id);
    }
    fn contains_trial_id(&self, trial_id: u32) -> bool {
        self.trial_id_to_study_number.contains_key(&trial_id)
    }
    fn get_study_id_trial_number(&self, trial_id: u32) -> Option<(u32, u32)> {
        self.trial_id_to_study_number.get(&trial_id).copied()
    }
    fn get_trial_id(&self, study_id: u32, trial_number: u32) -> Option<u32> {
        self.study_number_to_trial_id
            .get(&(study_id, trial_number))
            .copied()
    }
    fn remove_study(&mut self, study_id: u32) {
        self.trial_id_to_study_number
            .retain(|_, value| value.0 != study_id);
        self.study_number_to_trial_id
            .retain(|key, _| key.0 != study_id);
    }
}

fn check_trial_is_updatable(trial: &PersistedTrial) -> Result<()> {
    if trial.is_finished() {
        return Err(Error::new(ErrorKind::TrialAlreadyFinished));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distribution::Distribution;

    #[test]
    fn test_create_study_does_not_reuse_study_id() -> Result<()> {
        let mut storage = InMemoryStorage::new();

        let study1 = storage.create_new_study("study1", vec![Direction::Minimize])?;
        let study1_id = study1.id;
        storage.create_new_study("study2", vec![Direction::Minimize])?;
        storage.delete_study(study1_id)?;

        let err = storage
            .get_study(study1_id)
            .err()
            .expect("Expected StudyNotFound error");
        assert!(matches!(err.kind, ErrorKind::StudyNotFound));

        let study3 = storage.create_new_study("study3", vec![Direction::Minimize])?;
        assert_eq!(study3.id, 2);
        assert_ne!(study3.id, study1_id);

        assert_eq!(storage.get_studies()?.len(), 2);
        Ok(())
    }

    #[test]
    fn set_trial_param_rejects_incompatible_distribution_across_trials() -> Result<()> {
        let mut storage = InMemoryStorage::new();
        let study_id = storage
            .create_new_study("study", vec![Direction::Minimize])?
            .id;

        let float_dist = Distribution::new_float(0.0, 1.0, None, false);
        let int_dist = Distribution::new_int(0, 5, 1, false);

        let trial0_id = storage.create_new_trial(study_id)?.id;
        storage.set_trial_param(trial0_id, "x", &float_dist, 0.5)?;

        let trial1_id = storage.create_new_trial(study_id)?.id;
        let err = storage
            .set_trial_param(trial1_id, "x", &int_dist, 1.0)
            .err()
            .unwrap();
        assert!(matches!(err.kind, ErrorKind::IncompatibleDistribution));
        Ok(())
    }

    #[test]
    fn insert_trial_with_id_updates_next_trial_id() -> Result<()> {
        let mut storage = InMemoryStorage::new();
        let study_id = storage
            .create_new_study("study", vec![Direction::Minimize])?
            .id;

        let trial = storage.insert_trial_with_id(study_id, 10, 0)?;
        assert_eq!(trial.id, 10);
        assert_eq!(
            storage.get_trial_id_from_study_id_trial_number(study_id, 0)?,
            10
        );
        let trial = storage.create_new_trial(study_id)?;
        assert_eq!(trial.id, 11);
        Ok(())
    }

    #[test]
    fn get_trial_id_from_study_id_trial_number_uses_both_keys() -> Result<()> {
        let mut storage = InMemoryStorage::new();
        let study0_id = storage
            .create_new_study("study0", vec![Direction::Minimize])?
            .id;
        let study1_id = storage
            .create_new_study("study1", vec![Direction::Minimize])?
            .id;
        let study0_trial0_id = storage.create_new_trial(study0_id)?.id;
        let study0_trial1_id = storage.create_new_trial(study0_id)?.id;
        let study1_trial0_id = storage.create_new_trial(study1_id)?.id;

        assert_eq!(
            storage.get_trial_id_from_study_id_trial_number(study0_id, 0)?,
            study0_trial0_id
        );
        assert_eq!(
            storage.get_trial_id_from_study_id_trial_number(study0_id, 1)?,
            study0_trial1_id
        );
        assert_eq!(
            storage.get_trial_id_from_study_id_trial_number(study1_id, 0)?,
            study1_trial0_id
        );

        storage.delete_study(study0_id)?;
        let error = storage
            .get_trial_id_from_study_id_trial_number(study0_id, 0)
            .unwrap_err();
        assert!(matches!(error.kind, ErrorKind::TrialNotFound));
        assert_eq!(
            storage.get_trial_id_from_study_id_trial_number(study1_id, 0)?,
            study1_trial0_id
        );
        Ok(())
    }

    #[test]
    fn delete_study_removes_discarded_trial_mappings() -> Result<()> {
        let mut storage = InMemoryStorage::new_with_option(InMemoryStorageOptions {
            apply_discard: true,
        });
        let study_id = storage
            .create_new_study("study", vec![Direction::Minimize])?
            .id;
        let trial_id = storage.create_new_trial(study_id)?.id;
        storage.discard_trials(&[trial_id])?;

        storage.delete_study(study_id)?;

        let error = storage
            .get_trial_id_from_study_id_trial_number(study_id, 0)
            .unwrap_err();
        assert!(matches!(error.kind, ErrorKind::TrialNotFound));
        assert!(
            matches!(storage.get_trial(trial_id), Err(error) if matches!(error.kind, ErrorKind::TrialNotFound))
        );
        Ok(())
    }

    #[test]
    fn trials_are_stamped_with_naive_utc_datetimes() -> Result<()> {
        let before = now_naive_utc();
        let mut storage = InMemoryStorage::new();
        let study_id = storage.create_new_study("s", vec![Direction::Minimize])?.id;
        let trial_id = storage.create_new_trial(study_id)?.id;

        let trial = storage.get_trial(trial_id)?;
        let started = trial
            .datetime_start
            .clone()
            .expect("a new trial records when it started");
        assert!(trial.datetime_complete.is_none());

        storage.set_trial_state_values(trial_id, TrialStateValues::Complete(vec![1.0]))?;
        let trial = storage.get_trial(trial_id)?;
        let completed = trial
            .datetime_complete
            .clone()
            .expect("a finished trial records when it finished");
        assert_eq!(trial.datetime_start.as_deref(), Some(started.as_str()));

        // Naive UTC of a fixed width is lexicographically ordered, so the strings can be bracketed
        // by readings taken around the calls without any date arithmetic.
        let after = now_naive_utc();
        assert_eq!(started.len(), after.len());
        assert!(
            before <= started && started <= completed && completed <= after,
            "expected {before} <= {started} <= {completed} <= {after}"
        );
        Ok(())
    }

    #[test]
    fn create_new_trial_from_template_preserves_datetime() -> Result<()> {
        let mut storage = InMemoryStorage::new();
        let study_id = storage
            .create_new_study("study", vec![Direction::Minimize])?
            .id;
        let mut template = PersistedTrial::new(100, 200, 300);
        template.datetime_start = Some("2026-04-02 03:04:05.678".to_string());
        template.datetime_complete = Some("2026-04-02 03:14:15.678".to_string());
        template.state_values = TrialStateValues::Complete(vec![1.0]);

        let trial = storage.create_new_trial_from_template(study_id, &template)?;
        assert_eq!(trial.id, 0);
        assert_eq!(trial.study_id, study_id);
        assert_eq!(trial.number, 0);
        assert_eq!(trial.datetime_start, template.datetime_start);
        assert_eq!(trial.datetime_complete, template.datetime_complete);
        assert_eq!(trial.state_values, template.state_values);
        Ok(())
    }

    #[test]
    fn get_study_attr_returns_stored_value() -> Result<()> {
        let mut storage = InMemoryStorage::new();
        let study_id = storage
            .create_new_study("study", vec![Direction::Minimize])?
            .id;
        let key = AttrKey::User("owner".into());
        let mut attrs = Attrs::new();
        attrs.insert(key.clone(), "alice".to_string());
        storage.set_study_attrs(study_id, attrs, false)?;

        let value = storage.get_study_attr(study_id, key)?;
        assert_eq!(value, "alice");
        Ok(())
    }

    #[test]
    fn get_study_attr_returns_error_for_missing_key() -> Result<()> {
        let mut storage = InMemoryStorage::new();
        let study_id = storage
            .create_new_study("study", vec![Direction::Minimize])?
            .id;

        let err = storage
            .get_study_attr(study_id, AttrKey::User("missing".into()))
            .unwrap_err();
        assert!(matches!(err.kind, ErrorKind::AttrNotFound));
        Ok(())
    }

    #[test]
    fn discard_trials_omits_trials() -> Result<()> {
        let mut storage = InMemoryStorage::new_with_option(InMemoryStorageOptions {
            apply_discard: true,
        });
        let study_id = storage
            .create_new_study("study", vec![Direction::Minimize])?
            .id;
        let trial0_id = storage.create_new_trial(study_id)?.id;
        let trial1_id = storage.create_new_trial(study_id)?.id;

        storage.discard_trials(&[trial0_id])?;
        storage.discard_trials(&[trial0_id])?;

        let trials = storage.get_trials(study_id)?;
        assert!(trials[0].is_none());
        assert_eq!(trials[1].as_ref().unwrap().id, trial1_id);
        assert!(storage.may_omit_trials());
        let err = storage.get_trial(trial0_id).unwrap_err();
        assert!(matches!(err.kind, ErrorKind::TrialDiscarded));
        Ok(())
    }

    #[test]
    fn get_n_trials_counts_states() -> Result<()> {
        let mut storage = InMemoryStorage::new_with_option(InMemoryStorageOptions {
            apply_discard: true,
        });
        let study_id = storage
            .create_new_study("study", vec![Direction::Minimize])?
            .id;
        let running_trial_id = storage.create_new_trial(study_id)?.id;
        let complete_trial_id = storage.create_new_trial(study_id)?.id;
        storage.set_trial_state_values(complete_trial_id, TrialStateValues::Complete(vec![1.0]))?;
        let template = storage.get_trial(complete_trial_id)?.clone();
        storage.create_new_trial_from_template(study_id, &template)?;

        assert_eq!(storage.get_n_trials(study_id, None)?, 3);
        assert_eq!(
            storage.get_n_trials(study_id, Some(&[TrialState::Running]))?,
            1
        );
        assert_eq!(
            storage.get_n_trials(study_id, Some(&[TrialState::Complete]))?,
            2
        );
        assert_eq!(
            storage.get_n_trials(
                study_id,
                Some(&[TrialState::Complete, TrialState::Complete]),
            )?,
            2
        );
        assert_eq!(storage.get_n_trials(study_id, Some(&[]))?, 0);

        storage.discard_trials(&[complete_trial_id])?;
        assert_eq!(storage.get_n_trials(study_id, None)?, 3);
        assert_eq!(
            storage.get_n_trials(study_id, Some(&[TrialState::Complete]))?,
            2
        );
        assert!(storage.get_trial(running_trial_id).is_ok());
        Ok(())
    }
}
