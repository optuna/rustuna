use std::collections::HashMap;

use rustuna_core::attr::{
    category_labels_to_attrs, get_category_labels, AttrKey, Attrs, CategoryLabel,
};
use rustuna_core::distribution::Distribution;
use rustuna_core::study::{Direction, PersistedStudy};
use rustuna_core::study_cache::StudyCache;
use rustuna_core::trial::{PersistedTrial, TrialStateValues};
use rustuna_core::{Error, ErrorKind, Result};

/// Backend interface for [`CachedStorage`].
///
/// Unlike `rustuna_core::storage::Storage`, this trait returns owned values instead of references.
/// This allows the caching wrapper to materialize in-memory state and then hand out borrowed
/// references required by the core storage interface.
pub trait CachedStorageBackend: Send + Sync {
    // Design Note:
    // This trait is intended for backends that return owned values (not references) so that
    // a wrapper (e.g., CachedStorage) can materialize in-memory caches and then hand out
    // references required by the Storage trait. This mirrors Optuna's _CachedStorage pattern:
    // backend focuses on persistence, wrapper handles caching and reference semantics.
    fn create_new_study(
        &mut self,
        study_name: &str,
        directions: Vec<Direction>,
    ) -> Result<PersistedStudy>;
    fn delete_study(&mut self, study_id: u32) -> Result<()>;
    fn create_new_trial(&mut self, study_id: u32) -> Result<PersistedTrial>;
    fn create_new_trial_from_template(
        &mut self,
        study_id: u32,
        template: &PersistedTrial,
    ) -> Result<PersistedTrial>;
    fn set_trial_param(
        &mut self,
        trial_id: u32,
        name: &str,
        distribution: &Distribution,
        value: f64,
    ) -> Result<()>;
    fn set_trial_state_values(
        &mut self,
        trial_id: u32,
        state_values: TrialStateValues,
    ) -> Result<()>;
    fn set_trial_intermediate_values(
        &mut self,
        trial_id: u32,
        intermediate_values: HashMap<u32, f64>,
    ) -> Result<()>;
    fn get_studies(&mut self) -> Result<Vec<PersistedStudy>>;
    fn get_study(&mut self, study_id: u32) -> Result<PersistedStudy>;
    fn get_trial(&mut self, trial_id: u32) -> Result<PersistedTrial>;
    fn get_study_attr(&mut self, study_id: u32, key: AttrKey) -> Result<String>;
    fn set_study_attrs(
        &mut self,
        study_id: u32,
        attrs: Attrs,
        error_on_overwrite: bool,
    ) -> Result<()>;
    fn set_trial_attrs(
        &mut self,
        trial_id: u32,
        attrs: Attrs,
        error_on_overwrite: bool,
    ) -> Result<()>;

    fn discard_trials(&mut self, trial_ids: &[u32]) -> Result<()>;
    fn may_omit_discard_trials(&self) -> bool;

    // Return trials that need refreshing: unfinished trials in `included_numbers`
    // and trials with trial_number greater than `trial_number_greater_than`.
    // If `filter_discarded` is true, discarded trials are excluded when supported.
    fn get_trials_diff(
        &mut self,
        study_id: u32,
        included_numbers: &[u32],
        trial_number_greater_than: i32,
        filter_discarded: bool,
    ) -> Result<Vec<PersistedTrial>>;
}

/// Caching storage wrapper over a backend that returns owned values.
///
/// This type plays a role similar to Optuna's `_CachedStorage`: the backend is responsible for
/// persistence, while `CachedStorage` keeps studies and trials in memory and serves borrowed
/// references to callers.
pub struct CachedStorage {
    studies: Vec<PersistedStudy>,
    trials: HashMap<u32, Vec<Option<PersistedTrial>>>,
    trial_id_to_study_number: HashMap<u32, (u32, u32)>,
    study_caches: HashMap<u32, StudyCache>,
    unfinished_trials: HashMap<u32, Vec<u32>>,
    last_finished_trial_number: HashMap<u32, i32>,
    // Cache for category labels: (study_id, param_name) -> labels
    // Since category labels cannot be overwritten once set, this cache never needs invalidation.
    category_labels_cache: HashMap<(u32, String), Vec<CategoryLabel>>,

    apply_discard: bool,
    backend: Box<dyn CachedStorageBackend>,
}

impl CachedStorage {
    /// Creates a caching wrapper around the given backend.
    pub fn new(backend: Box<dyn CachedStorageBackend>, apply_discard: bool) -> CachedStorage {
        CachedStorage {
            studies: Vec::new(),
            trials: HashMap::new(),
            trial_id_to_study_number: HashMap::new(),
            study_caches: HashMap::new(),
            unfinished_trials: HashMap::new(),
            last_finished_trial_number: HashMap::new(),
            category_labels_cache: HashMap::new(),
            apply_discard,
            backend,
        }
    }

    fn refresh_trials(&mut self, study_id: u32) -> Result<()> {
        let unfinished = self
            .unfinished_trials
            .get(&study_id)
            .cloned()
            .unwrap_or_default();
        let last_finished = self
            .last_finished_trial_number
            .get(&study_id)
            .copied()
            .unwrap_or(-1);
        let loaded = self.backend.get_trials_diff(
            study_id,
            &unfinished,
            last_finished,
            self.apply_discard,
        )?;

        if loaded.is_empty() {
            return Ok(());
        }

        let trials = self.trials.entry(study_id).or_default();
        for trial in loaded {
            self.trial_id_to_study_number
                .insert(trial.id, (study_id, trial.number));
            let trial_number = trial.number as usize;
            if trials.len() <= trial_number {
                trials.resize(trial_number + 1, None);
            }
            trials[trial_number] = Some(trial);
        }

        let study_cache = self.study_caches.entry(study_id).or_default();
        study_cache.update(trials);

        let mut unfinished_next = vec![];
        let mut last_finished_next = last_finished;
        for trial in trials.iter().flatten() {
            if trial.is_finished() {
                last_finished_next = last_finished_next.max(trial.number as i32);
            } else {
                unfinished_next.push(trial.number);
            }
        }
        self.unfinished_trials.insert(study_id, unfinished_next);
        self.last_finished_trial_number
            .insert(study_id, last_finished_next);
        Ok(())
    }

    fn resolve_trial_location(&mut self, trial_id: u32) -> Result<(u32, u32)> {
        if let Some((study_id, trial_number)) = self.trial_id_to_study_number.get(&trial_id) {
            return Ok((*study_id, *trial_number));
        }

        let trial = self.backend.get_trial(trial_id)?;
        let study_id = trial.study_id;
        let trial_number = trial.number;
        let trials = self.trials.entry(study_id).or_default();
        let trial_index = trial_number as usize;
        if trials.len() <= trial_index {
            trials.resize(trial_index + 1, None);
        }
        trials[trial_index] = Some(trial);
        self.trial_id_to_study_number
            .insert(trial_id, (study_id, trial_number));
        Ok((study_id, trial_number))
    }

    fn is_trial_finished_in_cache(&self, study_id: u32, trial_number: u32) -> bool {
        self.trials
            .get(&study_id)
            .and_then(|trials| trials.get(trial_number as usize))
            .and_then(|trial| trial.as_ref())
            .is_some_and(|trial| trial.is_finished())
    }
}

impl rustuna_core::storage::Storage for CachedStorage {
    fn create_new_study(
        &mut self,
        study_name: &str,
        directions: Vec<Direction>,
    ) -> Result<&PersistedStudy> {
        let study = self.backend.create_new_study(study_name, directions)?;
        let study_id = study.id;
        self.studies.push(study);
        self.trials.insert(study_id, Vec::new());
        self.study_caches.insert(study_id, StudyCache::new());
        self.unfinished_trials.insert(study_id, vec![]);
        self.last_finished_trial_number.insert(study_id, -1);
        self.studies
            .last()
            .ok_or_else(|| Error::new(ErrorKind::StorageError))
    }

    fn delete_study(&mut self, study_id: u32) -> Result<()> {
        self.backend.delete_study(study_id)?;

        self.studies.retain(|s| s.id != study_id);
        if let Some(trials) = self.trials.remove(&study_id) {
            for trial in trials.into_iter().flatten() {
                self.trial_id_to_study_number.remove(&trial.id);
            }
        }
        self.study_caches.remove(&study_id);
        self.unfinished_trials.remove(&study_id);
        self.last_finished_trial_number.remove(&study_id);

        Ok(())
    }

    fn create_new_trial(&mut self, study_id: u32) -> Result<&PersistedTrial> {
        let trial = self.backend.create_new_trial(study_id)?;
        let number = trial.number;
        self.trial_id_to_study_number
            .insert(trial.id, (study_id, number));
        // Trials persisted by previous runs may not be cached yet. Load them before
        // registering the new trial so that StudyCache does not advance its cursor
        // past the unloaded (None) entries and skip them forever.
        if self.trials.get(&study_id).map_or(0, |trials| trials.len()) < number as usize {
            self.refresh_trials(study_id)?;
        }
        let trials = self.trials.entry(study_id).or_default();
        let trial_index = number as usize;
        if trials.len() <= trial_index {
            trials.resize(trial_index + 1, None);
        }
        trials[trial_index] = Some(trial);
        let trial_ref = trials
            .get(trial_index)
            .and_then(|trial| trial.as_ref())
            .ok_or_else(|| Error::new(ErrorKind::TrialNotFound))?;

        let study_cache = self.study_caches.entry(study_id).or_default();
        study_cache.update(trials);
        self.unfinished_trials
            .entry(study_id)
            .or_default()
            .push(trial_ref.number);
        Ok(trial_ref)
    }

    fn create_new_trial_from_template(
        &mut self,
        study_id: u32,
        template: &PersistedTrial,
    ) -> Result<&PersistedTrial> {
        let trial = self
            .backend
            .create_new_trial_from_template(study_id, template)?;
        let number = trial.number;
        self.trial_id_to_study_number
            .insert(trial.id, (study_id, number));
        // Trials persisted by previous runs may not be cached yet. Load them before
        // registering the new trial so that StudyCache does not advance its cursor
        // past the unloaded (None) entries and skip them forever.
        if self.trials.get(&study_id).map_or(0, |trials| trials.len()) < number as usize {
            self.refresh_trials(study_id)?;
        }
        let trials = self.trials.entry(study_id).or_default();
        let trial_index = number as usize;
        if trials.len() <= trial_index {
            trials.resize(trial_index + 1, None);
        }
        trials[trial_index] = Some(trial);
        let trial_ref = trials
            .get(trial_index)
            .and_then(|trial| trial.as_ref())
            .ok_or_else(|| Error::new(ErrorKind::TrialNotFound))?;

        let study_cache = self.study_caches.entry(study_id).or_default();
        study_cache.update(trials);
        if !trial_ref.is_finished() {
            self.unfinished_trials
                .entry(study_id)
                .or_default()
                .push(trial_ref.number);
        } else {
            let last_finished = self
                .last_finished_trial_number
                .get(&study_id)
                .copied()
                .unwrap_or(-1);
            self.last_finished_trial_number
                .insert(study_id, last_finished.max(trial_ref.number as i32));
        }
        Ok(trial_ref)
    }

    fn set_trial_param(
        &mut self,
        trial_id: u32,
        name: &str,
        distribution: &Distribution,
        value: f64,
    ) -> Result<()> {
        let (study_id, trial_number) = self.resolve_trial_location(trial_id)?;
        if self.is_trial_finished_in_cache(study_id, trial_number) {
            return Err(Error::new(ErrorKind::TrialAlreadyFinished));
        }
        self.refresh_trials(study_id)?;
        if let Some(trials) = self.trials.get(&study_id) {
            if let Some(trial) = trials
                .get(trial_number as usize)
                .and_then(|trial| trial.as_ref())
            {
                if trial.is_finished() {
                    return Err(Error::new(ErrorKind::TrialAlreadyFinished));
                }
            }
        }

        if let Some(existing) = self
            .study_caches
            .entry(study_id)
            .or_default()
            .param_distribution
            .get(name)
        {
            existing.check_compatibility(distribution)?;
        }

        self.backend
            .set_trial_param(trial_id, name, distribution, value)?;
        self.unfinished_trials
            .entry(study_id)
            .or_default()
            .push(trial_number);
        self.refresh_trials(study_id)?;

        let trials = self
            .trials
            .get_mut(&study_id)
            .ok_or_else(|| Error::new(ErrorKind::StudyNotFound))?;
        let trial = trials
            .get_mut(trial_number as usize)
            .and_then(|trial| trial.as_mut())
            .ok_or_else(|| Error::new(ErrorKind::TrialDiscarded))?;
        trial
            .distributions
            .insert(name.to_string(), distribution.clone());
        trial.internal_params.insert(name.to_string(), value);

        let study_cache = self.study_caches.entry(study_id).or_default();
        study_cache
            .param_distribution
            .insert(name.to_string(), distribution.clone());
        study_cache.update(trials);
        Ok(())
    }

    fn set_trial_state_values(
        &mut self,
        trial_id: u32,
        state_values: TrialStateValues,
    ) -> Result<()> {
        self.backend
            .set_trial_state_values(trial_id, state_values.clone())?;
        let (study_id, trial_number) = self.resolve_trial_location(trial_id)?;

        self.unfinished_trials
            .entry(study_id)
            .or_default()
            .push(trial_number);
        self.refresh_trials(study_id)?;

        let trials = self
            .trials
            .get_mut(&study_id)
            .ok_or_else(|| Error::new(ErrorKind::StudyNotFound))?;
        let trial = trials
            .get_mut(trial_number as usize)
            .and_then(|trial| trial.as_mut())
            .ok_or_else(|| Error::new(ErrorKind::TrialDiscarded))?;
        trial.state_values = state_values;

        self.study_caches
            .entry(study_id)
            .or_default()
            .update(trials);
        Ok(())
    }

    fn set_trial_intermediate_values(
        &mut self,
        trial_id: u32,
        intermediate_values: HashMap<u32, f64>,
    ) -> Result<()> {
        if intermediate_values.is_empty() {
            return Ok(());
        }
        let (study_id, trial_number) = self.resolve_trial_location(trial_id)?;
        if !self.is_trial_finished_in_cache(study_id, trial_number) {
            self.refresh_trials(study_id)?;
        }
        if self.is_trial_finished_in_cache(study_id, trial_number) {
            return Err(Error::new(ErrorKind::TrialAlreadyFinished));
        }

        self.backend
            .set_trial_intermediate_values(trial_id, intermediate_values.clone())?;
        let trials = self
            .trials
            .get_mut(&study_id)
            .ok_or_else(|| Error::new(ErrorKind::StudyNotFound))?;
        let trial = trials
            .get_mut(trial_number as usize)
            .and_then(|trial| trial.as_mut())
            .ok_or_else(|| Error::new(ErrorKind::TrialDiscarded))?;
        trial.intermediate_values.extend(intermediate_values);
        Ok(())
    }

    fn get_studies(&mut self) -> Result<&Vec<PersistedStudy>> {
        let loaded = self.backend.get_studies()?;
        self.studies = loaded;
        Ok(&self.studies)
    }

    fn get_study(&mut self, study_id: u32) -> Result<&PersistedStudy> {
        let loaded = self.backend.get_study(study_id)?;
        if let Some(study) = self.studies.iter_mut().find(|s| s.id == study_id) {
            *study = loaded;
        } else {
            self.studies.push(loaded);
        }
        self.studies
            .iter()
            .find(|s| s.id == study_id)
            .ok_or_else(|| Error::new(ErrorKind::StudyNotFound))
    }

    fn get_trials(&mut self, study_id: u32) -> Result<&Vec<Option<PersistedTrial>>> {
        self.refresh_trials(study_id)?;
        let trials = self
            .trials
            .get(&study_id)
            .ok_or_else(|| Error::new(ErrorKind::StudyNotFound))?;
        self.study_caches
            .entry(study_id)
            .or_default()
            .update(trials);
        Ok(trials)
    }

    fn get_trial(&mut self, trial_id: u32) -> Result<&PersistedTrial> {
        let (study_id, trial_number) = self.resolve_trial_location(trial_id)?;
        if !self.is_trial_finished_in_cache(study_id, trial_number) {
            self.refresh_trials(study_id)?;
        }
        self.get_cached_trial(trial_id)
    }

    fn get_study_attr(&mut self, study_id: u32, key: AttrKey) -> Result<String> {
        self.backend.get_study_attr(study_id, key)
    }

    fn get_cached_trial(&self, trial_id: u32) -> Result<&PersistedTrial> {
        let (study_id, trial_number) = self
            .trial_id_to_study_number
            .get(&trial_id)
            .copied()
            .ok_or_else(|| Error::new(ErrorKind::TrialNotFound))?;
        let trials = self
            .trials
            .get(&study_id)
            .ok_or_else(|| Error::new(ErrorKind::StudyNotFound))?;
        let trial = trials
            .get(trial_number as usize)
            .and_then(|trial| trial.as_ref())
            .ok_or_else(|| Error::new(ErrorKind::TrialDiscarded))?;
        Ok(trial)
    }

    fn set_category_labels(
        &mut self,
        study_id: u32,
        param_name: &str,
        labels: Vec<CategoryLabel>,
    ) -> Result<()> {
        let attrs = category_labels_to_attrs(param_name, &labels);
        self.backend.set_study_attrs(study_id, attrs, true)?;
        self.category_labels_cache
            .insert((study_id, param_name.to_string()), labels);
        Ok(())
    }

    fn get_category_labels(
        &mut self,
        study_id: u32,
        param_name: &str,
        cardinality: usize,
    ) -> Result<Option<Vec<CategoryLabel>>> {
        let key = (study_id, param_name.to_string());
        if let Some(labels) = self.category_labels_cache.get(&key) {
            return Ok(Some(labels.clone()));
        }
        let study = self.get_study(study_id)?;
        if let Some(labels) = get_category_labels(&study.attrs, param_name, cardinality) {
            self.category_labels_cache.insert(key, labels.clone());
            return Ok(Some(labels));
        }
        Ok(None)
    }

    fn get_trial_id_from_study_id_trial_number(
        &mut self,
        study_id: u32,
        trial_number: u32,
    ) -> Result<u32> {
        self.refresh_trials(study_id)?;
        let trials = self
            .trials
            .get(&study_id)
            .ok_or_else(|| Error::new(ErrorKind::StudyNotFound))?;
        let trial = trials
            .get(trial_number as usize)
            .and_then(|trial| trial.as_ref())
            .ok_or_else(|| Error::new(ErrorKind::TrialDiscarded))?;
        Ok(trial.id)
    }

    fn set_study_attrs(
        &mut self,
        study_id: u32,
        attrs: Attrs,
        error_on_overwrite: bool,
    ) -> Result<()> {
        self.backend
            .set_study_attrs(study_id, attrs.clone(), error_on_overwrite)?;
        self.studies = self.backend.get_studies()?;
        let study = self
            .studies
            .iter_mut()
            .find(|s| s.id == study_id)
            .ok_or_else(|| Error::new(ErrorKind::StudyNotFound))?;
        for (k, v) in attrs {
            study.attrs.insert(k, v);
        }
        Ok(())
    }

    fn set_trial_attrs(
        &mut self,
        trial_id: u32,
        attrs: Attrs,
        error_on_overwrite: bool,
    ) -> Result<()> {
        let (study_id, trial_number) = self.resolve_trial_location(trial_id)?;
        if self.is_trial_finished_in_cache(study_id, trial_number) {
            return Err(Error::new(ErrorKind::TrialAlreadyFinished));
        }
        self.refresh_trials(study_id)?;
        {
            let trials = self
                .trials
                .get(&study_id)
                .ok_or_else(|| Error::new(ErrorKind::StudyNotFound))?;
            let trial = trials
                .get(trial_number as usize)
                .and_then(|trial| trial.as_ref())
                .ok_or_else(|| Error::new(ErrorKind::TrialDiscarded))?;
            if trial.is_finished() {
                return Err(Error::new(ErrorKind::TrialAlreadyFinished));
            }
        }
        self.backend
            .set_trial_attrs(trial_id, attrs.clone(), error_on_overwrite)?;
        self.refresh_trials(study_id)?;
        let trials = self
            .trials
            .get_mut(&study_id)
            .ok_or_else(|| Error::new(ErrorKind::StudyNotFound))?;
        let trial = trials
            .get_mut(trial_number as usize)
            .and_then(|trial| trial.as_mut())
            .ok_or_else(|| Error::new(ErrorKind::TrialDiscarded))?;
        for (k, v) in attrs {
            trial.attrs.insert(k, v);
        }
        Ok(())
    }

    fn get_joint_search_space(&mut self, study_id: u32) -> Result<HashMap<String, Distribution>> {
        // get_trials updates the study cache as a side effect.
        self.get_trials(study_id)?;
        let cache = self.study_caches.entry(study_id).or_default();
        Ok(cache.get_joint_search_space())
    }

    fn discard_trials(&mut self, trial_ids: &[u32]) -> Result<()> {
        if !self.apply_discard {
            return Ok(());
        }

        let mut locations = Vec::with_capacity(trial_ids.len());
        for trial_id in trial_ids {
            if locations.iter().any(|(id, _, _)| id == trial_id) {
                continue;
            }
            let (study_id, trial_number) = self.resolve_trial_location(*trial_id)?;
            locations.push((*trial_id, study_id, trial_number));
        }

        self.backend.discard_trials(trial_ids)?;

        for (_, study_id, trial_number) in locations {
            if let Some(trials) = self.trials.get_mut(&study_id) {
                if let Some(trial) = trials.get_mut(trial_number as usize) {
                    *trial = None;
                }
            }
            self.study_caches.remove(&study_id);
            self.unfinished_trials.insert(study_id, Vec::new());
            self.last_finished_trial_number.insert(study_id, -1);
        }
        Ok(())
    }

    fn may_omit_trials(&self) -> bool {
        self.apply_discard && self.backend.may_omit_discard_trials()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustuna_core::attr::AttrKey;
    use rustuna_core::storage::Storage;
    use rustuna_core::ErrorKind;

    struct DummyBackend {
        inner: rustuna_core::storage::InMemoryStorage,
        discarded_trial_ids: std::collections::HashSet<u32>,
    }

    impl DummyBackend {
        fn new() -> Self {
            DummyBackend {
                inner: rustuna_core::storage::InMemoryStorage::new(),
                discarded_trial_ids: std::collections::HashSet::new(),
            }
        }
    }

    impl CachedStorageBackend for DummyBackend {
        fn create_new_study(
            &mut self,
            study_name: &str,
            directions: Vec<Direction>,
        ) -> Result<PersistedStudy> {
            let study = self.inner.create_new_study(study_name, directions)?.clone();
            Ok(study)
        }

        fn delete_study(&mut self, study_id: u32) -> Result<()> {
            self.inner.delete_study(study_id)?;
            Ok(())
        }

        fn create_new_trial(&mut self, study_id: u32) -> Result<PersistedTrial> {
            let trial = self.inner.create_new_trial(study_id)?.clone();
            Ok(trial)
        }

        fn create_new_trial_from_template(
            &mut self,
            study_id: u32,
            template: &PersistedTrial,
        ) -> Result<PersistedTrial> {
            let trial = self
                .inner
                .create_new_trial_from_template(study_id, template)?
                .clone();
            Ok(trial)
        }

        fn set_trial_param(
            &mut self,
            trial_id: u32,
            name: &str,
            distribution: &Distribution,
            value: f64,
        ) -> Result<()> {
            self.inner
                .set_trial_param(trial_id, name, distribution, value)
        }

        fn set_trial_state_values(
            &mut self,
            trial_id: u32,
            state_values: TrialStateValues,
        ) -> Result<()> {
            self.inner.set_trial_state_values(trial_id, state_values)
        }

        fn set_trial_intermediate_values(
            &mut self,
            trial_id: u32,
            intermediate_values: HashMap<u32, f64>,
        ) -> Result<()> {
            self.inner
                .set_trial_intermediate_values(trial_id, intermediate_values)
        }

        fn get_studies(&mut self) -> Result<Vec<PersistedStudy>> {
            Ok(self.inner.get_studies()?.clone())
        }

        fn get_study(&mut self, study_id: u32) -> Result<PersistedStudy> {
            Ok(self.inner.get_study(study_id)?.clone())
        }

        fn get_trials_diff(
            &mut self,
            study_id: u32,
            included_numbers: &[u32],
            trial_number_greater_than: i32,
            filter_discarded: bool,
        ) -> Result<Vec<PersistedTrial>> {
            let all = self.inner.get_trials(study_id)?.clone();
            let mut trials = Vec::new();
            for t in all.into_iter().flatten() {
                if filter_discarded && self.discarded_trial_ids.contains(&t.id) {
                    continue;
                }
                if included_numbers.contains(&t.number)
                    || (t.number as i32) > trial_number_greater_than
                {
                    trials.push(t);
                }
            }
            Ok(trials)
        }

        fn get_trial(&mut self, trial_id: u32) -> Result<PersistedTrial> {
            Ok(self.inner.get_trial(trial_id)?.clone())
        }

        fn get_study_attr(&mut self, study_id: u32, key: AttrKey) -> Result<String> {
            self.inner.get_study_attr(study_id, key)
        }

        fn set_study_attrs(
            &mut self,
            study_id: u32,
            attrs: Attrs,
            error_on_overwrite: bool,
        ) -> Result<()> {
            self.inner
                .set_study_attrs(study_id, attrs, error_on_overwrite)
        }

        fn set_trial_attrs(
            &mut self,
            trial_id: u32,
            attrs: Attrs,
            error_on_overwrite: bool,
        ) -> Result<()> {
            self.inner
                .set_trial_attrs(trial_id, attrs, error_on_overwrite)
        }

        fn discard_trials(&mut self, trial_ids: &[u32]) -> Result<()> {
            self.discarded_trial_ids.extend(trial_ids.iter().copied());
            Ok(())
        }

        fn may_omit_discard_trials(&self) -> bool {
            true
        }
    }
    #[test]
    fn create_new_study_updates_cache() -> Result<()> {
        let mut storage = CachedStorage::new(Box::new(DummyBackend::new()), false);
        let (study_id, name, directions) = {
            let study = storage.create_new_study("example", vec![Direction::Minimize])?;
            (study.id, study.name.clone(), study.directions.clone())
        };
        assert_eq!(name, "example");
        assert_eq!(directions, vec![Direction::Minimize]);
        assert_eq!(storage.studies.len(), 1);
        assert!(storage.trials.contains_key(&study_id));
        Ok(())
    }

    #[test]
    fn create_new_study_rejects_duplicate() -> Result<()> {
        let mut storage = CachedStorage::new(Box::new(DummyBackend::new()), false);
        storage.create_new_study("example", vec![Direction::Minimize])?;
        let res = storage.create_new_study("example", vec![Direction::Minimize]);
        match res {
            Err(e) => assert!(matches!(e.kind, ErrorKind::DuplicatedStudy)),
            Ok(_) => panic!("Expected duplicate study error"),
        }
        Ok(())
    }

    #[test]
    fn test_create_study_does_not_reuse_study_id() -> Result<()> {
        let mut storage = CachedStorage::new(Box::new(DummyBackend::new()), false);

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
    fn get_study_and_get_studies_use_cache() -> Result<()> {
        let mut storage = CachedStorage::new(Box::new(DummyBackend::new()), false);
        storage.create_new_study("s1", vec![Direction::Minimize])?;
        storage.create_new_study("s2", vec![Direction::Maximize])?;

        let all = storage.get_studies()?;
        assert_eq!(all.len(), 2);

        let s1 = storage.get_study(0)?;
        assert_eq!(s1.name, "s1");
        let s2 = storage.get_study(1)?;
        assert_eq!(s2.name, "s2");
        Ok(())
    }

    #[test]
    fn create_new_trial_appends_cache() -> Result<()> {
        let mut storage = CachedStorage::new(Box::new(DummyBackend::new()), false);
        let study = storage.create_new_study("s", vec![Direction::Minimize])?.id;
        let t0_num = storage.create_new_trial(study)?.number;
        let t1_num = storage.create_new_trial(study)?.number;
        assert_eq!(t0_num, 0);
        assert_eq!(t1_num, 1);
        let trials = storage
            .trials
            .get(&study)
            .expect("Trials map should contain study");
        assert_eq!(trials.len(), 2);
        Ok(())
    }

    #[test]
    fn get_trials_and_get_trial_return_cached_refs() -> Result<()> {
        let mut storage = CachedStorage::new(Box::new(DummyBackend::new()), false);
        let study_id = storage.create_new_study("s", vec![Direction::Minimize])?.id;
        let t0_id = storage.create_new_trial(study_id)?.id;
        let t1_id = storage.create_new_trial(study_id)?.id;

        let trials = storage.get_trials(study_id)?;
        assert_eq!(trials.len(), 2);
        let t0 = storage.get_trial(t0_id)?;
        assert_eq!(t0.number, 0);
        let t1 = storage.get_trial(t1_id)?;
        assert_eq!(t1.number, 1);
        Ok(())
    }

    #[test]
    fn get_trials_loads_from_backend_when_empty() -> Result<()> {
        let mut storage = CachedStorage::new(Box::new(DummyBackend::new()), false);
        let study_id = storage.create_new_study("s", vec![Direction::Minimize])?.id;
        storage.create_new_trial(study_id)?;

        storage.trials.clear();
        let trials = storage.get_trials(study_id)?;
        assert_eq!(trials.len(), 1);
        Ok(())
    }

    #[test]
    fn get_studies_refreshes_from_backend_every_time() -> Result<()> {
        let mut backend = DummyBackend::new();
        let study = backend.create_new_study("s", vec![Direction::Minimize])?;
        let mut storage = CachedStorage::new(Box::new(backend), false);

        let studies = storage.get_studies()?;
        assert_eq!(studies.len(), 1);

        storage
            .backend
            .create_new_study("s2", vec![Direction::Maximize])?;
        let studies = storage.get_studies()?;
        assert_eq!(studies.len(), 2);
        assert!(studies.iter().any(|s| s.name == study.name));
        assert!(studies.iter().any(|s| s.name == "s2"));
        Ok(())
    }

    #[test]
    fn get_trials_refreshes_when_backend_updates() -> Result<()> {
        let mut backend = DummyBackend::new();
        let study_id = backend.create_new_study("s", vec![Direction::Minimize])?.id;
        backend.create_new_trial(study_id)?;

        let mut storage = CachedStorage::new(Box::new(backend), false);
        let trials1 = storage.get_trials(study_id)?;
        assert_eq!(trials1.len(), 1);

        storage.backend.create_new_trial(study_id)?;
        let trials2 = storage.get_trials(study_id)?;
        assert_eq!(trials2.len(), 2);
        Ok(())
    }

    #[test]
    fn set_trial_state_values_updates_cache() -> Result<()> {
        let mut backend = DummyBackend::new();
        let study_id = backend.create_new_study("s", vec![Direction::Minimize])?.id;
        let trial_id = backend.create_new_trial(study_id)?.id;

        let mut storage = CachedStorage::new(Box::new(backend), false);
        storage.set_trial_state_values(trial_id, TrialStateValues::Complete(vec![1.0]))?;
        let trial = storage.get_trial(trial_id)?;
        assert!(matches!(trial.state_values, TrialStateValues::Complete(_)));
        Ok(())
    }

    #[test]
    fn set_trial_intermediate_values_updates_backend_and_cache() -> Result<()> {
        let mut backend = DummyBackend::new();
        let study_id = backend.create_new_study("s", vec![Direction::Minimize])?.id;
        let trial_id = backend.create_new_trial(study_id)?.id;

        let mut storage = CachedStorage::new(Box::new(backend), false);
        let mut values = HashMap::new();
        values.insert(0, 0.1);
        values.insert(2, 0.3);
        storage.set_trial_intermediate_values(trial_id, values)?;

        let trial = storage.get_trial(trial_id)?;
        assert_eq!(trial.intermediate_values[&0], 0.1);
        assert_eq!(trial.intermediate_values[&2], 0.3);

        let backend_trial = storage.backend.get_trial(trial_id)?;
        assert_eq!(backend_trial.intermediate_values[&0], 0.1);
        assert_eq!(backend_trial.intermediate_values[&2], 0.3);
        Ok(())
    }

    #[test]
    fn get_joint_search_space_uses_cache_update() -> Result<()> {
        let mut storage = CachedStorage::new(Box::new(DummyBackend::new()), false);
        let study_id = storage.create_new_study("s", vec![Direction::Minimize])?.id;

        let dist = Distribution::new_float(0.0, 1.0, None, false);
        let trial_id = storage.create_new_trial(study_id)?.id;
        storage.set_trial_param(trial_id, "x", &dist, 0.5)?;
        storage.set_trial_state_values(trial_id, TrialStateValues::Complete(vec![0.0]))?;

        let search_space = storage.get_joint_search_space(study_id)?;
        assert!(search_space.contains_key("x"));
        Ok(())
    }

    #[test]
    fn set_study_and_trial_attrs_update_cache() -> Result<()> {
        let mut storage = CachedStorage::new(Box::new(DummyBackend::new()), false);
        let study_id = storage.create_new_study("s", vec![Direction::Minimize])?.id;
        let trial_id = storage.create_new_trial(study_id)?.id;

        let mut s_attrs = Attrs::new();
        s_attrs.insert(AttrKey::User("foo".into()), "bar".to_string());
        storage.set_study_attrs(study_id, s_attrs, false)?;
        let study = storage.get_study(study_id)?;
        assert_eq!(
            study
                .attrs
                .get(&AttrKey::User("foo".into()))
                .expect("User attr 'foo' should exist"),
            "bar"
        );

        let mut t_attrs = Attrs::new();
        t_attrs.insert(AttrKey::System("key".into()), "val".to_string());
        storage.set_trial_attrs(trial_id, t_attrs, false)?;
        let trial = storage.get_trial(trial_id)?;
        assert_eq!(
            trial
                .attrs
                .get(&AttrKey::System("key".into()))
                .expect("System attr 'key' should exist"),
            "val"
        );
        Ok(())
    }

    #[test]
    fn set_trial_param_updates_cache_and_refreshes() -> Result<()> {
        let mut backend = DummyBackend::new();
        let study_id = backend.create_new_study("s", vec![Direction::Minimize])?.id;
        backend.create_new_trial(study_id)?;

        let mut storage = CachedStorage::new(Box::new(backend), false);
        let dist = Distribution::new_float(0.0, 1.0, None, false);
        let trial_id = storage.get_trials(study_id)?[0].as_ref().unwrap().id;
        storage.set_trial_param(trial_id, "x", &dist, 0.5)?;

        let trial = storage.get_trial(trial_id)?;
        assert_eq!(trial.internal_params.get("x"), Some(&0.5));
        assert_eq!(
            trial.distributions.get("x"),
            Some(&Distribution::new_float(0.0, 1.0, None, false))
        );
        Ok(())
    }

    #[test]
    fn set_trial_param() -> Result<()> {
        let mut storage = CachedStorage::new(Box::new(DummyBackend::new()), false);

        // Setup test across multiple studies and trials.
        let study_id = storage
            .create_new_study("test1", vec![Direction::Minimize])?
            .id;
        storage.create_new_study("test2", vec![Direction::Minimize])?;
        let trial_1 = storage.create_new_trial(study_id)?;
        let trial_1_id = trial_1.id;
        let trial_2 = storage.create_new_trial(study_id)?;
        let trial_2_id = trial_2.id;

        // Setup distributions
        let distribution_x = Distribution::new_float(1.0, 2.0, None, false);
        let distribution_y_1 = Distribution::new_categorical(3);
        // let distribution_y_2 = Distribution::new_categorical(2);
        let distribution_z = Distribution::new_float(1.0, 100.0, None, true);

        // Set new params.
        storage.set_trial_param(trial_1_id, "x", &distribution_x, 0.5)?;
        storage.set_trial_param(trial_1_id, "y", &distribution_y_1, 2.0)?;
        let trial = storage.get_trial(trial_1_id)?;
        assert_eq!(trial.internal_params["x"], 0.5);
        assert_eq!(trial.internal_params["y"], 2.0);

        // Set params to another trial
        storage.set_trial_param(trial_2_id, "x", &distribution_x, 0.3)?;
        storage.set_trial_param(trial_2_id, "z", &distribution_z, 0.1)?;
        let trial = storage.get_trial(trial_2_id)?;
        assert_eq!(trial.internal_params["x"], 0.3);
        assert_eq!(trial.internal_params["z"], 0.1);

        Ok(())
    }

    #[test]
    fn set_trial_param_rejects_incompatible_distribution_across_trials() -> Result<()> {
        let mut storage = CachedStorage::new(Box::new(DummyBackend::new()), false);
        let study_id = storage
            .create_new_study("test", vec![Direction::Minimize])?
            .id;

        let float_dist = Distribution::new_float(0.0, 1.0, None, false);
        let int_dist = Distribution::new_int(0, 5, 1, false);

        let trial0_id = storage.create_new_trial(study_id)?.id;
        storage.set_trial_param(trial0_id, "x", &float_dist, 0.5)?;

        let trial1_id = storage.create_new_trial(study_id)?.id;
        let err = storage
            .set_trial_param(trial1_id, "x", &int_dist, 1.0)
            .expect_err("Expected IncompatibleDistribution error");
        assert!(matches!(err.kind, ErrorKind::IncompatibleDistribution));
        Ok(())
    }

    #[test]
    fn discard_trials_invalidates_cache_and_backend() -> Result<()> {
        let mut storage = CachedStorage::new(Box::new(DummyBackend::new()), true);
        let study_id = storage
            .create_new_study("study", vec![Direction::Minimize])?
            .id;
        let trial0_id = storage.create_new_trial(study_id)?.id;
        let trial1_id = storage.create_new_trial(study_id)?.id;
        storage.get_trials(study_id)?;

        storage.discard_trials(&[trial0_id])?;

        let trials = storage.get_trials(study_id)?;
        assert!(trials[0].is_none());
        assert_eq!(trials[1].as_ref().unwrap().id, trial1_id);
        assert!(storage.may_omit_trials());
        assert!(matches!(
            storage.get_trial(trial0_id).unwrap_err().kind,
            ErrorKind::TrialDiscarded
        ));
        let backend_trials = storage.backend.get_trials_diff(study_id, &[], -1, true)?;
        assert_eq!(backend_trials.len(), 1);
        assert_eq!(backend_trials[0].id, trial1_id);
        Ok(())
    }

    #[test]
    fn discard_trials_keeps_cache_when_apply_discard_is_false() -> Result<()> {
        let mut storage = CachedStorage::new(Box::new(DummyBackend::new()), false);
        let study_id = storage
            .create_new_study("study", vec![Direction::Minimize])?
            .id;
        let trial_id = storage.create_new_trial(study_id)?.id;
        storage.get_trials(study_id)?;

        storage.discard_trials(&[trial_id])?;

        let trials = storage.get_trials(study_id)?;
        assert_eq!(trials[0].as_ref().unwrap().id, trial_id);
        assert!(!storage.may_omit_trials());
        Ok(())
    }
}
