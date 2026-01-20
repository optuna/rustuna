use std::collections::HashMap;

use crate::attr::{category_labels_to_attrs, get_category_labels, Attrs, CategoryLabel};
use crate::distribution::Distribution;
use crate::study::{Direction, PersistedStudy};
use crate::study_cache::StudyCache;
use crate::trial::{PersistedTrial, TrialStateValues};
use crate::{Error, ErrorKind, Result};

pub trait Storage: Send + Sync {
    fn create_new_study(
        &mut self,
        study_name: &str,
        directions: Vec<Direction>,
    ) -> Result<&PersistedStudy>;
    fn delete_study(&mut self, study_id: u32) -> Result<()>;
    fn create_new_trial(&mut self, study_id: u32) -> Result<&PersistedTrial>;
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
    // Design Note:
    // Pruning (early-stopping) is not currently supported in Rustuna. This method is
    // provided to maintain compatibility with Optuna storage APIs.
    fn set_trial_intermediate_values(
        &mut self,
        trial_id: u32,
        intermediate_values: HashMap<u32, f64>,
    ) -> Result<()>;
    // Design Note:
    // get_* methods take &mut self to allow in-place cache refresh in wrapper implementations
    // (e.g., CachedStorage). With &self it is impossible to safely update caches and return
    // references without relying on unsafe patterns.
    fn get_studies(&mut self) -> Result<&Vec<PersistedStudy>>;
    fn get_study(&mut self, study_id: u32) -> Result<&PersistedStudy>;
    fn get_trials(&mut self, study_id: u32) -> Result<&Vec<PersistedTrial>>;
    fn get_trial(&mut self, trial_id: u32) -> Result<&PersistedTrial>;
    // Design Note:
    // Category labels are stored in study system attrs internally, but exposed via dedicated
    // APIs for caching efficiency. Since category labels cannot be overwritten once set for
    // a given (study_id, param_name), implementations can safely cache them without
    // invalidation concerns.
    fn set_category_labels(
        &mut self,
        study_id: u32,
        param_name: &str,
        labels: Vec<CategoryLabel>,
    ) -> Result<()>;
    fn get_category_labels(
        &mut self,
        study_id: u32,
        param_name: &str,
        cardinality: usize,
    ) -> Result<Option<Vec<CategoryLabel>>>;
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
    fn get_joint_search_space(&mut self, study_id: u32) -> Result<HashMap<String, Distribution>>;
}

#[derive(Default)]
pub struct InMemoryStorage {
    studies: Vec<PersistedStudy>,
    trials: HashMap<u32, Vec<PersistedTrial>>,
    trial_id_to_study_number: HashMap<u32, (u32, u32)>,
    study_caches: HashMap<u32, StudyCache>,
    next_study_id: u32,
    next_trial_id: u32,
}
impl InMemoryStorage {
    pub fn new() -> InMemoryStorage {
        InMemoryStorage {
            studies: vec![],
            trials: HashMap::new(),
            trial_id_to_study_number: HashMap::new(),
            study_caches: HashMap::new(),
            next_study_id: 0,
            next_trial_id: 0,
        }
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
            let trial = trials
                .get(number as usize)
                .ok_or(Error::new(ErrorKind::TrialNotFound))?;
            if trial.id != trial_id {
                return Err(Error::new(ErrorKind::StorageError));
            }
            return Ok(trial);
        }
        if self.trial_id_to_study_number.contains_key(&trial_id) {
            return Err(Error::new(ErrorKind::StorageError));
        }
        let trial = PersistedTrial::new(trial_id, study_id, number);
        trials.push(trial);
        self.trial_id_to_study_number
            .insert(trial_id, (study_id, number));
        if trial_id >= self.next_trial_id {
            self.next_trial_id = trial_id + 1;
        }
        Ok(&trials[number as usize])
    }
}
fn get_trials_by_study_id(
    all_trials: &HashMap<u32, Vec<PersistedTrial>>,
    study_id: u32,
) -> Result<&Vec<PersistedTrial>> {
    let trials = all_trials
        .get(&study_id)
        .ok_or(Error::new(ErrorKind::StudyNotFound))?;
    Ok(trials)
}
fn get_mut_trials_by_study_id(
    all_trials: &mut HashMap<u32, Vec<PersistedTrial>>,
    study_id: u32,
) -> Result<&mut Vec<PersistedTrial>> {
    let trials = all_trials
        .get_mut(&study_id)
        .ok_or(Error::new(ErrorKind::StudyNotFound))?;
    Ok(trials)
}
fn get_study_id_trial_number_by_trial_id(
    trial_id_to_study_number: &HashMap<u32, (u32, u32)>,
    trial_id: u32,
) -> Result<(u32, u32)> {
    trial_id_to_study_number
        .get(&trial_id)
        .copied()
        .ok_or(Error::new(ErrorKind::TrialNotFound))
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
        if let Some(trials) = self.trials.remove(&study_id) {
            for trial in trials {
                self.trial_id_to_study_number.remove(&trial.id);
            }
        }
        self.study_caches.remove(&study_id);

        Ok(())
    }

    fn create_new_trial(&mut self, study_id: u32) -> Result<&PersistedTrial> {
        let trials = get_mut_trials_by_study_id(&mut self.trials, study_id)?;
        let trial_id = self.next_trial_id;
        self.next_trial_id += 1;
        let number = trials.len() as u32;
        let trial = PersistedTrial::new(trial_id, study_id, number);
        trials.push(trial);
        self.trial_id_to_study_number
            .insert(trial_id, (study_id, number));
        Ok(&trials[number as usize])
    }

    fn set_trial_param(
        &mut self,
        trial_id: u32,
        name: &str,
        distribution: &Distribution,
        value: f64,
    ) -> Result<()> {
        let (study_id, trial_number) =
            get_study_id_trial_number_by_trial_id(&self.trial_id_to_study_number, trial_id)?;
        let trial = get_mut_trials_by_study_id(&mut self.trials, study_id)?
            .get_mut(trial_number as usize)
            .ok_or(Error::new(ErrorKind::TrialNotFound))?;
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
            get_study_id_trial_number_by_trial_id(&self.trial_id_to_study_number, trial_id)?;
        let trial = get_mut_trials_by_study_id(&mut self.trials, study_id)?
            .get_mut(trial_number as usize)
            .ok_or(Error::new(ErrorKind::TrialNotFound))?;
        check_trial_is_updatable(trial)?;
        trial.state_values = state_values;
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
        let (study_id, trial_number) =
            get_study_id_trial_number_by_trial_id(&self.trial_id_to_study_number, trial_id)?;
        let trial = get_mut_trials_by_study_id(&mut self.trials, study_id)?
            .get_mut(trial_number as usize)
            .ok_or(Error::new(ErrorKind::TrialNotFound))?;
        check_trial_is_updatable(trial)?;
        for (step, value) in intermediate_values {
            trial.intermediate_values.insert(step, value);
        }
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

    fn get_trials(&mut self, study_id: u32) -> Result<&Vec<PersistedTrial>> {
        get_trials_by_study_id(&self.trials, study_id)
    }

    fn get_trial(&mut self, trial_id: u32) -> Result<&PersistedTrial> {
        let (study_id, trial_number) =
            get_study_id_trial_number_by_trial_id(&self.trial_id_to_study_number, trial_id)?;
        let trial = get_trials_by_study_id(&self.trials, study_id)?
            .get(trial_number as usize)
            .ok_or(Error::new(ErrorKind::TrialNotFound))?;
        Ok(trial)
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
        self.trial_id_to_study_number
            .iter()
            .find(|(_, &(s_id, t_num))| s_id == study_id && t_num == trial_number)
            .map(|(&trial_id, _)| trial_id)
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
            get_study_id_trial_number_by_trial_id(&self.trial_id_to_study_number, trial_id)?;
        let trial = get_mut_trials_by_study_id(&mut self.trials, study_id)?
            .get_mut(trial_number as usize)
            .ok_or(Error::new(ErrorKind::TrialNotFound))?;
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

        let float_dist = Distribution::Float {
            low: 0.0,
            high: 1.0,
            step: None,
            log: false,
        };
        let int_dist = Distribution::Int {
            low: 0,
            high: 5,
            step: 1,
            log: false,
        };

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
        let trial = storage.create_new_trial(study_id)?;
        assert_eq!(trial.id, 11);
        Ok(())
    }
}
