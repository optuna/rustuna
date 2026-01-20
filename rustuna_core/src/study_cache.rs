use std::collections::HashMap;

use crate::{
    distribution::Distribution,
    trial::{PersistedTrial, TrialStateValues},
};

#[derive(Debug, Clone)]
pub struct StudyCache {
    trial_number_cursor: usize,
    joint_search_space: Option<HashMap<String, Distribution>>,
    pub param_distribution: HashMap<String, Distribution>,
    // TODO(c-bata): Cache following values.
    // best_trial: Option<PersistedTrial>,
}

impl Default for StudyCache {
    fn default() -> Self {
        Self::new()
    }
}

impl StudyCache {
    pub fn new() -> StudyCache {
        StudyCache {
            trial_number_cursor: 0,
            joint_search_space: None,
            param_distribution: HashMap::new(),
        }
    }

    pub fn get_joint_search_space(&self) -> HashMap<String, Distribution> {
        match self.joint_search_space {
            Some(ref search_space) => search_space.clone(),
            None => HashMap::new(),
        }
    }

    pub fn update(&mut self, trials: &[PersistedTrial]) {
        for i in (self.trial_number_cursor..trials.len()).rev() {
            let trial = &trials[i];
            // Update trial number cursor to the oldest unfinished trial.
            if !trial.is_finished() {
                self.trial_number_cursor = i;
                continue;
            }

            // Update joint search space of complete trials.
            if let TrialStateValues::Complete(_) = trial.state_values {
                match self.joint_search_space {
                    Some(ref mut search_space) => {
                        let mut joint_space = HashMap::new();
                        for (name, distribution) in search_space.iter() {
                            if trial.distributions.get(name) == Some(distribution) {
                                joint_space.insert(name.clone(), distribution.clone());
                            }
                        }
                        *search_space = joint_space;
                    }
                    None => {
                        self.joint_search_space = Some(trial.distributions.clone());
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::attr::Attrs;
    use crate::study_cache::StudyCache;
    use crate::trial::{PersistedTrial, TrialStateValues};

    #[test]
    fn test_get_joint_search_space_empty() {
        let cache = StudyCache::new();
        let search_space = cache.get_joint_search_space();
        assert!(search_space.is_empty());
    }

    #[test]
    fn test_get_joint_search_space() {
        let mut cache = StudyCache::new();
        let trials = vec![
            PersistedTrial {
                id: 0,
                study_id: 0,
                number: 0,
                state_values: TrialStateValues::Complete(vec![0.0]),
                internal_params: [("x".to_string(), 0.0), ("y".to_string(), 1.0)]
                    .iter()
                    .cloned()
                    .collect(),
                distributions: [
                    (
                        "x".to_string(),
                        Distribution::Float {
                            low: 0.0,
                            high: 1.0,
                            step: None,
                            log: false,
                        },
                    ),
                    (
                        "y".to_string(),
                        Distribution::Float {
                            low: 1.0,
                            high: 2.0,
                            step: None,
                            log: false,
                        },
                    ),
                ]
                .iter()
                .cloned()
                .collect(),
                attrs: Attrs::new(),
                intermediate_values: HashMap::new(),
                datetime_start: None,
                datetime_complete: None,
            },
            PersistedTrial {
                id: 1,
                study_id: 0,
                number: 1,
                state_values: TrialStateValues::Fail,
                internal_params: HashMap::new(),
                distributions: HashMap::new(),
                attrs: Attrs::new(),
                intermediate_values: HashMap::new(),
                datetime_start: None,
                datetime_complete: None,
            },
            PersistedTrial {
                id: 2,
                study_id: 0,
                number: 2,
                state_values: TrialStateValues::Running,
                internal_params: HashMap::new(),
                distributions: HashMap::new(),
                attrs: Attrs::new(),
                intermediate_values: HashMap::new(),
                datetime_start: None,
                datetime_complete: None,
            },
            PersistedTrial {
                id: 3,
                study_id: 0,
                number: 3,
                state_values: TrialStateValues::Complete(vec![0.0]),
                internal_params: [("x".to_string(), 0.5), ("y".to_string(), 10.0)]
                    .iter()
                    .cloned()
                    .collect(),
                distributions: [
                    (
                        "x".to_string(),
                        Distribution::Float {
                            low: 0.0,
                            high: 1.0,
                            step: None,
                            log: false,
                        },
                    ),
                    (
                        "y".to_string(),
                        Distribution::Float {
                            low: 1.0,
                            high: 10.0,
                            step: None,
                            log: false,
                        },
                    ),
                ]
                .iter()
                .cloned()
                .collect(),
                attrs: Attrs::new(),
                intermediate_values: HashMap::new(),
                datetime_start: None,
                datetime_complete: None,
            },
        ];

        cache.update(&trials);
        let search_space = cache.get_joint_search_space();
        assert!(search_space.len() == 1); // Only "x" is common ("y" is dynamic).
        assert!(search_space.contains_key("x"));
    }

    #[test]
    fn test_trial_number_cursor() {
        let mut cache = StudyCache::new();
        let trials = vec![
            PersistedTrial {
                id: 0,
                study_id: 0,
                number: 0,
                state_values: TrialStateValues::Complete(vec![0.0]),
                internal_params: HashMap::new(),
                distributions: HashMap::new(),
                attrs: Attrs::new(),
                intermediate_values: HashMap::new(),
                datetime_start: None,
                datetime_complete: None,
            },
            PersistedTrial {
                id: 1,
                study_id: 0,
                number: 1,
                state_values: TrialStateValues::Pruned,
                internal_params: HashMap::new(),
                distributions: HashMap::new(),
                attrs: Attrs::new(),
                intermediate_values: HashMap::new(),
                datetime_start: None,
                datetime_complete: None,
            },
            PersistedTrial {
                id: 2,
                study_id: 0,
                number: 2,
                state_values: TrialStateValues::Fail,
                internal_params: HashMap::new(),
                distributions: HashMap::new(),
                attrs: Attrs::new(),
                intermediate_values: HashMap::new(),
                datetime_start: None,
                datetime_complete: None,
            },
            PersistedTrial {
                id: 3,
                study_id: 0,
                number: 3,
                state_values: TrialStateValues::Running,
                internal_params: HashMap::new(),
                distributions: HashMap::new(),
                attrs: Attrs::new(),
                intermediate_values: HashMap::new(),
                datetime_start: None,
                datetime_complete: None,
            },
            PersistedTrial {
                id: 4,
                study_id: 0,
                number: 4,
                state_values: TrialStateValues::Complete(vec![0.0]),
                internal_params: HashMap::new(),
                distributions: HashMap::new(),
                attrs: Attrs::new(),
                intermediate_values: HashMap::new(),
                datetime_start: None,
                datetime_complete: None,
            },
        ];

        assert!(cache.trial_number_cursor == 0);
        cache.update(&trials);
        println!("{:?}", cache.trial_number_cursor);
        assert!(cache.trial_number_cursor == 3);
        cache.update(&trials);
        assert!(cache.trial_number_cursor == 3);
    }
}
