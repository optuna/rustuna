use rustuna_core::study::Study;
use rustuna_core::trial::{PersistedTrial, TrialStateValues};
use rustuna_core::{Error, ErrorKind, Result};
use std::collections::HashMap;

/// Options shared by parameter-importance evaluators.
pub struct ImportanceOptions<'a> {
    pub target: Option<&'a dyn Fn(&PersistedTrial) -> f64>,
    pub normalize: bool,
    pub params: Option<Vec<String>>,
}

impl<'a> Default for ImportanceOptions<'a> {
    fn default() -> Self {
        Self {
            target: None,
            normalize: true,
            params: None,
        }
    }
}

impl<'a> ImportanceOptions<'a> {
    /// Creates the default options.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the target value used to evaluate importances.
    ///
    /// By default, evaluators use the first objective value of each completed trial. For
    /// multi-objective studies, this target must be specified explicitly.
    pub fn with_target(mut self, target: &'a dyn Fn(&PersistedTrial) -> f64) -> Self {
        self.target = Some(target);
        self
    }

    /// Sets whether the returned importances should be normalized to sum to `1.0`.
    pub fn normalize(mut self, normalize: bool) -> Self {
        self.normalize = normalize;
        self
    }

    /// Sets the list of parameter names to evaluate importances for.
    pub fn with_params(mut self, params: Vec<String>) -> Self {
        self.params = Some(params);
        self
    }
}

/// Evaluates parameter importances for completed trials in a study.
///
/// The returned map associates parameter names with non-negative importance values. By default,
/// the importances are normalized to sum to `1.0`.
pub fn get_param_importances(
    study: &Study,
    evaluator: &impl ImportanceEvaluator,
) -> Result<HashMap<String, f64>> {
    get_param_importances_with(study, evaluator, ImportanceOptions::default())
}

/// Evaluates parameter importances with explicit options.
///
/// This variant allows callers to provide a custom target function and to disable normalization.
pub fn get_param_importances_with(
    study: &Study,
    evaluator: &impl ImportanceEvaluator,
    opts: ImportanceOptions<'_>,
) -> Result<HashMap<String, f64>> {
    let normalize = opts.normalize;
    let importances = evaluator.evaluate_with(study, opts)?;
    if normalize {
        Ok(normalize_importances(importances))
    } else {
        Ok(importances)
    }
}

/// Trait implemented by parameter-importance evaluators.
pub trait ImportanceEvaluator {
    /// Evaluates parameter importances with the default options.
    fn evaluate(&self, study: &Study) -> Result<HashMap<String, f64>> {
        self.evaluate_with(study, ImportanceOptions::default())
    }

    /// Evaluates parameter importances with explicit options.
    fn evaluate_with(
        &self,
        study: &Study,
        opts: ImportanceOptions<'_>,
    ) -> Result<HashMap<String, f64>>;
}

fn normalize_importances(importances: HashMap<String, f64>) -> HashMap<String, f64> {
    let total = importances.values().sum::<f64>();
    if total == 0.0 {
        let n_params = importances.len() as f64;
        importances
            .into_keys()
            .map(|k| (k, 1.0 / n_params))
            .collect()
    } else {
        importances
            .into_iter()
            .map(|(k, v)| (k, v / total))
            .collect()
    }
}

fn default_target(t: &PersistedTrial) -> f64 {
    match &t.state_values {
        TrialStateValues::Complete(values) => values[0],
        _ => unreachable!("Only completed trials should be evaluated."),
    }
}

pub(crate) fn resolve_target(
    target: Option<&dyn Fn(&PersistedTrial) -> f64>,
) -> &dyn Fn(&PersistedTrial) -> f64 {
    target.unwrap_or(&default_target)
}

pub(crate) fn get_filtered_trials(
    study: &Study,
    target: Option<&dyn Fn(&PersistedTrial) -> f64>,
) -> Result<Vec<PersistedTrial>> {
    let mut guard = study.storage.write().map_err(|e| {
        Error::with_reason(
            ErrorKind::Unexpected,
            format!("Failed to acquire storage guard: {e}"),
        )
    })?;
    // TODO(c-bata): Avoid to clone trials.
    let completed_trials = guard
        .get_trials(study.id)?
        .iter()
        .flatten()
        .filter(|t| matches!(t.state_values, TrialStateValues::Complete(_)))
        .filter(|t| resolve_target(target)(t).is_finite())
        .cloned()
        .collect::<Vec<_>>();
    Ok(completed_trials)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ped_anova::PedAnovaImportanceEvaluator;
    use crate::test_utils;
    use crate::test_utils::ObjectiveType;
    use rustuna_core::distribution::Distribution;
    use rustuna_core::sampler::RandomSampler;
    use rustuna_core::storage::InMemoryStorage;
    use rustuna_core::study::{self, Direction};
    use rustuna_core::trial::PersistedTrial;
    use rustuna_core::{ErrorKind, Result};
    use std::collections::HashSet;

    #[test]
    fn test_error_multi_objective_wo_target() -> Result<()> {
        let evaluators = vec![PedAnovaImportanceEvaluator::default()];
        let study = test_utils::get_study(42, 5, ObjectiveType::Multi, Direction::Minimize)?;
        for evaluator in evaluators {
            let err = get_param_importances(&study, &evaluator).unwrap_err();
            assert!(matches!(err.kind, ErrorKind::ImportanceEvaluatorError));
        }
        Ok(())
    }

    #[test]
    fn test_evaluator_error_multi_objective_wo_target() -> Result<()> {
        let evaluators = vec![PedAnovaImportanceEvaluator::default()];
        let study = test_utils::get_study(42, 5, ObjectiveType::Multi, Direction::Minimize)?;
        for evaluator in evaluators {
            let err = evaluator.evaluate(&study).unwrap_err();
            assert!(matches!(err.kind, ErrorKind::ImportanceEvaluatorError));
        }
        Ok(())
    }

    #[test]
    fn test_get_param_importances() -> Result<()> {
        let evaluators = vec![PedAnovaImportanceEvaluator::default()];
        let study = test_utils::get_study(42, 20, ObjectiveType::Single, Direction::Minimize)?;
        for evaluator in evaluators {
            for normalize in [true, false] {
                let importances = get_param_importances_with(
                    &study,
                    &evaluator,
                    ImportanceOptions::new().normalize(normalize),
                )?;
                assert_eq!(importances.len(), 6, "{importances:?}");
                if normalize {
                    assert!(
                        importances
                            .values()
                            .all(|v| (-1e-12..=1.0 + 1e-12).contains(v)),
                        "{importances:?}"
                    );
                    assert!(
                        (importances.values().sum::<f64>() - 1.0).abs() < 1e-12,
                        "{importances:?}"
                    );
                }
            }
        }
        Ok(())
    }

    #[test]
    fn test_get_param_importances_with_target() -> Result<()> {
        let evaluators = vec![PedAnovaImportanceEvaluator::default()];
        let study = test_utils::get_study(42, 20, ObjectiveType::Single, Direction::Minimize)?;
        let target =
            |t: &PersistedTrial| -> f64 { t.internal_params["x1"] + t.internal_params["x2"] };
        for evaluator in evaluators {
            for normalize in [true, false] {
                let importances = get_param_importances_with(
                    &study,
                    &evaluator,
                    ImportanceOptions::new()
                        .with_target(&target)
                        .normalize(normalize),
                )?;
                assert_eq!(importances.len(), 6, "{importances:?}");
                if normalize {
                    assert!(
                        importances
                            .values()
                            .all(|v| (-1e-12..=1.0 + 1e-12).contains(v)),
                        "{importances:?}"
                    );
                    assert!(
                        (importances.values().sum::<f64>() - 1.0).abs() < 1e-12,
                        "{importances:?}"
                    );
                }

                let importances_wo_target = get_param_importances_with(
                    &study,
                    &evaluator,
                    ImportanceOptions::new().normalize(normalize),
                )?;
                assert_ne!(
                    importances, importances_wo_target,
                    "{importances:?}, {importances_wo_target:?}"
                );
            }
        }
        Ok(())
    }

    #[test]
    fn test_evaluator_evaluate_with_target() -> Result<()> {
        let evaluators = vec![PedAnovaImportanceEvaluator::default()];
        let study = test_utils::get_study(42, 20, ObjectiveType::Single, Direction::Minimize)?;
        let target =
            |t: &PersistedTrial| -> f64 { t.internal_params["x1"] + t.internal_params["x2"] };
        for evaluator in evaluators {
            let importances =
                evaluator.evaluate_with(&study, ImportanceOptions::new().with_target(&target))?;
            assert_eq!(importances.len(), 6, "{importances:?}");
            let importances_wo_target =
                evaluator.evaluate_with(&study, ImportanceOptions::new())?;
            assert_ne!(
                importances, importances_wo_target,
                "{importances:?}, {importances_wo_target:?}"
            );
        }
        Ok(())
    }

    #[test]
    fn test_get_param_importances_empty_study() -> Result<()> {
        let evaluators = vec![PedAnovaImportanceEvaluator::default()];
        let study = study::create_study(
            "empty-study",
            InMemoryStorage::new(),
            RandomSampler::new(),
            vec![Direction::Minimize],
        )?;
        for evaluator in evaluators {
            let importance = get_param_importances(&study, &evaluator)?;
            assert!(importance.is_empty(), "{importance:?}");
        }
        Ok(())
    }

    #[test]
    fn test_get_param_importances_single_search_space() -> Result<()> {
        let study = study::create_study(
            "empty-search-space",
            InMemoryStorage::new(),
            RandomSampler::new(),
            vec![Direction::Minimize],
        )?;
        study.optimize(
            |mut t| {
                let x1 = t.suggest_float("x1", 0.0, 5.0)?;
                let x2 = t.suggest("x2", &Distribution::new_float(0.0, 1.0, Some(1.0), false))?;
                Ok(vec![x1 + x2])
            },
            5,
        )?;
        let evaluators = vec![PedAnovaImportanceEvaluator::default()];
        for evaluator in evaluators {
            let importances = get_param_importances(&study, &evaluator)?;
            let keys = importances
                .keys()
                .map(String::as_str)
                .collect::<HashSet<_>>();
            let expected = HashSet::from(["x1", "x2"]);
            assert_eq!(keys, expected, "{importances:?}");
            assert!(importances["x1"] > 0.0, "{importances:?}");
            assert!(importances["x2"] == 0.0, "{importances:?}");
        }
        Ok(())
    }
}
