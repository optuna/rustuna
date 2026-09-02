use crate::common::{self, ImportanceEvaluator, ImportanceOptions};
use rustuna_core::distribution::Distribution;
use rustuna_core::internal::parzen_estimator::ParzenEstimator;
use rustuna_core::study::{Direction, Study};
use rustuna_core::trial::PersistedTrial;
use rustuna_core::Result;
use rustuna_core::{Error, ErrorKind};
use std::collections::{BTreeSet, HashMap, HashSet};

/// PED-ANOVA importance evaluator.
///
/// This evaluator implements the PED-ANOVA hyperparameter importance algorithm.
///
/// PED-ANOVA fits Parzen estimators to completed trials that perform better than a
/// user-specified baseline quantile and measures how much each parameter contributes to achieving
/// values better than that baseline.
///
/// For further information, see the original papers
/// [PED-ANOVA: Efficiently Quantifying Hyperparameter Importance in Arbitrary Subspaces](https://arxiv.org/abs/2304.10255) (IJCAI 2023).
/// [Conditional PED-ANOVA: Hyperparameter Importance in Hierarchical & Dynamic Search Spaces](https://arxiv.org/abs/2601.20800) (KDD 2026).
///
/// The quality of the result depends on how many trials are included above the target quantile.
/// In practice, it is preferable to have at least several top trials in the selected region.
///
/// `evaluate_on_local` controls whether importances are measured against the empirical search
/// region explored during optimization or against the full search space. Local evaluation is
/// especially useful when the effective search region changes during the study.
///
/// For a multi-objective study without an explicit target, trials are ranked by non-domination
/// rank, and ties within a rank are resolved by hypervolume subset selection (HSSP). To compute
/// the importance for a single objective instead, specify it with [`ImportanceOptions::with_target`].
/// [`PedAnovaImportanceEvaluator`] assumes minimization, so negate the target value when the
/// selected objective is being maximized.
///
/// # Examples
///
/// ```no_run
/// use rustuna_core::storage::InMemoryStorage;
/// use rustuna_core::study::{create_study, Direction};
/// use rustuna_core::sampler::RandomSampler;
/// use rustuna_core::Result;
/// use rustuna_importance::{get_param_importances, PedAnovaImportanceEvaluator};
///
/// fn main() -> Result<()> {
///     let storage = InMemoryStorage::new();
///     let study = create_study(
///         "ped-anova",
///         storage,
///         RandomSampler::new(),
///         vec![Direction::Minimize],
///     )?;
///
///     study.optimize(
///         |mut trial| {
///             let x1 = trial.suggest_float("x1", -10.0, 10.0)?;
///             let x2 = trial.suggest_float("x2", -10.0, 10.0)?;
///             Ok(vec![x1 + x2 / 1000.0])
///         },
///         100,
///     )?;
///
///     let evaluator = PedAnovaImportanceEvaluator::default();
///     let importances = get_param_importances(&study, &evaluator)?;
///     println!("{importances:?}");
///     Ok(())
/// }
/// ```
pub struct PedAnovaImportanceEvaluator {
    target_quantile: f64,
    region_quantile: f64,
    evaluate_on_local: bool,
    n_steps: usize,
    prior_weight: f64,
    min_n_trials_in_regime: usize,
}

impl Default for PedAnovaImportanceEvaluator {
    fn default() -> Self {
        Self {
            target_quantile: 0.1,
            region_quantile: 1.0,
            evaluate_on_local: true,
            n_steps: 50,
            prior_weight: 1.0,
            min_n_trials_in_regime: 2,
        }
    }
}

impl PedAnovaImportanceEvaluator {
    /// Creates a PED-ANOVA evaluator.
    ///
    /// `target_quantile` selects the top fraction of completed trials used as the target region.
    /// For example, `0.1` evaluates which parameters were important for achieving the top 10% of
    /// observed objective values.
    ///
    /// `region_quantile` selects the reference region against which the target region is compared.
    /// The default behavior in [`Default`] compares against all completed trials.
    ///
    /// `evaluate_on_local` controls whether the reference density is estimated from the explored
    /// region (`true`) or from the full search space (`false`).
    pub fn new(
        target_quantile: f64,
        region_quantile: f64,
        evaluate_on_local: bool,
    ) -> Result<Self> {
        if !(0.0 < target_quantile && target_quantile < region_quantile && region_quantile <= 1.0) {
            return Err(Error::with_reason(
                ErrorKind::ImportanceEvaluatorError,
                "condition 0.0 < `target_quantile` < `region_quantile` <= 1.0 must be satisfied",
            ));
        }
        Ok(Self {
            target_quantile,
            region_quantile,
            evaluate_on_local,
            n_steps: 50,
            prior_weight: 1.0,
            min_n_trials_in_regime: 2,
        })
    }

    fn get_top_quantile_trials<'a>(
        &self,
        study: &Study,
        trials: &'a [PersistedTrial],
        quantile: f64,
        target: Option<&dyn Fn(&PersistedTrial) -> f64>,
    ) -> Vec<&'a PersistedTrial> {
        if quantile == 1.0 {
            return trials.iter().collect();
        }
        if study.directions.len() > 1 && target.is_none() {
            let n_below = (trials.len() as f64 * quantile).ceil() as usize;
            let values = trials
                .iter()
                .map(|t| match &t.state_values {
                    TrialStateValues::Complete(v) => v.as_slice(),
                    _ => unreachable!("Only completed trials are passed to this function"),
                })
                .collect::<Vec<_>>();
            let (top_indices, _) = multi_objective::split_feasible_observation_indices(
                &values,
                &(0..trials.len()).collect::<Vec<_>>(),
                &study.directions,
                n_below,
            );
            return top_indices.into_iter().map(|i| &trials[i]).collect();
        }
        let is_lower_better = target.is_some() || study.directions[0] == Direction::Minimize;
        let objective_values = trials
            .iter()
            .map(|t| {
                let v = common::resolve_target(target)(t);
                if is_lower_better {
                    v
                } else {
                    -v
                }
            })
            .collect::<Vec<_>>();
        let num_top_trials = ((quantile * trials.len() as f64).ceil() as usize).max(1) - 1;
        let (_, &mut threshold, _) = objective_values
            .clone()
            .select_nth_unstable_by(num_top_trials, |a, b| a.total_cmp(b));
        let top_trials = trials
            .iter()
            .zip(objective_values.iter())
            .filter_map(|(t, &v)| (v <= threshold).then_some(t))
            .collect();
        top_trials
    }

    fn compute_pearson_divergence(
        &self,
        param_name: &str,
        dist: &Distribution,
        target_trials: &[&PersistedTrial],
        region_trials: &[&PersistedTrial],
    ) -> f64 {
        let (pe_top, grid_size) = build_parzen_estimator_on_grid(
            param_name,
            dist,
            target_trials,
            self.n_steps,
            self.prior_weight,
        );
        let pdf_top = (0..grid_size)
            .map(|i| pe_top.log_pdf(&[(param_name.to_string(), i as f64)].into()))
            .map(|p| p.exp() + 1e-12);
        let pdf_local: Vec<_> = if self.evaluate_on_local {
            // The importance of param during the study.
            let (pe_local, _) = build_parzen_estimator_on_grid(
                param_name,
                dist,
                region_trials,
                self.n_steps,
                self.prior_weight,
            );
            (0..grid_size)
                .map(|i| pe_local.log_pdf(&[(param_name.to_string(), i as f64)].into()))
                .map(|p| p.exp() + 1e-12)
                .collect()
        } else {
            // The importance of param in the search space.
            std::iter::repeat_n(1.0 / (grid_size as f64), grid_size).collect()
        };
        pdf_top
            .zip(pdf_local)
            .map(|(p_top, p_local)| p_local * (p_top / p_local - 1.0).powi(2))
            .sum()
    }
}

impl ImportanceEvaluator for PedAnovaImportanceEvaluator {
    /// Evaluates parameter importances from completed trials in the study.
    fn evaluate_with(
        &self,
        study: &Study,
        opts: ImportanceOptions,
    ) -> Result<HashMap<String, f64>> {
        let trials = common::get_filtered_trials(study, opts.target)?;
        common::ensure_target_for_multi_objective(&trials, opts.target)?;
        let params = resolve_params(&trials, opts.params)?;

        if trials.len() <= 1 {
            return Ok(params.into_iter().map(|name| (name, 0.0)).collect());
        }

        let target_trials =
            self.get_top_quantile_trials(study, &trials, self.target_quantile, opts.target);
        let region_trials =
            self.get_top_quantile_trials(study, &trials, self.region_quantile, opts.target);
        if target_trials.is_empty() {
            return Ok(params.into_iter().map(|name| (name, 0.0)).collect());
        }
        let quantile = target_trials.len() as f64 / region_trials.len() as f64;

        let target_trial_ids = target_trials.iter().map(|t| t.id).collect::<HashSet<_>>();
        let region_trial_ids = region_trials.iter().map(|t| t.id).collect::<HashSet<_>>();
        // Since HSSP is approximately implemented using a greedy algorithm, target trials
        // are guaranteed to be included in region trials, even when target is None for
        // multi-objective studies.
        assert!(target_trial_ids.is_subset(&region_trial_ids));

        // Theorem 4.2 and Algorithm 1 in the original paper:
        // https://arxiv.org/abs/2601.20800
        let importances = params.into_iter().map(|name| {
            let regime_trials =
                partition_by_regime(&name, &region_trials, self.min_n_trials_in_regime);
            let importance = regime_trials
                .into_iter()
                .map(|(dist, region_trials_regime)| {
                    let target_trials_regime = region_trials_regime
                        .iter()
                        .filter(|t| target_trial_ids.contains(&t.id))
                        .copied()
                        .collect::<Vec<_>>();
                    let regime_prob_target =
                        target_trials_regime.len() as f64 / target_trials.len() as f64; // alpha_i
                    let regime_prob_region =
                        region_trials_regime.len() as f64 / region_trials.len() as f64; // beta_i
                    match dist {
                        Some(dist) if !dist.is_single() && !target_trials_regime.is_empty() => {
                            regime_prob_target.powi(2) / regime_prob_region
                                * self.compute_pearson_divergence(
                                    &name,
                                    dist,
                                    &target_trials_regime,
                                    &region_trials_regime,
                                )
                        }
                        _ => 0.0,
                    }
                })
                .sum::<f64>();
            (name, importance)
        });
        Ok(importances
            .map(|(k, v)| (k, quantile.powi(2) * v))
            .collect())
    }
}

fn resolve_params(trials: &[PersistedTrial], params: Option<Vec<String>>) -> Result<Vec<String>> {
    let all_params = trials
        .iter()
        .flat_map(|t| t.distributions.keys())
        .cloned()
        .collect::<BTreeSet<_>>();
    match params {
        Some(params) => {
            let missing = params
                .iter()
                .filter(|p| !all_params.contains(*p))
                .cloned()
                .collect::<Vec<_>>();
            if missing.is_empty() {
                Ok(params)
            } else {
                Err(Error::with_reason(
                    ErrorKind::ImportanceEvaluatorError,
                    format!("Study must contain at least one completed trial for each specified parameter. Missing parameters: {missing:?}.")
                ))
            }
        }
        None => Ok(all_params.into_iter().collect()),
    }
}

fn partition_by_regime<'a>(
    param_name: &str,
    trials: &'a [&'a PersistedTrial],
    min_n_trials_in_regime: usize,
) -> Vec<(Option<&'a Distribution>, Vec<&'a PersistedTrial>)> {
    // Distribution does not implement Eq or Hash, so we use Vec instead of HashMap.
    let mut regime_trials: Vec<(Option<&'a Distribution>, Vec<&'a PersistedTrial>)> = vec![];
    for trial in trials {
        let dist = trial.distributions.get(param_name);
        if let Some((_, group_trials)) = regime_trials
            .iter_mut()
            .find(|(existing_regime, _)| *existing_regime == dist)
        {
            group_trials.push(trial);
        } else {
            regime_trials.push((dist, vec![trial]));
        }
    }
    regime_trials.retain(|(_, group_trials)| group_trials.len() >= min_n_trials_in_regime);
    regime_trials
}

fn count_numerical_param_in_grid(
    param_name: &str,
    dist: &Distribution,
    trials: &[&PersistedTrial],
    n_steps: usize,
) -> Vec<u32> {
    let (low, high, log, n_steps) = match dist {
        Distribution::Int {
            low,
            high,
            step,
            log,
        } => {
            let n_steps = if *log {
                let log2_domain_size = ((high - low + 1) as f64).log2().ceil() as usize + 1;
                n_steps.min(log2_domain_size)
            } else {
                n_steps.min(((high - low) / step + 1) as usize)
            };
            (*low as f64, *high as f64, *log, n_steps)
        }
        Distribution::Float {
            low,
            high,
            step,
            log,
        } => {
            let n_steps = if let Some(step) = step {
                n_steps.min(((high - low) / step).round() as usize + 1)
            } else {
                n_steps
            };
            (*low, *high, *log, n_steps)
        }
        _ => unreachable!("Invalid distribution type for numerical calculation"),
    };
    let (low, high) = if log {
        (low.ln(), high.ln())
    } else {
        (low, high)
    };
    let param_values = trials.iter().map(|t| {
        let v = t.internal_params[param_name];
        if log {
            v.ln()
        } else {
            v
        }
    });
    let step_size = (high - low) / (n_steps as f64 - 1.0);
    let mut counts = vec![0u32; n_steps];
    for v in param_values {
        let idx = ((v - low) / step_size - 0.5)
            .ceil()
            .max(0.0)
            .min((n_steps - 1) as f64) as usize;
        counts[idx] += 1;
    }
    counts
}

fn count_categorical_param_in_grid(
    param_name: &str,
    dist: &Distribution,
    trials: &[&PersistedTrial],
) -> Vec<u32> {
    let cardinality = match dist {
        Distribution::Categorical { cardinality } => *cardinality,
        _ => unreachable!("Invalid distribution type for categorical calculation"),
    };
    let mut counts = vec![0u32; cardinality];
    for t in trials {
        let v = t.internal_params[param_name];
        counts[v as usize] += 1;
    }
    counts
}

fn build_parzen_estimator_on_grid(
    param_name: &str,
    dist: &Distribution,
    trials: &[&PersistedTrial],
    n_steps: usize,
    prior_weight: f64,
) -> (ParzenEstimator, usize) {
    let (counts, rounded_dist) = match dist {
        Distribution::Int { .. } | Distribution::Float { .. } => {
            let counts = count_numerical_param_in_grid(param_name, dist, trials, n_steps);
            let rounded_dist = Distribution::new_int(0, (counts.len() - 1) as i64, 1, false);
            (counts, rounded_dist)
        }
        Distribution::Categorical { .. } => {
            let counts = count_categorical_param_in_grid(param_name, dist, trials);
            (counts, dist.clone())
        }
    };
    let observation = counts
        .iter()
        .enumerate()
        .filter(|(_, &c)| c > 0)
        .map(|(i, _)| i as f64)
        .collect();
    let weights = counts
        .iter()
        .filter(|&&c| c > 0)
        .map(|&c| c as f64)
        .collect::<Vec<_>>();
    let pe = ParzenEstimator::with_scott(
        &[(param_name.to_string(), observation)].into(),
        &[(param_name.to_string(), rounded_dist)].into(),
        &weights,
        prior_weight,
    );
    (pe, counts.len())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils;
    use crate::test_utils::ObjectiveType;
    use rustuna_core::study::Direction;
    use rustuna_core::Result;

    #[test]
    fn test_n_trials_less_than_three() -> Result<()> {
        let evaluator = PedAnovaImportanceEvaluator::default();
        for n_trials in 0..=2 {
            let study =
                test_utils::get_study(42, n_trials, ObjectiveType::Single, Direction::Minimize)?;
            let importances = evaluator.evaluate(&study)?;
            assert!(
                (n_trials == 2) ^ importances.values().all(|v| v.abs() <= 1e-12),
                "{importances:?}"
            );
        }
        Ok(())
    }

    #[test]
    fn test_direction() -> Result<()> {
        let study_minimize =
            test_utils::get_study(42, 20, ObjectiveType::Single, Direction::Minimize)?;
        let study_maximize =
            test_utils::get_study(42, 20, ObjectiveType::Single, Direction::Maximize)?;
        let evaluator = PedAnovaImportanceEvaluator::default();
        let importances_minimize = evaluator.evaluate(&study_minimize)?;
        let importances_maximize = evaluator.evaluate(&study_maximize)?;
        assert_ne!(importances_minimize, importances_maximize);
        Ok(())
    }

    #[test]
    fn test_target_quantile() -> Result<()> {
        let study = test_utils::get_study(42, 20, ObjectiveType::Single, Direction::Minimize)?;
        let evaluator_default = PedAnovaImportanceEvaluator::default();
        let evaluator = PedAnovaImportanceEvaluator::new(0.3, 1.0, true)?;
        let importances_default = evaluator_default.evaluate(&study)?;
        let importances = evaluator.evaluate(&study)?;
        assert_ne!(importances_default, importances);
        Ok(())
    }

    #[test]
    fn test_region_quantile_less_than_one() -> Result<()> {
        let study = test_utils::get_study(42, 20, ObjectiveType::Single, Direction::Minimize)?;
        let evaluator_default = PedAnovaImportanceEvaluator::default();
        let evaluator = PedAnovaImportanceEvaluator::new(0.1, 0.5, true)?;
        let importances_default = evaluator_default.evaluate(&study)?;
        let importances = evaluator.evaluate(&study)?;
        assert_ne!(importances_default, importances);
        Ok(())
    }

    #[test]
    fn test_get_top_quantile_trials_multi_objective_without_target() -> Result<()> {
        let study = test_utils::get_study(42, 20, ObjectiveType::Multi, Direction::Minimize)?;
        let trials = common::get_filtered_trials(&study, None)?;
        let evaluator = PedAnovaImportanceEvaluator::new(0.1, 0.5, true)?;

        let target_trials = evaluator.get_top_quantile_trials(&study, &trials, 0.1, None);
        let region_trials = evaluator.get_top_quantile_trials(&study, &trials, 0.5, None);
        let target_ids = target_trials.iter().map(|t| t.id).collect::<HashSet<_>>();
        let region_ids = region_trials.iter().map(|t| t.id).collect::<HashSet<_>>();

        assert_eq!(target_trials.len(), 2);
        assert_eq!(region_trials.len(), 10);
        assert!(target_ids.is_subset(&region_ids));
        Ok(())
    }

    #[test]
    fn test_evaluate_on_local() -> Result<()> {
        let study = test_utils::get_study(42, 20, ObjectiveType::Single, Direction::Minimize)?;
        let evaluator_default = PedAnovaImportanceEvaluator::default();
        let evaluator = PedAnovaImportanceEvaluator::new(0.1, 1.0, false)?;
        let importances_default = evaluator_default.evaluate(&study)?;
        let importances = evaluator.evaluate(&study)?;
        assert_ne!(importances_default, importances);
        Ok(())
    }

    #[test]
    fn test_conditional() -> Result<()> {
        let study = test_utils::get_study(42, 20, ObjectiveType::Conditional, Direction::Minimize)?;

        let evaluator = PedAnovaImportanceEvaluator::default();
        let params_cases = [
            None,
            Some(vec![]),
            Some(vec!["c"]),
            Some(vec!["x"]),
            Some(vec!["c", "x"]),
            Some(vec!["x", "y"]),
            Some(vec!["c", "x", "y"]),
            Some(vec!["d"]),
            Some(vec!["c", "d"]),
        ];
        for params in params_cases {
            let importances = match &params {
                Some(params) => evaluator.evaluate_with(
                    &study,
                    ImportanceOptions::new()
                        .with_params(params.iter().map(|p| (*p).to_string()).collect()),
                ),
                None => evaluator.evaluate(&study),
            };
            if params.as_ref().is_some_and(|params| params.contains(&"d")) {
                assert!(
                    matches!(
                        importances.unwrap_err().kind,
                        ErrorKind::ImportanceEvaluatorError
                    ),
                    "{params:?}"
                );
                continue;
            }

            let importances = importances?;
            if params.as_ref().is_some_and(Vec::is_empty) {
                assert!(importances.is_empty());
                continue;
            }
            let expected = params
                .unwrap_or_else(|| vec!["c", "x", "y"])
                .into_iter()
                .collect::<HashSet<_>>();
            assert_eq!(
                importances
                    .keys()
                    .map(String::as_str)
                    .collect::<HashSet<_>>(),
                expected
            );
            assert!(!importances.values().all(|&v| v == 0.0), "{importances:?}");
        }
        Ok(())
    }
}
