use std::collections::{HashMap, HashSet};

use crate::multi_objective::{hssp, nds};
use crate::study::Direction;
use crate::trial::{PersistedTrial, TrialStateValues};
use crate::Result;

const EPS: f64 = 1e-12;

pub fn split_trials_for_multi_objective<'a>(
    trials: &[&'a PersistedTrial],
    directions: &[Direction],
    gamma: usize,
) -> Result<(Vec<&'a PersistedTrial>, Vec<&'a PersistedTrial>)> {
    let n = trials.len();
    assert!(
        gamma <= n,
        "gamma must be less than or equal to the number of trials"
    );

    if n == 0 {
        return Ok((Vec::new(), Vec::new()));
    }
    if gamma == n {
        return Ok((trials.to_vec(), Vec::new()));
    }

    let values = trials
        .iter()
        .map(|trial| match &trial.state_values {
            TrialStateValues::Complete(values) => values.as_slice(),
            _ => panic!("Unexpected non-complete trial found during multi-objective split"),
        })
        .collect::<Vec<_>>();
    let feasibles_violations = trials
        .iter()
        .map(|trial| {
            let constraints = trial.constraints()?;
            let feasible = constraints.values().all(|x| *x <= 0.0);
            let violation = constraints.values().filter(|&x| *x > 0.0).sum::<f64>();
            Ok((feasible, violation))
        })
        .collect::<Result<Vec<_>>>()?;
    let (good_indices, poor_indices) = split_observation_indices_for_multi_objective(
        &values,
        &feasibles_violations,
        directions,
        gamma,
    );
    let good_trials = good_indices.into_iter().map(|i| trials[i]).collect();
    let poor_trials = poor_indices.into_iter().map(|i| trials[i]).collect();
    Ok((good_trials, poor_trials))
}

/// Splits objective observations into promising and non-promising index sets.
///
/// `values[i]` holds the observed objective values (before direction
/// normalization) and `feasibles_violations[i]` the constraint feasibility and
/// total violation of the same observation. Both returned vectors contain
/// indices into the input slices: feasible observations are ranked by
/// non-dominated sorting and hypervolume subset selection, and when fewer than
/// `gamma` observations are feasible the promising set is padded with the
/// least-violating infeasible ones.
pub fn split_observation_indices_for_multi_objective<T: AsRef<[f64]>>(
    values: &[T],
    feasibles_violations: &[(bool, f64)],
    directions: &[Direction],
    gamma: usize,
) -> (Vec<usize>, Vec<usize>) {
    let n = values.len();
    assert_eq!(
        n,
        feasibles_violations.len(),
        "values and feasibles_violations must have the same length"
    );
    assert!(
        gamma <= n,
        "gamma must be less than or equal to the number of trials"
    );

    if n == 0 {
        return (Vec::new(), Vec::new());
    }
    if gamma == n {
        return ((0..n).collect(), Vec::new());
    }

    let mut feasible_indices = Vec::new();
    let mut infeasible_index_violations = Vec::new();
    for (index, &(feasible, violation)) in feasibles_violations.iter().enumerate() {
        if feasible {
            feasible_indices.push(index);
        } else {
            infeasible_index_violations.push((index, violation));
        }
    }

    let feasible_gamma = gamma.min(feasible_indices.len());
    let (mut good_indices, mut poor_indices) =
        split_feasible_observation_indices(values, &feasible_indices, directions, feasible_gamma);
    let infeasible_gamma = gamma.saturating_sub(good_indices.len());
    let (infeasible_good_indices, infeasible_poor_indices) =
        split_infeasible_observation_indices(&mut infeasible_index_violations, infeasible_gamma);
    good_indices.extend(infeasible_good_indices);
    poor_indices.extend(infeasible_poor_indices);
    (good_indices, poor_indices)
}

fn split_feasible_observation_indices<T: AsRef<[f64]>>(
    values: &[T],
    feasible_indices: &[usize],
    directions: &[Direction],
    gamma: usize,
) -> (Vec<usize>, Vec<usize>) {
    let n = feasible_indices.len();
    assert!(
        gamma <= n,
        "gamma must be less than or equal to the number of trials"
    );

    if n == 0 {
        return (Vec::new(), Vec::new());
    }
    if gamma == n {
        return (feasible_indices.to_vec(), Vec::new());
    }

    // Assume minimization (negate value if maximization)
    let loss_values: Vec<Vec<f64>> = feasible_indices
        .iter()
        .map(|&index| {
            values[index]
                .as_ref()
                .iter()
                .zip(directions.iter())
                .map(|(&val, dir)| match dir {
                    Direction::Minimize => val,
                    Direction::Maximize => -val,
                })
                .collect()
        })
        .collect();
    let nondomination_ranks = nds::fast_non_dominated_sort_partial(&loss_values, gamma);
    let mut rank_to_indices: HashMap<usize, Vec<usize>> = HashMap::new();
    for (i, &rank) in nondomination_ranks.iter().enumerate() {
        rank_to_indices.entry(rank).or_default().push(i);
    }

    let mut good_local = Vec::with_capacity(gamma);
    let mut poor_local = Vec::with_capacity(n - gamma);

    let mut current_rank = 0usize;
    while good_local.len() + rank_to_indices.get(&current_rank).map_or(0, |v| v.len()) <= gamma {
        if let Some(indices) = rank_to_indices.get(&current_rank) {
            for &i in indices.iter() {
                good_local.push(i);
            }
        }
        current_rank += 1;
    }
    let hss_subset_size = gamma - good_local.len();
    if hss_subset_size > 0 {
        let rank_i_loss_vals = rank_to_indices
            .get(&current_rank)
            .unwrap()
            .iter()
            .map(|&i| loss_values[i].as_slice())
            .collect::<Vec<&[f64]>>();
        let rank_i_indices = rank_to_indices.get(&current_rank).unwrap();

        // Partition rank-i rows by their inf signature so HSSP only sees finite
        // points against a finite reference. NDS has already certified every row
        // here as Pareto-incomparable within the rank — a `+inf` row is not
        // dominated, it just has zero hypervolume contribution against any finite
        // reference (the point lies outside the reference box). The split below
        // is a static realization of HV-greedy ordering, not a dominance call:
        //
        //   * Any `-inf` → inclusive HV factor `ref - (-inf) = +inf`, so HV-greedy
        //     would pick these first. Take them up-front.
        //   * Any `+inf` (and no `-inf`) → at least one factor `ref - point <= 0`,
        //     so the inclusive HV is 0 and HV-greedy picks these last. Defer to
        //     padding once the finite candidates are exhausted.
        //   * All finite → regular HSSP candidates.
        //
        // Doing the split here also protects HSSP from numerical poison: a `-inf`
        // would propagate `+inf` through the lazy contributions, and a `+inf`
        // would blow up the reference itself if naively folded into the worst.
        let n_dims = directions.len();
        let mut neg_inf_local: Vec<usize> = Vec::new();
        let mut finite_local: Vec<usize> = Vec::new();
        let mut pos_inf_local: Vec<usize> = Vec::new();
        for (local, loss_val) in rank_i_loss_vals.iter().enumerate() {
            let has_neg_inf = loss_val.contains(&f64::NEG_INFINITY);
            let has_pos_inf = loss_val.contains(&f64::INFINITY);
            if has_neg_inf {
                neg_inf_local.push(local);
            } else if has_pos_inf {
                pos_inf_local.push(local);
            } else {
                finite_local.push(local);
            }
        }

        let mut remaining = hss_subset_size;
        for &local in neg_inf_local.iter().take(remaining) {
            good_local.push(rank_i_indices[local]);
        }
        remaining = remaining.saturating_sub(neg_inf_local.len());

        if remaining > 0 && !finite_local.is_empty() {
            let finite_rows: Vec<&[f64]> =
                finite_local.iter().map(|&l| rank_i_loss_vals[l]).collect();
            let finite_indices: Vec<usize> =
                finite_local.iter().map(|&l| rank_i_indices[l]).collect();
            let mut worst_point = vec![f64::NEG_INFINITY; n_dims];
            let mut dim_has_finite = vec![false; n_dims];
            for row in finite_rows.iter() {
                for d in 0..n_dims {
                    let v = row[d];
                    if v > worst_point[d] {
                        worst_point[d] = v;
                        dim_has_finite[d] = true;
                    }
                }
            }
            if dim_has_finite.iter().all(|&b| b) {
                let mut reference_point = Vec::with_capacity(n_dims);
                for &w in worst_point.iter() {
                    let r = (1.1 * w).max(0.9 * w);
                    reference_point.push(if r == 0.0 { EPS } else { r });
                }
                let take = remaining.min(finite_rows.len());
                let selected_indices = hssp::hypervolume_subset_selection(
                    &finite_rows,
                    &finite_indices,
                    &reference_point,
                    take,
                );
                for &i in selected_indices.iter() {
                    good_local.push(i);
                }
                remaining = remaining.saturating_sub(take);
            } else {
                // No finite candidates contributed to some dimension's worst
                // (only possible when the finite subset is empty for that dim,
                // which the outer guard already excluded). Defensive fallback.
                let take = remaining.min(finite_local.len());
                for &local in finite_local.iter().take(take) {
                    good_local.push(rank_i_indices[local]);
                }
                remaining = remaining.saturating_sub(take);
            }
        }

        for &local in pos_inf_local.iter().take(remaining) {
            good_local.push(rank_i_indices[local]);
        }
    }
    let good_local_set: HashSet<usize> = good_local.iter().copied().collect();
    for local in 0..n {
        if !good_local_set.contains(&local) {
            poor_local.push(local);
        }
    }

    (
        good_local
            .into_iter()
            .map(|local| feasible_indices[local])
            .collect(),
        poor_local
            .into_iter()
            .map(|local| feasible_indices[local])
            .collect(),
    )
}

fn split_infeasible_observation_indices(
    index_violations: &mut [(usize, f64)],
    gamma: usize,
) -> (Vec<usize>, Vec<usize>) {
    let n = index_violations.len();
    assert!(
        gamma <= n,
        "gamma must be less than or equal to the number of trials"
    );
    if n == 0 {
        return (Vec::new(), Vec::new());
    }
    if gamma == 0 {
        return (
            Vec::new(),
            index_violations.iter().map(|&(i, _)| i).collect(),
        );
    }
    if gamma == n {
        return (
            index_violations.iter().map(|&(i, _)| i).collect(),
            Vec::new(),
        );
    }

    index_violations.select_nth_unstable_by(gamma, |(_, violation_i), (_, violation_j)| {
        violation_i
            .partial_cmp(violation_j)
            .expect("constraint is non-Nan value")
    });
    let (good_indices, poor_indices) = index_violations.split_at(gamma);
    (
        good_indices.iter().map(|&(i, _)| i).collect(),
        poor_indices.iter().map(|&(i, _)| i).collect(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attr::AttrKey;

    const FEASIBLE: (bool, f64) = (true, 0.0);

    #[test]
    fn split_observation_indices_returns_original_positions() {
        let values = vec![vec![3.0], vec![1.0], vec![4.0], vec![2.0]];
        let feasibles_violations = vec![FEASIBLE; values.len()];
        let (good, poor) = split_observation_indices_for_multi_objective(
            &values,
            &feasibles_violations,
            &[Direction::Minimize],
            2,
        );

        assert_eq!(good, vec![1, 3]);
        assert_eq!(poor, vec![0, 2]);
    }

    #[test]
    fn split_observation_indices_pads_with_least_violating_infeasible() {
        let values = vec![vec![1.0], vec![2.0], vec![3.0], vec![4.0]];
        let feasibles_violations = vec![(false, 5.0), (true, 0.0), (false, 1.0), (false, 3.0)];
        let (good, poor) = split_observation_indices_for_multi_objective(
            &values,
            &feasibles_violations,
            &[Direction::Minimize],
            2,
        );

        // The single feasible observation is promising; the second slot is
        // filled by the infeasible observation with the smallest violation.
        assert_eq!(good, vec![1, 2]);
        assert_eq!(poor.len(), 2);
        assert!(poor.contains(&0) && poor.contains(&3));
    }

    #[test]
    fn split_observation_indices_all_infeasible() {
        let values = vec![vec![1.0], vec![2.0], vec![3.0]];
        let feasibles_violations = vec![(false, 2.0), (false, 3.0), (false, 1.0)];
        let (good, poor) = split_observation_indices_for_multi_objective(
            &values,
            &feasibles_violations,
            &[Direction::Minimize],
            1,
        );

        assert_eq!(good, vec![2]);
        assert_eq!(poor.len(), 2);
        assert!(poor.contains(&0) && poor.contains(&1));
    }

    #[test]
    fn split_observation_indices_gamma_equals_n() {
        let values = vec![vec![3.0], vec![1.0]];
        let feasibles_violations = vec![(true, 0.0), (false, 1.0)];
        let (good, poor) = split_observation_indices_for_multi_objective(
            &values,
            &feasibles_violations,
            &[Direction::Minimize],
            2,
        );

        assert_eq!(good, vec![0, 1]);
        assert!(poor.is_empty());
    }

    #[test]
    fn split_observation_indices_empty_input() {
        let values: Vec<Vec<f64>> = Vec::new();
        let (good, poor) =
            split_observation_indices_for_multi_objective(&values, &[], &[Direction::Minimize], 0);

        assert!(good.is_empty());
        assert!(poor.is_empty());
    }

    fn complete_trial(number: u32, values: Vec<f64>, constraint: Option<f64>) -> PersistedTrial {
        let mut trial = PersistedTrial::new(number, 0, number);
        trial.state_values = TrialStateValues::Complete(values);
        if let Some(c) = constraint {
            trial
                .attrs
                .insert(AttrKey::System("constraints:c0".into()), c.to_string());
        }
        trial
    }

    #[test]
    fn split_trials_matches_index_based_split() {
        // The mutually non-dominated feasible trials 0, 4, 5, and 6 form a rank-0
        // front larger than gamma, so the promising set is completed by
        // hypervolume subset selection rather than by whole-rank inclusion.
        let trials = [
            complete_trial(0, vec![1.0, 8.0], None),
            complete_trial(1, vec![f64::INFINITY, 1.0], Some(-1.0)),
            complete_trial(2, vec![2.0, 2.0], Some(0.5)),
            complete_trial(3, vec![3.0, 3.0], Some(2.0)),
            complete_trial(4, vec![4.0, 1.0], None),
            complete_trial(5, vec![0.5, 9.0], Some(-0.5)),
            complete_trial(6, vec![3.0, 2.0], None),
        ];
        let trial_refs: Vec<&PersistedTrial> = trials.iter().collect();
        let directions = [Direction::Minimize, Direction::Minimize];
        let gamma = 3;

        let (good_trials, poor_trials) =
            split_trials_for_multi_objective(&trial_refs, &directions, gamma).unwrap();

        let values: Vec<&[f64]> = trials
            .iter()
            .map(|t| match &t.state_values {
                TrialStateValues::Complete(v) => v.as_slice(),
                _ => unreachable!(),
            })
            .collect();
        let feasibles_violations: Vec<(bool, f64)> = trials
            .iter()
            .map(|t| {
                let constraints = t.constraints().unwrap();
                (
                    constraints.values().all(|x| *x <= 0.0),
                    constraints.values().filter(|&x| *x > 0.0).sum(),
                )
            })
            .collect();
        let (good_indices, poor_indices) = split_observation_indices_for_multi_objective(
            &values,
            &feasibles_violations,
            &directions,
            gamma,
        );

        let good_numbers: Vec<u32> = good_trials.iter().map(|t| t.number).collect();
        let poor_numbers: Vec<u32> = poor_trials.iter().map(|t| t.number).collect();
        let good_mapped: Vec<u32> = good_indices.iter().map(|&i| trials[i].number).collect();
        let poor_mapped: Vec<u32> = poor_indices.iter().map(|&i| trials[i].number).collect();
        assert_eq!(good_numbers, good_mapped);
        assert_eq!(poor_numbers, poor_mapped);
        assert_eq!(good_numbers.len(), gamma);
    }
}
