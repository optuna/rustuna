use std::collections::{HashMap, HashSet};

use crate::multi_objective::{hssp, nds};
use crate::study::Direction;
use crate::trial::{PersistedTrial, TrialStateValues};

const EPS: f64 = 1e-12;

pub fn split_trials_for_multi_objective<'a>(
    trials: &[&'a PersistedTrial],
    directions: &[Direction],
    gamma: usize,
) -> (Vec<&'a PersistedTrial>, Vec<&'a PersistedTrial>) {
    let n = trials.len();
    assert!(
        gamma <= n,
        "gamma must be less than or equal to the number of trials"
    );

    if n == 0 {
        return (Vec::new(), Vec::new());
    }
    if gamma == n {
        return (trials.to_vec(), Vec::new());
    }

    // Assume minimization (negate value if maximization)
    let loss_values: Vec<Vec<f64>> = trials
        .iter()
        .map(|t| {
            let vals = match &t.state_values {
                TrialStateValues::Complete(v) => v.as_slice(),
                _ => panic!("Unexpected non-complete trial found during TPE sampling"),
            };
            vals.iter()
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

    let mut good_trials = Vec::with_capacity(gamma);
    let mut poor_trials = Vec::with_capacity(n - gamma);

    let mut current_rank = 0usize;
    while good_trials.len() + rank_to_indices.get(&current_rank).map_or(0, |v| v.len()) <= gamma {
        if let Some(indices) = rank_to_indices.get(&current_rank) {
            for &i in indices.iter() {
                good_trials.push(trials[i]);
            }
        }
        current_rank += 1;
    }
    let hss_subset_size = gamma - good_trials.len();
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
            good_trials.push(trials[rank_i_indices[local]]);
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
                    good_trials.push(trials[i]);
                }
                remaining = remaining.saturating_sub(take);
            } else {
                // No finite candidates contributed to some dimension's worst
                // (only possible when the finite subset is empty for that dim,
                // which the outer guard already excluded). Defensive fallback.
                let take = remaining.min(finite_local.len());
                for &local in finite_local.iter().take(take) {
                    good_trials.push(trials[rank_i_indices[local]]);
                }
                remaining = remaining.saturating_sub(take);
            }
        }

        for &local in pos_inf_local.iter().take(remaining) {
            good_trials.push(trials[rank_i_indices[local]]);
        }
    }
    let good_numbers: HashSet<u32> = good_trials.iter().map(|t| t.number).collect();
    for &trial in trials.iter() {
        if !good_numbers.contains(&trial.number) {
            poor_trials.push(trial);
        }
    }

    (good_trials, poor_trials)
}
