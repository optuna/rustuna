//! Non-dominated sorting and Pareto-front filtering.

use std::cmp::Ordering;

/// Sort `indices` in place so that `loss_values[indices[k]]` is non-decreasing under
/// lexicographic order. Uses `f64::total_cmp`, which is a total order even in the
/// presence of NaN / signed zero — callers that need to react to NaN should do an
/// explicit pre-check via [`loss_values_have_nan`].
pub(super) fn lex_sort_indices<R: AsRef<[f64]>>(loss_values: &[R], indices: &mut [usize]) {
    if indices.is_empty() {
        return;
    }
    let m = loss_values[indices[0]].as_ref().len();
    indices.sort_unstable_by(|&a, &b| {
        let ra = loss_values[a].as_ref();
        let rb = loss_values[b].as_ref();
        for k in 0..m {
            match ra[k].total_cmp(&rb[k]) {
                Ordering::Equal => continue,
                ord => return ord,
            }
        }
        Ordering::Equal
    });
}

/// O(N×M) scan that reports whether any cell of any row contains NaN. Used as a guard
/// before running dominance / hypervolume code that doesn't define behaviour on NaN.
pub(super) fn loss_values_have_nan<R: AsRef<[f64]>>(loss_values: &[R]) -> bool {
    loss_values
        .iter()
        .any(|r| r.as_ref().iter().any(|v| v.is_nan()))
}

/// Assigns non-domination ranks (0 for Pareto-optimal, 1 for the next layer, …) but
/// stops once at least `n_below` points have been ranked. Unranked points receive a
/// sentinel rank larger than every emitted rank, so callers that only need the top-
/// `n_below` layers (typical when feeding TPE's good-vs-poor split) get correct
/// results without paying for the long tail. Passing `n_below = loss_values.len()`
/// yields a full ranking.
pub fn fast_non_dominated_sort_partial<R>(loss_values: &[R], n_below: usize) -> Vec<usize>
where
    R: AsRef<[f64]>,
{
    let n = loss_values.len();
    let mut ranks: Vec<usize> = vec![usize::MAX; n];
    if n == 0 {
        return ranks;
    }
    let n_below = n_below.min(n);

    // Uniform shape precondition: every row must have the same M as row 0. Both fast
    // paths below index by row index without per-row length checks, so a mixed-length
    // input would panic OOB deep inside the algorithm instead of failing here cleanly.
    let m = loss_values[0].as_ref().len();
    for (i, r) in loss_values.iter().enumerate() {
        assert_eq!(
            r.as_ref().len(),
            m,
            "fast_non_dominated_sort_partial: row {i} has length {} but row 0 has length {m}",
            r.as_ref().len()
        );
    }

    // NaN policy: dominance is ill-defined when any coordinate is NaN. Treat NaN
    // trials as *worse* than any clean trial so they end up in `poor_trials` rather
    // than poisoning the good/poor split (rank 0 means Pareto-optimal — assigning
    // NaN trials there would feed garbage into the Parzen estimator).
    //
    // Implementation: recurse on the clean subset, then assign every NaN-containing
    // row a sentinel rank one larger than the maximum rank emitted for clean rows.
    if loss_values_have_nan(loss_values) {
        let mut clean: Vec<usize> = Vec::with_capacity(n);
        let mut nan_rows: Vec<usize> = Vec::new();
        for (i, r) in loss_values.iter().enumerate() {
            if r.as_ref().iter().any(|v| v.is_nan()) {
                nan_rows.push(i);
            } else {
                clean.push(i);
            }
        }
        let clean_rows: Vec<&[f64]> = clean.iter().map(|&i| loss_values[i].as_ref()).collect();
        let sub_ranks = fast_non_dominated_sort_partial(&clean_rows, n_below);
        let sentinel = sub_ranks.iter().copied().max().map(|r| r + 1).unwrap_or(0);
        for (sub_i, &orig) in clean.iter().enumerate() {
            ranks[orig] = sub_ranks[sub_i];
        }
        for &orig in &nan_rows {
            ranks[orig] = sentinel;
        }
        return ranks;
    }

    // Fast O(N log N) path for the bi-objective case (M==2). Uses a patience-sort-style
    // sweep over points sorted by (f1 asc, f2 asc) — this already produces every layer
    // in linearithmic time, so the `n_below` hint isn't needed.
    //
    // Exact duplicates (same (f1, f2)) do *not* dominate each other and must share a
    // rank. lex-sorted order makes them adjacent, so we just carry forward the previous
    // point's rank when we see an exact match instead of letting the binary search push
    // the duplicate to the next front.
    if loss_values[0].as_ref().len() == 2 {
        let mut order: Vec<usize> = (0..n).collect();
        lex_sort_indices(loss_values, &mut order);
        let mut min_f2_per_rank: Vec<f64> = Vec::new();
        let mut prev_loss: Option<(f64, f64)> = None;
        let mut prev_rank: usize = 0;
        for &i in &order {
            let row = loss_values[i].as_ref();
            let x = row[0];
            let y = row[1];
            if prev_loss == Some((x, y)) {
                ranks[i] = prev_rank;
                continue;
            }
            let mut lo = 0usize;
            let mut hi = min_f2_per_rank.len();
            while lo < hi {
                let mid = (lo + hi) / 2;
                if min_f2_per_rank[mid] > y {
                    hi = mid;
                } else {
                    lo = mid + 1;
                }
            }
            if lo == min_f2_per_rank.len() {
                min_f2_per_rank.push(y);
            } else {
                min_f2_per_rank[lo] = y;
            }
            ranks[i] = lo;
            prev_loss = Some((x, y));
            prev_rank = lo;
        }
        return ranks;
    }

    // General-M peel-onion. One lex-sort up front; each layer extracts the Pareto front
    // of the still-unranked remainder via Kung's-style sweep. Stops when `n_below` points
    // have been assigned a rank.
    let mut lex_order: Vec<usize> = (0..n).collect();
    lex_sort_indices(loss_values, &mut lex_order);

    // `remaining` keeps lex order across layers; we only ever shrink it.
    let mut remaining: Vec<usize> = lex_order;
    let mut current_rank: usize = 0;
    let mut n_ranked: usize = 0;
    // Reusable scratch buffers — `is_pareto_front_lex_sweep` and the peel-onion driver
    // both need O(N) bool / usize storage that shrinks layer by layer; preallocating
    // here and passing through avoids `vec![…; n]` churn per layer.
    let mut on_front: Vec<bool> = Vec::new();
    let mut status: Vec<bool> = Vec::new();
    let mut next: Vec<usize> = Vec::new();

    while !remaining.is_empty() && n_ranked < n_below {
        is_pareto_front_lex_sweep(loss_values, &remaining, &mut on_front, &mut status);
        next.clear();
        for (i, &orig_idx) in remaining.iter().enumerate() {
            if on_front[i] {
                ranks[orig_idx] = current_rank;
                n_ranked += 1;
            } else {
                next.push(orig_idx);
            }
        }
        std::mem::swap(&mut remaining, &mut next);
        current_rank += 1;
    }

    // Sentinel rank for every point we did not need to rank precisely. Strictly greater
    // than any emitted rank, so callers iterating ranks 0, 1, … in order never visit it
    // before they have already pulled `n_below` points out.
    for &orig_idx in &remaining {
        ranks[orig_idx] = current_rank;
    }

    debug_assert!(
        ranks.iter().all(|&r| r != usize::MAX),
        "Some ranks were not assigned"
    );
    ranks
}

/// Mark Pareto-front membership over already lex-sorted `indices`. After the sweep,
/// `on_front[i] == true` iff `loss_values[indices[i]]` is on the front.
///
/// `drop_duplicates=true` collapses exact-duplicate points to a single front
/// representative — HV computation receives unique inputs.
///
/// `drop_duplicates=false` keeps exact duplicates on the same front — NDS layers
/// where ties must share a rank.
///
/// `status` is scratch reused across calls — its prior contents are discarded.
fn lex_pareto_mark<R>(
    loss_values: &[R],
    indices: &[usize],
    on_front: &mut Vec<bool>,
    status: &mut Vec<bool>,
    drop_duplicates: bool,
) where
    R: AsRef<[f64]>,
{
    let n = indices.len();
    on_front.clear();
    on_front.resize(n, false);
    if n == 0 {
        return;
    }
    let m = loss_values[indices[0]].as_ref().len();
    // status[i] starts true ("still in the active set"). It is flipped to false once
    // some earlier (lex-smaller) front member dominates point i.
    status.clear();
    status.resize(n, true);
    for top_pos in 0..n {
        if !status[top_pos] {
            continue;
        }
        on_front[top_pos] = true;
        let top = loss_values[indices[top_pos]].as_ref();
        for i in (top_pos + 1)..n {
            if !status[i] {
                continue;
            }
            let pt = loss_values[indices[i]].as_ref();
            // Track `not_dom` (some pt[k] < top[k] → top can't dominate) and
            // `top_strict` (some top[k] < pt[k] → strictness witness). The drop
            // condition depends on the `drop_duplicates` policy:
            //   * drop_duplicates=false → only strict dominance ((!not_dom) && top_strict)
            //   * drop_duplicates=true  → weak dominance counts too (!not_dom alone),
            //     which collapses `pt == top` into the existing front entry.
            let mut not_dom = false;
            let mut top_strict = false;
            for k in 0..m {
                if pt[k] < top[k] {
                    not_dom = true;
                    break;
                }
                if pt[k] > top[k] {
                    top_strict = true;
                }
            }
            let dropped = if drop_duplicates {
                !not_dom
            } else {
                !not_dom && top_strict
            };
            if dropped {
                status[i] = false;
            }
        }
    }
}

/// Kung's-style Pareto-front extraction for a single NDS layer. Exact duplicates
/// share the rank; see [`lex_pareto_mark`] for the underlying sweep.
fn is_pareto_front_lex_sweep<R>(
    loss_values: &[R],
    remaining: &[usize],
    on_front: &mut Vec<bool>,
    status: &mut Vec<bool>,
) where
    R: AsRef<[f64]>,
{
    lex_pareto_mark(loss_values, remaining, on_front, status, false);
}

/// In-place reduce `indices` to those whose loss vectors form the Pareto front.
/// Exact duplicates are collapsed (one representative kept) so downstream HV code
/// receives unique inputs.
///
/// NaN policy: rows containing NaN are treated as worse than any clean row and are
/// dropped from `indices` — the Pareto front is the set of *best* points, and a
/// failed-evaluation row should never be reported as "best".
pub(super) fn filter_pareto_front<R>(loss_values: &[R], indices: &mut Vec<usize>)
where
    R: AsRef<[f64]>,
{
    if indices.is_empty() {
        return;
    }

    indices.retain(|&i| !loss_values[i].as_ref().iter().any(|v| v.is_nan()));
    if indices.is_empty() {
        return;
    }

    lex_sort_indices(loss_values, indices.as_mut_slice());

    let mut on_front: Vec<bool> = Vec::new();
    let mut status: Vec<bool> = Vec::new();
    lex_pareto_mark(
        loss_values,
        indices.as_slice(),
        &mut on_front,
        &mut status,
        true,
    );
    let mut write = 0usize;
    for i in 0..indices.len() {
        if on_front[i] {
            indices[write] = indices[i];
            write += 1;
        }
    }
    indices.truncate(write);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fast_non_dominated_sort_basic() {
        // simple 2D example
        let pts = [
            vec![0.0, 0.0], // 0 -> front 0
            vec![1.0, 0.0], // 1 dominated by 0 -> front 1
            vec![0.0, 1.0], // 2 dominated by 0 -> front 1
            vec![2.0, 2.0], // dominated by both -> front 2
            vec![0.5, 0.5], // non-dominated w.r.t 1 & 2 -> front 1
        ];
        let refs: Vec<&[f64]> = pts.iter().map(|r| r.as_slice()).collect();
        let ranks = fast_non_dominated_sort_partial(&refs, refs.len());
        assert_eq!(ranks[0], 0);
        assert!(ranks[1] > 0);
        assert!(ranks[2] > 0);
        assert!(ranks[3] > ranks[1] && ranks[3] > ranks[2]);
    }

    #[test]
    fn partial_sort_with_small_n_below_returns_sentinel_for_excess_points() {
        // 3D ranks for 6 points: pts 0, 1, 2 should land in earlier fronts; with
        // n_below = 2 we only need front 0, so the remaining points get a sentinel
        // rank strictly greater than every emitted rank.
        let pts = [
            vec![0.0, 0.0, 0.0], // front 0
            vec![1.0, 0.0, 0.0], // dominated by 0 → front 1
            vec![0.0, 1.0, 0.0], // dominated by 0 → front 1
            vec![1.0, 1.0, 0.0], // dominated by all earlier → front 2
            vec![2.0, 2.0, 2.0], // dominated by all → front 3
            vec![1.5, 1.5, 1.5], // dominated by all → front 3
        ];
        let refs: Vec<&[f64]> = pts.iter().map(|r| r.as_slice()).collect();
        let ranks = fast_non_dominated_sort_partial(&refs, 2);

        assert_eq!(ranks[0], 0, "front-0 point must be ranked precisely");
        let sentinel = ranks.iter().copied().max().unwrap();
        assert!(
            sentinel > 0,
            "sentinel must exceed every precise rank emitted"
        );
        // Every non-front-0 point should be either ranked precisely OR carry the
        // sentinel — never `usize::MAX`.
        assert!(ranks.iter().all(|&r| r != usize::MAX));
        // n_below = 2 means we stop after at least 2 points are ranked. Since front
        // 0 holds 1 point, we keep peeling and front 1 (2 points) is fully assigned.
        // Whatever is left (≥1 point) carries the sentinel rank.
        let precise_count = ranks.iter().filter(|&&r| r < sentinel).count();
        assert!(
            precise_count >= 2,
            "must rank at least n_below = 2 points precisely (got {precise_count})"
        );
    }

    #[test]
    fn partial_sort_with_n_below_equal_to_n_assigns_every_rank_precisely() {
        let pts = [
            vec![0.0, 0.0, 0.0],
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![1.0, 1.0, 0.0],
        ];
        let refs: Vec<&[f64]> = pts.iter().map(|r| r.as_slice()).collect();
        let ranks_partial = fast_non_dominated_sort_partial(&refs, refs.len());
        // With n_below = N the function must reproduce a complete ranking — every
        // entry should be a "real" rank, never the sentinel-bigger-than-max.
        assert!(ranks_partial.iter().all(|&r| r != usize::MAX));
        assert_eq!(ranks_partial[0], 0);
        assert!(ranks_partial[3] > ranks_partial[1] && ranks_partial[3] > ranks_partial[2]);
    }

    #[test]
    fn duplicate_loss_vectors_share_rank_2d() {
        // Two identical (0, 0) points must share rank 0 since neither dominates the
        // other; (1, 1) is then rank 1. The patience-sort fast path must carry the
        // previous point's rank forward on exact duplicates.
        let pts = [vec![0.0, 0.0], vec![0.0, 0.0], vec![1.0, 1.0]];
        let refs: Vec<&[f64]> = pts.iter().map(|r| r.as_slice()).collect();
        let ranks = fast_non_dominated_sort_partial(&refs, refs.len());
        assert_eq!(ranks, vec![0, 0, 1]);
    }

    #[test]
    fn duplicate_loss_vectors_share_rank_3d() {
        // Same invariant in the general-M peel-onion: exact duplicates must share a
        // rank, so the strict-dominance check has to be enforced inside Kung's sweep.
        let pts = [
            vec![0.0, 0.0, 0.0],
            vec![0.0, 0.0, 0.0],
            vec![1.0, 1.0, 1.0],
        ];
        let refs: Vec<&[f64]> = pts.iter().map(|r| r.as_slice()).collect();
        let ranks = fast_non_dominated_sort_partial(&refs, refs.len());
        assert_eq!(ranks, vec![0, 0, 1]);
    }

    #[test]
    #[should_panic(expected = "row 1 has length 1 but row 0 has length 2")]
    fn fast_nds_partial_panics_clearly_on_ragged_input() {
        // Mixed-length rows must be rejected up front with a clear message rather than
        // OOB-panicking deep inside the patience sweep.
        let pts: Vec<Vec<f64>> = vec![vec![0.0, 1.0], vec![0.5]];
        let refs: Vec<&[f64]> = pts.iter().map(|v| v.as_slice()).collect();
        let _ = fast_non_dominated_sort_partial(&refs, refs.len());
    }

    #[test]
    fn nan_rows_get_sentinel_rank_worse_than_clean_rows() {
        // NaN policy: a NaN-containing trial must be ranked *worse* than any clean
        // trial. Rank 0 (Pareto-optimal) is reserved for clean rows.
        let pts = [
            vec![0.0, 0.0],
            vec![1.0, f64::NAN], // NaN → sentinel rank
            vec![0.5, 0.5],
            vec![1.0, 1.0],
        ];
        let refs: Vec<&[f64]> = pts.iter().map(|r| r.as_slice()).collect();
        let ranks = fast_non_dominated_sort_partial(&refs, refs.len());
        assert_eq!(ranks[0], 0, "clean dominator is rank 0");
        let clean_max = ranks[0].max(ranks[2]).max(ranks[3]);
        assert!(
            ranks[1] > clean_max,
            "NaN row {} must be strictly worse than any clean row (max clean = {})",
            ranks[1],
            clean_max
        );
    }

    #[test]
    fn filter_pareto_front_drops_nan_rows() {
        // NaN policy: rows with NaN are never on the Pareto front; they're dropped.
        let pts = [
            vec![1.0, 2.0],
            vec![f64::NAN, 0.5], // dropped
            vec![0.5, 0.5],      // dominates the others
            vec![3.0, 3.0],      // dominated by index 2
        ];
        let refs: Vec<&[f64]> = pts.iter().map(|r| r.as_slice()).collect();
        let mut indices: Vec<usize> = (0..refs.len()).collect();
        filter_pareto_front(&refs, &mut indices);
        assert!(!indices.contains(&1), "NaN row must be dropped");
        assert!(indices.contains(&2), "Pareto-optimal row must be kept");
        assert!(!indices.contains(&3), "dominated row must be dropped");
    }

    #[test]
    fn filter_pareto_front_keeps_only_non_dominated() {
        let pts = [
            vec![1.0, 2.0],
            vec![2.0, 1.0],
            vec![3.0, 3.0], // dominated by 0 and 1
            vec![0.5, 4.0], // non-dominated (smaller in f0 than 0)
            vec![1.0, 2.0], // duplicate of 0 → dropped by the dedup-aware sweep
        ];
        let refs: Vec<&[f64]> = pts.iter().map(|r| r.as_slice()).collect();
        let mut indices: Vec<usize> = (0..refs.len()).collect();
        filter_pareto_front(&refs, &mut indices);

        // Pareto front of this input is {0, 1, 3}. The duplicate at index 4 should be
        // dropped; whichever of 0 or 4 is kept depends on the lex sort, but their
        // coordinates are identical so the resulting `indices` is still a valid
        // Pareto front.
        let kept: std::collections::HashSet<_> = indices.iter().copied().collect();
        assert_eq!(
            kept.len(),
            indices.len(),
            "filter_pareto_front must not produce duplicates"
        );
        assert!(kept.contains(&1));
        assert!(kept.contains(&3));
        assert!(!kept.contains(&2), "dominated points must be removed");
        // Exactly one of the duplicates (0 or 4) survives.
        assert_eq!(
            usize::from(kept.contains(&0)) + usize::from(kept.contains(&4)),
            1
        );
    }
}
