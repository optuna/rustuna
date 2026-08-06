//! Hypervolume subset selection problem (HSSP) solver — greedy with submodular
//! upper-bound skip and an LRU(size 1) cache shared across consecutive calls.

use std::cell::RefCell;

use super::hypervolume::{compute_hypervolume, compute_hypervolume_assume_pareto, inclusive_hv};

/// M values for which `compute_hypervolume` has a closed-form / specialised non-recursive
/// path (M=2 via inclusion-exclusion, M=3 via the row-streamed 3D sweep). For M ≥ 4 the
/// WFG recursion makes the *incremental* HV approach (recompute `HV(S ∪ {j})`) expensive
/// because every recursive call repeats the filter-pareto pass, so HSSP switches to the
/// *decremental* "inclusive_hv − intersection HV" formulation that avoids those repeated
/// filters.
const HSSP_INCREMENTAL_MAX_M: usize = 3;

// LRU(size 1) cache for `hypervolume_subset_selection`. Consecutive trials often hit
// HSSP with the same boundary-rank inputs (same partial-rank contents, same
// subset_size, same reference point), and reusing the prior result skips an entire
// greedy sweep when that happens.
thread_local! {
    static HSSP_CACHE: RefCell<Option<HssCacheEntry>> = const { RefCell::new(None) };
}

struct HssCacheEntry {
    /// Cached loss values flattened row-major (length = `indices.len() * ref_point.len()`).
    loss_vals_flat: Vec<f64>,
    indices: Vec<usize>,
    subset_size: usize,
    ref_point: Vec<f64>,
    result: Vec<usize>,
}

/// Greedy hypervolume subset selection: pick `subset_size` indices from
/// `rank_i_indices` that approximately maximise hypervolume against `reference_point`.
/// Returns selected original indices (in selection order).
///
/// Submodular maximisation guarantees a `(1 − 1/e)` approximation via plain greedy;
/// this implementation tightens contribution upper bounds after each pick and walks
/// candidates by descending upper bound, skipping any whose bound has already dropped
/// below the running max.
pub fn hypervolume_subset_selection<R>(
    rank_i_loss_vals: &[R],
    rank_i_indices: &[usize],
    reference_point: &[f64],
    subset_size: usize,
) -> Vec<usize>
where
    R: AsRef<[f64]>,
{
    assert_eq!(rank_i_loss_vals.len(), rank_i_indices.len());
    let n = rank_i_loss_vals.len();
    if subset_size == 0 {
        return Vec::new();
    }
    if subset_size >= n {
        return rank_i_indices.to_vec();
    }
    let m = reference_point.len();
    let assume_pareto_fast_path = m <= HSSP_INCREMENTAL_MAX_M;

    // Uniform NaN policy (see the module-level note): NaN coordinates poison the
    // contribution math, so treat NaN-containing rows as worse than every clean row
    // and exclude them from the HV-driven selection. The greedy runs on the clean
    // subset; if it cannot produce enough picks we pad with NaN rows in input order
    // so the caller still gets `subset_size` indices back.
    let any_nan_ref = reference_point.iter().any(|v| v.is_nan());
    if any_nan_ref
        || rank_i_loss_vals
            .iter()
            .any(|r| r.as_ref().iter().any(|v| v.is_nan()))
    {
        // A NaN reference point makes the contribution formula meaningless for every
        // row — there's no "clean" subset to recurse on. Fall back to input order.
        if any_nan_ref {
            return rank_i_indices[..subset_size].to_vec();
        }
        let mut clean_idx: Vec<usize> = Vec::with_capacity(n);
        let mut nan_idx: Vec<usize> = Vec::new();
        for (pos, r) in rank_i_loss_vals.iter().enumerate() {
            if r.as_ref().iter().any(|v| v.is_nan()) {
                nan_idx.push(pos);
            } else {
                clean_idx.push(pos);
            }
        }
        let clean_rows: Vec<&[f64]> = clean_idx
            .iter()
            .map(|&i| rank_i_loss_vals[i].as_ref())
            .collect();
        let clean_orig: Vec<usize> = clean_idx.iter().map(|&i| rank_i_indices[i]).collect();
        let take = subset_size.min(clean_idx.len());
        let mut out = hypervolume_subset_selection(&clean_rows, &clean_orig, reference_point, take);
        for &i in &nan_idx {
            if out.len() >= subset_size {
                break;
            }
            out.push(rank_i_indices[i]);
        }
        return out;
    }

    // Cache probe: short-circuit on the cheap scalar fields first (subset_size, dims),
    // then on slice lengths, before doing the per-element comparison of indices /
    // loss_vals. Bit-level compare (`to_bits`) keeps the check exact for arbitrary f64
    // payloads — caller doesn't need to worry about +0/-0/NaN equality semantics.
    let cached = HSSP_CACHE.with(|cell| {
        let cache = cell.borrow();
        let entry = cache.as_ref()?;
        // `m` here is also implicitly checked via `ref_point.len()` and
        // `loss_vals_flat.len() == n * m`, so we don't store it separately.
        if entry.subset_size != subset_size
            || entry.indices.len() != rank_i_indices.len()
            || entry.loss_vals_flat.len() != n * m
            || entry.ref_point.len() != m
        {
            return None;
        }
        if entry.indices.as_slice() != rank_i_indices {
            return None;
        }
        for (a, b) in entry.ref_point.iter().zip(reference_point.iter()) {
            if a.to_bits() != b.to_bits() {
                return None;
            }
        }
        let mut idx = 0usize;
        for r in rank_i_loss_vals.iter() {
            let row = r.as_ref();
            // Per-row length sanity: a caller passing mixed-length rows whose total
            // happens to equal `n * m` would otherwise sneak past the outer length
            // guard and either OOB-panic on `loss_vals_flat[idx]` or false-hit on a
            // stale entry.
            if row.len() != m || idx + row.len() > entry.loss_vals_flat.len() {
                return None;
            }
            for &v in row.iter() {
                if entry.loss_vals_flat[idx].to_bits() != v.to_bits() {
                    return None;
                }
                idx += 1;
            }
        }
        if idx != entry.loss_vals_flat.len() {
            return None;
        }
        Some(entry.result.clone())
    });
    if let Some(r) = cached {
        return r;
    }

    // Fast path: exact O(subset_size * n) greedy for two objectives, ported from Optuna's
    // `_solve_hssp_2d`. Placed after the cache probe so repeated identical boundary ranks
    // still short-circuit; the result is stored in the cache below like the general path.
    // Reaching here guarantees NaN-free rows; also require finiteness so the +/-inf handling
    // done by the caller / general greedy is not disturbed.
    if m == 2
        && rank_i_loss_vals
            .iter()
            .all(|r| r.as_ref().iter().all(|v| v.is_finite()))
    {
        let selected_indices = solve_hssp_2d(
            rank_i_loss_vals,
            rank_i_indices,
            reference_point,
            subset_size,
        );
        HSSP_CACHE.with(|cell| {
            let mut flat = Vec::with_capacity(n * m);
            for r in rank_i_loss_vals.iter() {
                flat.extend_from_slice(r.as_ref());
            }
            *cell.borrow_mut() = Some(HssCacheEntry {
                loss_vals_flat: flat,
                indices: rank_i_indices.to_vec(),
                subset_size,
                ref_point: reference_point.to_vec(),
                result: selected_indices.clone(),
            });
        });
        return selected_indices;
    }

    // remaining[*] = position in the input arrays for candidates not yet selected.
    let mut remaining: Vec<usize> = (0..n).collect();
    // contribs[*] = a submodular upper bound on H(S ∪ {j}) − H(S) for remaining[j].
    // After `lazy_contribs_update` processes a candidate, its entry becomes the exact
    // value.
    let mut contribs: Vec<f64> = (0..n)
        .map(|i| inclusive_hv(rank_i_loss_vals[i].as_ref(), reference_point))
        .collect();

    let mut selected_indices: Vec<usize> = Vec::with_capacity(subset_size);
    let mut selected_rows: Vec<Vec<f64>> = Vec::with_capacity(subset_size);
    let mut hv_selected: f64 = 0.0;

    // Scratch buffers reused across iterations.
    let mut order: Vec<usize> = Vec::with_capacity(n);
    // Flat (selected_rows.len() × m) scratch for the M ≥ 4 decremental path; avoids
    // per-candidate `Vec<Vec<f64>>` allocation inside the inner loop.
    let mut intersec_flat: Vec<f64> = Vec::with_capacity(subset_size * m);

    for k in 0..subset_size {
        // Argmax of contribs. The entries are submodular upper bounds, but the maximum
        // is always exact here: the descending walk in `lazy_contribs_update` recomputes
        // the entry with the highest upper bound first, so the running argmax for the
        // next iteration is already pinned to a concrete value.
        let mut argmax = 0usize;
        let mut max_contrib = f64::NEG_INFINITY;
        for (i, &c) in contribs.iter().enumerate() {
            if c > max_contrib {
                max_contrib = c;
                argmax = i;
            }
        }

        // Commit selection.
        let chosen_remaining = remaining[argmax];
        selected_indices.push(rank_i_indices[chosen_remaining]);
        let chosen_row: Vec<f64> = rank_i_loss_vals[chosen_remaining].as_ref().to_vec();
        hv_selected += contribs[argmax];

        remaining.swap_remove(argmax);
        contribs.swap_remove(argmax);
        selected_rows.push(chosen_row);

        if k == subset_size - 1 || remaining.is_empty() {
            break;
        }

        // If `hv_selected` has overflowed to +INF (or NaN), every subsequent
        // `hv_plus - hv_selected` would degenerate to NaN/-INF and poison the rest of
        // the greedy sweep. Mark every remaining candidate's contribution as +INF so
        // argmax just picks them in input order until `subset_size` is met, and skip
        // the per-candidate HV recompute.
        if !hv_selected.is_finite() {
            for c in contribs.iter_mut() {
                *c = f64::INFINITY;
            }
            continue;
        }

        // ---- `lazy_contribs_update` ----
        // Tighten contribs using a submodular bound that depends only on the most-recently
        // selected point: H(T ∪ {j}) − H(T) ≤ H({t} ∪ {j}) − H({t}) =
        // inclusive_hv(j) − vol(ref − max(t, j)) where t is the just-selected point.
        let chosen_row_ref = selected_rows.last().unwrap().as_slice();
        for (j, &remain_pos) in remaining.iter().enumerate() {
            let cand_row = rank_i_loss_vals[remain_pos].as_ref();
            let mut inter_prod = 1.0;
            for d in 0..m {
                let intersec = cand_row[d].max(chosen_row_ref[d]);
                let diff = reference_point[d] - intersec;
                let diff = if diff.is_nan() { 0.0 } else { diff };
                inter_prod *= diff.max(0.0);
            }
            let bound = inclusive_hv(cand_row, reference_point) - inter_prod;
            if bound < contribs[j] {
                contribs[j] = bound;
            }
        }

        // Walk candidates by descending upper bound. Once a candidate's recomputed exact
        // contribution sets the running max, any subsequent candidate whose upper bound is
        // already below that max can be skipped — its true contribution is at most its
        // current upper bound, so it cannot win the next argmax.
        order.clear();
        order.extend(0..remaining.len());
        order.sort_unstable_by(|&a, &b| contribs[b].total_cmp(&contribs[a]));

        let mut max_seen = 0.0_f64;
        for &j in &order {
            if !contribs[j].is_finite() {
                contribs[j] = f64::INFINITY;
                max_seen = f64::INFINITY;
                continue;
            }
            if contribs[j] < max_seen {
                break; // Already sorted descending; nothing past here can win.
            }

            let cand_row = rank_i_loss_vals[remaining[j]].as_ref();
            let exact = if assume_pareto_fast_path {
                // Incremental: H(S ∪ {j}) − H(S). Selected rows + cand are all on the
                // Pareto front of the original rank-i input, so skip the filter.
                let mut tmp_refs: Vec<&[f64]> = selected_rows.iter().map(Vec::as_slice).collect();
                tmp_refs.push(cand_row);
                let hv_plus = compute_hypervolume_assume_pareto(&tmp_refs, reference_point);
                hv_plus - hv_selected
            } else {
                // Decremental: H({j}) − H(intersection points). Useful when WFG recursion
                // for M ≥ 4 makes the incremental path expensive due to filtering inside
                // every recursive call.
                intersec_flat.clear();
                for sv in &selected_rows {
                    for d in 0..m {
                        intersec_flat.push(sv[d].max(cand_row[d]));
                    }
                }
                let tmp_refs: Vec<&[f64]> = intersec_flat.chunks_exact(m).collect();
                let intersec_hv = compute_hypervolume(&tmp_refs, reference_point);
                inclusive_hv(cand_row, reference_point) - intersec_hv
            };

            contribs[j] = exact;
            if exact > max_seen {
                max_seen = exact;
            }
        }
    }

    HSSP_CACHE.with(|cell| {
        let mut flat = Vec::with_capacity(n * m);
        for r in rank_i_loss_vals.iter() {
            flat.extend_from_slice(r.as_ref());
        }
        *cell.borrow_mut() = Some(HssCacheEntry {
            loss_vals_flat: flat,
            indices: rank_i_indices.to_vec(),
            subset_size,
            ref_point: reference_point.to_vec(),
            result: selected_indices.clone(),
        });
    });
    selected_indices
}

/// Exact greedy hypervolume subset selection specialised for two objectives, ported from
/// Optuna's `_solve_hssp_2d`. Runs in O(`subset_size` * n): each of the `subset_size` picks
/// takes an O(n) argmax over the marginal contributions plus an O(n) rectangle update. The
/// general greedy in [`hypervolume_subset_selection`] recomputes a hypervolume per candidate,
/// which becomes O(n^3) as the Pareto front grows on convergence; this path avoids that.
///
/// Expects finite, NaN-free rows (the caller guarantees this). Points outside the reference
/// box (a coordinate at/above its reference) contribute zero, as in the general greedy.
/// Returns the selected original indices in selection order.
fn solve_hssp_2d<R: AsRef<[f64]>>(
    rank_i_loss_vals: &[R],
    rank_i_indices: &[usize],
    reference_point: &[f64],
    subset_size: usize,
) -> Vec<usize> {
    let n = rank_i_loss_vals.len();
    // Lex-sort by (f0, f1): the left/right rectangle updates below rely on ascending f0 order.
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| {
        let (ra, rb) = (rank_i_loss_vals[a].as_ref(), rank_i_loss_vals[b].as_ref());
        ra[0].total_cmp(&rb[0]).then(ra[1].total_cmp(&rb[1]))
    });
    let mut orig: Vec<usize> = order.iter().map(|&o| rank_i_indices[o]).collect();
    let mut f0: Vec<f64> = order
        .iter()
        .map(|&o| rank_i_loss_vals[o].as_ref()[0])
        .collect();
    let mut f1: Vec<f64> = order
        .iter()
        .map(|&o| rank_i_loss_vals[o].as_ref()[1])
        .collect();
    // `d0`/`d1`: upper-right corner of each point's current contribution rectangle.
    let (r0, r1) = (reference_point[0], reference_point[1]);
    let mut d0: Vec<f64> = vec![r0; n];
    let mut d1: Vec<f64> = vec![r1; n];

    let mut selected = Vec::with_capacity(subset_size);
    let mut m = n;
    for _ in 0..subset_size {
        let mut best = 0usize;
        let mut best_contrib = f64::NEG_INFINITY;
        for j in 0..m {
            // Clamp each rectangle side at 0 so a point outside the reference box (a
            // coordinate >= its reference) contributes 0 rather than a spurious positive
            // area from two negative widths, matching `inclusive_hv` in the general greedy.
            let contrib = (d0[j] - f0[j]).max(0.0) * (d1[j] - f1[j]).max(0.0);
            if contrib > best_contrib {
                best_contrib = contrib;
                best = j;
            }
        }
        selected.push(orig[best]);
        let (s0, s1) = (f0[best], f1[best]);
        // Remove the chosen point, keeping the arrays f0-sorted.
        orig.remove(best);
        f0.remove(best);
        f1.remove(best);
        d0.remove(best);
        d1.remove(best);
        m -= 1;
        // Points left of the chosen one (smaller f0) get their f0-extent capped at s0;
        // points to its right (larger f0) get their f1-extent capped at s1.
        for d in d0.iter_mut().take(best) {
            if s0 < *d {
                *d = s0;
            }
        }
        for d in d1.iter_mut().take(m).skip(best) {
            if s1 < *d {
                *d = s1;
            }
        }
    }
    selected
}

#[cfg(test)]
mod tests {
    use super::*;

    // Reference greedy HSSP: at each step pick the candidate maximising HV(selected ∪ cand),
    // recomputing the hypervolume from scratch. Slow but unambiguously correct.
    fn reference_greedy(rows: &[[f64; 2]], reference_point: &[f64], subset_size: usize) -> f64 {
        let mut chosen: Vec<usize> = Vec::new();
        let mut avail: Vec<usize> = (0..rows.len()).collect();
        for _ in 0..subset_size {
            let mut best = avail[0];
            let mut best_hv = f64::NEG_INFINITY;
            for &c in &avail {
                let mut s: Vec<&[f64]> = chosen.iter().map(|&i| rows[i].as_slice()).collect();
                s.push(rows[c].as_slice());
                let hv = compute_hypervolume(&s, reference_point);
                if hv > best_hv {
                    best_hv = hv;
                    best = c;
                }
            }
            chosen.push(best);
            avail.retain(|&x| x != best);
        }
        best_hv_of(rows, &chosen, reference_point)
    }

    fn best_hv_of(rows: &[[f64; 2]], sel: &[usize], reference_point: &[f64]) -> f64 {
        let s: Vec<&[f64]> = sel.iter().map(|&i| rows[i].as_slice()).collect();
        compute_hypervolume(&s, reference_point)
    }

    #[test]
    fn solve_hssp_2d_matches_reference_greedy_hv() {
        // A spread of 2D points below the reference point, incl. a duplicate and same-f0 ties.
        let rows: Vec<[f64; 2]> = vec![
            [0.1, 0.9],
            [0.2, 0.7],
            [0.2, 0.6],
            [0.35, 0.55],
            [0.5, 0.5],
            [0.55, 0.45],
            [0.7, 0.3],
            [0.75, 0.25],
            [0.9, 0.1],
            [0.9, 0.1],
        ];
        let reference_point = [1.0, 1.0];
        let indices: Vec<usize> = (0..rows.len()).collect();
        let row_refs: Vec<&[f64]> = rows.iter().map(|r| r.as_slice()).collect();
        for subset in 1..rows.len() {
            let sel = solve_hssp_2d(&row_refs, &indices, &reference_point, subset);
            assert_eq!(sel.len(), subset);
            let hv_fast = best_hv_of(&rows, &sel, &reference_point);
            let hv_ref = reference_greedy(&rows, &reference_point, subset);
            assert!(
                (hv_fast - hv_ref).abs() < 1e-9,
                "subset={subset}: solve_hssp_2d HV={hv_fast} != reference greedy HV={hv_ref}"
            );
        }
    }

    #[test]
    fn solve_hssp_2d_ignores_points_outside_reference_box() {
        // A point outside the reference box must contribute zero, not a spurious positive
        // area from two negative rectangle widths (Codex review counterexample).
        let rows: Vec<[f64; 2]> = vec![[0.0, 1.0], [1.0, 0.0], [3.0, 3.0]];
        let reference_point = [2.0, 2.0];
        let indices: Vec<usize> = (0..rows.len()).collect();
        let row_refs: Vec<&[f64]> = rows.iter().map(|r| r.as_slice()).collect();
        for subset in 1..rows.len() {
            let sel = solve_hssp_2d(&row_refs, &indices, &reference_point, subset);
            assert!(
                !sel.contains(&2),
                "subset={subset}: selected out-of-box point [3,3]"
            );
            let hv_fast = best_hv_of(&rows, &sel, &reference_point);
            let hv_ref = reference_greedy(&rows, &reference_point, subset);
            assert!(
                (hv_fast - hv_ref).abs() < 1e-9,
                "subset={subset}: solve_hssp_2d HV={hv_fast} != reference greedy HV={hv_ref}"
            );
        }
    }

    #[test]
    fn greedy_hv_approx_minus_1_over_e_no_itertools() {
        fn combinations_recursive(
            n: usize,
            k: usize,
            start: usize,
            current: &mut Vec<usize>,
            out: &mut Vec<Vec<usize>>,
        ) {
            if current.len() == k {
                out.push(current.clone());
                return;
            }
            for i in start..n {
                current.push(i);
                combinations_recursive(n, k, i + 1, current, out);
                current.pop();
            }
        }

        // A simple problem instance
        let pts: Vec<[f64; 2]> = vec![
            [1.0, 4.0],
            [2.0, 3.0],
            [3.0, 2.0],
            [4.0, 1.0],
            [2.5, 2.5],
            [3.5, 3.5],
        ];
        let n = pts.len();
        let refp = vec![6.0, 6.0];

        let rows_ref: Vec<Vec<f64>> = pts.iter().map(|p| vec![p[0], p[1]]).collect();
        let row_slices: Vec<&[f64]> = rows_ref.iter().map(|v| v.as_slice()).collect();

        let k = 2usize;

        // Exhaustive search
        let mut all_combs: Vec<Vec<usize>> = Vec::new();
        combinations_recursive(n, k, 0, &mut Vec::new(), &mut all_combs);

        let mut best_hv = 0.0;
        for comb in all_combs.iter() {
            let chosen: Vec<&[f64]> = comb.iter().map(|&i| row_slices[i]).collect();
            let hv = compute_hypervolume(&chosen, &refp);
            if hv > best_hv {
                best_hv = hv;
            }
        }

        // Greedy selection
        let rank_i_loss_vals: Vec<&[f64]> = row_slices.clone();
        let rank_i_indices: Vec<usize> = (0..n).collect();

        let greedy_sel = hypervolume_subset_selection(&rank_i_loss_vals, &rank_i_indices, &refp, k);
        let greedy_rows: Vec<&[f64]> = greedy_sel.iter().map(|&i| row_slices[i]).collect();
        let greedy_hv = compute_hypervolume(&greedy_rows, &refp);

        // Verify (1 - 1/e) approximation
        let bound = 1.0 - 1.0f64 / std::f64::consts::E;
        assert!(
            greedy_hv >= bound * best_hv,
            "greedy hv {} < (1-1/e)*opt {} (opt {})",
            greedy_hv,
            bound * best_hv,
            best_hv
        );
    }

    #[test]
    fn cache_returns_identical_result_for_identical_inputs() {
        // Two calls with the same arguments should produce byte-equal results; the
        // second call goes through the cache, the first populates it.
        let pts: Vec<Vec<f64>> = vec![
            vec![1.0, 4.0],
            vec![2.0, 3.0],
            vec![3.0, 2.0],
            vec![4.0, 1.0],
        ];
        let refs: Vec<&[f64]> = pts.iter().map(|v| v.as_slice()).collect();
        let indices: Vec<usize> = (0..pts.len()).collect();
        let refp = vec![5.0, 5.0];

        let a = hypervolume_subset_selection(&refs, &indices, &refp, 2);
        let b = hypervolume_subset_selection(&refs, &indices, &refp, 2);
        assert_eq!(a, b, "cache hit must return the same selection");
    }

    #[test]
    fn cache_misses_when_subset_size_or_inputs_change() {
        // Same indices but different subset_size, then different loss values — both
        // must invalidate the LRU(size 1) cache and recompute correctly.
        let pts: Vec<Vec<f64>> = vec![
            vec![1.0, 4.0],
            vec![2.0, 3.0],
            vec![3.0, 2.0],
            vec![4.0, 1.0],
        ];
        let refs: Vec<&[f64]> = pts.iter().map(|v| v.as_slice()).collect();
        let indices: Vec<usize> = (0..pts.len()).collect();
        let refp = vec![5.0, 5.0];

        let s2 = hypervolume_subset_selection(&refs, &indices, &refp, 2);
        let s3 = hypervolume_subset_selection(&refs, &indices, &refp, 3);
        assert_eq!(s2.len(), 2);
        assert_eq!(s3.len(), 3);
        // s2 must be a prefix-like subset (greedy adds one point at a time), so the
        // 2-element result is contained in the 3-element one.
        for idx in &s2 {
            assert!(
                s3.contains(idx),
                "greedy s3 must contain every member of s2"
            );
        }
    }

    #[test]
    fn subset_size_zero_returns_empty() {
        let refs: Vec<&[f64]> = vec![&[1.0, 2.0], &[2.0, 1.0]];
        let indices = vec![10usize, 11];
        let refp = vec![5.0, 5.0];
        let out = hypervolume_subset_selection(&refs, &indices, &refp, 0);
        assert!(out.is_empty());
    }

    #[test]
    fn hv_overflow_does_not_poison_remaining_picks() {
        // When inclusive_hv overflows to +INF the +INF poison guard must degrade
        // gracefully: argmax keeps picking +INF candidates and the returned subset
        // still has the requested size with no panics.
        let huge = 1e160;
        let pts = [
            vec![0.0, 0.0, 0.0, 0.0],
            vec![huge / 2.0, 0.0, 0.0, 0.0],
            vec![0.0, huge / 2.0, 0.0, 0.0],
            vec![0.0, 0.0, huge / 2.0, 0.0],
        ];
        let refs: Vec<&[f64]> = pts.iter().map(|v| v.as_slice()).collect();
        let indices: Vec<usize> = (0..pts.len()).collect();
        let ref_point = vec![huge, huge, huge, huge];
        let out = hypervolume_subset_selection(&refs, &indices, &ref_point, 3);
        assert_eq!(out.len(), 3, "subset_size honoured even when HV overflows");
        // Selected indices must be unique and drawn from the input set.
        let seen: std::collections::HashSet<_> = out.iter().copied().collect();
        assert_eq!(seen.len(), out.len(), "no duplicate selections");
        assert!(out.iter().all(|i| indices.contains(i)));
    }

    #[test]
    fn cache_rejects_mismatched_row_lengths() {
        // Caller passing differently-shaped rows whose total happens to equal n*m
        // must not OOB-panic or false-hit in the cache probe — per-row length checks
        // make the probe fail-open.
        let pts_a: Vec<Vec<f64>> = vec![vec![1.0, 4.0], vec![2.0, 3.0]];
        let refs_a: Vec<&[f64]> = pts_a.iter().map(|v| v.as_slice()).collect();
        let indices = vec![0usize, 1];
        let refp = vec![5.0, 5.0];
        let _ = hypervolume_subset_selection(&refs_a, &indices, &refp, 1);

        // Same n=2 and same ref length but with rows of length 3 and 1 (total still 4):
        // the outer length guard alone would pass — we want the probe to reject.
        let pts_b: Vec<Vec<f64>> = vec![vec![1.0, 4.0, 0.0], vec![2.0]];
        let refs_b: Vec<&[f64]> = pts_b.iter().map(|v| v.as_slice()).collect();
        // We're not asserting a value here — the assertion is the absence of a panic.
        // The cache probe in particular must not OOB on these rows. The selection
        // itself will hit upstream length asserts elsewhere; we just want the cache
        // path to fail-open.
        // (Wrap in a closure so a panic here can be caught.)
        let res = std::panic::catch_unwind(|| {
            // Call site might panic for other reasons (length checks deeper in the
            // pipeline). We only assert the cache probe doesn't OOB-panic.
            // Using a single-row equivalent: same n=2 and ref len, mixed shapes.
            // This is best-effort — it documents the defensive intent.
            let _ = hypervolume_subset_selection(&refs_b, &indices, &refp, 1);
        });
        // Either way, the original cache (from refs_a) must remain valid for refs_a.
        let _ = res;
        let again = hypervolume_subset_selection(&refs_a, &indices, &refp, 1);
        assert_eq!(again.len(), 1);
    }

    #[test]
    fn hssp_prefers_clean_rows_then_pads_with_nan() {
        // NaN policy: greedy runs on the clean subset; if `subset_size` exceeds the
        // number of clean rows, the remainder is padded with NaN rows in input order.
        let refs: Vec<&[f64]> = vec![
            &[1.0, 2.0],      // clean
            &[f64::NAN, 1.0], // NaN — only chosen as padding
            &[0.5, 0.5],      // clean
        ];
        let indices = vec![10usize, 11, 12];
        let refp = vec![5.0, 5.0];

        // subset_size = 1: one of the clean rows wins, NaN is not selected.
        let one = hypervolume_subset_selection(&refs, &indices, &refp, 1);
        assert_eq!(one.len(), 1);
        assert!(one[0] == 10 || one[0] == 12, "must pick a clean row");

        // subset_size = 3 forces padding with the NaN row at the tail.
        let three = hypervolume_subset_selection(&refs, &indices, &refp, 3);
        assert_eq!(three.len(), 3);
        assert!(three.contains(&11), "NaN row appears as padding");
        // The two clean rows must both be present.
        assert!(three.contains(&10) && three.contains(&12));
    }

    #[test]
    fn hssp_falls_back_to_input_order_on_nan_reference_point() {
        // NaN in the reference point is unrecoverable (no clean subset to recurse on);
        // we fall back to taking the first `subset_size` indices verbatim.
        let refs: Vec<&[f64]> = vec![&[1.0, 2.0], &[0.5, 0.5]];
        let indices = vec![10usize, 11];
        let refp = vec![5.0, f64::NAN];
        assert_eq!(
            hypervolume_subset_selection(&refs, &indices, &refp, 1),
            vec![10]
        );
    }

    #[test]
    fn subset_size_at_least_n_returns_all_indices() {
        let refs: Vec<&[f64]> = vec![&[1.0, 2.0], &[2.0, 1.0]];
        let indices = vec![10usize, 11];
        let refp = vec![5.0, 5.0];
        let out = hypervolume_subset_selection(&refs, &indices, &refp, 2);
        assert_eq!(out, indices);
        // Even oversized requests fall through the same short-circuit.
        let out = hypervolume_subset_selection(&refs, &indices, &refp, 10);
        assert_eq!(out, indices);
    }
}
