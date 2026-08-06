//! Exact hypervolume of a solution set against a reference point.
//!
//! Dispatches by number of points and number of objectives:
//!   * 0/1/2 points        → closed-form inclusion-exclusion
//!   * M = 3, any N        → [`compute_hypervolume_3d_sorted`] (O(N²) sweep)
//!   * M ≥ 4               → WFG-style recursion with a flat scratch buffer

use super::nds::{filter_pareto_front, lex_sort_indices};

/// Inclusive hypervolume of a single point against `reference_point`: ∏(ref[k] − row[k])
/// clamped to ≥ 0 (NaN treated as 0, so dominated coordinates do not invert the volume).
/// Shared between [`compute_hypervolume_with`] and the HSSP module.
pub(super) fn inclusive_hv(row: &[f64], reference_point: &[f64]) -> f64 {
    let mut prod = 1.0;
    for k in 0..reference_point.len() {
        let diff = reference_point[k] - row[k];
        let diff = if diff.is_nan() { 0.0 } else { diff };
        prod *= diff.max(0.0);
    }
    prod
}

pub fn compute_hypervolume<R>(loss_vals: &[R], reference_point: &[f64]) -> f64
where
    R: AsRef<[f64]>,
{
    compute_hypervolume_with(loss_vals, reference_point, false)
}

/// Variant of [`compute_hypervolume`] that skips the Pareto-front filter when callers
/// can guarantee the input is already non-dominated, saving an `O(N²M)` pass.
/// Crate-private because the precondition is only checked at call sites we control.
pub(super) fn compute_hypervolume_assume_pareto<R>(loss_vals: &[R], reference_point: &[f64]) -> f64
where
    R: AsRef<[f64]>,
{
    compute_hypervolume_with(loss_vals, reference_point, true)
}

fn compute_hypervolume_with<R>(loss_vals: &[R], reference_point: &[f64], assume_pareto: bool) -> f64
where
    R: AsRef<[f64]>,
{
    if loss_vals.is_empty() {
        return 0.0;
    }
    let n = loss_vals.len();
    let m = reference_point.len();

    assert!(m > 0, "reference_point must be non-empty");
    for (k, &v) in reference_point.iter().enumerate() {
        assert!(!v.is_nan(), "reference_point[{k}] is NaN");
    }

    for (i, r) in loss_vals.iter().enumerate() {
        let row = r.as_ref();
        assert!(
            row.len() == m,
            "dim mismatch: loss_vals[{}].len() == {}, expected {}",
            i,
            row.len(),
            m
        );
    }

    // Uniform NaN policy (see the module-level note): rows with any NaN coordinate
    // contribute no useful HV — they're treated as worse than every clean row and
    // dropped here. If nothing clean remains, HV is 0.
    let mut indices: Vec<usize> = (0..n)
        .filter(|&i| !loss_vals[i].as_ref().iter().any(|v| v.is_nan()))
        .collect();
    if indices.is_empty() {
        return 0.0;
    }
    if !assume_pareto {
        filter_pareto_front(loss_vals, &mut indices);
    }

    // Full lex sort (not just by f[0]) so that points tied on f[0] still get a
    // deterministic order. The downstream 3D sweep accumulates `row_sum` left-to-right
    // and FP non-associativity would otherwise let `sort_unstable_by` swap tied points
    // at runtime and change the reported HV by a ULP.
    lex_sort_indices(loss_vals, indices.as_mut_slice());

    // The M=3 fast path computes everything from scratch inside
    // `compute_hypervolume_3d_sorted`; `inclusive_hv` is only used by the small-N
    // shortcuts (1/2 points) and the M≥4 WFG recursion below.
    match indices.len() {
        0 => 0.0,
        1 => inclusive_hv(loss_vals[indices[0]].as_ref(), reference_point),
        2 => {
            let i = indices[0];
            let j = indices[1];
            let ri = loss_vals[i].as_ref();
            let rj = loss_vals[j].as_ref();
            let mut inter = 1.0;
            for k in 0..m {
                let maxval = ri[k].max(rj[k]);
                let diff = reference_point[k] - maxval;
                let diff = if diff.is_nan() { 0.0 } else { diff };
                inter *= diff.max(0.0);
            }
            inclusive_hv(ri, reference_point) + inclusive_hv(rj, reference_point) - inter
        }
        _ if m == 3 => {
            // Specialised O(N²) algorithm for M=3. The sweep assumes every kept point
            // lies within `reference_point`; otherwise their inflated cell widths
            // inflate the reported hypervolume. Drop ref-overshooting points up front
            // to match the clamping semantics of the WFG path's `inclusive_hv`.
            indices.retain(|&i| {
                let row = loss_vals[i].as_ref();
                row.iter().zip(reference_point.iter()).all(|(v, r)| v < r)
            });
            match indices.len() {
                0 => 0.0,
                1 => inclusive_hv(loss_vals[indices[0]].as_ref(), reference_point),
                2 => {
                    let i = indices[0];
                    let j = indices[1];
                    let ri = loss_vals[i].as_ref();
                    let rj = loss_vals[j].as_ref();
                    let mut inter = 1.0;
                    for k in 0..m {
                        let maxval = ri[k].max(rj[k]);
                        let diff = reference_point[k] - maxval;
                        let diff = if diff.is_nan() { 0.0 } else { diff };
                        inter *= diff.max(0.0);
                    }
                    inclusive_hv(ri, reference_point) + inclusive_hv(rj, reference_point) - inter
                }
                _ => compute_hypervolume_3d_sorted(loss_vals, &indices, reference_point),
            }
        }
        _ => {
            let mut total = 0.0;
            let len = indices.len();
            let max_rows = len - 1;
            // Flat (rows*m) scratch for the limited points, reused across i_pos.
            let mut limited_flat: Vec<f64> = Vec::with_capacity(max_rows * m);
            for (i_pos, &i_idx) in indices.iter().enumerate() {
                let inc_hv = inclusive_hv(loss_vals[i_idx].as_ref(), reference_point);
                let rows = len - (i_pos + 1);
                if rows == 0 {
                    total += inc_hv;
                    continue;
                }
                limited_flat.clear();
                let ri = loss_vals[i_idx].as_ref();
                for j_rel in 0..rows {
                    let j_idx = indices[i_pos + 1 + j_rel];
                    let rj = loss_vals[j_idx].as_ref();
                    for k in 0..m {
                        limited_flat.push(ri[k].max(rj[k]));
                    }
                }
                let limited_refs: Vec<&[f64]> = limited_flat.chunks_exact(m).collect();
                total += compute_exclusive_hypervolume(&limited_refs, inc_hv, reference_point);
            }
            total
        }
    }
}

/// O(N²) 3-objective hypervolume.
///
/// Cell `(i, j)` of an implicit `N × N` matrix would normally hold `ref_z − z_p` for
/// the unique point `p` having x-rank `i` and y-rank `j`. A 2D max-accumulate
/// (down-then-right) turns each cell into the maximum `ref_z − z` reachable by any
/// point with smaller x-rank and y-rank, so the dominated volume is
/// `Σ_{i,j} dx[i] · dy[j] · z_height[i, j]`. The matrix is streamed row-by-row to
/// keep memory at O(N) instead of O(N²).
///
/// `pareto_indices` must already be sorted by `loss_vals[idx][0]` ascending and
/// contain only Pareto-optimal points.
fn compute_hypervolume_3d_sorted<R>(
    loss_vals: &[R],
    pareto_indices: &[usize],
    reference_point: &[f64],
) -> f64
where
    R: AsRef<[f64]>,
{
    let n = pareto_indices.len();
    if n == 0 {
        return 0.0;
    }
    let row = |k: usize| -> &[f64] { loss_vals[pareto_indices[k]].as_ref() };

    // Permutation that maps x-rank → y-rank.
    let mut y_order: Vec<usize> = (0..n).collect();
    y_order.sort_unstable_by(|&a, &b| row(a)[1].total_cmp(&row(b)[1]));
    let mut y_rank_of: Vec<usize> = vec![0; n];
    for (yr, &xr) in y_order.iter().enumerate() {
        y_rank_of[xr] = yr;
    }

    // y-axis cell widths.
    let mut dy: Vec<f64> = Vec::with_capacity(n);
    for k in 0..n - 1 {
        let next = row(y_order[k + 1])[1];
        let curr = row(y_order[k])[1];
        dy.push((next - curr).max(0.0));
    }
    dy.push((reference_point[1] - row(y_order[n - 1])[1]).max(0.0));

    // Stream the 2D max-prefix row by row.
    let mut prev_row: Vec<f64> = vec![0.0; n];
    let mut curr_row: Vec<f64> = vec![0.0; n];

    let mut total = 0.0;
    for (i, &inject_at) in y_rank_of.iter().enumerate() {
        let injected = {
            let diff = reference_point[2] - row(i)[2];
            if diff.is_nan() {
                0.0
            } else {
                diff.max(0.0)
            }
        };

        // curr_row[j] = max(prev_row[j], curr_row[j-1], original[i, j]). The "original"
        // is non-zero only at column `inject_at` for this row.
        let mut running_left = 0.0_f64;
        for j in 0..n {
            let from_top = prev_row[j];
            let from_self = if j == inject_at { injected } else { 0.0 };
            let cell = from_top.max(running_left).max(from_self);
            curr_row[j] = cell;
            running_left = cell;
        }

        // Accumulate this row's contribution: dx[i] · Σ_j dy[j] · curr_row[j].
        let dx_i = if i + 1 < n {
            (row(i + 1)[0] - row(i)[0]).max(0.0)
        } else {
            (reference_point[0] - row(i)[0]).max(0.0)
        };
        let mut row_sum = 0.0;
        for j in 0..n {
            row_sum += dy[j] * curr_row[j];
        }
        total += dx_i * row_sum;

        std::mem::swap(&mut prev_row, &mut curr_row);
    }
    total
}

fn compute_exclusive_hypervolume(
    limited_sols: &[&[f64]],
    inclusive_hv: f64,
    reference_point: &[f64],
) -> f64 {
    if limited_sols.is_empty() {
        inclusive_hv
    } else {
        inclusive_hv - compute_hypervolume(limited_sols, reference_point)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_hypervolume_float_simple() {
        let loss_vals = [vec![1.0, 1.0], vec![2.0, 0.5]];
        let refs: Vec<&[f64]> = loss_vals.iter().map(|v| v.as_slice()).collect();
        let reference_point = vec![3.0, 3.0];

        let hv = compute_hypervolume(&refs, &reference_point);
        assert!(hv > 0.0);
    }

    #[test]
    fn test_compute_hypervolume() {
        let loss_vals = [vec![1.0, 2.0], vec![2.0, 1.5], vec![1.5, 1.0]];
        let refs: Vec<&[f64]> = loss_vals.iter().map(|v| v.as_slice()).collect();
        let reference_point = vec![3.0, 3.0];

        let hv = compute_hypervolume(&refs, &reference_point);
        // Manually computed expected hypervolume
        let expected_hv = 3.5;
        assert!(
            (hv - expected_hv).abs() < 1e-6,
            "hv: {hv}, expected: {expected_hv}"
        );
    }

    #[test]
    fn compute_hypervolume_3d_single_point() {
        // Closed form for a single point: ∏ (ref[k] − p[k]).
        let pts = [vec![1.0, 2.0, 0.5]];
        let refs: Vec<&[f64]> = pts.iter().map(|v| v.as_slice()).collect();
        let ref_point = vec![3.0, 3.0, 3.0];
        let hv = compute_hypervolume(&refs, &ref_point);
        let expected = (3.0 - 1.0) * (3.0 - 2.0) * (3.0 - 0.5);
        assert!(
            (hv - expected).abs() < 1e-9,
            "hv {hv} vs expected {expected}"
        );
    }

    #[test]
    fn compute_hypervolume_3d_two_disjoint_boxes() {
        // Two non-overlapping (in (x,y)) boxes share the same z and don't overlap each
        // other's dominated region. HV = sum of inclusive volumes.
        let pts = [vec![0.0, 2.0, 0.0], vec![2.0, 0.0, 0.0]];
        let refs: Vec<&[f64]> = pts.iter().map(|v| v.as_slice()).collect();
        let ref_point = vec![3.0, 3.0, 3.0];
        let hv = compute_hypervolume(&refs, &ref_point);
        let inc_a = (3.0 - 0.0) * (3.0 - 2.0) * (3.0 - 0.0);
        let inc_b = (3.0 - 2.0) * (3.0 - 0.0) * (3.0 - 0.0);
        let inter = (3.0 - 2.0) * (3.0 - 2.0) * (3.0 - 0.0);
        let expected = inc_a + inc_b - inter;
        assert!(
            (hv - expected).abs() < 1e-9,
            "hv {hv} vs expected {expected}"
        );
    }

    #[test]
    fn compute_hypervolume_3d_matches_inclusion_exclusion_for_two_points() {
        // Two arbitrary points — the 3D path should agree with the closed-form
        // 2-point inclusion-exclusion the 2D arm uses.
        let pts = [vec![0.5, 1.0, 1.5], vec![1.0, 0.5, 0.8]];
        let refs: Vec<&[f64]> = pts.iter().map(|v| v.as_slice()).collect();
        let ref_point = vec![3.0, 3.0, 3.0];
        let hv = compute_hypervolume(&refs, &ref_point);

        let mut inc = 0.0;
        for p in pts.iter() {
            inc += (ref_point[0] - p[0]) * (ref_point[1] - p[1]) * (ref_point[2] - p[2]);
        }
        let max = [
            pts[0][0].max(pts[1][0]),
            pts[0][1].max(pts[1][1]),
            pts[0][2].max(pts[1][2]),
        ];
        let inter = (ref_point[0] - max[0]) * (ref_point[1] - max[1]) * (ref_point[2] - max[2]);
        let expected = inc - inter;
        assert!(
            (hv - expected).abs() < 1e-9,
            "hv {hv} vs expected {expected}"
        );
    }

    #[test]
    fn compute_hypervolume_excludes_nan_rows_and_uses_the_clean_subset() {
        // NaN policy: NaN rows are dropped, HV is computed over the clean subset.
        let pts_nan = [vec![1.0, f64::NAN, 0.5], vec![0.5, 1.0, 1.0]];
        let refs_nan: Vec<&[f64]> = pts_nan.iter().map(|v| v.as_slice()).collect();
        let ref_point = vec![3.0, 3.0, 3.0];
        let hv_nan = compute_hypervolume(&refs_nan, &ref_point);

        // The clean subset is just `[[0.5, 1.0, 1.0]]`; its HV must match.
        let pts_clean = [vec![0.5, 1.0, 1.0]];
        let refs_clean: Vec<&[f64]> = pts_clean.iter().map(|v| v.as_slice()).collect();
        let hv_clean = compute_hypervolume(&refs_clean, &ref_point);

        assert!((hv_nan - hv_clean).abs() < 1e-9);
        assert!(hv_nan > 0.0, "clean subset is non-empty so HV must be > 0");
    }

    #[test]
    fn compute_hypervolume_returns_zero_when_all_rows_have_nan() {
        let pts = [vec![1.0, f64::NAN, 0.5], vec![f64::NAN, 0.5, 0.5]];
        let refs: Vec<&[f64]> = pts.iter().map(|v| v.as_slice()).collect();
        let ref_point = vec![3.0, 3.0, 3.0];
        assert_eq!(compute_hypervolume(&refs, &ref_point), 0.0);
    }

    #[test]
    fn compute_hypervolume_3d_clamps_points_outside_reference() {
        // M=3 fast path must drop points with coords >= reference (otherwise their
        // inflated cell widths over-count). Verify the result matches the
        // closed-form on the in-reference subset.
        let pts = [
            vec![0.0, 0.0, 1.0],
            vec![0.0, 1.0, 0.0],
            vec![3.0, 0.0, 0.0],
        ];
        let refs: Vec<&[f64]> = pts.iter().map(|v| v.as_slice()).collect();
        let ref_point = vec![2.0, 2.0, 2.0];
        let hv = compute_hypervolume(&refs, &ref_point);
        // Inclusion-exclusion over the two within-reference points {[0,0,1], [0,1,0]}:
        let inc_a = (2.0 - 0.0) * (2.0 - 0.0) * (2.0 - 1.0); // [0,0,1]
        let inc_b = (2.0 - 0.0) * (2.0 - 1.0) * (2.0 - 0.0); // [0,1,0]
        let inter = (2.0 - 0.0) * (2.0 - 1.0) * (2.0 - 1.0); // max-corner [0,1,1]
        let expected = inc_a + inc_b - inter; // 4 + 4 - 2 = 6
        assert!(
            (hv - expected).abs() < 1e-9,
            "hv {hv} vs expected {expected}"
        );
    }

    #[test]
    fn assume_pareto_matches_unfiltered_when_input_is_pareto() {
        // For an input that already is a Pareto front, the assume_pareto fast path
        // must produce the same answer as the filter-then-compute path.
        let pts = [vec![0.0, 2.0], vec![1.0, 1.0], vec![2.0, 0.0]];
        let refs: Vec<&[f64]> = pts.iter().map(|v| v.as_slice()).collect();
        let ref_point = vec![3.0, 3.0];
        let hv_filtered = compute_hypervolume(&refs, &ref_point);
        let hv_assumed = compute_hypervolume_assume_pareto(&refs, &ref_point);
        assert!(
            (hv_filtered - hv_assumed).abs() < 1e-9,
            "filtered {hv_filtered} vs assume_pareto {hv_assumed}"
        );
    }
}
