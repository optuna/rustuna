//! Primitives for multi-objective optimization.
//!
//! Split into three concerns:
//!
//! * [`nds`] — non-dominated sorting with optional `n_below` early termination and the
//!   single-rank `filter_pareto_front` used by `compute_hypervolume`. Shared helpers
//!   (`lex_sort_indices`, `loss_values_have_nan`) live here.
//! * [`hypervolume`] — exact hypervolume of a set of solutions, with closed-form
//!   M=2 / O(N²) M=3 / WFG-recursive M≥4 paths.
//! * [`hssp`] — greedy hypervolume subset selection problem solver with submodular
//!   upper-bound skip and an LRU(size 1) cache.
//!
//! Only a minimal set of items is re-exported from this module; implementation
//! details stay crate-private so refactors remain local.
//!
//! ## NaN policy
//!
//! Dominance, hypervolume, and HSSP contributions are all ill-defined when an
//! objective value is NaN. Across this module **NaN rows are treated as worse than
//! every clean row** — rank 0 / the Pareto front / "best in subset" must never be
//! occupied by a failed-evaluation trial, otherwise downstream algorithms would
//! consume invalid objective values. Every public entry point follows the same rule:
//!
//! | Function                              | NaN handling                                       |
//! |---------------------------------------|----------------------------------------------------|
//! | `fast_non_dominated_sort_partial`     | NaN rows assigned a sentinel rank one larger than the maximum rank emitted for clean rows |
//! | `filter_pareto_front`                 | NaN rows dropped from `indices` (never on the front) |
//! | `compute_hypervolume`                 | NaN rows excluded from the HV computation (all NaN → 0.0) |
//! | `hypervolume_subset_selection`        | greedy runs on the clean subset; remaining picks pad with NaN rows in input order |
//!
//! A NaN reference point is treated as an unrecoverable input (`compute_hypervolume`
//! returns 0.0 / HSSP falls back to input order) because there's no meaningful
//! "clean subset" to recurse on. Callers can still filter NaN trials upstream if a
//! tighter contract is needed.

mod hssp;
mod hypervolume;
mod nds;
mod split;

pub use hssp::hypervolume_subset_selection;
pub use nds::fast_non_dominated_sort_partial;
pub use split::{
    split_feasible_observation_indices, split_observation_indices_for_multi_objective,
    split_trials_for_multi_objective,
};
