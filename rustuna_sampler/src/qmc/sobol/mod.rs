//! Sobol' low-discrepancy sequence generator.
//!
//! This module reproduces `scipy.stats.qmc.Sobol(d, scramble=False)` exactly: the same Joe-Kuo
//! direction numbers, the same Antonov-Saleev Gray code construction, and the origin at index 0.
//!
//! # Balance properties
//!
//! Sobol' points are a quadrature rule, not independent samples. They form a `(t, m, d)`-net only
//! when the `2^m` points at indices `0..2^m` are taken together, so a non-power-of-two count,
//! skipping index 0, or thinning the sequence all degrade their uniformity.
//!
//! # Examples
//!
//! ```
//! use rustuna_core::Result;
//! use rustuna_sampler::qmc::sobol;
//!
//! fn main() -> Result<()> {
//!     assert_eq!(sobol::nth_point(2, 0)?, vec![0.0, 0.0]);
//!     assert_eq!(sobol::nth_point(2, 1)?, vec![0.5, 0.5]);
//!     assert_eq!(sobol::nth_point(2, 2)?, vec![0.75, 0.25]);
//!     assert_eq!(sobol::nth_point(2, 3)?, vec![0.25, 0.75]);
//!     Ok(())
//! }
//! ```

mod direction_numbers;
mod engine;

pub use engine::nth_point;
