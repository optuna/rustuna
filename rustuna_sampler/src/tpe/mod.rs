//! Tree-structured Parzen Estimator (TPE) samplers.
//!
//! This module provides Rustuna's TPE implementation for both single-objective and
//! multi-objective optimization.

mod sampler;

pub use sampler::{TpeConfig, TpeSampler};
