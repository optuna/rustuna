//! Sampler implementations for Rustuna.
//!
//! This crate provides concrete implementations of the `rustuna_core::sampler::Sampler` trait,
//! including TPE-based samplers, NSGA-II for multi-objective optimization, and a
//! quasi-Monte Carlo sampler behind the `qmc` feature.
//! This crate includes data that is licensed by third-party developers.
//! See LICENSE_THIRD_PARTY for details.

pub mod nsgaii;
#[cfg(feature = "qmc")]
pub mod qmc;
pub mod tpe;
