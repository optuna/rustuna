//! Sampler implementations for Rustuna.
//!
//! This crate provides concrete implementations of the `rustuna_core::sampler::Sampler` trait,
//! including TPE-based samplers, NSGA-II for multi-objective optimization, and a
//! quasi-Monte Carlo sampler built on the Sobol' sequence.

pub mod nsgaii;
pub mod qmc;
pub mod tpe;
