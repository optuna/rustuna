//! Core components for Rustuna.
//!
//! This crate provides the central study, trial, storage, sampler, queue, and distribution
//! components used across Rustuna. Concrete storage backends, sampler implementations, and
//! language bindings are implemented in other crates in the workspace.

pub use error::{Error, ErrorKind};

pub mod attr;
pub mod distribution;
pub mod sampler;
pub mod storage;
pub mod study;
pub mod transform;
pub mod trial;
pub mod trial_queue;

mod datetime;
mod error;
mod multi_objective;
mod parzen_estimator;
mod string_interner;
mod study_cache;

/// Implementation details shared by Rustuna crates.
///
/// This module is not part of Rustuna's stable public API. Items in this module
/// are not covered by Rustuna's semantic-versioning guarantees and may be
/// changed or removed in any release without a major version bump.
#[doc(hidden)]
pub mod internal {
    pub mod parzen_estimator {
        pub use crate::parzen_estimator::*;
    }
    pub mod study_cache {
        pub use crate::study_cache::StudyCache;
    }
    pub mod datetime {
        pub use crate::datetime::now_naive_utc;
    }
    pub mod multi_objective {
        pub use crate::multi_objective::*;
    }
}

/// A crate-specific [`std::result::Result`] alias.
pub type Result<T, E = Error> = std::result::Result<T, E>;
