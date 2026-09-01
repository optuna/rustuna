//! Storage and queue backends for Rustuna.
//!
//! This crate provides concrete persistence layers and trial-queue implementations built on top
//! of `rustuna_core`, including caching wrappers, SQLite-backed storage, journal-based storage,
//! and shared queue backends used by `Study::enqueue_trial`. These are concrete backends for the
//! `rustuna_core::storage::Storage` and `rustuna_core::trial_queue::TrialQueue` traits.

pub mod cache;
pub mod directory_queue;
pub mod journal;
pub mod sqlite3;
pub mod sqlite3_queue;

mod datetime;

#[cfg(test)]
mod test_utils;
