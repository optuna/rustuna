#![allow(non_local_definitions)]

use pyo3::prelude::*;

mod attrs;
mod distribution;
mod exception;
mod importance;
mod sampler;
mod storage;
mod study;
mod trial;
mod trial_queue;

/// A Python module implemented in Rust.
#[pymodule]
#[pyo3(name = "_rustuna")]
fn rustuna(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<attrs::AttrsDictView>()?;
    // trial
    m.add_class::<trial::PyTrial>()?;
    m.add_class::<trial::PyPersistedTrial>()?;
    m.add_class::<trial::PyTrialState>()?;
    m.add_function(wrap_pyfunction!(trial::py_create_trial, m)?)?;
    // study
    m.add_function(wrap_pyfunction!(study::py_create_study, m)?)?;
    m.add_function(wrap_pyfunction!(study::py_load_study, m)?)?;
    m.add_function(wrap_pyfunction!(study::py_copy_study, m)?)?;
    m.add_class::<study::PyStudy>()?;
    m.add_class::<study::PyDirection>()?;
    m.add_class::<study::PyPersistedStudy>()?;
    // distribution
    m.add_class::<distribution::PyDistribution>()?;
    // storage
    m.add_class::<storage::in_memory::PyInMemoryStorage>()?;
    m.add_class::<storage::cached::PyCachedStorage>()?;
    m.add_class::<storage::journal::PyJournalFileStorage>()?;
    m.add_class::<storage::sqlite3::PySQLite3Storage>()?;
    m.add_class::<storage::to_rust::PyToRustStorage>()?;
    // sampler
    m.add_class::<sampler::tpe::PyTpeSampler>()?;
    m.add_class::<sampler::nsgaii::PyNSGAIISampler>()?;
    m.add_class::<sampler::PySamplerContext>()?;
    m.add_class::<sampler::cmaes::PyCmaEsSampler>()?;
    m.add_class::<sampler::random::PyRandomSampler>()?;
    m.add_class::<sampler::qmc::PyQmcSampler>()?;
    // importance
    m.add_function(wrap_pyfunction!(importance::py_get_param_importances, m)?)?;
    m.add_class::<importance::PyPedAnovaImportanceEvaluator>()?;
    // trial_queue
    m.add_class::<trial_queue::directory::PyDirectoryTrialQueue>()?;
    m.add_class::<trial_queue::inmemory::PyInMemoryTrialQueue>()?;
    m.add_class::<trial_queue::sqlite3::PySQLite3TrialQueue>()?;
    Ok(())
}
