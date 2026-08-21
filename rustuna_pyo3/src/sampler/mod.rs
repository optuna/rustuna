use std::sync::{Arc, RwLock};

use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::Py;

use rustuna_core::storage::Storage;

use crate::storage::cached::PyCachedStorage;
use crate::storage::in_memory::PyInMemoryStorage;
use crate::storage::journal::PyJournalFileStorage;
use crate::storage::to_rust::PyToRustStorage;

pub mod cmaes;
mod context;
pub mod nsgaii;
pub mod qmc;
pub mod random;
pub mod to_rust;
pub mod tpe;
pub use context::PySamplerContext;

fn extract_storage(storage: Py<PyAny>) -> PyResult<Arc<RwLock<dyn Storage>>> {
    Python::attach(|py| {
        let storage_ref = storage.bind(py);
        if let Ok(py_inmemory_storage) = storage_ref.extract::<PyInMemoryStorage>() {
            Ok(py_inmemory_storage.storage())
        } else if let Ok(py_cached_storage) = storage_ref.extract::<PyCachedStorage>() {
            Ok(py_cached_storage.storage())
        } else if let Ok(py_journal_storage) = storage_ref.extract::<PyJournalFileStorage>() {
            Ok(py_journal_storage.storage())
        } else if let Ok(py_obj_storage) = storage_ref.extract::<PyToRustStorage>() {
            Ok(py_obj_storage.storage.clone() as Arc<RwLock<dyn Storage>>)
        } else {
            Err(PyRuntimeError::new_err(
                "Invalid storage type. Use rustuna.Storage for Rust-native storages or rustuna.PyObjectStorage for Python StorageProtocol implementations.",
            ))
        }
    })
}
