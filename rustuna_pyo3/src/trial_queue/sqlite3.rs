use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use rustuna_core::trial_queue::TrialQueue;
use rustuna_core::ErrorKind;
use rustuna_storage::sqlite3_queue::SQLite3TrialQueue;
use std::sync::Arc;

use crate::exception::err_to_exceptions;

#[derive(Clone)]
#[pyclass(name = "SQLite3TrialQueue", from_py_object)]
#[pyo3(module = "rustuna")]
pub struct PySQLite3TrialQueue {
    pub queue: Arc<SQLite3TrialQueue>,
}

#[pymethods]
impl PySQLite3TrialQueue {
    #[new]
    fn py_new(py: Python<'_>, db_path: String, namespace: String) -> PyResult<Self> {
        let queue = py
            .detach(|| SQLite3TrialQueue::new(db_path, namespace))
            .map_err(|error| {
                PyRuntimeError::new_err(format!(
                    "Failed to create SQLite3TrialQueue: {}",
                    error.reason
                ))
            })?;
        Ok(Self {
            queue: Arc::new(queue),
        })
    }

    fn enqueue(&self, py: Python<'_>, trial_id: u32) -> PyResult<()> {
        py.detach(|| self.queue.enqueue(trial_id).map_err(err_to_exceptions))
    }

    fn dequeue(&self, py: Python<'_>) -> PyResult<Option<u32>> {
        py.detach(|| match self.queue.dequeue() {
            Ok(trial_id) => Ok(Some(trial_id)),
            Err(error) if matches!(error.kind, ErrorKind::TrialQueueEmpty) => Ok(None),
            Err(error) => Err(err_to_exceptions(error)),
        })
    }
}
