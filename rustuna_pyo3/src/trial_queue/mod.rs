use std::sync::Arc;

use pyo3::prelude::*;

use crate::trial_queue::directory::PyDirectoryTrialQueue;
use crate::trial_queue::inmemory::PyInMemoryTrialQueue;
use crate::trial_queue::sqlite3::PySQLite3TrialQueue;
use crate::trial_queue::to_rust::ToRustTrialQueue;

pub mod directory;
pub mod inmemory;
pub mod sqlite3;
pub(crate) mod to_python;
pub mod to_rust;

pub(crate) type SharedTrialQueue = Arc<dyn rustuna_core::trial_queue::TrialQueue>;

/// Resolves a Python trial queue into a Rust trial queue and the Python object that owns it.
///
/// Built-in queues share their Rust queue directly, other objects are wrapped in
/// `ToRustTrialQueue`, and `None` creates a new in-memory queue.
pub(crate) fn into_trial_queue_pyobj(
    py: Python<'_>,
    trial_queue: Option<Py<PyAny>>,
) -> PyResult<(SharedTrialQueue, Py<PyAny>)> {
    match trial_queue {
        Some(trial_queue) => {
            let trial_queue_ref = trial_queue.bind(py);
            if let Ok(py_inmemory_trial_queue) = trial_queue_ref.extract::<PyInMemoryTrialQueue>() {
                Ok((
                    py_inmemory_trial_queue.queue.clone() as SharedTrialQueue,
                    trial_queue.clone_ref(py),
                ))
            } else if let Ok(py_sqlite3_trial_queue) =
                trial_queue_ref.extract::<PySQLite3TrialQueue>()
            {
                Ok((
                    py_sqlite3_trial_queue.queue.clone() as SharedTrialQueue,
                    trial_queue.clone_ref(py),
                ))
            } else if let Ok(py_directory_trial_queue) =
                trial_queue_ref.extract::<PyDirectoryTrialQueue>()
            {
                Ok((
                    py_directory_trial_queue.queue.clone() as SharedTrialQueue,
                    trial_queue.clone_ref(py),
                ))
            } else {
                let queue: SharedTrialQueue =
                    Arc::new(ToRustTrialQueue::new(trial_queue.clone_ref(py)));
                Ok((queue, trial_queue.clone_ref(py)))
            }
        }
        None => {
            let trial_queue = PyInMemoryTrialQueue::new();
            let queue = trial_queue.queue.clone();
            let py_trial_queue = Py::new(py, trial_queue)?.into_any();
            Ok((queue, py_trial_queue))
        }
    }
}
