use std::sync::Arc;

use pyo3::prelude::*;

use rustuna_core::trial_queue::TrialQueue;

use self::directory::PyDirectoryTrialQueue;
use self::inmemory::PyInMemoryTrialQueue;
use self::sqlite3::PySQLite3TrialQueue;
use self::to_python::ToPythonTrialQueue;
use self::to_rust::ToRustTrialQueue;

pub mod directory;
pub mod inmemory;
pub mod sqlite3;
pub(crate) mod to_python;
pub mod to_rust;

pub(crate) fn extract_trial_queue(trial_queue: Py<PyAny>) -> PyResult<Arc<dyn TrialQueue>> {
    Python::attach(|py| {
        let trial_queue_ref = trial_queue.bind(py);
        if let Ok(queue) = trial_queue_ref.extract::<PyInMemoryTrialQueue>() {
            Ok(queue.queue.clone() as Arc<dyn TrialQueue>)
        } else if let Ok(queue) = trial_queue_ref.extract::<PySQLite3TrialQueue>() {
            Ok(queue.queue.clone() as Arc<dyn TrialQueue>)
        } else if let Ok(queue) = trial_queue_ref.extract::<PyDirectoryTrialQueue>() {
            Ok(queue.queue.clone() as Arc<dyn TrialQueue>)
        } else if let Ok(queue) = trial_queue_ref.extract::<ToPythonTrialQueue>() {
            Ok(queue.queue())
        } else {
            Ok(Arc::new(ToRustTrialQueue::new(trial_queue)) as Arc<dyn TrialQueue>)
        }
    })
}
