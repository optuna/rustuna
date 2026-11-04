use pyo3::prelude::*;
use rustuna_core::trial_queue::{InMemoryTrialQueue, TrialQueue};
use rustuna_core::ErrorKind;
use std::sync::Arc;

use crate::exception::err_to_exceptions;

#[derive(Clone)]
#[pyclass(name = "InMemoryTrialQueue", from_py_object)]
#[pyo3(module = "rustuna")]
pub struct PyInMemoryTrialQueue {
    pub queue: Arc<InMemoryTrialQueue>,
}

impl Default for PyInMemoryTrialQueue {
    fn default() -> Self {
        Self::new()
    }
}

impl PyInMemoryTrialQueue {
    pub fn new() -> Self {
        Self {
            queue: Arc::new(InMemoryTrialQueue::new()),
        }
    }
}

#[pymethods]
impl PyInMemoryTrialQueue {
    #[new]
    fn py_new() -> Self {
        Self::new()
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
