use std::sync::Arc;

use pyo3::prelude::*;
use rustuna_core::trial_queue::TrialQueue;
use rustuna_core::ErrorKind;

use crate::exception::err_to_exceptions;

#[derive(Clone)]
#[pyclass(name = "ToPythonTrialQueue", module = "rustuna", from_py_object)]
pub(crate) struct ToPythonTrialQueue {
    queue: Arc<dyn TrialQueue>,
}

impl ToPythonTrialQueue {
    pub(crate) fn new(queue: Arc<dyn TrialQueue>) -> Self {
        Self { queue }
    }

    pub(crate) fn queue(&self) -> Arc<dyn TrialQueue> {
        Arc::clone(&self.queue)
    }
}

#[pymethods]
impl ToPythonTrialQueue {
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
