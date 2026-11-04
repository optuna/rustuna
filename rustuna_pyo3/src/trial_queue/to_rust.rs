use pyo3::prelude::*;
use rustuna_core::trial_queue::TrialQueue;
use rustuna_core::{Error, ErrorKind, Result};

// ToRustTrialQueue adapts a Python object implementing rustuna.TrialQueueProtocol to
// rustuna_core::trial_queue::TrialQueue. Rustuna can therefore use Python trial queues through the
// same interface as native queues.
pub struct ToRustTrialQueue {
    obj: Py<PyAny>,
}

impl ToRustTrialQueue {
    pub fn new(obj: Py<PyAny>) -> Self {
        Self { obj }
    }

    fn map_pyerr(&self, operation: &str, err: PyErr) -> Error {
        Python::attach(|py| {
            Error::with_reason(
                ErrorKind::Unexpected,
                format!("TrialQueueProtocol.{operation} failed: {}", err.value(py)),
            )
        })
    }
}

impl TrialQueue for ToRustTrialQueue {
    fn enqueue(&self, trial_id: u32) -> Result<()> {
        Python::attach(|py| {
            self.obj
                .call_method1(py, "enqueue", (trial_id,))
                .map(|_| ())
                .map_err(|err| self.map_pyerr("enqueue", err))
        })
    }

    fn dequeue(&self) -> Result<u32> {
        Python::attach(|py| {
            let trial_id = self
                .obj
                .call_method0(py, "dequeue")
                .and_then(|value| value.extract::<Option<u32>>(py))
                .map_err(|err| self.map_pyerr("dequeue", err))?;
            trial_id.ok_or_else(|| Error::new(ErrorKind::TrialQueueEmpty))
        })
    }
}
