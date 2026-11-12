use std::sync::Arc;

use pyo3::prelude::*;

use rustuna_core::sampler::Context as SamplerContext;
use rustuna_core::trial_queue::InMemoryTrialQueue;

use crate::study::PyDirection;
use crate::trial_queue::to_python::ToPythonTrialQueue;

#[derive(Clone)]
#[pyclass(name = "SamplerContext", skip_from_py_object)]
#[pyo3(module = "rustuna")]
pub struct PySamplerContext {
    pub(crate) context: SamplerContext,
}
impl From<SamplerContext> for PySamplerContext {
    fn from(item: SamplerContext) -> Self {
        PySamplerContext { context: item }
    }
}
#[allow(non_local_definitions)]
#[pymethods]
impl PySamplerContext {
    #[new]
    #[pyo3(signature = (*, study_id, trial_number, trial_id, directions))]
    pub fn py_new(
        study_id: u32,
        trial_number: u32,
        trial_id: u32,
        directions: Vec<PyDirection>,
    ) -> PyResult<Self> {
        Ok(PySamplerContext {
            context: SamplerContext {
                study_id,
                trial_number,
                trial_id,
                directions: directions.into_iter().map(|d| d.into()).collect(),
                trial_queue: Arc::new(InMemoryTrialQueue::new()),
            },
        })
    }
    #[getter]
    fn study_id(&self) -> u32 {
        self.context.study_id
    }
    #[getter]
    fn trial_number(&self) -> u32 {
        self.context.trial_number
    }
    #[getter]
    fn trial_id(&self) -> u32 {
        self.context.trial_id
    }
    #[getter]
    fn directions(&self) -> Vec<PyDirection> {
        self.context
            .directions
            .iter()
            .map(|d| d.clone().into())
            .collect()
    }

    #[getter]
    fn trial_queue(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let queue = ToPythonTrialQueue::new(Arc::clone(&self.context.trial_queue));
        Ok(Py::new(py, queue)?.into_any())
    }
}
