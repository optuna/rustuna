use std::collections::HashMap;

use pyo3::exceptions::PyUserWarning;
use pyo3::prelude::*;
use pyo3::PyResult;

use rustuna_core::trial::PersistedTrial;
use rustuna_importance::{
    get_param_importances_with, ImportanceEvaluator, ImportanceOptions, PedAnovaImportanceEvaluator,
};

use crate::exception::err_to_exceptions;
use crate::study::PyStudy;
use crate::trial::PyTrialState;

#[pyfunction]
#[pyo3(name = "get_param_importances", signature = (study, *, evaluator = None, params = None, target = None, normalize = true))]
pub fn py_get_param_importances(
    py: Python<'_>,
    study: &PyStudy,
    evaluator: Option<&PyPedAnovaImportanceEvaluator>,
    params: Option<Vec<String>>,
    target: Option<Py<PyAny>>,
    normalize: bool,
) -> PyResult<HashMap<String, f64>> {
    let rust_target = convert_py_target(py, study, target)?;
    let options = ImportanceOptions {
        target: rust_target
            .as_ref()
            .map(|target| target as &dyn Fn(&PersistedTrial) -> f64),
        normalize,
        params,
    };
    let default_evaluator = PedAnovaImportanceEvaluator::default();
    let evaluator = evaluator
        .map(|wrapper| &wrapper.evaluator)
        .unwrap_or(&default_evaluator);
    let importances =
        get_param_importances_with(&study.study, evaluator, options).map_err(err_to_exceptions)?;
    Ok(importances)
}

#[pyclass(name = "PedAnovaImportanceEvaluator")]
#[pyo3(module = "rustuna.importance")]
pub struct PyPedAnovaImportanceEvaluator {
    evaluator: PedAnovaImportanceEvaluator,
}

#[pymethods]
impl PyPedAnovaImportanceEvaluator {
    #[new]
    #[pyo3(signature = (*, target_quantile = 0.1, region_quantile = 1.0, evaluate_on_local = true))]
    fn py_new(
        py: Python<'_>,
        target_quantile: f64,
        region_quantile: f64,
        evaluate_on_local: bool,
    ) -> PyResult<Self> {
        let evaluator = PedAnovaImportanceEvaluator::builder()
            .target_quantile(target_quantile)
            .region_quantile(region_quantile)
            .evaluate_on_local(evaluate_on_local)
            .build()
            .map_err(err_to_exceptions)?;
        if region_quantile != 1.0 && !evaluate_on_local {
            PyErr::warn(
                py,
                &py.get_type::<PyUserWarning>(),
                pyo3::ffi::c_str!(
                    "If `evaluate_on_local` is False, `region_quantile` has no effect."
                ),
                1,
            )?;
        }
        Ok(Self { evaluator })
    }

    #[pyo3(signature = (study, params = None, *, target = None))]
    fn evaluate(
        &self,
        py: Python<'_>,
        study: &PyStudy,
        params: Option<Vec<String>>,
        target: Option<Py<PyAny>>,
    ) -> PyResult<HashMap<String, f64>> {
        let rust_target = convert_py_target(py, study, target)?;
        let options = ImportanceOptions {
            target: rust_target
                .as_ref()
                .map(|target| target as &dyn Fn(&PersistedTrial) -> f64),
            normalize: true,
            params,
        };
        let importances = self
            .evaluator
            .evaluate_with(&study.study, options)
            .map_err(err_to_exceptions)?;
        Ok(importances)
    }
}

fn convert_py_target(
    py: Python<'_>,
    study: &PyStudy,
    target: Option<Py<PyAny>>,
) -> PyResult<Option<impl Fn(&PersistedTrial) -> f64>> {
    let target_values = target
        .map(|target| {
            study
                .py_get_trials(Some(vec![PyTrialState::COMPLETE]))?
                .into_iter()
                .map(|trial| {
                    let trial_id = trial.with_trial(|t| Ok(t.id))?;
                    let py_trial = Py::new(py, trial)?;
                    let value = target.call1(py, (py_trial,))?.extract::<f64>(py)?;
                    Ok((trial_id, value))
                })
                .collect::<PyResult<HashMap<_, _>>>()
        })
        .transpose()?;
    let rust_target = target_values.map(|values| {
        move |trial: &PersistedTrial| values.get(&trial.id).copied().unwrap_or(f64::NAN)
    });
    Ok(rust_target)
}
