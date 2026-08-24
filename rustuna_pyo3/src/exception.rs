use pyo3::exceptions::{PyImportError, PyKeyError, PyRuntimeError, PyValueError};
use pyo3::prelude::*;

mod exceptions {
    pyo3::import_exception!(rustuna.exceptions, DuplicatedStudyError);
    pyo3::import_exception!(rustuna.exceptions, TrialPruned);
    pyo3::import_exception!(rustuna.exceptions, TrialDiscarded);
    pyo3::import_exception!(rustuna.exceptions, UpdateFinishedTrialError);
    pyo3::import_exception!(rustuna.exceptions, StorageInternalError);
}

pub fn err_to_exceptions(e: rustuna_core::Error) -> PyErr {
    match e.kind {
        rustuna_core::ErrorKind::TrialNotFound => PyKeyError::new_err("Trial not found"),
        rustuna_core::ErrorKind::StudyNotFound => PyKeyError::new_err("Study not found"),
        rustuna_core::ErrorKind::AttrNotFound => PyKeyError::new_err("Attribute not found"),
        rustuna_core::ErrorKind::TrialDiscarded => {
            exceptions::TrialDiscarded::new_err("Trial discarded")
        }
        rustuna_core::ErrorKind::TrialAlreadyFinished => {
            exceptions::UpdateFinishedTrialError::new_err("Trial already finished")
        }
        rustuna_core::ErrorKind::StorageError => {
            exceptions::StorageInternalError::new_err(if e.reason.is_empty() {
                "storage internal error".to_string()
            } else {
                format!("storage internal error: {}", e.reason)
            })
        }
        rustuna_core::ErrorKind::DuplicatedStudy => {
            exceptions::DuplicatedStudyError::new_err("Duplicate study name")
        }
        rustuna_core::ErrorKind::IncompatibleDistribution => {
            PyValueError::new_err("Incompatible distribution for the parameter")
        }
        rustuna_core::ErrorKind::ImportanceEvaluatorError => PyValueError::new_err(e.reason),
        rustuna_core::ErrorKind::UnsupportedMultiObjective => {
            PyRuntimeError::new_err("Multi-objective study is not supported")
        }
        rustuna_core::ErrorKind::MissingDependency => PyImportError::new_err(e.reason),
        rustuna_core::ErrorKind::InvalidObjectiveValues => PyValueError::new_err(e.reason),
        _ => PyRuntimeError::new_err(format!("{e:?}")),
    }
}
