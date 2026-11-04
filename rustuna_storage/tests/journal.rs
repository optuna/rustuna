use std::process::Command;

use rustuna_core::distribution::Distribution;
use rustuna_core::storage::Storage;
use rustuna_core::trial::TrialStateValues;
use rustuna_core::{Error, ErrorKind, Result};
use rustuna_storage::journal::file::JournalFileBackend;
use rustuna_storage::journal::storage::JournalStorage;

#[path = "../src/test_utils.rs"]
mod test_utils;
use test_utils::TempDir;

fn run_optuna_python_script(python: &str, journal_path: &str, script: &str) -> Result<()> {
    let output = Command::new(python)
        .args(["-c", script, journal_path])
        .output()
        .map_err(|e| {
            Error::with_reason(
                ErrorKind::Unexpected,
                format!("Failed to execute Python: {e}"),
            )
        })?;
    if output.status.success() {
        return Ok(());
    }
    eprintln!(
        "Optuna script failed (status={}):\nstdout:\n{}\nstderr:\n{}",
        output.status,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    Err(Error::with_reason(
        ErrorKind::Unexpected,
        format!("Optuna script failed with status: {}", output.status),
    ))
}

#[test]
#[ignore = "Requires external Python + Optuna to populate journal for integration test"]
fn load_studies_from_optuna_journal() -> Result<()> {
    let python = std::env::var("PYTHON_BIN").unwrap_or_else(|_| "python3".to_string());
    let dir = TempDir::new().map_err(|e| {
        Error::with_reason(
            ErrorKind::Unexpected,
            format!("Failed to create temp dir: {e}"),
        )
    })?;
    let journal_path = dir.path().join("optuna.journal");

    let script = r#"
import sys
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
import optuna

backend = JournalFileBackend(sys.argv[1])
storage = JournalStorage(backend)
optuna.create_study(storage=storage, study_name="test-0")
optuna.create_study(storage=storage, study_name="test-1", directions=["maximize", "minimize"])
"#;
    run_optuna_python_script(&python, journal_path.to_string_lossy().as_ref(), script)?;

    let mut storage = JournalStorage::new(Box::new(JournalFileBackend::new(&journal_path, None)?))?;
    let studies = storage.get_studies()?;
    assert_eq!(studies.len(), 2);
    assert_eq!(studies[0].name, "test-0");
    assert_eq!(studies[1].name, "test-1");

    assert_eq!(studies[0].directions.len(), 1);
    assert_eq!(
        studies[0].directions[0],
        rustuna_core::study::Direction::Minimize
    );

    assert_eq!(studies[1].directions.len(), 2);
    assert_eq!(
        studies[1].directions[0],
        rustuna_core::study::Direction::Maximize
    );
    assert_eq!(
        studies[1].directions[1],
        rustuna_core::study::Direction::Minimize
    );
    Ok(())
}

#[test]
#[ignore = "Requires external Python + Optuna to populate journal for integration test"]
fn load_trial_from_optuna_journal() -> Result<()> {
    let python = std::env::var("PYTHON_BIN").unwrap_or_else(|_| "python3".to_string());
    let dir = TempDir::new().map_err(|e| {
        Error::with_reason(
            ErrorKind::Unexpected,
            format!("Failed to create temp dir: {e}"),
        )
    })?;
    let journal_path = dir.path().join("optuna.journal");

    let script = r#"
import sys
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
import optuna

def objective(trial: optuna.Trial) -> float:
    x = trial.suggest_float("x", 1, 10, log=True)
    y = trial.suggest_int("y", -10, 10)
    trial.suggest_categorical("z", [True, False, "foo", 10])
    return x ** 2 + y

backend = JournalFileBackend(sys.argv[1])
storage = JournalStorage(backend)
study = optuna.create_study(storage=storage)
study.optimize(objective, n_trials=10)
"#;

    run_optuna_python_script(&python, journal_path.to_string_lossy().as_ref(), script)?;

    let mut storage = JournalStorage::new(Box::new(JournalFileBackend::new(&journal_path, None)?))?;
    let study_id = {
        let studies = storage.get_studies()?;
        assert_eq!(studies.len(), 1);
        studies[0].id
    };

    let trial_id = storage.get_trials(study_id)?[0].as_ref().unwrap().id;
    let trial0 = storage.get_trial(trial_id)?;
    assert_eq!(trial0.number, 0);

    // Distributions
    assert_eq!(trial0.distributions.len(), 3);
    assert_eq!(
        trial0.distributions["x"],
        rustuna_core::distribution::Distribution::new_float(1.0, 10.0, None, true)
    );
    assert_eq!(
        trial0.distributions["y"],
        rustuna_core::distribution::Distribution::new_int(-10, 10, 1, false)
    );
    assert_eq!(
        trial0.distributions["z"],
        rustuna_core::distribution::Distribution::new_categorical(4)
    );

    // Objective value
    assert!(matches!(trial0.state_values, TrialStateValues::Complete(_)));
    Ok(())
}

#[test]
#[ignore = "Requires external Python + Optuna to populate journal for integration test"]
fn get_trials_from_optuna_journal() -> Result<()> {
    let python = std::env::var("PYTHON_BIN").unwrap_or_else(|_| "python3".to_string());
    let dir = TempDir::new().map_err(|e| {
        Error::with_reason(
            ErrorKind::Unexpected,
            format!("Failed to create temp dir: {e}"),
        )
    })?;
    let journal_path = dir.path().join("optuna.journal");

    let script = r#"
import sys
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
import optuna

def objective(trial: optuna.Trial) -> float:
    x = trial.suggest_float("x", 1, 10, log=True)
    y = trial.suggest_int("y", -10, 10)
    trial.suggest_categorical("z", [True, False, "foo", 10])
    trial.set_user_attr("str_key", "value")
    trial.set_user_attr("int_key", 1)
    return x ** 2 + y

backend = JournalFileBackend(sys.argv[1])
storage = JournalStorage(backend)
study = optuna.create_study(storage=storage, study_name="foo", load_if_exists=True)
study.optimize(objective, n_trials=10)
"#;
    // Evaluate 10 trials
    run_optuna_python_script(&python, journal_path.to_string_lossy().as_ref(), script)?;
    let mut storage = JournalStorage::new(Box::new(JournalFileBackend::new(&journal_path, None)?))?;
    let study_id = {
        let studies = storage.get_studies()?;
        assert_eq!(studies.len(), 1);
        studies[0].id
    };
    let trials = storage.get_trials(study_id)?;
    assert_eq!(trials.len(), 10);

    // Evaluate more 10 trials with load_if_exists=True
    run_optuna_python_script(&python, journal_path.to_string_lossy().as_ref(), script)?;
    let mut storage = JournalStorage::new(Box::new(JournalFileBackend::new(&journal_path, None)?))?;
    let trials = storage.get_trials(study_id)?;
    assert_eq!(trials.len(), 20);
    assert_eq!(trials[0].as_ref().unwrap().distributions.len(), 3);
    assert_eq!(
        trials[0].as_ref().unwrap().distributions["x"],
        Distribution::new_float(1.0, 10.0, None, true)
    );
    assert_eq!(
        trials[0].as_ref().unwrap().distributions["y"],
        Distribution::new_int(-10, 10, 1, false)
    );
    assert_eq!(
        trials[0].as_ref().unwrap().distributions["z"],
        Distribution::new_categorical(4)
    );
    assert_eq!(trials[0].as_ref().unwrap().internal_params.len(), 3);
    // Optuna's JSON-encoded `user_attr` fields are restored. The default (plain) format unquotes
    // JSON strings and keeps other values as JSON texts.
    let attrs = &trials[0].as_ref().unwrap().attrs;
    let user_attr = |key: &str| attrs.get(&rustuna_core::attr::AttrKey::User(key.into()));
    assert_eq!(user_attr("str_key"), Some(&"value".to_string()));
    assert_eq!(user_attr("int_key"), Some(&"1".to_string()));
    assert!(matches!(
        trials[0].as_ref().unwrap().state_values,
        TrialStateValues::Complete(_)
    ));
    Ok(())
}
