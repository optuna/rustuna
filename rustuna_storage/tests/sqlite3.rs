use std::path::Path;
use std::process::Command;

use rustuna_core::distribution::Distribution;
use rustuna_core::storage::Storage;
use rustuna_core::trial::TrialStateValues;
use rustuna_core::{Error, ErrorKind, Result};
use rustuna_storage::cache::{CachedStorage, CachedStorageBackend};
use rustuna_storage::sqlite3::SQLite3Storage;

fn run_optuna_script(python: &str, db_path: &Path, script: &str) -> Result<()> {
    let output = Command::new(python)
        .args(["-c", script, db_path.to_string_lossy().as_ref()])
        .output()
        .map_err(|_| Error::new(ErrorKind::Unexpected))?;
    if output.status.success() {
        return Ok(());
    }
    eprintln!(
        "Optuna script failed (status={}):\nstdout:\n{}\nstderr:\n{}",
        output.status,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    Err(Error::new(ErrorKind::Unexpected))
}

#[test]
#[ignore = "Requires external Python + Optuna to populate SQLite for integration test"]
fn load_studies_from_optuna_sqlite() -> Result<()> {
    let python = std::env::var("PYTHON_BIN").unwrap_or_else(|_| "python3".to_string());
    let dir = tempfile::tempdir().map_err(|_| Error::new(ErrorKind::Unexpected))?;
    let db_path = dir.path().join("optuna.sqlite3");
    let script = r#"
import optuna, sys

storage = "sqlite:///" + sys.argv[1]
optuna.create_study(storage=storage, study_name="test-0")
optuna.create_study(storage=storage, study_name="test-1", directions=["maximize", "minimize"])
"#;
    run_optuna_script(&python, &db_path, script)?;

    let mut storage = SQLite3Storage::new(db_path.to_string_lossy().as_ref())?;
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
#[ignore = "Requires external Python + Optuna to populate SQLite for integration test"]
fn load_trial() -> Result<()> {
    let python = std::env::var("PYTHON_BIN").unwrap_or_else(|_| "python3".to_string());
    let dir = tempfile::tempdir().map_err(|_| Error::new(ErrorKind::Unexpected))?;
    let db_path = dir.path().join("optuna.sqlite3");
    let script = r#"
import optuna, sys

def objective(trial: optuna.Trial) -> float:
    x = trial.suggest_float("x", 1, 10, log=True)
    y = trial.suggest_int("y", -10, 10)
    trial.suggest_categorical("z", [True, False, "foo", 10])
    return x ** 2 + y

storage = "sqlite:///" + sys.argv[1]
study = optuna.create_study(storage=storage)
study.optimize(objective, n_trials=10)
"#;

    run_optuna_script(&python, &db_path, script)?;

    let mut storage = SQLite3Storage::new(db_path.to_string_lossy().as_ref())?;
    let studies = storage.get_studies()?;
    assert_eq!(studies.len(), 1);

    let trials = storage.get_trials_diff(studies[0].id, &[], -1, false)?;
    let trial0 = trials
        .into_iter()
        .find(|trial| trial.number == 0)
        .expect("trial number 0 should exist");
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
#[ignore = "Requires external Python + Optuna to populate SQLite for integration test"]
fn get_trials() -> Result<()> {
    let python = std::env::var("PYTHON_BIN").unwrap_or_else(|_| "python3".to_string());
    let dir = tempfile::tempdir().map_err(|_| Error::new(ErrorKind::Unexpected))?;
    let db_path = dir.path().join("optuna.sqlite3");
    let script = r#"
import optuna, sys

def objective(trial: optuna.Trial) -> float:
    x = trial.suggest_float("x", 1, 10, log=True)
    y = trial.suggest_int("y", -10, 10)
    trial.suggest_categorical("z", [True, False, "foo", 10])
    trial.set_user_attr("key", "value")
    return x ** 2 + y

storage = "sqlite:///" + sys.argv[1]
study = optuna.create_study(storage=storage, study_name="foo", load_if_exists=True)
study.optimize(objective, n_trials=10)
"#;
    // Evaluate 10 trials
    run_optuna_script(&python, &db_path, script)?;
    let mut storage = CachedStorage::new(
        Box::new(SQLite3Storage::new(db_path.to_string_lossy().as_ref())?),
        false,
    );
    let study_id = {
        let studies = storage.get_studies()?;
        assert_eq!(studies.len(), 1);
        studies[0].id
    };
    let trials = storage.get_trials(study_id)?;
    assert_eq!(trials.len(), 10);

    // Evaluate more 10 trials
    run_optuna_script(&python, &db_path, script)?;
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
    let user_attrs_count = trials[0]
        .as_ref()
        .unwrap()
        .attrs
        .keys()
        .filter(|k| matches!(k, rustuna_core::attr::AttrKey::User(_)))
        .count();
    assert_eq!(user_attrs_count, 1);
    assert!(matches!(
        trials[0].as_ref().unwrap().state_values,
        TrialStateValues::Complete(_)
    ));
    Ok(())
}
