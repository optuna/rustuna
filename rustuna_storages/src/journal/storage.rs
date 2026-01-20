use std::collections::HashMap;
use std::thread;
use std::time::{SystemTime, UNIX_EPOCH};

use serde_json::{Map, Number, Value};

use rustuna_core::attr::{
    category_labels_to_attrs, get_category_labels, AttrKey, Attrs, CategoryLabel,
};
use rustuna_core::distribution::Distribution;
use rustuna_core::storage::Storage;
use rustuna_core::study::{Direction, PersistedStudy};
use rustuna_core::study_cache::StudyCache;
use rustuna_core::trial::{PersistedTrial, TrialStateValues};
use rustuna_core::{Error, ErrorKind, Result};

use super::{JournalBackend, JournalLog, JournalOperation};

pub struct JournalStorage {
    backend: Box<dyn JournalBackend>,
    replay: JournalReplayState,
}

impl JournalStorage {
    pub fn new(backend: Box<dyn JournalBackend>) -> Result<Self> {
        let worker_id_prefix = format!("{}-{}-", unique_prefix(), std::process::id());
        let mut storage = JournalStorage {
            backend,
            replay: JournalReplayState::new(worker_id_prefix),
        };
        storage.sync_with_backend()?;
        Ok(storage)
    }

    fn worker_id(&self) -> String {
        format!(
            "{}{:?}",
            self.replay.worker_id_prefix,
            thread::current().id()
        )
    }

    fn write_log(&mut self, op_code: JournalOperation, fields: Map<String, Value>) -> Result<()> {
        let worker_id = self.worker_id();
        let log = JournalLog {
            op_code: op_code as i32,
            worker_id: worker_id.clone(),
            fields,
        };
        self.backend.append_logs(&[log])
    }

    fn sync_with_backend(&mut self) -> Result<()> {
        let worker_id = self.worker_id();
        self.backend
            .read_logs(self.replay.log_number_read, &mut |log| {
                self.replay
                    .apply_logs(std::slice::from_ref(&log), &worker_id)
            })
    }
}

impl Storage for JournalStorage {
    fn create_new_study(
        &mut self,
        study_name: &str,
        directions: Vec<Direction>,
    ) -> Result<&PersistedStudy> {
        let mut fields = Map::new();
        fields.insert(
            "study_name".to_string(),
            Value::String(study_name.to_string()),
        );
        let directions_val = directions
            .iter()
            .map(|d| {
                let v = match d {
                    Direction::Minimize => 1,
                    Direction::Maximize => 2,
                };
                Value::Number(Number::from(v))
            })
            .collect::<Vec<_>>();
        fields.insert("directions".to_string(), Value::Array(directions_val));
        self.write_log(JournalOperation::CreateStudy, fields)?;
        self.sync_with_backend()?;
        let study = self
            .replay
            .studies
            .values()
            .find(|s| s.name == study_name)
            .ok_or_else(|| {
                Error::with_reason(
                    ErrorKind::StorageError,
                    format!("Failed to find study by name in replay state: {study_name}"),
                )
            })?;
        Ok(study)
    }

    fn delete_study(&mut self, study_id: u32) -> Result<()> {
        let mut fields = Map::new();
        fields.insert(
            "study_id".to_string(),
            Value::Number(Number::from(study_id)),
        );
        self.write_log(JournalOperation::DeleteStudy, fields)?;
        self.sync_with_backend()?;
        Ok(())
    }

    fn create_new_trial(&mut self, study_id: u32) -> Result<&PersistedTrial> {
        let mut fields = Map::new();
        fields.insert(
            "study_id".to_string(),
            Value::Number(Number::from(study_id)),
        );
        fields.insert(
            "datetime_start".to_string(),
            Value::String(chrono::Local::now().naive_local().to_string()),
        );
        self.write_log(JournalOperation::CreateTrial, fields)?;
        self.sync_with_backend()?;
        let trial_id = self
            .replay
            .last_created_trial_id_by_this_process
            .ok_or_else(|| {
                Error::with_reason(
                    ErrorKind::StorageError,
                    "Failed to get last created trial id by this process",
                )
            })?;
        let (trial_study_id, trial_number) = self
            .replay
            .trial_id_to_study_number
            .get(&trial_id)
            .copied()
            .ok_or_else(|| {
                Error::with_reason(
                    ErrorKind::StorageError,
                    format!(
                        "Failed to get study number for newly created trial: trial_id={trial_id}"
                    ),
                )
            })?;
        if trial_study_id != study_id {
            return Err(Error::with_reason(
                ErrorKind::StorageError,
                format!(
                    "Trial study id mismatch: trial_study_id={trial_study_id}, expected study_id={study_id}"
                ),
            ));
        }
        let trials = self.replay.trials_by_study.get(&study_id).ok_or_else(|| {
            Error::with_reason(
                ErrorKind::StorageError,
                format!("Failed to find trials for study: study_id={study_id}"),
            )
        })?;
        trials.get(trial_number as usize).ok_or_else(|| {
            Error::with_reason(
                ErrorKind::TrialNotFound,
                format!("Failed to find trial at given number: trial_number={trial_number}"),
            )
        })
    }

    fn set_trial_param(
        &mut self,
        trial_id: u32,
        name: &str,
        distribution: &Distribution,
        value: f64,
    ) -> Result<()> {
        self.sync_with_backend()?;
        self.replay.ensure_trial_updatable(trial_id)?;
        let (study_id, _trial_number) = self
            .replay
            .trial_id_to_study_number
            .get(&trial_id)
            .copied()
            .ok_or_else(|| {
                Error::with_reason(
                    ErrorKind::TrialNotFound,
                    format!("Trial not found in storage: trial_id={trial_id}"),
                )
            })?;
        self.replay
            .check_param_compatibility(study_id, name, distribution)?;
        let labels = self.replay.labels_for_param(study_id, name);
        let dist_json = distribution_to_json(distribution, labels.as_deref())?;

        let mut fields = Map::new();
        fields.insert(
            "trial_id".to_string(),
            Value::Number(Number::from(trial_id)),
        );
        fields.insert("param_name".to_string(), Value::String(name.to_string()));
        fields.insert(
            "param_value_internal".to_string(),
            Value::String(format!("{value:.17e}")),
        );
        fields.insert("distribution".to_string(), Value::String(dist_json));
        self.write_log(JournalOperation::SetTrialParam, fields)?;
        self.sync_with_backend()?;
        Ok(())
    }

    fn set_trial_state_values(
        &mut self,
        trial_id: u32,
        state_values: TrialStateValues,
    ) -> Result<()> {
        self.sync_with_backend()?;
        self.replay.ensure_trial_updatable(trial_id)?;
        let (study_id, trial_number) = self
            .replay
            .trial_id_to_study_number
            .get(&trial_id)
            .copied()
            .ok_or_else(|| {
                Error::with_reason(
                    ErrorKind::TrialNotFound,
                    format!("Trial not found in storage: trial_id={trial_id}"),
                )
            })?;
        let existing_trial = self
            .replay
            .trial_from_study_number(study_id, trial_number)?;

        let (state_code, values) = match state_values {
            TrialStateValues::Running => (0, None),
            TrialStateValues::Complete(v) => (1, Some(v)),
            TrialStateValues::Pruned => (2, None),
            TrialStateValues::Waiting => (3, None),
            TrialStateValues::Fail => (4, None),
        };
        let mut fields = Map::new();
        fields.insert(
            "trial_id".to_string(),
            Value::Number(Number::from(trial_id)),
        );
        fields.insert("state".to_string(), Value::Number(Number::from(state_code)));
        if let Some(values) = values {
            let arr = values
                .iter()
                .map(|value| value_to_json(*value))
                .collect::<Vec<_>>();
            fields.insert("values".to_string(), Value::Array(arr));
        } else {
            fields.insert("values".to_string(), Value::Null);
        }
        if state_code == 0
            && (!matches!(existing_trial.state_values, TrialStateValues::Running)
                || existing_trial.datetime_start.is_none())
        {
            fields.insert(
                "datetime_start".to_string(),
                Value::String(chrono::Local::now().naive_local().to_string()),
            );
        } else if matches!(state_code, 1 | 2 | 4) {
            fields.insert(
                "datetime_complete".to_string(),
                Value::String(chrono::Local::now().naive_local().to_string()),
            );
        }
        self.write_log(JournalOperation::SetTrialStateValues, fields)?;
        self.sync_with_backend()?;
        Ok(())
    }

    fn set_trial_intermediate_values(
        &mut self,
        trial_id: u32,
        intermediate_values: HashMap<u32, f64>,
    ) -> Result<()> {
        if intermediate_values.is_empty() {
            return Ok(());
        }
        self.sync_with_backend()?;
        self.replay.ensure_trial_updatable(trial_id)?;
        let worker_id = self.worker_id();
        let mut logs = Vec::with_capacity(intermediate_values.len());
        for (step, value) in intermediate_values {
            let mut fields = Map::new();
            fields.insert(
                "trial_id".to_string(),
                Value::Number(Number::from(trial_id)),
            );
            fields.insert("step".to_string(), Value::Number(Number::from(step)));
            fields.insert("intermediate_value".to_string(), value_to_json(value));
            logs.push(JournalLog {
                op_code: JournalOperation::SetTrialIntermediateValue as i32,
                worker_id: worker_id.clone(),
                fields,
            });
        }
        self.backend.append_logs(&logs)?;
        self.sync_with_backend()?;
        Ok(())
    }

    fn get_studies(&mut self) -> Result<&Vec<PersistedStudy>> {
        self.sync_with_backend()?;
        self.replay.studies_sorted.clear();
        self.replay
            .studies_sorted
            .extend(self.replay.studies.values().cloned());
        self.replay.studies_sorted.sort_by_key(|s| s.id);
        Ok(&self.replay.studies_sorted)
    }

    fn get_study(&mut self, study_id: u32) -> Result<&PersistedStudy> {
        self.sync_with_backend()?;
        self.replay.studies.get(&study_id).ok_or_else(|| {
            Error::with_reason(
                ErrorKind::StudyNotFound,
                format!("Study not found in storage: study_id={study_id}"),
            )
        })
    }

    fn get_trials(&mut self, study_id: u32) -> Result<&Vec<PersistedTrial>> {
        self.sync_with_backend()?;
        let trials = self.replay.trials_by_study.get(&study_id).ok_or_else(|| {
            Error::with_reason(
                ErrorKind::StudyNotFound,
                format!("Failed to get trials for study: study_id={study_id}"),
            )
        })?;
        Ok(trials)
    }

    fn get_trial(&mut self, trial_id: u32) -> Result<&PersistedTrial> {
        self.sync_with_backend()?;
        let (study_id, trial_number) = self
            .replay
            .trial_id_to_study_number
            .get(&trial_id)
            .copied()
            .ok_or_else(|| {
                Error::with_reason(
                    ErrorKind::TrialNotFound,
                    format!("Trial not found in storage: trial_id={trial_id}"),
                )
            })?;
        let trials = self.replay.trials_by_study.get(&study_id).ok_or_else(|| {
            Error::with_reason(
                ErrorKind::StudyNotFound,
                format!("Failed to get trials for study: study_id={study_id}"),
            )
        })?;
        trials.get(trial_number as usize).ok_or_else(|| {
            Error::with_reason(
                ErrorKind::TrialNotFound,
                format!(
                    "Trial number out of bounds: trial_number={}, num_trials={}",
                    trial_number,
                    trials.len()
                ),
            )
        })
    }

    fn get_category_labels(
        &mut self,
        study_id: u32,
        param_name: &str,
        cardinality: usize,
    ) -> Result<Option<Vec<CategoryLabel>>> {
        self.sync_with_backend()?;
        let study = self
            .replay
            .studies
            .values()
            .find(|s| s.id == study_id)
            .ok_or_else(|| Error::new(ErrorKind::StudyNotFound))?;
        Ok(get_category_labels(&study.attrs, param_name, cardinality))
    }

    fn set_category_labels(
        &mut self,
        study_id: u32,
        param_name: &str,
        labels: Vec<CategoryLabel>,
    ) -> Result<()> {
        let attrs = category_labels_to_attrs(param_name, &labels);
        self.set_study_attrs(study_id, attrs, true)
    }

    fn get_trial_id_from_study_id_trial_number(
        &mut self,
        study_id: u32,
        trial_number: u32,
    ) -> Result<u32> {
        self.sync_with_backend()?;
        self.replay
            .trial_id_from_study_number(study_id, trial_number)
    }

    fn set_study_attrs(
        &mut self,
        study_id: u32,
        attrs: Attrs,
        error_on_overwrite: bool,
    ) -> Result<()> {
        self.sync_with_backend()?;
        let study = self.replay.studies.get(&study_id).ok_or_else(|| {
            Error::with_reason(
                ErrorKind::StudyNotFound,
                format!("Study not found in storage: study_id={study_id}"),
            )
        })?;
        if error_on_overwrite {
            for key in attrs.keys() {
                if study.attrs.contains_key(key) {
                    return Err(Error::with_reason(
                        ErrorKind::AttrOverwriteNotAllowed,
                        format!("Attribute already exists and overwrite not allowed: key={key:?}"),
                    ));
                }
            }
        }

        for (key, value) in attrs {
            let mut fields = Map::new();
            fields.insert(
                "study_id".to_string(),
                Value::Number(Number::from(study_id)),
            );
            match key {
                AttrKey::User(k) => {
                    let mut map = Map::new();
                    map.insert(k.to_string(), Value::String(value));
                    fields.insert("user_attr".to_string(), Value::Object(map));
                    self.write_log(JournalOperation::SetStudyUserAttr, fields)?;
                }
                AttrKey::System(k) => {
                    let mut map = Map::new();
                    map.insert(k.to_string(), Value::String(value));
                    fields.insert("system_attr".to_string(), Value::Object(map));
                    self.write_log(JournalOperation::SetStudySystemAttr, fields)?;
                }
            }
        }
        self.sync_with_backend()?;
        Ok(())
    }

    fn set_trial_attrs(
        &mut self,
        trial_id: u32,
        attrs: Attrs,
        error_on_overwrite: bool,
    ) -> Result<()> {
        self.sync_with_backend()?;
        let (study_id, trial_number) = self
            .replay
            .trial_id_to_study_number
            .get(&trial_id)
            .copied()
            .ok_or_else(|| {
                Error::with_reason(
                    ErrorKind::TrialNotFound,
                    format!("Trial not found during set_trial_attrs: trial_id={trial_id}"),
                )
            })?;
        let trial = self
            .replay
            .trial_from_study_number(study_id, trial_number)?;
        if error_on_overwrite {
            for key in attrs.keys() {
                if trial.attrs.contains_key(key) {
                    return Err(Error::with_reason(
                        ErrorKind::AttrOverwriteNotAllowed,
                        format!("Attribute already exists and overwrite not allowed: key={key:?}"),
                    ));
                }
            }
        }
        for (key, value) in attrs {
            let mut fields = Map::new();
            fields.insert(
                "trial_id".to_string(),
                Value::Number(Number::from(trial_id)),
            );
            match key {
                AttrKey::User(k) => {
                    let mut map = Map::new();
                    map.insert(k.to_string(), Value::String(value));
                    fields.insert("user_attr".to_string(), Value::Object(map));
                    self.write_log(JournalOperation::SetTrialUserAttr, fields)?;
                }
                AttrKey::System(k) => {
                    let mut map = Map::new();
                    map.insert(k.to_string(), Value::String(value));
                    fields.insert("system_attr".to_string(), Value::Object(map));
                    self.write_log(JournalOperation::SetTrialSystemAttr, fields)?;
                }
            }
        }
        self.sync_with_backend()?;
        Ok(())
    }

    fn get_joint_search_space(&mut self, study_id: u32) -> Result<HashMap<String, Distribution>> {
        self.sync_with_backend()?;
        let trials = self.replay.trials_by_study.get(&study_id).ok_or_else(|| {
            Error::with_reason(
                ErrorKind::StudyNotFound,
                format!("Failed to get trials for study: study_id={study_id}"),
            )
        })?;
        let cache = self.replay.study_caches.entry(study_id).or_default();
        cache.update(trials);
        Ok(cache.get_joint_search_space())
    }
}

struct JournalReplayState {
    log_number_read: usize,
    worker_id_prefix: String,
    studies: HashMap<u32, PersistedStudy>,
    trials_by_study: HashMap<u32, Vec<PersistedTrial>>,
    trial_id_to_study_number: HashMap<u32, (u32, u32)>,
    trial_id_to_study_id: HashMap<u32, u32>,
    study_id_to_trial_ids: HashMap<u32, Vec<u32>>,
    next_study_id: u32,
    worker_id_to_owned_trial_id: HashMap<String, u32>,
    last_created_trial_id_by_this_process: Option<u32>,
    studies_sorted: Vec<PersistedStudy>,
    study_caches: HashMap<u32, StudyCache>,
}

impl JournalReplayState {
    fn new(worker_id_prefix: String) -> Self {
        JournalReplayState {
            log_number_read: 0,
            worker_id_prefix,
            studies: HashMap::new(),
            trials_by_study: HashMap::new(),
            trial_id_to_study_number: HashMap::new(),
            trial_id_to_study_id: HashMap::new(),
            study_id_to_trial_ids: HashMap::new(),
            next_study_id: 0,
            worker_id_to_owned_trial_id: HashMap::new(),
            last_created_trial_id_by_this_process: None,
            studies_sorted: Vec::new(),
            study_caches: HashMap::new(),
        }
    }

    fn apply_logs(&mut self, logs: &[JournalLog], worker_id: &str) -> Result<()> {
        for log in logs {
            self.log_number_read += 1;
            let op = JournalOperation::from_i32(log.op_code)?;
            match op {
                JournalOperation::CreateStudy => self.apply_create_study(log, worker_id)?,
                JournalOperation::DeleteStudy => self.apply_delete_study(log, worker_id)?,
                JournalOperation::SetStudyUserAttr => {
                    self.apply_set_study_user_attr(log, worker_id)?
                }
                JournalOperation::SetStudySystemAttr => {
                    self.apply_set_study_system_attr(log, worker_id)?
                }
                JournalOperation::CreateTrial => self.apply_create_trial(log, worker_id)?,
                JournalOperation::SetTrialParam => self.apply_set_trial_param(log, worker_id)?,
                JournalOperation::SetTrialStateValues => {
                    self.apply_set_trial_state_values(log, worker_id)?
                }
                JournalOperation::SetTrialIntermediateValue => {
                    self.apply_set_trial_intermediate_value(log, worker_id)?
                }
                JournalOperation::SetTrialUserAttr => {
                    self.apply_set_trial_user_attr(log, worker_id)?
                }
                JournalOperation::SetTrialSystemAttr => {
                    self.apply_set_trial_system_attr(log, worker_id)?
                }
            }
        }
        Ok(())
    }

    fn is_issued_by_this_worker(&self, log: &JournalLog, worker_id: &str) -> bool {
        log.worker_id == worker_id
    }

    fn study_exists(&self, study_id: u32, log: &JournalLog, worker_id: &str) -> Result<bool> {
        if self.studies.contains_key(&study_id) {
            Ok(true)
        } else if self.is_issued_by_this_worker(log, worker_id) {
            Err(Error::with_reason(
                ErrorKind::StudyNotFound,
                format!("Study not found during replay: study_id={study_id}"),
            ))
        } else {
            Ok(false)
        }
    }

    fn trial_exists_and_updatable(
        &self,
        trial_id: u32,
        log: &JournalLog,
        worker_id: &str,
    ) -> Result<bool> {
        let (study_id, trial_number) = match self.trial_id_to_study_number.get(&trial_id) {
            Some(v) => *v,
            None => {
                if self.is_issued_by_this_worker(log, worker_id) {
                    return Err(Error::with_reason(
                        ErrorKind::TrialNotFound,
                        format!("Trial not found during replay: trial_id={trial_id}"),
                    ));
                }
                return Ok(false);
            }
        };
        let trial = self
            .trials_by_study
            .get(&study_id)
            .and_then(|trials| trials.get(trial_number as usize))
            .ok_or_else(|| {
                Error::with_reason(
                    ErrorKind::TrialNotFound,
                    format!("Trial not found at position: trial_number={trial_number}"),
                )
            })?;
        if trial.is_finished() {
            if self.is_issued_by_this_worker(log, worker_id) {
                return Err(Error::with_reason(
                    ErrorKind::TrialAlreadyFinished,
                    format!("Trial already finished: trial_id={trial_id}"),
                ));
            }
            return Ok(false);
        }
        Ok(true)
    }

    fn apply_create_study(&mut self, log: &JournalLog, worker_id: &str) -> Result<()> {
        let study_name = get_string(&log.fields, "study_name")?;
        let directions_raw = log
            .fields
            .get("directions")
            .and_then(|v| v.as_array())
            .ok_or_else(|| {
                Error::with_reason(
                    ErrorKind::StorageError,
                    "Failed to parse directions array from create study log",
                )
            })?;
        let mut directions = Vec::with_capacity(directions_raw.len());
        for d in directions_raw {
            let value = d.as_i64().ok_or_else(|| {
                Error::with_reason(
                    ErrorKind::StorageError,
                    "Failed to parse direction value as i64",
                )
            })?;
            let dir = match value {
                1 => Direction::Minimize,
                2 => Direction::Maximize,
                _ => {
                    return Err(Error::with_reason(
                        ErrorKind::StorageError,
                        format!("Invalid direction value: {value} (expected 1 for Minimize or 2 for Maximize per Optuna schema)"),
                    ))
                }
            };
            directions.push(dir);
        }

        if self.studies.values().any(|s| s.name == study_name) {
            if self.is_issued_by_this_worker(log, worker_id) {
                return Err(Error::with_reason(
                    ErrorKind::DuplicatedStudy,
                    format!("Study already exists: {study_name}"),
                ));
            }
            return Ok(());
        }

        let study_id = self.next_study_id;
        self.next_study_id += 1;
        self.studies.insert(
            study_id,
            PersistedStudy::new(study_id, study_name, directions),
        );
        self.trials_by_study.insert(study_id, Vec::new());
        self.study_id_to_trial_ids.insert(study_id, Vec::new());
        Ok(())
    }

    fn apply_delete_study(&mut self, log: &JournalLog, worker_id: &str) -> Result<()> {
        let study_id = get_u32(&log.fields, "study_id")?;
        if self.study_exists(study_id, log, worker_id)? {
            self.studies.remove(&study_id);
            if let Some(trial_ids) = self.study_id_to_trial_ids.remove(&study_id) {
                for trial_id in trial_ids {
                    self.trial_id_to_study_number.remove(&trial_id);
                    self.trial_id_to_study_id.remove(&trial_id);
                }
            }
            self.trials_by_study.remove(&study_id);
            self.study_caches.remove(&study_id);
        }
        Ok(())
    }

    fn apply_set_study_user_attr(&mut self, log: &JournalLog, worker_id: &str) -> Result<()> {
        let study_id = get_u32(&log.fields, "study_id")?;
        if !self.study_exists(study_id, log, worker_id)? {
            return Ok(());
        }
        let attrs = log
            .fields
            .get("user_attr")
            .and_then(|v| v.as_object())
            .ok_or_else(|| {
                Error::with_reason(
                    ErrorKind::StorageError,
                    "Failed to parse user_attr object from set study user attr log",
                )
            })?;
        if let Some(study) = self.studies.get_mut(&study_id) {
            for (key, value) in attrs {
                let value = json_value_to_attr(value)?;
                study.attrs.insert(AttrKey::User(key.clone().into()), value);
            }
        }
        Ok(())
    }

    fn apply_set_study_system_attr(&mut self, log: &JournalLog, worker_id: &str) -> Result<()> {
        let study_id = get_u32(&log.fields, "study_id")?;
        if !self.study_exists(study_id, log, worker_id)? {
            return Ok(());
        }
        let attrs = log
            .fields
            .get("system_attr")
            .and_then(|v| v.as_object())
            .ok_or_else(|| {
                Error::with_reason(
                    ErrorKind::StorageError,
                    "Failed to parse system_attr object from set study system attr log",
                )
            })?;
        if let Some(study) = self.studies.get_mut(&study_id) {
            for (key, value) in attrs {
                let v = if key.starts_with("category_labels:") {
                    value
                        .as_str()
                        .ok_or_else(|| {
                            Error::with_reason(
                                ErrorKind::StorageError,
                                format!("Failed to parse category label as string: key={key}"),
                            )
                        })?
                        .to_string()
                } else {
                    json_value_to_attr(value)?
                };
                study.attrs.insert(AttrKey::System(key.clone().into()), v);
            }
        }
        Ok(())
    }

    fn apply_create_trial(&mut self, log: &JournalLog, worker_id: &str) -> Result<()> {
        let study_id = get_u32(&log.fields, "study_id")?;
        if !self.study_exists(study_id, log, worker_id)? {
            return Ok(());
        }
        let trial_id = self.trial_id_to_study_number.len() as u32;
        let trial_number = self
            .study_id_to_trial_ids
            .get(&study_id)
            .map(|ids| ids.len())
            .unwrap_or(0) as u32;

        let mut trial = PersistedTrial::new(trial_id, study_id, trial_number);
        let state_code = log
            .fields
            .get("state")
            .and_then(|v| v.as_i64())
            .unwrap_or(0);
        trial.state_values = match state_code {
            0 => TrialStateValues::Running,
            1 => {
                if let Some(values) = log.fields.get("values").and_then(|v| v.as_array()) {
                    let parsed = values
                        .iter()
                        .map(parse_f64_value)
                        .collect::<Result<Vec<_>>>()?;
                    TrialStateValues::Complete(parsed)
                } else if let Some(value) = log.fields.get("value") {
                    let parsed = parse_f64_value(value)?;
                    TrialStateValues::Complete(vec![parsed])
                } else {
                    TrialStateValues::Complete(Vec::new())
                }
            }
            2 => TrialStateValues::Pruned,
            3 => TrialStateValues::Waiting,
            4 => TrialStateValues::Fail,
            _ => {
                return Err(Error::with_reason(
                    ErrorKind::StorageError,
                    format!("Invalid trial state code: {state_code}"),
                ))
            }
        };

        if let Some(params) = log.fields.get("params").and_then(|v| v.as_object()) {
            for (name, val) in params {
                if let Some(value) = val.as_f64() {
                    trial.internal_params.insert(name.clone(), value);
                }
            }
        }
        if let Some(dists) = log.fields.get("distributions").and_then(|v| v.as_object()) {
            for (name, dist_json) in dists {
                let dist_json = dist_json.as_str().ok_or_else(|| {
                    Error::with_reason(
                        ErrorKind::StorageError,
                        format!("Failed to parse distribution as string: param_name={name}"),
                    )
                })?;
                let (dist, labels) = json_to_distribution(dist_json)?;
                if let Some(labels) = labels {
                    let attrs = category_labels_to_attrs(name, &labels);
                    if let Some(study) = self.studies.get_mut(&study_id) {
                        for (k, v) in attrs {
                            study.attrs.entry(k).or_insert(v);
                        }
                    }
                }
                trial.distributions.insert(name.clone(), dist);
            }
        }

        let mut attrs: Attrs = Attrs::new();

        if let Some(user_attrs) = log.fields.get("user_attrs").and_then(|v| v.as_object()) {
            for (k, v) in user_attrs {
                let value = json_value_to_attr(v)?;
                attrs.insert(AttrKey::User(k.clone().into()), value);
            }
        }
        if let Some(system_attrs) = log.fields.get("system_attrs").and_then(|v| v.as_object()) {
            for (k, v) in system_attrs {
                let value = json_value_to_attr(v)?;
                attrs.insert(AttrKey::System(k.clone().into()), value);
            }
        }
        trial.attrs = attrs;
        if let Some(values) = log
            .fields
            .get("intermediate_values")
            .and_then(|v| v.as_object())
        {
            trial.intermediate_values = intermediate_values_from_map(values)?;
        }

        if let Some(dt) = log.fields.get("datetime_start").and_then(|v| v.as_str()) {
            trial.datetime_start = Some(dt.to_string());
        }
        if let Some(dt) = log.fields.get("datetime_complete").and_then(|v| v.as_str()) {
            trial.datetime_complete = Some(dt.to_string());
        }
        self.trials_by_study
            .entry(study_id)
            .or_default()
            .push(trial);
        self.study_id_to_trial_ids
            .entry(study_id)
            .or_default()
            .push(trial_id);
        self.trial_id_to_study_number
            .insert(trial_id, (study_id, trial_number));
        self.trial_id_to_study_id.insert(trial_id, study_id);

        if self.is_issued_by_this_worker(log, worker_id) {
            self.last_created_trial_id_by_this_process = Some(trial_id);
            if matches!(state_code, 0) {
                self.worker_id_to_owned_trial_id
                    .insert(worker_id.to_string(), trial_id);
            }
        }
        Ok(())
    }

    fn apply_set_trial_param(&mut self, log: &JournalLog, worker_id: &str) -> Result<()> {
        let trial_id = get_u32(&log.fields, "trial_id")?;
        if !self.trial_exists_and_updatable(trial_id, log, worker_id)? {
            return Ok(());
        }
        let param_name = get_string(&log.fields, "param_name")?;
        let param_value = get_f64(&log.fields, "param_value_internal")?;
        let dist_json = get_string(&log.fields, "distribution")?;
        let (dist, labels) = json_to_distribution(&dist_json)?;
        let (study_id, trial_number) = self
            .trial_id_to_study_number
            .get(&trial_id)
            .copied()
            .ok_or_else(|| {
                Error::with_reason(
                    ErrorKind::TrialNotFound,
                    format!("Trial not found during set param: trial_id={trial_id}"),
                )
            })?;
        if let Some(existing) = self.existing_distribution(study_id, &param_name) {
            if existing.check_compatibility(&dist).is_err() {
                if self.is_issued_by_this_worker(log, worker_id) {
                    return Err(Error::with_reason(
                        ErrorKind::IncompatibleDistribution,
                        format!("Incompatible distribution for parameter: param_name={param_name}"),
                    ));
                }
                return Ok(());
            }
        }

        if let Some(labels) = labels {
            let attrs = category_labels_to_attrs(&param_name, &labels);
            if let Some(study) = self.studies.get_mut(&study_id) {
                for (k, v) in attrs {
                    study.attrs.entry(k).or_insert(v);
                }
            }
        }

        let trials = self.trials_by_study.get_mut(&study_id).ok_or_else(|| {
            Error::with_reason(
                ErrorKind::StudyNotFound,
                format!("Study not found during set trial param: study_id={study_id}"),
            )
        })?;
        let trial = trials.get_mut(trial_number as usize).ok_or_else(|| {
            Error::with_reason(
                ErrorKind::TrialNotFound,
                format!(
                    "Trial not found at position during set param: trial_number={trial_number}"
                ),
            )
        })?;
        trial
            .internal_params
            .insert(param_name.clone(), param_value);
        trial.distributions.insert(param_name.clone(), dist);
        let cache = self.study_caches.entry(study_id).or_default();
        let dist_clone = trial
            .distributions
            .get(&param_name)
            .ok_or_else(|| {
                Error::with_reason(
                    ErrorKind::StorageError,
                    format!("Failed to retrieve inserted distribution: param_name={param_name}"),
                )
            })?
            .clone();
        cache
            .param_distribution
            .insert(param_name.clone(), dist_clone);
        Ok(())
    }

    fn apply_set_trial_state_values(&mut self, log: &JournalLog, worker_id: &str) -> Result<()> {
        let trial_id = get_u32(&log.fields, "trial_id")?;
        if !self.trial_exists_and_updatable(trial_id, log, worker_id)? {
            return Ok(());
        }
        let state_code = get_i64(&log.fields, "state")?;
        let (study_id, trial_number) = self
            .trial_id_to_study_number
            .get(&trial_id)
            .copied()
            .ok_or_else(|| {
                Error::with_reason(
                    ErrorKind::TrialNotFound,
                    format!("Trial not found during set state: trial_id={trial_id}"),
                )
            })?;
        let issued_by_this_worker = self.is_issued_by_this_worker(log, worker_id);
        let trials = self.trials_by_study.get_mut(&study_id).ok_or_else(|| {
            Error::with_reason(
                ErrorKind::StudyNotFound,
                format!("Study not found during set trial state: study_id={study_id}"),
            )
        })?;
        let trial = trials.get_mut(trial_number as usize).ok_or_else(|| {
            Error::with_reason(
                ErrorKind::TrialNotFound,
                format!(
                    "Trial not found at position during set state: trial_number={trial_number}"
                ),
            )
        })?;

        let state = match state_code {
            0 => TrialStateValues::Running,
            1 => {
                let values = log
                    .fields
                    .get("values")
                    .and_then(|v| v.as_array())
                    .map(|vals| vals.iter().map(parse_f64_value).collect::<Result<Vec<_>>>())
                    .transpose()?
                    .unwrap_or_default();
                TrialStateValues::Complete(values)
            }
            2 => TrialStateValues::Pruned,
            3 => TrialStateValues::Waiting,
            4 => TrialStateValues::Fail,
            _ => {
                return Err(Error::with_reason(
                    ErrorKind::StorageError,
                    format!("Invalid trial state code: {state_code}"),
                ))
            }
        };
        if state_code == 0 {
            if let Some(dt) = log.fields.get("datetime_start").and_then(|v| v.as_str()) {
                trial.datetime_start = Some(dt.to_string());
            }
            if issued_by_this_worker {
                self.worker_id_to_owned_trial_id
                    .insert(worker_id.to_string(), trial_id);
            }
        } else if matches!(state_code, 1 | 2 | 4) {
            if let Some(dt) = log.fields.get("datetime_complete").and_then(|v| v.as_str()) {
                trial.datetime_complete = Some(dt.to_string());
            }
        }
        trial.state_values = state;
        Ok(())
    }

    fn apply_set_trial_intermediate_value(
        &mut self,
        log: &JournalLog,
        worker_id: &str,
    ) -> Result<()> {
        let trial_id = get_u32(&log.fields, "trial_id")?;
        if !self.trial_exists_and_updatable(trial_id, log, worker_id)? {
            return Ok(());
        }
        let step = get_u32(&log.fields, "step")?;
        let value = get_f64(&log.fields, "intermediate_value")?;
        let (study_id, trial_number) = self
            .trial_id_to_study_number
            .get(&trial_id)
            .copied()
            .ok_or_else(|| {
                Error::with_reason(
                    ErrorKind::TrialNotFound,
                    format!("Trial not found during set intermediate value: trial_id={trial_id}"),
                )
            })?;
        let trials = self.trials_by_study.get_mut(&study_id).ok_or_else(|| {
            Error::with_reason(
                ErrorKind::StudyNotFound,
                format!("Study not found during set intermediate value: study_id={study_id}"),
            )
        })?;
        let trial = trials.get_mut(trial_number as usize).ok_or_else(|| {
            Error::with_reason(
                ErrorKind::TrialNotFound,
                format!(
                    "Trial not found at position during set intermediate value: trial_number={trial_number}"
                ),
            )
        })?;
        trial.intermediate_values.insert(step, value);
        Ok(())
    }

    fn apply_set_trial_user_attr(&mut self, log: &JournalLog, worker_id: &str) -> Result<()> {
        let trial_id = get_u32(&log.fields, "trial_id")?;
        if !self.trial_exists_and_updatable(trial_id, log, worker_id)? {
            return Ok(());
        }
        let attrs = log
            .fields
            .get("user_attr")
            .and_then(|v| v.as_object())
            .ok_or_else(|| {
                Error::with_reason(
                    ErrorKind::StorageError,
                    "Failed to parse user_attr object from set trial user attr log",
                )
            })?;
        let (study_id, trial_number) = self
            .trial_id_to_study_number
            .get(&trial_id)
            .copied()
            .ok_or_else(|| {
                Error::with_reason(
                    ErrorKind::TrialNotFound,
                    format!("Trial not found during set user attr: trial_id={trial_id}"),
                )
            })?;
        let trials = self.trials_by_study.get_mut(&study_id).ok_or_else(|| {
            Error::with_reason(
                ErrorKind::StudyNotFound,
                format!("Study not found during set trial user attr: study_id={study_id}"),
            )
        })?;
        let trial = trials.get_mut(trial_number as usize).ok_or_else(|| {
            Error::with_reason(
                ErrorKind::TrialNotFound,
                format!(
                    "Trial not found at position during set user attr: trial_number={trial_number}"
                ),
            )
        })?;
        for (key, value) in attrs {
            let v = json_value_to_attr(value)?;
            trial.attrs.insert(AttrKey::User(key.clone().into()), v);
        }
        Ok(())
    }

    fn apply_set_trial_system_attr(&mut self, log: &JournalLog, worker_id: &str) -> Result<()> {
        let trial_id = get_u32(&log.fields, "trial_id")?;
        let attrs = log
            .fields
            .get("system_attr")
            .and_then(|v| v.as_object())
            .ok_or_else(|| {
                Error::with_reason(
                    ErrorKind::StorageError,
                    "Failed to parse system_attr object from set trial system attr log",
                )
            })?;
        let (study_id, trial_number) = match self.trial_id_to_study_number.get(&trial_id) {
            Some(v) => *v,
            None => {
                if self.is_issued_by_this_worker(log, worker_id) {
                    return Err(Error::with_reason(
                        ErrorKind::TrialNotFound,
                        format!("Trial not found during set system attr: trial_id={trial_id}"),
                    ));
                }
                return Ok(());
            }
        };
        let trials = self.trials_by_study.get_mut(&study_id).ok_or_else(|| {
            Error::with_reason(
                ErrorKind::StudyNotFound,
                format!("Study not found during set trial system attr: study_id={study_id}"),
            )
        })?;
        let trial = trials.get_mut(trial_number as usize).ok_or_else(|| {
            Error::with_reason(
                ErrorKind::TrialNotFound,
                format!(
                    "Trial not found at position during set system attr: trial_number={trial_number}"
                ),
            )
        })?;
        let is_finished = trial.is_finished();
        for (key, value) in attrs {
            if is_finished {
                if self.is_issued_by_this_worker(log, worker_id) {
                    return Err(Error::with_reason(
                        ErrorKind::TrialAlreadyFinished,
                        format!("Trial already finished, cannot set system attr: key={key}"),
                    ));
                }
                return Ok(());
            }
            let v = json_value_to_attr(value)?;
            trial.attrs.insert(AttrKey::System(key.clone().into()), v);
        }
        Ok(())
    }

    fn trial_id_from_study_number(&self, study_id: u32, trial_number: u32) -> Result<u32> {
        let trial_ids = self.study_id_to_trial_ids.get(&study_id).ok_or_else(|| {
            Error::with_reason(
                ErrorKind::StudyNotFound,
                format!("Study not found in trial id lookup: study_id={study_id}"),
            )
        })?;
        let trial_id = trial_ids.get(trial_number as usize).ok_or_else(|| {
            Error::with_reason(
                ErrorKind::TrialNotFound,
                format!(
                    "Trial not found at position in trial id lookup: trial_number={trial_number}"
                ),
            )
        })?;
        Ok(*trial_id)
    }

    fn trial_from_study_number(&self, study_id: u32, trial_number: u32) -> Result<&PersistedTrial> {
        self.trials_by_study
            .get(&study_id)
            .and_then(|trials| trials.get(trial_number as usize))
            .ok_or_else(|| {
                Error::with_reason(
                    ErrorKind::TrialNotFound,
                    format!(
                        "Trial not found at position: study_id={study_id}, trial_number={trial_number}"
                    ),
                )
            })
    }

    fn ensure_trial_updatable(&self, trial_id: u32) -> Result<()> {
        let (study_id, trial_number) = self
            .trial_id_to_study_number
            .get(&trial_id)
            .copied()
            .ok_or_else(|| {
                Error::with_reason(
                    ErrorKind::TrialNotFound,
                    format!("Trial not found during updatable check: trial_id={trial_id}"),
                )
            })?;
        let trial = self
            .trials_by_study
            .get(&study_id)
            .and_then(|trials| trials.get(trial_number as usize))
            .ok_or_else(|| {
                Error::with_reason(
                    ErrorKind::TrialNotFound,
                    format!(
                        "Trial not found at position during updatable check: trial_number={trial_number}"
                    ),
                )
            })?;
        if trial.is_finished() {
            return Err(Error::with_reason(
                ErrorKind::TrialAlreadyFinished,
                format!("Trial already finished and cannot be updated: trial_id={trial_id}"),
            ));
        }
        Ok(())
    }

    fn check_param_compatibility(
        &self,
        study_id: u32,
        param_name: &str,
        distribution: &Distribution,
    ) -> Result<()> {
        if let Some(existing) = self.existing_distribution(study_id, param_name) {
            existing.check_compatibility(distribution)?;
        }
        Ok(())
    }

    fn labels_for_param(&self, study_id: u32, param_name: &str) -> Option<Vec<CategoryLabel>> {
        let study = self.studies.get(&study_id)?;
        let labels = extract_category_labels(&study.attrs, param_name)?;
        Some(labels)
    }

    fn existing_distribution(&self, study_id: u32, param_name: &str) -> Option<&Distribution> {
        let trials = self.trials_by_study.get(&study_id)?;
        for trial in trials {
            if let Some(dist) = trial.distributions.get(param_name) {
                return Some(dist);
            }
        }
        None
    }
}

fn get_string(map: &Map<String, Value>, key: &str) -> Result<String> {
    map.get(key)
        .and_then(|v| v.as_str())
        .map(|v| v.to_string())
        .ok_or_else(|| {
            Error::with_reason(
                ErrorKind::StorageError,
                format!("Failed to get string value from map: key={key}"),
            )
        })
}

fn get_u32(map: &Map<String, Value>, key: &str) -> Result<u32> {
    map.get(key)
        .and_then(|v| v.as_u64())
        .map(|v| v as u32)
        .ok_or_else(|| {
            Error::with_reason(
                ErrorKind::StorageError,
                format!("Failed to get u32 value from map: key={key}"),
            )
        })
}

fn get_i64(map: &Map<String, Value>, key: &str) -> Result<i64> {
    map.get(key).and_then(|v| v.as_i64()).ok_or_else(|| {
        Error::with_reason(
            ErrorKind::StorageError,
            format!("Failed to get i64 value from map: key={key}"),
        )
    })
}

fn get_f64(map: &Map<String, Value>, key: &str) -> Result<f64> {
    map.get(key)
        .map(parse_f64_value)
        .transpose()?
        .ok_or_else(|| {
            Error::with_reason(
                ErrorKind::StorageError,
                format!("Failed to get f64 value from map: key={key}"),
            )
        })
}

fn distribution_to_json(
    distribution: &Distribution,
    labels: Option<&[CategoryLabel]>,
) -> Result<String> {
    let (name, attributes) = match distribution {
        Distribution::Float {
            low,
            high,
            step,
            log,
        } => (
            "FloatDistribution",
            json_map(vec![
                (
                    "low",
                    Value::Number(Number::from_f64(*low).ok_or_else(|| {
                        Error::with_reason(
                            ErrorKind::StorageError,
                            format!("Failed to convert low f64 to JSON: {low}"),
                        )
                    })?),
                ),
                (
                    "high",
                    Value::Number(Number::from_f64(*high).ok_or_else(|| {
                        Error::with_reason(
                            ErrorKind::StorageError,
                            format!("Failed to convert high f64 to JSON: {high}"),
                        )
                    })?),
                ),
                (
                    "step",
                    step.map(|s| {
                        Number::from_f64(s).map(Value::Number).ok_or_else(|| {
                            Error::with_reason(
                                ErrorKind::StorageError,
                                format!("Failed to convert step f64 to JSON: {s}"),
                            )
                        })
                    })
                    .transpose()?
                    .unwrap_or(Value::Null),
                ),
                ("log", Value::Bool(*log)),
            ]),
        ),
        Distribution::Int {
            low,
            high,
            step,
            log,
        } => (
            "IntDistribution",
            json_map(vec![
                ("low", Value::Number(Number::from(*low))),
                ("high", Value::Number(Number::from(*high))),
                ("step", Value::Number(Number::from(*step))),
                ("log", Value::Bool(*log)),
            ]),
        ),
        Distribution::Categorical { cardinality } => {
            let choices = labels
                .map(|ls| ls.iter().map(category_label_to_value).collect::<Vec<_>>())
                .unwrap_or_else(|| {
                    (0..*cardinality as u32)
                        .map(|i| Value::Number(Number::from(i)))
                        .collect::<Vec<_>>()
                });
            (
                "CategoricalDistribution",
                json_map(vec![("choices", Value::Array(choices))]),
            )
        }
    };

    let mut root = Map::new();
    root.insert("name".to_string(), Value::String(name.to_string()));
    root.insert("attributes".to_string(), Value::Object(attributes));
    serde_json::to_string(&Value::Object(root)).map_err(|_| {
        Error::with_reason(
            ErrorKind::StorageError,
            "Failed to serialize distribution to JSON",
        )
    })
}

fn json_to_distribution(
    distribution_json: &str,
) -> Result<(Distribution, Option<Vec<CategoryLabel>>)> {
    let value: Value = serde_json::from_str(distribution_json).map_err(|_| {
        Error::with_reason(ErrorKind::StorageError, "Failed to parse distribution JSON")
    })?;
    let name = value.get("name").and_then(|v| v.as_str()).ok_or_else(|| {
        Error::with_reason(
            ErrorKind::StorageError,
            "Failed to get name field from distribution JSON",
        )
    })?;
    let attributes = value
        .get("attributes")
        .and_then(|v| v.as_object())
        .ok_or_else(|| {
            Error::with_reason(
                ErrorKind::StorageError,
                "Failed to get attributes field from distribution JSON",
            )
        })?;

    match name {
        "FloatDistribution" => {
            let low = attributes
                .get("low")
                .and_then(|v| v.as_f64())
                .ok_or_else(|| {
                    Error::with_reason(
                        ErrorKind::StorageError,
                        "Failed to get low value from float distribution",
                    )
                })?;
            let high = attributes
                .get("high")
                .and_then(|v| v.as_f64())
                .ok_or_else(|| {
                    Error::with_reason(
                        ErrorKind::StorageError,
                        "Failed to get high value from float distribution",
                    )
                })?;
            let log = attributes
                .get("log")
                .and_then(|v| v.as_bool())
                .ok_or_else(|| {
                    Error::with_reason(
                        ErrorKind::StorageError,
                        "Failed to get log flag from float distribution",
                    )
                })?;
            let step = match attributes.get("step") {
                Some(Value::Null) | None => None,
                Some(Value::Number(n)) => n.as_f64(),
                Some(Value::String(s)) => s.parse::<f64>().ok(),
                _ => None,
            };
            Ok((
                Distribution::Float {
                    low,
                    high,
                    step,
                    log,
                },
                None,
            ))
        }
        "IntDistribution" => {
            let low = attributes
                .get("low")
                .and_then(|v| v.as_i64())
                .ok_or_else(|| {
                    Error::with_reason(
                        ErrorKind::StorageError,
                        "Failed to get low value from int distribution",
                    )
                })?;
            let high = attributes
                .get("high")
                .and_then(|v| v.as_i64())
                .ok_or_else(|| {
                    Error::with_reason(
                        ErrorKind::StorageError,
                        "Failed to get high value from int distribution",
                    )
                })?;
            let log = attributes
                .get("log")
                .and_then(|v| v.as_bool())
                .ok_or_else(|| {
                    Error::with_reason(
                        ErrorKind::StorageError,
                        "Failed to get log flag from int distribution",
                    )
                })?;
            let step = match attributes.get("step") {
                Some(Value::Null) | None => 1,
                Some(Value::Number(n)) => n.as_i64().unwrap_or(1),
                Some(Value::String(s)) => s.parse::<i64>().unwrap_or(1),
                _ => 1,
            };
            Ok((
                Distribution::Int {
                    low,
                    high,
                    step,
                    log,
                },
                None,
            ))
        }
        "CategoricalDistribution" => {
            let size = match attributes.get("size") {
                Some(v) => v.as_u64(),
                None => attributes
                    .get("choices")
                    .and_then(|v| v.as_array())
                    .map(|a| a.len() as u64),
            }
            .ok_or_else(|| {
                Error::with_reason(
                    ErrorKind::StorageError,
                    "Failed to get size from categorical distribution",
                )
            })?;
            let labels = attributes.get("choices").and_then(|arr| {
                arr.as_array().map(|vals| {
                    vals.iter()
                        .filter_map(value_to_category_label)
                        .collect::<Vec<_>>()
                })
            });
            Ok((
                Distribution::Categorical {
                    cardinality: size as usize,
                },
                labels,
            ))
        }
        _ => Err(Error::with_reason(
            ErrorKind::StorageError,
            format!("Unknown distribution type: {name}"),
        )),
    }
}

fn category_label_to_value(label: &CategoryLabel) -> Value {
    match label {
        CategoryLabel::Float(f) => Number::from_f64(*f)
            .map(Value::Number)
            .unwrap_or(Value::Null),
        CategoryLabel::Int(i) => Value::Number(Number::from(*i)),
        CategoryLabel::String(s) => Value::String(s.clone()),
        CategoryLabel::Bool(b) => Value::Bool(*b),
        CategoryLabel::None => Value::Null,
    }
}

fn value_to_category_label(v: &Value) -> Option<CategoryLabel> {
    match v {
        Value::Null => Some(CategoryLabel::None),
        Value::Bool(b) => Some(CategoryLabel::Bool(*b)),
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Some(CategoryLabel::Int(i))
            } else {
                n.as_f64().map(CategoryLabel::Float)
            }
        }
        Value::String(s) => Some(CategoryLabel::String(s.clone())),
        _ => None,
    }
}

fn json_map(entries: Vec<(&str, Value)>) -> Map<String, Value> {
    let mut map = Map::new();
    for (k, v) in entries {
        map.insert(k.to_string(), v);
    }
    map
}

fn intermediate_values_from_map(map: &Map<String, Value>) -> Result<HashMap<u32, f64>> {
    let mut values = HashMap::with_capacity(map.len());
    for (k, v) in map {
        let step: u32 = k.parse().map_err(|_| {
            Error::with_reason(
                ErrorKind::StorageError,
                format!("Failed to parse step as u32: {k}"),
            )
        })?;
        let value = parse_f64_value(v)?;
        values.insert(step, value);
    }
    Ok(values)
}
fn unique_prefix() -> String {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    format!("{nanos}")
}

fn parse_f64_value(value: &Value) -> Result<f64> {
    if let Some(v) = value.as_f64() {
        return Ok(v);
    }
    if let Some(s) = value.as_str() {
        match s {
            "nan" | "NaN" => return Ok(f64::NAN),
            "inf" | "Infinity" | "INF" => return Ok(f64::INFINITY),
            "-inf" | "-Infinity" | "-INF" => return Ok(f64::NEG_INFINITY),
            _ => {}
        }
        if let Ok(v) = s.parse::<f64>() {
            return Ok(v);
        }
    }
    Err(Error::with_reason(
        ErrorKind::StorageError,
        format!("Failed to parse f64 value: {value:?}"),
    ))
}

fn json_value_to_attr(value: &Value) -> Result<String> {
    if let Value::String(s) = value {
        if matches!(s.as_str(), "Infinity" | "-Infinity" | "NaN") {
            return Ok(s.clone());
        }
        if serde_json::from_str::<Value>(s).is_ok() {
            return Ok(s.clone());
        }
    }
    serde_json::to_string(value).map_err(|_| {
        Error::with_reason(
            ErrorKind::StorageError,
            format!("Failed to serialize value to JSON: {value:?}"),
        )
    })
}

fn value_to_json(value: f64) -> Value {
    if value.is_nan() {
        return Value::String("NaN".to_string());
    }
    if value.is_infinite() {
        if value.is_sign_positive() {
            return Value::String("Infinity".to_string());
        }
        return Value::String("-Infinity".to_string());
    }
    Number::from_f64(value)
        .map(Value::Number)
        .unwrap_or(Value::Null)
}

fn extract_category_labels(attrs: &Attrs, param_name: &str) -> Option<Vec<CategoryLabel>> {
    let mut labels = Vec::new();
    let mut index = 0;
    loop {
        let key = AttrKey::System(format!("category_labels:{param_name}:{index}").into());
        match attrs.get(&key) {
            Some(raw) => {
                let label = CategoryLabel::deserialize(raw)?;
                labels.push(label);
                index += 1;
            }
            None => break,
        }
    }
    if labels.is_empty() {
        None
    } else {
        Some(labels)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rustuna_core::storage::Storage;
    use rustuna_core::ErrorKind;
    use std::sync::{Arc, Mutex};

    struct InMemoryJournalBackend {
        logs: Arc<Mutex<Vec<JournalLog>>>,
    }

    impl JournalBackend for InMemoryJournalBackend {
        fn read_logs(
            &mut self,
            log_number_from: usize,
            handler: &mut dyn FnMut(JournalLog) -> Result<()>,
        ) -> Result<()> {
            let logs = self.logs.lock().map_err(|_| {
                Error::with_reason(
                    ErrorKind::StorageError,
                    "Failed to acquire lock on journal logs",
                )
            })?;
            if log_number_from >= logs.len() {
                return Ok(());
            }
            for log in &logs[log_number_from..] {
                handler(log.clone())?;
            }
            Ok(())
        }

        fn append_logs(&mut self, logs: &[JournalLog]) -> Result<()> {
            let mut guard = self.logs.lock().map_err(|_| {
                Error::with_reason(
                    ErrorKind::StorageError,
                    "Failed to acquire lock on journal logs",
                )
            })?;
            guard.extend_from_slice(logs);
            Ok(())
        }
    }

    fn new_storage() -> Result<(JournalStorage, Arc<Mutex<Vec<JournalLog>>>)> {
        let logs = Arc::new(Mutex::new(Vec::new()));
        let backend = InMemoryJournalBackend { logs: logs.clone() };
        let storage = JournalStorage::new(Box::new(backend))?;
        Ok((storage, logs))
    }

    #[test]
    fn test_create_trial_order_across_storages() -> Result<()> {
        let logs = Arc::new(Mutex::new(Vec::new()));
        let backend1 = InMemoryJournalBackend { logs: logs.clone() };
        let backend2 = InMemoryJournalBackend { logs: logs.clone() };
        let mut storage1 = JournalStorage::new(Box::new(backend1))?;
        let mut storage2 = JournalStorage::new(Box::new(backend2))?;

        let study_id = storage1
            .create_new_study("shared", vec![Direction::Minimize])?
            .id;
        let study_id2 = storage2.get_studies()?.first().map(|s| s.id).unwrap_or(0);
        assert_eq!(study_id2, study_id);

        let trial1 = storage1.create_new_trial(study_id)?;
        let trial2 = storage2.create_new_trial(study_id)?;

        assert_eq!(trial1.number, 0);
        assert_eq!(trial2.number, 1);
        Ok(())
    }

    #[test]
    fn test_worker_id_prefix_includes_process_id() -> Result<()> {
        let (storage, _logs) = new_storage()?;
        let pid = std::process::id();
        assert!(storage
            .replay
            .worker_id_prefix
            .contains(&format!("-{pid}-")));
        Ok(())
    }

    fn append_external_log(logs: &Arc<Mutex<Vec<JournalLog>>>, log: JournalLog) {
        let mut guard = logs.lock().expect("lock logs");
        guard.push(log);
    }

    fn create_study_log(study_name: &str, directions: Vec<Direction>) -> JournalLog {
        let mut fields = Map::new();
        fields.insert(
            "study_name".to_string(),
            Value::String(study_name.to_string()),
        );
        let dirs = directions
            .into_iter()
            .map(|d| match d {
                Direction::Minimize => Value::Number(Number::from(1)),
                Direction::Maximize => Value::Number(Number::from(2)),
            })
            .collect::<Vec<_>>();
        fields.insert("directions".to_string(), Value::Array(dirs));
        JournalLog {
            op_code: JournalOperation::CreateStudy as i32,
            worker_id: "external".to_string(),
            fields,
        }
    }

    fn create_trial_log(study_id: u32) -> JournalLog {
        let mut fields = Map::new();
        fields.insert(
            "study_id".to_string(),
            Value::Number(Number::from(study_id)),
        );
        fields.insert(
            "datetime_start".to_string(),
            Value::String("2024-01-01 00:00:00".to_string()),
        );
        JournalLog {
            op_code: JournalOperation::CreateTrial as i32,
            worker_id: "external".to_string(),
            fields,
        }
    }

    #[test]
    fn create_new_study_updates_cache() -> Result<()> {
        let (mut storage, _logs) = new_storage()?;
        let study = storage.create_new_study("example", vec![Direction::Minimize])?;
        assert_eq!(study.name, "example");
        assert_eq!(study.directions, vec![Direction::Minimize]);
        assert_eq!(storage.get_studies()?.len(), 1);
        Ok(())
    }

    #[test]
    fn create_new_study_rejects_duplicate() -> Result<()> {
        let (mut storage, _logs) = new_storage()?;
        storage.create_new_study("example", vec![Direction::Minimize])?;
        let res = storage.create_new_study("example", vec![Direction::Minimize]);
        match res {
            Err(e) => assert!(matches!(e.kind, ErrorKind::DuplicatedStudy)),
            Ok(_) => panic!("Expected duplicate study error"),
        }
        Ok(())
    }

    #[test]
    fn test_create_study_does_not_reuse_study_id() -> Result<()> {
        let (mut storage, _logs) = new_storage()?;
        let study1 = storage.create_new_study("study1", vec![Direction::Minimize])?;
        let study1_id = study1.id;
        storage.create_new_study("study2", vec![Direction::Minimize])?;
        storage.delete_study(study1_id)?;

        let err = storage
            .get_study(study1_id)
            .err()
            .expect("Expected StudyNotFound error");
        assert!(matches!(err.kind, ErrorKind::StudyNotFound));

        let study3 = storage.create_new_study("study3", vec![Direction::Minimize])?;
        assert_eq!(study3.id, 2);
        assert_ne!(study3.id, study1_id);
        assert_eq!(storage.get_studies()?.len(), 2);
        Ok(())
    }

    #[test]
    fn get_study_and_get_studies_use_cache() -> Result<()> {
        let (mut storage, _logs) = new_storage()?;
        storage.create_new_study("s1", vec![Direction::Minimize])?;
        storage.create_new_study("s2", vec![Direction::Maximize])?;

        let all = storage.get_studies()?;
        assert_eq!(all.len(), 2);

        let s1 = storage.get_study(0)?;
        assert_eq!(s1.name, "s1");
        let s2 = storage.get_study(1)?;
        assert_eq!(s2.name, "s2");
        Ok(())
    }

    #[test]
    fn create_new_trial_appends_cache() -> Result<()> {
        let (mut storage, _logs) = new_storage()?;
        let study = storage.create_new_study("s", vec![Direction::Minimize])?.id;
        let t0_num = storage.create_new_trial(study)?.number;
        let t1_num = storage.create_new_trial(study)?.number;
        assert_eq!(t0_num, 0);
        assert_eq!(t1_num, 1);
        let trials = storage.get_trials(study)?;
        assert_eq!(trials.len(), 2);
        Ok(())
    }

    #[test]
    fn get_trials_and_get_trial_return_cached_refs() -> Result<()> {
        let (mut storage, _logs) = new_storage()?;
        let study_id = storage.create_new_study("s", vec![Direction::Minimize])?.id;
        let t0_id = storage.create_new_trial(study_id)?.id;
        let t1_id = storage.create_new_trial(study_id)?.id;

        let trials = storage.get_trials(study_id)?;
        assert_eq!(trials.len(), 2);
        let t0 = storage.get_trial(t0_id)?;
        assert_eq!(t0.number, 0);
        let t1 = storage.get_trial(t1_id)?;
        assert_eq!(t1.number, 1);
        Ok(())
    }

    #[test]
    fn get_trials_replays_from_backend_logs() -> Result<()> {
        let (mut storage, logs) = new_storage()?;
        let study_id = storage.create_new_study("s", vec![Direction::Minimize])?.id;
        let _trial = storage.create_new_trial(study_id)?;

        let backend = InMemoryJournalBackend { logs: logs.clone() };
        let mut storage2 = JournalStorage::new(Box::new(backend))?;
        let trials = storage2.get_trials(study_id)?;
        assert_eq!(trials.len(), 1);
        Ok(())
    }

    #[test]
    fn get_studies_refreshes_from_backend_every_time() -> Result<()> {
        let (mut storage, logs) = new_storage()?;
        storage.create_new_study("s", vec![Direction::Minimize])?;

        append_external_log(&logs, create_study_log("s2", vec![Direction::Maximize]));

        let studies = storage.get_studies()?;
        assert_eq!(studies.len(), 2);
        assert!(studies.iter().any(|s| s.name == "s"));
        assert!(studies.iter().any(|s| s.name == "s2"));
        Ok(())
    }

    #[test]
    fn get_trials_refreshes_when_backend_updates() -> Result<()> {
        let (mut storage, logs) = new_storage()?;
        let study_id = storage.create_new_study("s", vec![Direction::Minimize])?.id;
        storage.create_new_trial(study_id)?;

        append_external_log(&logs, create_trial_log(study_id));
        let trials = storage.get_trials(study_id)?;
        assert_eq!(trials.len(), 2);
        Ok(())
    }

    #[test]
    fn set_trial_state_values_updates_cache() -> Result<()> {
        let (mut storage, _logs) = new_storage()?;
        let study_id = storage.create_new_study("s", vec![Direction::Minimize])?.id;
        let trial_id = storage.create_new_trial(study_id)?.id;

        storage.set_trial_state_values(trial_id, TrialStateValues::Complete(vec![1.0]))?;
        let trial = storage.get_trial(trial_id)?;
        assert!(matches!(trial.state_values, TrialStateValues::Complete(_)));
        Ok(())
    }

    #[test]
    fn get_joint_search_space_uses_cache_update() -> Result<()> {
        let (mut storage, _logs) = new_storage()?;
        let study_id = storage.create_new_study("s", vec![Direction::Minimize])?.id;

        let dist = Distribution::Float {
            low: 0.0,
            high: 1.0,
            step: None,
            log: false,
        };
        let trial_id = storage.create_new_trial(study_id)?.id;
        storage.set_trial_param(trial_id, "x", &dist, 0.5)?;
        storage.set_trial_state_values(trial_id, TrialStateValues::Complete(vec![0.0]))?;

        let search_space = storage.get_joint_search_space(study_id)?;
        assert!(search_space.contains_key("x"));
        Ok(())
    }

    #[test]
    fn set_study_and_trial_attrs_update_cache() -> Result<()> {
        let (mut storage, _logs) = new_storage()?;
        let study_id = storage.create_new_study("s", vec![Direction::Minimize])?.id;
        let trial_id = storage.create_new_trial(study_id)?.id;

        let mut s_attrs = Attrs::new();
        s_attrs.insert(AttrKey::User("foo".into()), "bar".to_string());
        storage.set_study_attrs(study_id, s_attrs, false)?;
        let study = storage.get_study(study_id)?;
        assert_eq!(
            study
                .attrs
                .get(&AttrKey::User("foo".into()))
                .expect("User attr 'foo' should exist"),
            "\"bar\""
        );

        let mut t_attrs = Attrs::new();
        t_attrs.insert(AttrKey::System("key".into()), "val".to_string());
        storage.set_trial_attrs(trial_id, t_attrs, false)?;
        let trial = storage.get_trial(trial_id)?;
        assert_eq!(
            trial
                .attrs
                .get(&AttrKey::System("key".into()))
                .expect("System attr 'key' should exist"),
            "\"val\""
        );
        Ok(())
    }

    #[test]
    fn set_trial_param_updates_cache_and_refreshes() -> Result<()> {
        let (mut storage, _logs) = new_storage()?;
        let study_id = storage.create_new_study("s", vec![Direction::Minimize])?.id;
        storage.create_new_trial(study_id)?;

        let dist = Distribution::Float {
            low: 0.0,
            high: 1.0,
            step: None,
            log: false,
        };
        let trial_id = storage.get_trials(study_id)?[0].id;
        storage.set_trial_param(trial_id, "x", &dist, 0.5)?;

        let trial = storage.get_trial(trial_id)?;
        assert_eq!(trial.internal_params.get("x"), Some(&0.5));
        assert_eq!(
            trial.distributions.get("x"),
            Some(&Distribution::Float {
                low: 0.0,
                high: 1.0,
                step: None,
                log: false
            })
        );
        Ok(())
    }

    #[test]
    fn set_trial_param() -> Result<()> {
        let (mut storage, _logs) = new_storage()?;
        let study_id = storage
            .create_new_study("test1", vec![Direction::Minimize])?
            .id;
        storage.create_new_study("test2", vec![Direction::Minimize])?;
        let trial_1_id = storage.create_new_trial(study_id)?.id;
        let trial_2_id = storage.create_new_trial(study_id)?.id;

        let distribution_x = Distribution::Float {
            low: 1.0,
            high: 2.0,
            step: None,
            log: false,
        };
        let distribution_y_1 = Distribution::Categorical { cardinality: 3 };
        let distribution_z = Distribution::Float {
            low: 1.0,
            high: 100.0,
            step: None,
            log: true,
        };

        storage.set_trial_param(trial_1_id, "x", &distribution_x, 0.5)?;
        storage.set_trial_param(trial_1_id, "y", &distribution_y_1, 2.0)?;
        let trial = storage.get_trial(trial_1_id)?;
        assert_eq!(trial.internal_params["x"], 0.5);
        assert_eq!(trial.internal_params["y"], 2.0);

        storage.set_trial_param(trial_2_id, "x", &distribution_x, 0.3)?;
        storage.set_trial_param(trial_2_id, "z", &distribution_z, 0.1)?;
        let trial = storage.get_trial(trial_2_id)?;
        assert_eq!(trial.internal_params["x"], 0.3);
        assert_eq!(trial.internal_params["z"], 0.1);

        Ok(())
    }

    #[test]
    fn set_trial_param_rejects_incompatible_distribution_across_trials() -> Result<()> {
        let (mut storage, _logs) = new_storage()?;
        let study_id = storage
            .create_new_study("test", vec![Direction::Minimize])?
            .id;

        let float_dist = Distribution::Float {
            low: 0.0,
            high: 1.0,
            step: None,
            log: false,
        };
        let int_dist = Distribution::Int {
            low: 0,
            high: 5,
            step: 1,
            log: false,
        };

        let trial0_id = storage.create_new_trial(study_id)?.id;
        storage.set_trial_param(trial0_id, "x", &float_dist, 0.5)?;

        let trial1_id = storage.create_new_trial(study_id)?.id;
        let err = storage
            .set_trial_param(trial1_id, "x", &int_dist, 1.0)
            .expect_err("Expected IncompatibleDistribution error");
        assert!(matches!(err.kind, ErrorKind::IncompatibleDistribution));
        Ok(())
    }
}
