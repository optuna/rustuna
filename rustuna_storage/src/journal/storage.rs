use std::collections::HashMap;
use std::thread;
use std::time::{SystemTime, UNIX_EPOCH};

use serde::Serialize;
use serde_json::value::{to_raw_value, RawValue};
use serde_json::{Map, Number, Value};

use rustuna_core::attr::{
    category_labels_to_attrs, get_category_labels, AttrKey, Attrs, CategoryLabel,
};
use rustuna_core::distribution::Distribution;
use rustuna_core::internal::study_cache::StudyCache;
use rustuna_core::storage::Storage;
use rustuna_core::study::{Direction, PersistedStudy};
use rustuna_core::trial::{PersistedTrial, TrialState, TrialStateValues};
use rustuna_core::{Error, ErrorKind, Result};

use super::{JournalBackend, JournalLog, JournalOperation};
use crate::datetime::{journal_datetime_to_naive_utc, naive_utc_to_aware_utc, now_aware_utc};

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
/// Options for [`JournalStorage`].
pub struct JournalStorageOptions {
    /// If `true`, discarded trials are omitted when replaying the journal.
    /// Discard logs are written regardless of this option.
    pub apply_discard: bool,
}

/// Storage implementation backed by an append-only journal log.
///
/// Similar in spirit to Optuna's `JournalStorage`, this storage writes every state-changing
/// operation as a journal record and reconstructs the latest snapshot by replaying logs.
pub struct JournalStorage {
    backend: Box<dyn JournalBackend>,
    replay: JournalReplayState,
}

impl JournalStorage {
    /// Creates a journal storage and synchronizes its in-memory replay state from the backend.
    pub fn new(backend: Box<dyn JournalBackend>) -> Result<Self> {
        Self::new_with_options(backend, JournalStorageOptions::default())
    }

    /// Creates a journal storage and synchronizes its in-memory replay state from the backend.
    pub fn new_with_options(
        backend: Box<dyn JournalBackend>,
        options: JournalStorageOptions,
    ) -> Result<Self> {
        let worker_id_prefix = format!("{}-{}-", unique_prefix(), std::process::id());
        let mut storage = JournalStorage {
            backend,
            replay: JournalReplayState::new(worker_id_prefix, options.apply_discard),
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

    fn write_log(
        &mut self,
        op_code: JournalOperation,
        fields: HashMap<String, Box<RawValue>>,
    ) -> Result<()> {
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
        let mut first_error: Option<Error> = None;
        self.backend
            .read_logs(self.replay.log_number_read, &mut |log| {
                if first_error.is_none() {
                    if let Err(err) = self
                        .replay
                        .apply_logs(std::slice::from_ref(&log), &worker_id)
                    {
                        first_error = Some(err);
                    }
                } else {
                    let _ = self
                        .replay
                        .apply_logs(std::slice::from_ref(&log), &worker_id);
                }
                Ok(())
            })?;
        if let Some(err) = first_error {
            return Err(err);
        }
        Ok(())
    }
}

impl Storage for JournalStorage {
    fn create_new_study(
        &mut self,
        study_name: &str,
        directions: Vec<Direction>,
    ) -> Result<&PersistedStudy> {
        let mut fields = HashMap::new();
        fields.insert("study_name".to_string(), to_raw(&study_name.to_string())?);
        let directions_val = directions
            .iter()
            .map(|d| match d {
                Direction::Minimize => 1,
                Direction::Maximize => 2,
            })
            .collect::<Vec<_>>();
        fields.insert("directions".to_string(), to_raw(&directions_val)?);
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
        let mut fields = HashMap::new();
        fields.insert("study_id".to_string(), to_raw(&study_id)?);
        self.write_log(JournalOperation::DeleteStudy, fields)?;
        self.sync_with_backend()?;
        Ok(())
    }

    fn create_new_trial(&mut self, study_id: u32) -> Result<&PersistedTrial> {
        let mut fields = HashMap::new();
        fields.insert("study_id".to_string(), to_raw(&study_id)?);
        fields.insert("datetime_start".to_string(), to_raw(&now_aware_utc())?);
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
        trials
            .get(trial_number as usize)
            .and_then(|trial| trial.as_ref())
            .ok_or_else(|| {
                Error::with_reason(
                    ErrorKind::TrialNotFound,
                    format!("Failed to find trial at given number: trial_number={trial_number}"),
                )
            })
    }

    fn create_new_trial_from_template(
        &mut self,
        study_id: u32,
        template: &PersistedTrial,
    ) -> Result<&PersistedTrial> {
        let (state_code, values_raw) = match &template.state_values {
            TrialStateValues::Running => (0, None),
            TrialStateValues::Complete(values) => {
                let arr = values
                    .iter()
                    .map(|value| value_to_json(*value))
                    .collect::<Vec<_>>();
                (1, Some(arr))
            }
            TrialStateValues::Pruned => (2, None),
            TrialStateValues::Fail => (3, None),
            TrialStateValues::Waiting => (4, None),
        };

        let mut user_attrs = HashMap::new();
        let mut system_attrs = HashMap::new();
        for (key, value) in &template.attrs {
            match key {
                AttrKey::User(key) => {
                    user_attrs.insert(key.to_string(), value.clone());
                }
                AttrKey::System(key) => {
                    system_attrs.insert(key.to_string(), value.clone());
                }
            }
        }

        let mut params = HashMap::with_capacity(template.distributions.len());
        let mut distributions = HashMap::with_capacity(template.distributions.len());
        for (param_name, distribution) in &template.distributions {
            let param_value = template.internal_params.get(param_name).ok_or_else(|| {
                Error::with_reason(
                    ErrorKind::StorageError,
                    format!("Template trial has no internal param for '{param_name}'"),
                )
            })?;
            params.insert(param_name.clone(), param_value.to_string());
            let labels = self.replay.labels_for_param(study_id, param_name);
            let dist_json = distribution_to_json(distribution, labels.as_deref())?;
            distributions.insert(param_name.clone(), dist_json);
        }

        let mut intermediate_values_json =
            HashMap::with_capacity(template.intermediate_values.len());
        for (step, value) in &template.intermediate_values {
            intermediate_values_json.insert(step.to_string(), value_to_json(*value));
        }

        let mut fields = HashMap::with_capacity(10);
        fields.insert("study_id".to_string(), to_raw(&study_id)?);
        fields.insert("state".to_string(), to_raw(&state_code)?);
        if let Some(values_raw) = values_raw {
            fields.insert("values".to_string(), to_raw(&values_raw)?);
        } else {
            fields.insert("values".to_string(), to_raw(&Value::Null)?);
        }
        fields.insert("params".to_string(), to_raw(&params)?);
        fields.insert("distributions".to_string(), to_raw(&distributions)?);
        fields.insert("user_attrs".to_string(), to_raw(&user_attrs)?);
        fields.insert("system_attrs".to_string(), to_raw(&system_attrs)?);
        fields.insert(
            "intermediate_values".to_string(),
            to_raw(&intermediate_values_json)?,
        );
        fields.insert(
            "datetime_start".to_string(),
            to_raw(
                &template
                    .datetime_start
                    .as_deref()
                    .map(naive_utc_to_aware_utc),
            )?,
        );
        if let Some(ref dt) = template.datetime_complete {
            fields.insert(
                "datetime_complete".to_string(),
                to_raw(&naive_utc_to_aware_utc(dt))?,
            );
        }
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
        trials
            .get(trial_number as usize)
            .and_then(|trial| trial.as_ref())
            .ok_or_else(|| {
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

        let mut fields = HashMap::new();
        fields.insert("trial_id".to_string(), to_raw(&trial_id)?);
        fields.insert("param_name".to_string(), to_raw(&name.to_string())?);
        fields.insert(
            "param_value_internal".to_string(),
            to_raw(&value_to_json(value))?,
        );
        fields.insert("distribution".to_string(), to_raw(&dist_json)?);
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
            TrialStateValues::Fail => (3, None),
            TrialStateValues::Waiting => (4, None),
        };
        let mut fields = HashMap::new();
        fields.insert("trial_id".to_string(), to_raw(&trial_id)?);
        fields.insert("state".to_string(), to_raw(&state_code)?);
        if let Some(values) = values {
            let arr = values
                .iter()
                .map(|value| value_to_json(*value))
                .collect::<Vec<_>>();
            fields.insert("values".to_string(), to_raw(&arr)?);
        } else {
            fields.insert("values".to_string(), to_raw(&Value::Null)?);
        }
        if state_code == 0
            && (!matches!(existing_trial.state_values, TrialStateValues::Running)
                || existing_trial.datetime_start.is_none())
        {
            fields.insert("datetime_start".to_string(), to_raw(&now_aware_utc())?);
        } else if matches!(state_code, 1..=3) {
            fields.insert("datetime_complete".to_string(), to_raw(&now_aware_utc())?);
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
            let mut fields = HashMap::new();
            fields.insert("trial_id".to_string(), to_raw(&trial_id)?);
            fields.insert("step".to_string(), to_raw(&step)?);
            fields.insert(
                "intermediate_value".to_string(),
                to_raw(&value_to_json(value))?,
            );
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

    fn get_trials(&mut self, study_id: u32) -> Result<&Vec<Option<PersistedTrial>>> {
        self.sync_with_backend()?;
        self.replay.trials_by_study.get(&study_id).ok_or_else(|| {
            Error::with_reason(
                ErrorKind::StudyNotFound,
                format!("Failed to get trials for study: study_id={study_id}"),
            )
        })
    }

    fn get_trial(&mut self, trial_id: u32) -> Result<&PersistedTrial> {
        self.sync_with_backend()?;
        self.get_cached_trial(trial_id)
    }

    fn get_study_attr(&mut self, study_id: u32, key: AttrKey) -> Result<String> {
        let study = self.get_study(study_id)?;
        study
            .attrs
            .get(&key)
            .cloned()
            .ok_or(Error::new(ErrorKind::AttrNotFound))
    }

    fn get_cached_trial(&self, trial_id: u32) -> Result<&PersistedTrial> {
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
        trials
            .get(trial_number as usize)
            .and_then(|trial| trial.as_ref())
            .ok_or_else(|| {
                Error::with_reason(
                    ErrorKind::TrialDiscarded,
                    format!(
                        "Trial number out of bounds or discarded: trial_number={}, num_trials={}",
                        trial_number,
                        trials.len()
                    ),
                )
            })
    }

    fn get_trial_number_from_id(&mut self, trial_id: u32) -> Result<u32> {
        if let Some((_, trial_number)) = self.replay.trial_id_to_study_number.get(&trial_id) {
            return Ok(*trial_number);
        }

        self.sync_with_backend()?;
        self.replay
            .trial_id_to_study_number
            .get(&trial_id)
            .map(|(_, trial_number)| *trial_number)
            .ok_or_else(|| {
                Error::with_reason(
                    ErrorKind::TrialNotFound,
                    format!("Trial not found in storage: trial_id={trial_id}"),
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

        // Design note: Rustuna intentionally does not share Optuna's user/system attribute value
        // schema. Rustuna stores string maps in `user_attr_str`/`system_attr_str` without
        // double JSON encoding, and writes `{"rustuna": null}` to Optuna's fields. Optuna applies
        // only this dummy field and ignores the Rustuna-specific fields.
        let mut user_attrs = HashMap::new();
        let mut system_attrs = HashMap::new();
        for (key, value) in attrs {
            match key {
                AttrKey::User(k) => {
                    user_attrs.insert(k.to_string(), value);
                }
                AttrKey::System(k) => {
                    system_attrs.insert(k.to_string(), value);
                }
            }
        }
        if !user_attrs.is_empty() {
            let mut fields = HashMap::new();
            fields.insert("study_id".to_string(), to_raw(&study_id)?);
            fields.insert(
                "user_attr".to_string(),
                to_raw(&HashMap::from([("rustuna", Value::Null)]))?,
            );
            fields.insert("user_attr_str".to_string(), to_raw(&user_attrs)?);
            self.write_log(JournalOperation::SetStudyUserAttr, fields)?;
        }
        if !system_attrs.is_empty() {
            let mut fields = HashMap::new();
            fields.insert("study_id".to_string(), to_raw(&study_id)?);
            fields.insert(
                "system_attr".to_string(),
                to_raw(&HashMap::from([("rustuna", Value::Null)]))?,
            );
            fields.insert("system_attr_str".to_string(), to_raw(&system_attrs)?);
            self.write_log(JournalOperation::SetStudySystemAttr, fields)?;
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

        // Design note: Rustuna intentionally does not share Optuna's user/system attribute value
        // schema. Rustuna stores string maps in `user_attr_str`/`system_attr_str` without
        // double JSON encoding, and writes `{"rustuna": null}` to Optuna's fields. Optuna applies
        // only this dummy field and ignores the Rustuna-specific fields.
        let mut user_attrs = HashMap::new();
        let mut system_attrs = HashMap::new();
        for (key, value) in attrs {
            match key {
                AttrKey::User(k) => {
                    user_attrs.insert(k.to_string(), value);
                }
                AttrKey::System(k) => {
                    system_attrs.insert(k.to_string(), value);
                }
            }
        }
        if !user_attrs.is_empty() {
            let mut fields = HashMap::new();
            fields.insert("trial_id".to_string(), to_raw(&trial_id)?);
            fields.insert(
                "user_attr".to_string(),
                to_raw(&HashMap::from([("rustuna", Value::Null)]))?,
            );
            fields.insert("user_attr_str".to_string(), to_raw(&user_attrs)?);
            self.write_log(JournalOperation::SetTrialUserAttr, fields)?;
        }
        if !system_attrs.is_empty() {
            let mut fields = HashMap::new();
            fields.insert("trial_id".to_string(), to_raw(&trial_id)?);
            fields.insert(
                "system_attr".to_string(),
                to_raw(&HashMap::from([("rustuna", Value::Null)]))?,
            );
            fields.insert("system_attr_str".to_string(), to_raw(&system_attrs)?);
            self.write_log(JournalOperation::SetTrialSystemAttr, fields)?;
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

    fn get_n_trials(&mut self, study_id: u32, states: Option<&[TrialState]>) -> Result<u32> {
        self.sync_with_backend()?;
        self.replay.get_n_trials(study_id, states)
    }

    fn discard_trials(&mut self, trial_ids: &[u32]) -> Result<()> {
        self.sync_with_backend()?;
        let mut unique_trial_ids = Vec::new();
        for trial_id in trial_ids {
            if unique_trial_ids.contains(trial_id) {
                continue;
            }
            let (study_id, trial_number) = self
                .replay
                .trial_id_to_study_number
                .get(trial_id)
                .copied()
                .ok_or_else(|| Error::new(ErrorKind::TrialNotFound))?;
            let trials = self
                .replay
                .trials_by_study
                .get(&study_id)
                .ok_or_else(|| Error::new(ErrorKind::StudyNotFound))?;
            let slot = trials
                .get(trial_number as usize)
                .ok_or_else(|| Error::new(ErrorKind::TrialNotFound))?;
            if slot.is_none() {
                continue;
            }
            unique_trial_ids.push(*trial_id);
        }
        let fields = HashMap::from([(
            "trial_ids".to_string(),
            serde_json::value::to_raw_value(&unique_trial_ids).map_err(|_| {
                Error::with_reason(ErrorKind::StorageError, "Failed to serialize trial IDs")
            })?,
        )]);
        self.write_log(JournalOperation::DiscardTrials, fields)?;
        self.sync_with_backend()
    }

    fn may_omit_trials(&self) -> bool {
        self.replay
            .trials_by_study
            .values()
            .any(|trials| trials.iter().any(Option::is_none))
    }
}

struct JournalReplayState {
    log_number_read: usize,
    worker_id_prefix: String,
    studies: HashMap<u32, PersistedStudy>,
    trials_by_study: HashMap<u32, Vec<Option<PersistedTrial>>>,
    trial_id_to_study_number: HashMap<u32, (u32, u32)>,
    trial_id_to_study_id: HashMap<u32, u32>,
    study_id_to_trial_ids: HashMap<u32, Vec<u32>>,
    next_study_id: u32,
    worker_id_to_owned_trial_id: HashMap<String, u32>,
    last_created_trial_id_by_this_process: Option<u32>,
    studies_sorted: Vec<PersistedStudy>,
    study_caches: HashMap<u32, StudyCache>,
    discarded_state_counts: HashMap<(u32, TrialState), u32>,
    apply_discard: bool,
}

impl JournalReplayState {
    fn new(worker_id_prefix: String, apply_discard: bool) -> Self {
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
            discarded_state_counts: HashMap::new(),
            apply_discard,
        }
    }

    fn get_n_trials(&self, study_id: u32, states: Option<&[TrialState]>) -> Result<u32> {
        if !self.studies.contains_key(&study_id) {
            return Err(Error::new(ErrorKind::StudyNotFound));
        }
        let states: Option<std::collections::HashSet<TrialState>> =
            states.map(|states| states.iter().copied().collect());
        let live_count = self
            .trials_by_study
            .get(&study_id)
            .ok_or(Error::new(ErrorKind::StudyNotFound))?
            .iter()
            .flatten()
            .filter(|trial| {
                states
                    .as_ref()
                    .is_none_or(|states| states.contains(&trial.state_values.state()))
            })
            .count() as u32;
        let discarded_count = self
            .discarded_state_counts
            .iter()
            .filter(|((id, state), _)| {
                *id == study_id && states.as_ref().is_none_or(|states| states.contains(state))
            })
            .map(|(_, count)| count)
            .sum::<u32>();
        Ok(live_count + discarded_count)
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
                JournalOperation::DiscardTrials => {
                    if self.apply_discard {
                        self.apply_discard_trials(log, worker_id)?
                    }
                }
            };
        }
        Ok(())
    }

    fn is_issued_by_this_worker(&self, log: &JournalLog, worker_id: &str) -> bool {
        log.worker_id() == worker_id
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
            .and_then(|trial| trial.as_ref())
            .ok_or_else(|| {
                Error::with_reason(
                    ErrorKind::TrialDiscarded,
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
        let directions_raw = get_vec_i32(&log.fields, "directions")?;
        let mut directions = Vec::with_capacity(directions_raw.len());
        for value in directions_raw {
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
            self.discarded_state_counts
                .retain(|(id, _), _| *id != study_id);
            self.study_caches.remove(&study_id);
        }
        Ok(())
    }

    fn apply_set_study_user_attr(&mut self, log: &JournalLog, worker_id: &str) -> Result<()> {
        let study_id = get_u32(&log.fields, "study_id")?;
        let Some(attrs) = get_optional_raw_map(&log.fields, "user_attr_str")? else {
            return Ok(());
        };
        if !self.study_exists(study_id, log, worker_id)? {
            return Ok(());
        }
        if let Some(study) = self.studies.get_mut(&study_id) {
            for (key, value) in attrs {
                let value = raw_value_to_attr_string(&value)?;
                study.attrs.insert(AttrKey::User(key.clone().into()), value);
            }
        }
        Ok(())
    }

    fn apply_set_study_system_attr(&mut self, log: &JournalLog, worker_id: &str) -> Result<()> {
        let study_id = get_u32(&log.fields, "study_id")?;
        let Some(attrs) = get_optional_raw_map(&log.fields, "system_attr_str")? else {
            return Ok(());
        };
        if !self.study_exists(study_id, log, worker_id)? {
            return Ok(());
        }
        if let Some(study) = self.studies.get_mut(&study_id) {
            for (key, value) in attrs {
                let v = if key.starts_with("category_labels:") {
                    raw_value_to_plain_string(&value)?
                } else {
                    raw_value_to_attr_string(&value)?
                };
                study.attrs.insert(AttrKey::System(key.clone().into()), v);
            }
        }
        Ok(())
    }

    fn apply_create_trial(&mut self, log: &JournalLog, worker_id: &str) -> Result<()> {
        let study_id = get_u32(&log.fields, "study_id")?;
        let state = get_optional_i64(&log.fields, "state")?;
        let values = get_optional_raw_vec(&log.fields, "values")?;
        let value = get_optional_raw(&log.fields, "value");
        let params = get_optional_raw_map(&log.fields, "params")?;
        let distributions = get_optional_raw_map(&log.fields, "distributions")?;
        let user_attrs = get_optional_raw_map(&log.fields, "user_attrs")?;
        let system_attrs = get_optional_raw_map(&log.fields, "system_attrs")?;
        let intermediate_values = get_optional_raw_map(&log.fields, "intermediate_values")?;
        let datetime_start = get_optional_string(&log.fields, "datetime_start")?
            .as_deref()
            .map(journal_datetime_to_naive_utc);
        let datetime_complete = get_optional_string(&log.fields, "datetime_complete")?
            .as_deref()
            .map(journal_datetime_to_naive_utc);
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
        let state_code = state.unwrap_or(0);
        trial.state_values = match state_code {
            0 => TrialStateValues::Running,
            1 => {
                if let Some(values) = values.as_ref() {
                    let parsed = parse_f64_vec_raw(values)?;
                    TrialStateValues::Complete(parsed)
                } else if let Some(value) = value {
                    let parsed = parse_f64_raw(value)?;
                    TrialStateValues::Complete(vec![parsed])
                } else {
                    TrialStateValues::Complete(Vec::new())
                }
            }
            2 => TrialStateValues::Pruned,
            3 => TrialStateValues::Fail,
            4 => TrialStateValues::Waiting,
            _ => {
                return Err(Error::with_reason(
                    ErrorKind::StorageError,
                    format!("Invalid trial state code: {state_code}"),
                ))
            }
        };

        if let Some(params) = params {
            for (name, raw) in params {
                let value = parse_f64_raw(&raw)?;
                trial.internal_params.insert(name.clone(), value);
            }
        }
        if let Some(dists) = distributions {
            for (name, dist_json) in dists {
                let dist_json = raw_value_to_json_string(&dist_json)?;
                let (dist, labels) = json_to_distribution(&dist_json)?;
                if let Some(labels) = labels {
                    let attrs = category_labels_to_attrs(&name, &labels);
                    if let Some(study) = self.studies.get_mut(&study_id) {
                        for (k, v) in attrs {
                            study.attrs.entry(k).or_insert(v);
                        }
                    }
                }
                trial.distributions.insert(name.to_string(), dist);
            }
        }

        let mut attrs: Attrs = Attrs::new();

        if let Some(user_attrs) = user_attrs {
            for (k, v) in user_attrs {
                let value = raw_value_to_attr_string(&v)?;
                attrs.insert(AttrKey::User(k.clone().into()), value);
            }
        }
        if let Some(system_attrs) = system_attrs {
            for (k, v) in system_attrs {
                let value = raw_value_to_attr_string(&v)?;
                attrs.insert(AttrKey::System(k.clone().into()), value);
            }
        }
        if let Some(values) = intermediate_values {
            trial
                .intermediate_values
                .extend(intermediate_values_from_raw(&values)?);
        }
        trial.attrs = attrs;

        trial.datetime_start = datetime_start;
        trial.datetime_complete = datetime_complete;
        self.trials_by_study
            .entry(study_id)
            .or_default()
            .push(Some(trial));
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
        let param_name = get_string(&log.fields, "param_name")?;
        let param_value_internal = get_raw(&log.fields, "param_value_internal")?;
        let distribution = get_raw(&log.fields, "distribution")?;
        if !self.trial_exists_and_updatable(trial_id, log, worker_id)? {
            return Ok(());
        }
        let param_value = parse_f64_raw(param_value_internal)?;
        let dist_json = raw_value_to_json_string(distribution)?;
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
        let trial = trial.as_mut().ok_or_else(|| {
            Error::with_reason(
                ErrorKind::TrialDiscarded,
                format!("Trial discarded during set param: trial_number={trial_number}"),
            )
        })?;
        trial
            .internal_params
            .insert(param_name.clone(), param_value);
        trial.distributions.insert(param_name.clone(), dist);
        let cache = self.study_caches.entry(study_id).or_default();
        let dist_clone = trial
            .distributions
            .get(param_name.as_str())
            .ok_or_else(|| {
                Error::with_reason(
                    ErrorKind::StorageError,
                    format!("Failed to retrieve inserted distribution: param_name={param_name}"),
                )
            })?
            .clone();
        cache.param_distribution.insert(param_name, dist_clone);
        Ok(())
    }

    fn apply_set_trial_state_values(&mut self, log: &JournalLog, worker_id: &str) -> Result<()> {
        let trial_id = get_u32(&log.fields, "trial_id")?;
        let state_code = get_i64(&log.fields, "state")?;
        let values = get_optional_raw_vec(&log.fields, "values")?;
        let datetime_start = get_optional_string(&log.fields, "datetime_start")?
            .as_deref()
            .map(journal_datetime_to_naive_utc);
        let datetime_complete = get_optional_string(&log.fields, "datetime_complete")?
            .as_deref()
            .map(journal_datetime_to_naive_utc);
        if !self.trial_exists_and_updatable(trial_id, log, worker_id)? {
            return Ok(());
        }
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
        let trial = trial.as_mut().ok_or_else(|| {
            Error::with_reason(
                ErrorKind::TrialDiscarded,
                format!("Trial discarded during set state: trial_number={trial_number}"),
            )
        })?;

        let state = match state_code {
            0 => TrialStateValues::Running,
            1 => {
                let values = match values.as_ref() {
                    Some(raws) => parse_f64_vec_raw(raws)?,
                    None => Vec::new(),
                };
                TrialStateValues::Complete(values)
            }
            2 => TrialStateValues::Pruned,
            3 => TrialStateValues::Fail,
            4 => TrialStateValues::Waiting,
            _ => {
                return Err(Error::with_reason(
                    ErrorKind::StorageError,
                    format!("Invalid trial state code: {state_code}"),
                ))
            }
        };
        if state_code == 0 {
            if let Some(dt) = datetime_start {
                trial.datetime_start = Some(dt);
            }
            if issued_by_this_worker {
                self.worker_id_to_owned_trial_id
                    .insert(worker_id.to_string(), trial_id);
            }
        } else if matches!(state_code, 1..=3) {
            if let Some(dt) = datetime_complete {
                trial.datetime_complete = Some(dt);
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
        let step = get_u32(&log.fields, "step")?;
        let intermediate_value = get_raw(&log.fields, "intermediate_value")?;
        if !self.trial_exists_and_updatable(trial_id, log, worker_id)? {
            return Ok(());
        }
        let value = parse_f64_raw(intermediate_value)?;
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
        let trial = trial.as_mut().ok_or_else(|| {
            Error::with_reason(
                ErrorKind::TrialDiscarded,
                format!(
                    "Trial discarded during set intermediate value: trial_number={trial_number}"
                ),
            )
        })?;
        trial.intermediate_values.insert(step, value);
        Ok(())
    }

    fn apply_set_trial_user_attr(&mut self, log: &JournalLog, worker_id: &str) -> Result<()> {
        let trial_id = get_u32(&log.fields, "trial_id")?;
        let Some(attrs) = get_optional_raw_map(&log.fields, "user_attr_str")? else {
            return Ok(());
        };
        if !self.trial_exists_and_updatable(trial_id, log, worker_id)? {
            return Ok(());
        }
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
        let trial = trial.as_mut().ok_or_else(|| {
            Error::with_reason(
                ErrorKind::TrialDiscarded,
                format!("Trial discarded during set user attr: trial_number={trial_number}"),
            )
        })?;
        for (key, value) in attrs {
            let v = raw_value_to_attr_string(&value)?;
            trial.attrs.insert(AttrKey::User(key.clone().into()), v);
        }
        Ok(())
    }

    fn apply_set_trial_system_attr(&mut self, log: &JournalLog, worker_id: &str) -> Result<()> {
        let trial_id = get_u32(&log.fields, "trial_id")?;
        let Some(attrs) = get_optional_raw_map(&log.fields, "system_attr_str")? else {
            return Ok(());
        };
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
        let trial = trial.as_mut().ok_or_else(|| {
            Error::with_reason(
                ErrorKind::TrialDiscarded,
                format!("Trial discarded during set system attr: trial_number={trial_number}"),
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
            let v = raw_value_to_attr_string(&value)?;
            trial.attrs.insert(AttrKey::System(key.clone().into()), v);
        }
        Ok(())
    }

    fn apply_discard_trials(&mut self, log: &JournalLog, _worker_id: &str) -> Result<()> {
        let raw = log.fields.get("trial_ids").ok_or_else(|| {
            Error::with_reason(
                ErrorKind::StorageError,
                "Missing trial_ids in discard trials log",
            )
        })?;
        let trial_ids: Vec<u32> = serde_json::from_str(raw.get()).map_err(|_| {
            Error::with_reason(
                ErrorKind::StorageError,
                "Failed to deserialize trial_ids in discard trials log",
            )
        })?;
        let mut targets = Vec::new();
        for trial_id in trial_ids {
            if targets
                .iter()
                .any(|(_, _, existing_trial_id, _)| existing_trial_id == &trial_id)
            {
                continue;
            }
            let (study_id, trial_number) = self
                .trial_id_to_study_number
                .get(&trial_id)
                .copied()
                .ok_or_else(|| {
                    Error::with_reason(
                        ErrorKind::TrialNotFound,
                        format!("Trial not found during discard: trial_id={trial_id}"),
                    )
                })?;
            let trials = self.trials_by_study.get(&study_id).ok_or_else(|| {
                Error::with_reason(
                    ErrorKind::StudyNotFound,
                    format!("Study not found during discard: study_id={study_id}"),
                )
            })?;
            let slot = trials.get(trial_number as usize).ok_or_else(|| {
                Error::with_reason(
                    ErrorKind::TrialNotFound,
                    format!(
                        "Trial not found at position during discard: trial_number={trial_number}"
                    ),
                )
            })?;
            let Some(trial) = slot.as_ref() else {
                continue;
            };
            let state = trial.state_values.state();
            targets.push((study_id, trial_number, trial_id, state));
        }
        for (study_id, trial_number, _, state) in targets {
            let discarded = {
                let trials = self.trials_by_study.get_mut(&study_id).ok_or_else(|| {
                    Error::with_reason(
                        ErrorKind::StudyNotFound,
                        format!("Study not found during discard: study_id={study_id}"),
                    )
                })?;
                let slot = trials.get_mut(trial_number as usize).ok_or_else(|| {
                    Error::with_reason(
                        ErrorKind::TrialNotFound,
                        format!(
                            "Trial not found at position during discard: trial_number={trial_number}"
                        ),
                    )
                })?;
                slot.take().is_some()
            };
            if discarded {
                *self
                    .discarded_state_counts
                    .entry((study_id, state))
                    .or_default() += 1;
            }
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
            .and_then(|trial| trial.as_ref())
            .ok_or_else(|| {
                Error::with_reason(
                    ErrorKind::TrialDiscarded,
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
            .and_then(|trial| trial.as_ref())
            .ok_or_else(|| {
                Error::with_reason(
                    ErrorKind::TrialDiscarded,
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
            if let Some(trial) = trial.as_ref() {
                if let Some(dist) = trial.distributions.get(param_name) {
                    return Some(dist);
                }
            }
        }
        None
    }
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
            Ok((Distribution::new_float(low, high, step, log), None))
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
            Ok((Distribution::new_int(low, high, step, log), None))
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
            Ok((Distribution::new_categorical(size as usize), labels))
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

fn unique_prefix() -> String {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    format!("{nanos}")
}

fn to_raw<T: Serialize>(value: &T) -> Result<Box<RawValue>> {
    to_raw_value(value).map_err(|e| Error::with_reason(ErrorKind::StorageError, e.to_string()))
}

fn get_raw<'a>(fields: &'a HashMap<String, Box<RawValue>>, key: &str) -> Result<&'a RawValue> {
    fields.get(key).map(|v| v.as_ref()).ok_or_else(|| {
        Error::with_reason(
            ErrorKind::StorageError,
            format!("Missing field in journal log: key={key}"),
        )
    })
}

fn get_optional_raw<'a>(
    fields: &'a HashMap<String, Box<RawValue>>,
    key: &str,
) -> Option<&'a RawValue> {
    fields.get(key).and_then(|v| {
        if v.get() == "null" {
            None
        } else {
            Some(v.as_ref())
        }
    })
}

fn get_string(fields: &HashMap<String, Box<RawValue>>, key: &str) -> Result<String> {
    raw_value_to_plain_string(get_raw(fields, key)?)
}

fn get_optional_string(
    fields: &HashMap<String, Box<RawValue>>,
    key: &str,
) -> Result<Option<String>> {
    match get_optional_raw(fields, key) {
        Some(raw) => raw_value_to_plain_string(raw).map(Some),
        None => Ok(None),
    }
}

fn get_u32(fields: &HashMap<String, Box<RawValue>>, key: &str) -> Result<u32> {
    parse_u32_raw(get_raw(fields, key)?)
}

fn get_i64(fields: &HashMap<String, Box<RawValue>>, key: &str) -> Result<i64> {
    parse_i64_raw(get_raw(fields, key)?)
}

fn get_optional_i64(fields: &HashMap<String, Box<RawValue>>, key: &str) -> Result<Option<i64>> {
    match get_optional_raw(fields, key) {
        Some(raw) => parse_i64_raw(raw).map(Some),
        None => Ok(None),
    }
}

fn get_vec_i32(fields: &HashMap<String, Box<RawValue>>, key: &str) -> Result<Vec<i32>> {
    let raw = get_raw(fields, key)?;
    serde_json::from_str::<Vec<i32>>(raw.get()).map_err(|_| {
        Error::with_reason(
            ErrorKind::StorageError,
            format!("Failed to parse i32 array from field: key={key}"),
        )
    })
}

fn get_optional_raw_map(
    fields: &HashMap<String, Box<RawValue>>,
    key: &str,
) -> Result<Option<HashMap<String, Box<RawValue>>>> {
    match get_optional_raw(fields, key) {
        Some(raw) => serde_json::from_str::<HashMap<String, Box<RawValue>>>(raw.get())
            .map(Some)
            .map_err(|_| {
                Error::with_reason(
                    ErrorKind::StorageError,
                    format!("Failed to parse object from field: key={key}"),
                )
            }),
        None => Ok(None),
    }
}

fn get_optional_raw_vec(
    fields: &HashMap<String, Box<RawValue>>,
    key: &str,
) -> Result<Option<Vec<Box<RawValue>>>> {
    match get_optional_raw(fields, key) {
        Some(raw) => serde_json::from_str::<Vec<Box<RawValue>>>(raw.get())
            .map(Some)
            .map_err(|_| {
                Error::with_reason(
                    ErrorKind::StorageError,
                    format!("Failed to parse array from field: key={key}"),
                )
            }),
        None => Ok(None),
    }
}

fn parse_u32_raw(value: &RawValue) -> Result<u32> {
    serde_json::from_str::<u64>(value.get())
        .map_err(|_| {
            Error::with_reason(
                ErrorKind::StorageError,
                format!("Failed to parse u32 value: {}", value.get()),
            )
        })
        .map(|v| v as u32)
}

fn parse_i64_raw(value: &RawValue) -> Result<i64> {
    serde_json::from_str::<i64>(value.get()).map_err(|_| {
        Error::with_reason(
            ErrorKind::StorageError,
            format!("Failed to parse i64 value: {}", value.get()),
        )
    })
}

fn parse_f64_raw(value: &RawValue) -> Result<f64> {
    parse_f64_json(value.get())
}

fn parse_f64_json(raw: &str) -> Result<f64> {
    if raw.starts_with('"') {
        let s: String = serde_json::from_str(raw).map_err(|_| {
            Error::with_reason(ErrorKind::StorageError, "Failed to parse string as JSON")
        })?;
        return parse_f64_from_str(&s);
    }
    parse_f64_from_str(raw)
}

fn parse_f64_from_str(raw: &str) -> Result<f64> {
    match raw {
        "nan" | "NaN" => Ok(f64::NAN),
        "inf" | "Infinity" | "INF" => Ok(f64::INFINITY),
        "-inf" | "-Infinity" | "-INF" => Ok(f64::NEG_INFINITY),
        _ => raw.parse::<f64>().map_err(|_| {
            Error::with_reason(
                ErrorKind::StorageError,
                format!("Failed to parse f64 value: {raw}"),
            )
        }),
    }
}

fn raw_value_to_plain_string(value: &RawValue) -> Result<String> {
    serde_json::from_str::<String>(value.get())
        .map_err(|_| Error::with_reason(ErrorKind::StorageError, "Failed to parse value as string"))
}

fn raw_value_to_json_string(value: &RawValue) -> Result<String> {
    let raw = value.get();
    if raw.starts_with('"') {
        return raw_value_to_plain_string(value);
    }
    Ok(raw.to_string())
}

fn raw_value_to_attr_string(value: &RawValue) -> Result<String> {
    let raw = value.get();
    if raw.starts_with('"') {
        return raw_value_to_plain_string(value);
    }
    Ok(raw.to_string())
}

fn parse_f64_vec_raw(values: &[Box<RawValue>]) -> Result<Vec<f64>> {
    values.iter().map(|v| parse_f64_raw(v)).collect()
}

fn intermediate_values_from_raw(map: &HashMap<String, Box<RawValue>>) -> Result<HashMap<u32, f64>> {
    let mut values = HashMap::new();
    for (k, v) in map {
        let step: u32 = k.parse().map_err(|_| {
            Error::with_reason(
                ErrorKind::StorageError,
                format!("Failed to parse step as u32: {k}"),
            )
        })?;
        let value = parse_f64_raw(v)?;
        values.insert(step, value);
    }
    Ok(values)
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
    use rustuna_core::attr::AttrKey;
    use rustuna_core::storage::Storage;
    use rustuna_core::trial::PersistedTrial;
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
    fn trial_datetimes_are_logged_as_aware_utc() -> Result<()> {
        let (mut storage, logs) = new_storage()?;
        let study_id = storage
            .create_new_study("example", vec![Direction::Minimize])?
            .id;
        let trial_id = storage.create_new_trial(study_id)?.id;
        storage.set_trial_state_values(trial_id, TrialStateValues::Complete(vec![1.0]))?;

        let trial = storage.get_trial(trial_id)?;
        let reported_start = trial.datetime_start.clone().expect("datetime_start is set");
        let reported_complete = trial
            .datetime_complete
            .clone()
            .expect("datetime_complete is set");

        // Collect the datetimes as they appear in the log itself.
        let logged: Vec<String> = {
            let guard = logs.lock().expect("lock logs");
            guard
                .iter()
                .flat_map(|log| {
                    ["datetime_start", "datetime_complete"]
                        .iter()
                        .filter_map(|key| get_optional_string(&log.fields, key).ok().flatten())
                        .collect::<Vec<_>>()
                })
                .collect()
        };
        assert_eq!(logged.len(), 2, "expected both datetimes in the log");

        for value in &logged {
            // Journal logs carry timezone-aware UTC, matching
            // datetime.now(tz=timezone.utc).isoformat(timespec="microseconds") in Optuna.
            assert!(value.ends_with("+00:00"), "not aware UTC: {value}");
            assert_eq!(value.as_bytes()[10], b'T', "not RFC 3339: {value}");
            assert_eq!(
                value.len(),
                "1970-01-01T00:00:00.000000+00:00".len(),
                "not microsecond precision: {value}"
            );
        }

        // What PersistedTrial carries is the very same instants as naive UTC, so the log entries
        // are the reported values plus an explicit offset.
        assert_eq!(
            logged,
            vec![
                naive_utc_to_aware_utc(&reported_start),
                naive_utc_to_aware_utc(&reported_complete),
            ]
        );
        Ok(())
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
        let mut fields = HashMap::new();
        fields.insert(
            "study_name".to_string(),
            to_raw(&study_name.to_string()).unwrap(),
        );
        let dirs = directions
            .into_iter()
            .map(|d| match d {
                Direction::Minimize => 1,
                Direction::Maximize => 2,
            })
            .collect::<Vec<_>>();
        fields.insert("directions".to_string(), to_raw(&dirs).unwrap());
        JournalLog {
            op_code: JournalOperation::CreateStudy as i32,
            worker_id: "external".to_string(),
            fields,
        }
    }

    fn create_trial_log(study_id: u32) -> JournalLog {
        let mut fields = HashMap::new();
        fields.insert("study_id".to_string(), to_raw(&study_id).unwrap());
        fields.insert(
            "datetime_start".to_string(),
            to_raw(&"2024-01-01 00:00:00".to_string()).unwrap(),
        );
        JournalLog {
            op_code: JournalOperation::CreateTrial as i32,
            worker_id: "external".to_string(),
            fields,
        }
    }

    #[test]
    fn duplicate_discard_trials_logs_do_not_poison_replay() -> Result<()> {
        let (mut storage, logs) = new_storage()?;
        let study_id = storage
            .create_new_study("study", vec![Direction::Minimize])?
            .id;
        let trial_id = storage.create_new_trial(study_id)?.id;

        storage.discard_trials(&[trial_id])?;
        {
            let mut guard = logs.lock().map_err(|_| {
                Error::with_reason(
                    ErrorKind::StorageError,
                    "Failed to acquire lock on journal logs",
                )
            })?;
            let discard_log = guard
                .iter()
                .find(|log| log.op_code == JournalOperation::DiscardTrials as i32)
                .cloned()
                .ok_or_else(|| {
                    Error::with_reason(ErrorKind::StorageError, "Discard log not found")
                })?;
            guard.push(discard_log);
        }

        let backend = InMemoryJournalBackend { logs: logs.clone() };
        let mut reloaded = JournalStorage::new_with_options(
            Box::new(backend),
            JournalStorageOptions {
                apply_discard: true,
            },
        )?;
        let trials = reloaded.get_trials(study_id)?;
        assert!(trials[0].is_none());
        Ok(())
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
    fn apply_discard_false_keeps_discarded_trials() -> Result<()> {
        let (mut storage, logs) = new_storage()?;
        let study_id = storage.create_new_study("s", vec![Direction::Minimize])?.id;
        let trial_id = storage.create_new_trial(study_id)?.id;
        storage.discard_trials(&[trial_id])?;

        let backend = InMemoryJournalBackend { logs: logs.clone() };
        let mut storage2 = JournalStorage::new_with_options(
            Box::new(backend),
            JournalStorageOptions {
                apply_discard: false,
            },
        )?;
        let trials = storage2.get_trials(study_id)?;
        assert_eq!(trials.len(), 1);
        assert_eq!(trials[0].as_ref().unwrap().id, trial_id);
        assert_eq!(storage2.get_trial(trial_id)?.id, trial_id);
        Ok(())
    }

    #[test]
    fn get_n_trials_counts_states_including_discarded_trials() -> Result<()> {
        let logs = Arc::new(Mutex::new(Vec::new()));
        let backend = InMemoryJournalBackend { logs };
        let mut storage = JournalStorage::new_with_options(
            Box::new(backend),
            JournalStorageOptions {
                apply_discard: true,
            },
        )?;
        let study_id = storage.create_new_study("s", vec![Direction::Minimize])?.id;
        let running_trial_id = storage.create_new_trial(study_id)?.id;
        let complete_trial_id = storage.create_new_trial(study_id)?.id;
        storage.set_trial_state_values(complete_trial_id, TrialStateValues::Complete(vec![1.0]))?;

        assert_eq!(storage.get_n_trials(study_id, None)?, 2);
        assert_eq!(
            storage.get_n_trials(study_id, Some(&[TrialState::Running]))?,
            1
        );
        assert_eq!(
            storage.get_n_trials(study_id, Some(&[TrialState::Complete]))?,
            1
        );

        storage.discard_trials(&[complete_trial_id])?;
        assert_eq!(storage.get_n_trials(study_id, None)?, 2);
        assert_eq!(
            storage.get_n_trials(study_id, Some(&[TrialState::Complete]))?,
            1
        );
        assert!(storage.get_trial(running_trial_id).is_ok());
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
    fn set_trial_intermediate_values_replays_to_trial_field() -> Result<()> {
        let (mut storage, logs) = new_storage()?;
        let study_id = storage.create_new_study("s", vec![Direction::Minimize])?.id;
        let trial_id = storage.create_new_trial(study_id)?.id;

        let mut values = HashMap::new();
        values.insert(0, 0.1);
        values.insert(1, f64::INFINITY);
        values.insert(2, f64::NAN);
        storage.set_trial_intermediate_values(trial_id, values)?;

        let backend = InMemoryJournalBackend { logs: logs.clone() };
        let mut reloaded = JournalStorage::new(Box::new(backend))?;
        let trial = reloaded.get_trial(trial_id)?;
        assert_eq!(trial.intermediate_values[&0], 0.1);
        assert_eq!(trial.intermediate_values[&1], f64::INFINITY);
        assert!(trial.intermediate_values[&2].is_nan());
        assert!(!trial
            .attrs
            .contains_key(&AttrKey::System("intermediate_values".into())));
        Ok(())
    }

    #[test]
    fn get_joint_search_space_uses_cache_update() -> Result<()> {
        let (mut storage, _logs) = new_storage()?;
        let study_id = storage.create_new_study("s", vec![Direction::Minimize])?.id;

        let dist = Distribution::new_float(0.0, 1.0, None, false);
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
            "bar"
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
            "val"
        );
        Ok(())
    }

    #[test]
    fn raw_value_to_attr_string_preserves_plain_and_json_strings() -> Result<()> {
        let plain = to_raw(&"abc.json".to_string())?;
        assert_eq!(raw_value_to_attr_string(&plain)?, "abc.json");

        let json_string = to_raw(&"\"abc.json\"".to_string())?;
        assert_eq!(raw_value_to_attr_string(&json_string)?, "\"abc.json\"");
        Ok(())
    }

    #[test]
    fn create_new_trial_from_template_preserves_user_attr_str() -> Result<()> {
        let (mut storage, _logs) = new_storage()?;
        let study_id = storage
            .create_new_study("study", vec![Direction::Minimize])?
            .id;

        let mut template = PersistedTrial::new(100, study_id, 0);
        template
            .attrs
            .insert(AttrKey::User("plain".into()), "abc.json".to_string());
        template
            .attrs
            .insert(AttrKey::User("json".into()), "\"abc.json\"".to_string());

        let trial_id = storage
            .create_new_trial_from_template(study_id, &template)?
            .id;
        let trial = storage.get_trial(trial_id)?;

        assert_eq!(
            trial.attrs.get(&AttrKey::User("plain".into())),
            Some(&"abc.json".to_string())
        );
        assert_eq!(
            trial.attrs.get(&AttrKey::User("json".into())),
            Some(&"\"abc.json\"".to_string())
        );
        Ok(())
    }

    #[test]
    fn late_multi_attr_write_does_not_break_future_replay() -> Result<()> {
        let (mut storage, _logs) = new_storage()?;
        let study_id = storage
            .create_new_study("study", vec![Direction::Minimize])?
            .id;
        let trial_id = storage.create_new_trial(study_id)?.id;

        let mut attrs = Attrs::new();
        attrs.insert(AttrKey::User("x".into()), "1".to_string());
        attrs.insert(AttrKey::User("y".into()), "2".to_string());
        storage.set_trial_attrs(trial_id, attrs.clone(), false)?;
        storage.set_trial_state_values(trial_id, TrialStateValues::Complete(vec![1.0]))?;

        let err = storage
            .set_trial_attrs(trial_id, attrs, false)
            .expect_err("late write should fail");
        assert!(matches!(err.kind, ErrorKind::TrialAlreadyFinished));

        let trials = storage.get_trials(study_id)?;
        assert_eq!(trials.len(), 1);
        assert_eq!(trials[0].as_ref().unwrap().id, trial_id);
        Ok(())
    }

    #[test]
    fn set_trial_param_updates_cache_and_refreshes() -> Result<()> {
        let (mut storage, _logs) = new_storage()?;
        let study_id = storage.create_new_study("s", vec![Direction::Minimize])?.id;
        storage.create_new_trial(study_id)?;

        let dist = Distribution::new_float(0.0, 1.0, None, false);
        let trial_id = storage.get_trials(study_id)?[0].as_ref().unwrap().id;
        storage.set_trial_param(trial_id, "x", &dist, 0.5)?;

        let trial = storage.get_trial(trial_id)?;
        assert_eq!(trial.internal_params.get("x"), Some(&0.5));
        assert_eq!(
            trial.distributions.get("x"),
            Some(&Distribution::new_float(0.0, 1.0, None, false))
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

        let distribution_x = Distribution::new_float(1.0, 2.0, None, false);
        let distribution_y_1 = Distribution::new_categorical(3);
        let distribution_z = Distribution::new_float(1.0, 100.0, None, true);

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

        let float_dist = Distribution::new_float(0.0, 1.0, None, false);
        let int_dist = Distribution::new_int(0, 5, 1, false);

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
