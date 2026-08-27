use crate::cache::{CachedStorageBackend, DiscardedTrialsDiff};
use rusqlite::{
    params, Connection, Error as RusqliteError, OptionalExtension, TransactionBehavior,
};
use rustuna_core::attr::{
    category_labels_to_attrs, get_category_labels, AttrKey, Attrs, CategoryLabel,
};
use rustuna_core::distribution::Distribution;
use rustuna_core::internal::datetime::now_naive_utc;
use rustuna_core::study::{Direction, PersistedStudy};
use rustuna_core::trial::{PersistedTrial, TrialState, TrialStateValues};
use rustuna_core::{Error, ErrorKind, Result};
use serde_json::{json, Number, Value};
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;

/// Options for [`SQLite3Storage`].
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct SQLite3StorageOptions {
    /// If `true`, discarded trials are omitted from subsequent reads.
    ///
    /// As in `JournalStorageOptions`, this only gates reads: `discard_trials` marks the trials
    /// in the database regardless of this option.
    pub apply_discard: bool,
}

/// SQLite-backed storage backend.
///
/// This backend persists studies and trials in a local SQLite database and is typically wrapped
/// by [`crate::cache::CachedStorage`] to provide the reference-based `Storage` API from
/// `rustuna_core`.
pub struct SQLite3Storage {
    conn: Mutex<Connection>,
    options: SQLite3StorageOptions,
    has_discarded_at_column: AtomicBool,
}

const SCHEMA_SQL: &str = include_str!("sqlite3_schema.sql");
const TRIALS_STUDY_ID_NUMBER_INDEX_SQL: &str =
    "CREATE INDEX IF NOT EXISTS trials_study_id_number_key ON trials (study_id, number)";
const TRIALS_DISCARDED_AT_COLUMN_SQL: &str = "ALTER TABLE trials ADD COLUMN discarded_at DATETIME";
const TRIALS_DISCARDED_AT_INDEX_SQL: &str =
    "CREATE INDEX IF NOT EXISTS trials_study_id_discarded_at_key ON trials (study_id, discarded_at)";

type TrialRow = (u32, u32, String, Option<String>, Option<String>);

impl SQLite3Storage {
    /// Opens a SQLite database file.
    pub fn new(file_path: &str) -> Result<SQLite3Storage> {
        Self::new_with_option(file_path, SQLite3StorageOptions::default())
    }

    /// Opens a SQLite database file with the given options.
    ///
    /// When `apply_discard` is enabled, call [`Self::validate_discard_support`] after
    /// [`Self::create_database`] to reject databases whose schema predates the discard column.
    pub fn new_with_option(
        file_path: &str,
        options: SQLite3StorageOptions,
    ) -> Result<SQLite3Storage> {
        let conn = Connection::open(file_path).map_err(|e| {
            Error::with_reason(
                ErrorKind::StorageError,
                format!("Failed to open {file_path}: {e}"),
            )
        })?;
        let has_discarded_at_column = Self::has_discarded_at_column(&conn)?;
        Ok(SQLite3Storage {
            conn: Mutex::new(conn),
            options,
            has_discarded_at_column: AtomicBool::new(has_discarded_at_column),
        })
    }

    /// Returns an error when discards were requested but the database cannot record them.
    ///
    /// [`Self::create_database`] migrates the column in, so this only fails for databases opened
    /// without initialization.
    pub fn validate_discard_support(&self) -> Result<()> {
        if self.options.apply_discard && !self.has_discarded_at_column.load(Ordering::Acquire) {
            return Err(Error::with_reason(
                ErrorKind::StorageError,
                "apply_discard requires the Rustuna-specific `discarded_at` column on the \
                 trials table. Open the database with create_database enabled to add it.",
            ));
        }
        Ok(())
    }

    /// Creates the Rustuna schema if the database has not been initialized yet.
    pub fn create_database(&self) -> Result<()> {
        let conn = self
            .conn
            .lock()
            .map_err(|_| Error::new(ErrorKind::StorageError))?;
        conn.execute_batch("BEGIN IMMEDIATE").map_err(|e| {
            Error::with_reason(
                ErrorKind::StorageError,
                format!("Failed to start database initialization: {e}"),
            )
        })?;

        let result = (|| -> Result<()> {
            if !Self::is_initialized(&conn)? {
                conn.execute_batch(SCHEMA_SQL).map_err(|e| {
                    Error::with_reason(
                        ErrorKind::StorageError,
                        format!("Failed to create database: {e}"),
                    )
                })?;
            }

            // Optuna's SQLite schema does not include this composite index. Keep it available
            // when opening a database initialized by an older Rustuna version or by Optuna,
            // because `SCHEMA_SQL` is skipped for initialized databases.
            conn.execute_batch(TRIALS_STUDY_ID_NUMBER_INDEX_SQL)
                .map_err(|e| {
                    Error::with_reason(
                        ErrorKind::StorageError,
                        format!("Failed to create trial lookup index: {e}"),
                    )
                })?;

            // Same reasoning for the discard column: a database created by Optuna or by an
            // older Rustuna has a `trials` table without it, and `SCHEMA_SQL` above is skipped
            // for such databases. Adding it here keeps `apply_discard` from silently degrading
            // into a no-op that only looks correct until the storage is reopened.
            if !Self::has_discarded_at_column(&conn)? {
                conn.execute_batch(TRIALS_DISCARDED_AT_COLUMN_SQL)
                    .map_err(|e| {
                        Error::with_reason(
                            ErrorKind::StorageError,
                            format!("Failed to add the discarded_at column: {e}"),
                        )
                    })?;
            }
            conn.execute_batch(TRIALS_DISCARDED_AT_INDEX_SQL)
                .map_err(|e| {
                    Error::with_reason(
                        ErrorKind::StorageError,
                        format!("Failed to create discard lookup index: {e}"),
                    )
                })?;
            Ok(())
        })();

        match result {
            Ok(()) => conn.execute_batch("COMMIT").map_err(|e| {
                Error::with_reason(
                    ErrorKind::StorageError,
                    format!("Failed to commit database initialization: {e}"),
                )
            })?,
            Err(err) => {
                let _ = conn.execute_batch("ROLLBACK");
                return Err(err);
            }
        }

        let has_discarded_at_column = Self::has_discarded_at_column(&conn)?;
        self.has_discarded_at_column
            .store(has_discarded_at_column, Ordering::Release);

        Ok(())
    }

    fn is_initialized(conn: &Connection) -> Result<bool> {
        let exists: Option<String> = conn
            .query_row(
                "SELECT name FROM sqlite_master WHERE type = 'table' AND name = 'version_info'",
                [],
                |row| row.get(0),
            )
            .optional()
            .map_err(|e| {
                Error::with_reason(
                    ErrorKind::StorageError,
                    format!("Failed to inspect database schema: {e}"),
                )
            })?;
        Ok(exists.is_some())
    }

    fn has_discarded_at_column(conn: &Connection) -> Result<bool> {
        let mut stmt = conn.prepare("PRAGMA table_info(trials)").map_err(|e| {
            Error::with_reason(
                ErrorKind::StorageError,
                format!("Failed to inspect trials schema: {e}"),
            )
        })?;
        let columns = stmt
            .query_map([], |row| row.get::<_, String>(1))
            .map_err(|e| {
                Error::with_reason(
                    ErrorKind::StorageError,
                    format!("Failed to inspect trials schema: {e}"),
                )
            })?;
        for column in columns {
            if column.map_err(|e| {
                Error::with_reason(
                    ErrorKind::StorageError,
                    format!("Failed to inspect trials schema: {e}"),
                )
            })? == "discarded_at"
            {
                return Ok(true);
            }
        }
        Ok(false)
    }

    fn validate_study_id(&self, study_id: u32) -> Result<()> {
        let guard = self
            .conn
            .lock()
            .map_err(|_| Error::new(ErrorKind::StorageError))?;
        let study_exists: Option<u32> = guard
            .query_row(
                "SELECT study_id FROM studies WHERE study_id = ?",
                params![study_id],
                |row| row.get(0),
            )
            .optional()
            .map_err(|e| {
                Error::with_reason(
                    ErrorKind::StorageError,
                    format!("Database query failed: {e}"),
                )
            })?;
        if study_exists.is_none() {
            return Err(Error::new(ErrorKind::StudyNotFound));
        }
        drop(guard);
        Ok(())
    }
}

impl CachedStorageBackend for SQLite3Storage {
    fn apply_discard(&self) -> bool {
        self.options.apply_discard
    }

    fn discard_trials(&mut self, trial_ids: &[u32]) -> Result<()> {
        if !self.has_discarded_at_column.load(Ordering::Acquire) {
            // Failing loudly rather than silently: a no-op here would let the caller believe the
            // trials were discarded until the database is reopened and they all reappear.
            return Err(Error::with_reason(
                ErrorKind::StorageError,
                "Cannot discard trials: the trials table has no `discarded_at` column. Open the \
                 database with create_database enabled to add it.",
            ));
        }
        if trial_ids.is_empty() {
            return Ok(());
        }

        let mut guard = self
            .conn
            .lock()
            .map_err(|_| Error::new(ErrorKind::StorageError))?;
        let tx = guard.transaction().map_err(|e| {
            Error::with_reason(
                ErrorKind::StorageError,
                format!("Database query failed: {e}"),
            )
        })?;
        for trial_id in trial_ids {
            // `discarded_at IS NULL` keeps the first timestamp when the same trial is discarded
            // twice. Restamping it would move the trial ahead of readers that already passed it,
            // making them replay a discard they have applied.
            let updated = tx
                .execute(
                    "UPDATE trials SET discarded_at = ? WHERE trial_id = ? AND discarded_at IS NULL",
                    params![now_naive_utc(), trial_id],
                )
                .map_err(|e| {
                    Error::with_reason(
                        ErrorKind::StorageError,
                        format!("Database query failed: {e}"),
                    )
                })?;
            if updated == 0 {
                // Either the trial does not exist, or it was already discarded.
                let exists: Option<u32> = tx
                    .query_row(
                        "SELECT trial_id FROM trials WHERE trial_id = ?",
                        params![trial_id],
                        |row| row.get(0),
                    )
                    .optional()
                    .map_err(|e| {
                        Error::with_reason(
                            ErrorKind::StorageError,
                            format!("Database query failed: {e}"),
                        )
                    })?;
                if exists.is_none() {
                    return Err(Error::new(ErrorKind::TrialNotFound));
                }
            }
        }
        tx.commit().map_err(|e| {
            Error::with_reason(
                ErrorKind::StorageError,
                format!("Database query failed: {e}"),
            )
        })
    }

    fn get_discarded_trials_diff(
        &mut self,
        study_id: u32,
        cursor: Option<&str>,
    ) -> Result<DiscardedTrialsDiff> {
        if !self.has_discarded_at_column.load(Ordering::Acquire) {
            return Ok(DiscardedTrialsDiff::default());
        }
        let guard = self
            .conn
            .lock()
            .map_err(|_| Error::new(ErrorKind::StorageError))?;

        // `>=` instead of `>`: several trials can share a timestamp, and the cursor only carries
        // the timestamp itself. Re-reading the boundary batch costs at most one discard call and
        // applying a discard twice is a no-op, whereas `>` would drop the trials tied with it.
        //
        // Trials discarded by another process whose clock lags behind the cursor are missed. That
        // only costs memory (the trial stays cached), never correctness, so it is not worth
        // ordering discards across machines.
        let mut stmt = guard
            .prepare(
                "SELECT number, discarded_at FROM trials \
                 WHERE study_id = ? AND discarded_at IS NOT NULL AND discarded_at >= ?",
            )
            .map_err(|e| {
                Error::with_reason(
                    ErrorKind::StorageError,
                    format!("Database query failed: {e}"),
                )
            })?;
        // Every non-NULL timestamp compares greater than or equal to the empty string.
        let rows = stmt
            .query_map(params![study_id, cursor.unwrap_or("")], |row| {
                Ok((row.get::<_, u32>(0)?, row.get::<_, String>(1)?))
            })
            .map_err(|e| {
                Error::with_reason(
                    ErrorKind::StorageError,
                    format!("Database query failed: {e}"),
                )
            })?;

        let mut diff = DiscardedTrialsDiff::default();
        for row in rows {
            let (number, discarded_at) = row.map_err(|e| {
                Error::with_reason(
                    ErrorKind::StorageError,
                    format!("Database query failed: {e}"),
                )
            })?;
            diff.numbers.push(number);
            if diff.cursor.as_ref().is_none_or(|max| *max < discarded_at) {
                diff.cursor = Some(discarded_at);
            }
        }
        Ok(diff)
    }

    fn get_n_trials(
        &mut self,
        study_id: u32,
        states: Option<&[TrialState]>,
    ) -> rustuna_core::Result<u32> {
        self.validate_study_id(study_id)?;

        let mut sql = "SELECT COUNT(*) FROM trials WHERE study_id = ?".to_string();
        let mut params: Vec<Box<dyn rusqlite::ToSql>> = vec![Box::new(study_id)];
        if let Some(states) = states {
            if states.is_empty() {
                return Ok(0);
            }
            let placeholders = states.iter().map(|_| "?").collect::<Vec<_>>().join(", ");
            sql.push_str(&format!(" AND state IN ({placeholders})"));
            params.extend(states.iter().map(|state| {
                let state = match state {
                    TrialState::Running => "RUNNING",
                    TrialState::Complete => "COMPLETE",
                    TrialState::Pruned => "PRUNED",
                    TrialState::Waiting => "WAITING",
                    TrialState::Fail => "FAIL",
                };
                Box::new(state.to_string()) as Box<dyn rusqlite::ToSql>
            }));
        }

        let guard = self
            .conn
            .lock()
            .map_err(|_| Error::new(ErrorKind::StorageError))?;
        let param_refs: Vec<&dyn rusqlite::ToSql> = params.iter().map(|p| p.as_ref()).collect();
        guard
            .query_row(&sql, param_refs.as_slice(), |row| row.get(0))
            .map_err(|e| {
                Error::with_reason(
                    ErrorKind::StorageError,
                    format!("Database query failed: {e}"),
                )
            })
    }

    fn create_new_study(
        &mut self,
        study_name: &str,
        directions: Vec<rustuna_core::study::Direction>,
    ) -> rustuna_core::Result<rustuna_core::study::PersistedStudy> {
        let guard = self
            .conn
            .lock()
            .map_err(|_| Error::new(ErrorKind::StorageError))?;

        let existing: Option<u32> = guard
            .query_row(
                "SELECT study_id FROM studies WHERE study_name = ?",
                params![study_name],
                |row| row.get(0),
            )
            .optional()
            .map_err(|e| {
                Error::with_reason(
                    ErrorKind::StorageError,
                    format!("Database query failed: {e}"),
                )
            })?;
        if existing.is_some() {
            return Err(Error::new(ErrorKind::DuplicatedStudy));
        }

        guard
            .execute(
                "INSERT INTO studies (study_name) VALUES (?)",
                params![study_name],
            )
            .map_err(|e| {
                Error::with_reason(
                    ErrorKind::StorageError,
                    format!("Database query failed: {e}"),
                )
            })?;

        let study_id = guard.last_insert_rowid() as u32;

        for (objective, direction) in directions.iter().enumerate() {
            let direction_str = match direction {
                Direction::Minimize => "MINIMIZE",
                Direction::Maximize => "MAXIMIZE",
            };

            guard.execute(
                "INSERT INTO study_directions (direction, study_id, objective) VALUES (?, ?, ?)",
                params![direction_str, study_id, objective as u32],
            )
            .map_err(|e| Error::with_reason(ErrorKind::StorageError, format!("Database query failed: {e}")))?;
        }
        drop(guard);

        let persisted_study = PersistedStudy::new(study_id, study_name.to_string(), directions);
        Ok(persisted_study)
    }

    fn create_new_trial(
        &mut self,
        study_id: u32,
    ) -> rustuna_core::Result<rustuna_core::trial::PersistedTrial> {
        let mut guard = self
            .conn
            .lock()
            .map_err(|_| Error::new(ErrorKind::StorageError))?;
        // A trial must not become visible until its number has been assigned. Without a
        // transaction, another process can observe the row inserted below with number=NULL and
        // fail while decoding it as a u32. IMMEDIATE also serializes number allocation among
        // concurrent writers before either of them computes COUNT(...).
        let tx = guard
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .map_err(|e| {
                Error::with_reason(
                    ErrorKind::StorageError,
                    format!("Database query failed: {e}"),
                )
            })?;
        let study_exists: Option<u32> = tx
            .query_row(
                "SELECT study_id FROM studies WHERE study_id = ?",
                params![study_id],
                |row| row.get(0),
            )
            .optional()
            .map_err(|e| {
                Error::with_reason(
                    ErrorKind::StorageError,
                    format!("Database query failed: {e}"),
                )
            })?;
        if study_exists.is_none() {
            return Err(Error::new(ErrorKind::StudyNotFound));
        }
        tx.execute(
            "INSERT INTO trials (number, study_id, state, datetime_start, datetime_complete) \
                 VALUES (NULL, ?, ?, ?, NULL)",
            params![study_id, "RUNNING", now_naive_utc()],
        )
        .map_err(|e| {
            Error::with_reason(
                ErrorKind::StorageError,
                format!("Database query failed: {e}"),
            )
        })?;

        let trial_id = tx.last_insert_rowid() as u32;
        let number: u32 = tx
            .query_row(
                "SELECT COUNT(trial_id) FROM trials WHERE study_id = ? AND trial_id < ?",
                params![study_id, trial_id],
                |row| row.get(0),
            )
            .map_err(|e| {
                Error::with_reason(
                    ErrorKind::StorageError,
                    format!("Database query failed: {e}"),
                )
            })?;

        tx.execute(
            "UPDATE trials SET number = ? WHERE trial_id = ?",
            params![number, trial_id],
        )
        .map_err(|e| {
            Error::with_reason(
                ErrorKind::StorageError,
                format!("Database query failed: {e}"),
            )
        })?;

        let datetime_start: Option<String> = tx
            .query_row(
                "SELECT datetime_start FROM trials WHERE trial_id = ?",
                params![trial_id],
                |row| row.get(0),
            )
            .optional()
            .map_err(|e| {
                Error::with_reason(
                    ErrorKind::StorageError,
                    format!("Database query failed: {e}"),
                )
            })?;
        tx.commit().map_err(|e| {
            Error::with_reason(
                ErrorKind::StorageError,
                format!("Database query failed: {e}"),
            )
        })?;

        let mut trial = PersistedTrial::new(trial_id, study_id, number);
        trial.datetime_start = datetime_start;
        Ok(trial)
    }

    fn create_new_trial_from_template(
        &mut self,
        study_id: u32,
        template: &PersistedTrial,
    ) -> rustuna_core::Result<PersistedTrial> {
        for param_name in template.internal_params.keys() {
            if !template.distributions.contains_key(param_name) {
                return Err(Error::with_reason(
                    ErrorKind::StorageError,
                    format!(
                        "Template trial has internal_params['{param_name}'] but no matching distribution."
                    ),
                ));
            }
        }
        for param_name in template.distributions.keys() {
            if !template.internal_params.contains_key(param_name) {
                return Err(Error::with_reason(
                    ErrorKind::StorageError,
                    format!(
                        "Template trial has distributions['{param_name}'] but no matching internal_params."
                    ),
                ));
            }
        }

        let new_trial = self.create_new_trial(study_id)?;
        let trial_id = new_trial.id;

        for (param_name, distribution) in &template.distributions {
            let internal_value = template.internal_params.get(param_name).ok_or_else(|| {
                Error::with_reason(
                    ErrorKind::StorageError,
                    format!("Template trial has no internal param for '{param_name}'"),
                )
            })?;
            self.set_trial_param(trial_id, param_name, distribution, *internal_value)?;
        }

        if !template.attrs.is_empty() {
            self.set_trial_attrs(trial_id, template.attrs.clone(), false)?;
        }

        if !template.intermediate_values.is_empty() {
            self.set_trial_intermediate_values(trial_id, template.intermediate_values.clone())?;
        }

        if !matches!(template.state_values, TrialStateValues::Running) {
            self.set_trial_state_values(trial_id, template.state_values.clone())?;
        }

        let guard = self
            .conn
            .lock()
            .map_err(|_| Error::new(ErrorKind::StorageError))?;
        guard
            .execute(
                "UPDATE trials SET datetime_start = ?, datetime_complete = ? WHERE trial_id = ?",
                params![
                    template.datetime_start,
                    template.datetime_complete,
                    trial_id
                ],
            )
            .map_err(|e| {
                Error::with_reason(
                    ErrorKind::StorageError,
                    format!("Database query failed: {e}"),
                )
            })?;
        drop(guard);

        self.get_trial(trial_id)
    }

    fn set_trial_param(
        &mut self,
        trial_id: u32,
        name: &str,
        distribution: &rustuna_core::distribution::Distribution,
        value: f64,
    ) -> rustuna_core::Result<()> {
        // Note: Compatibility between distributions across trials is enforced by CachedStorage.
        let guard = self
            .conn
            .lock()
            .map_err(|_| Error::new(ErrorKind::StorageError))?;
        let study_id: Option<u32> = guard
            .query_row(
                "SELECT study_id FROM trials WHERE trial_id = ?",
                params![trial_id],
                |row| row.get(0),
            )
            .optional()
            .map_err(|e| {
                Error::with_reason(
                    ErrorKind::StorageError,
                    format!("Database query failed: {e}"),
                )
            })?;
        let study_id = study_id.ok_or_else(|| Error::new(ErrorKind::TrialNotFound))?;

        let labels = match distribution {
            Distribution::Categorical { cardinality } => {
                read_category_labels(&guard, study_id, name, *cardinality)?
            }
            _ => None,
        };
        let distribution_json = distribution_to_json(distribution, labels.as_deref());
        guard
            .execute(
                "INSERT INTO trial_params (trial_id, param_name, param_value, distribution_json) \
             VALUES (?, ?, ?, ?) \
             ON CONFLICT(trial_id, param_name) DO UPDATE SET \
                 param_value=excluded.param_value, distribution_json=excluded.distribution_json",
                params![trial_id, name, value, distribution_json],
            )
            .map_err(|e| {
                Error::with_reason(
                    ErrorKind::StorageError,
                    format!("Database query failed: {e}"),
                )
            })?;
        Ok(())
    }

    fn set_trial_state_values(
        &mut self,
        trial_id: u32,
        state_values: rustuna_core::trial::TrialStateValues,
    ) -> rustuna_core::Result<()> {
        if matches!(&state_values, TrialStateValues::Complete(values) if values.is_empty()) {
            return Err(Error::with_reason(
                ErrorKind::InvalidObjectiveValues,
                format!("Cannot complete trial {trial_id} without objective values"),
            ));
        }

        let mut guard = self
            .conn
            .lock()
            .map_err(|_| Error::new(ErrorKind::StorageError))?;
        // State and objective values form one logical record. Keep both the finished-state check
        // and all writes in one IMMEDIATE transaction so another process can never observe
        // COMPLETE before its trial_values rows exist, and a failed values write cannot leave a
        // permanently incomplete trial behind.
        let tx = guard
            .transaction_with_behavior(TransactionBehavior::Immediate)
            .map_err(|e| {
                Error::with_reason(
                    ErrorKind::StorageError,
                    format!("Database query failed: {e}"),
                )
            })?;
        let result: Option<String> = tx
            .query_row(
                "SELECT state FROM trials WHERE trial_id = ?",
                params![trial_id],
                |row| row.get(0),
            )
            .optional()
            .map_err(|e| {
                Error::with_reason(
                    ErrorKind::StorageError,
                    format!("Database query failed: {e}"),
                )
            })?;
        let current_state = result.ok_or(Error::new(ErrorKind::TrialNotFound))?;

        if matches!(current_state.as_str(), "COMPLETE" | "FAIL" | "PRUNED") {
            return Err(Error::new(ErrorKind::TrialAlreadyFinished));
        }

        match &state_values {
            TrialStateValues::Complete(values) => {
                let placeholders = values
                    .iter()
                    .enumerate()
                    .map(|(i, _)| format!("({trial_id}, {i}, ?, 'FINITE')"))
                    .collect::<Vec<_>>()
                    .join(", ");
                let sql = format!(
                    "INSERT INTO trial_values (trial_id, objective, value, value_type) VALUES {placeholders} \
                     ON CONFLICT(trial_id, objective) DO UPDATE SET value=excluded.value, value_type=excluded.value_type"
                );
                let params: Vec<&dyn rusqlite::ToSql> =
                    values.iter().map(|v| v as &dyn rusqlite::ToSql).collect();
                tx.execute(&sql, params.as_slice()).map_err(|e| {
                    Error::with_reason(
                        ErrorKind::StorageError,
                        format!("Database query failed: {e}"),
                    )
                })?;

                tx.execute(
                    "UPDATE trials SET state = ?, datetime_complete = ? WHERE trial_id = ?",
                    params!["COMPLETE", now_naive_utc(), trial_id],
                )
                .map_err(|e| {
                    Error::with_reason(
                        ErrorKind::StorageError,
                        format!("Database query failed: {e}"),
                    )
                })?;
            }
            TrialStateValues::Pruned => {
                tx.execute(
                    "UPDATE trials SET state = ?, datetime_complete = ? WHERE trial_id = ?",
                    params!["PRUNED", now_naive_utc(), trial_id],
                )
                .map_err(|e| {
                    Error::with_reason(
                        ErrorKind::StorageError,
                        format!("Database query failed: {e}"),
                    )
                })?;
            }
            TrialStateValues::Fail => {
                tx.execute(
                    "UPDATE trials SET state = ?, datetime_complete = ? WHERE trial_id = ?",
                    params!["FAIL", now_naive_utc(), trial_id],
                )
                .map_err(|e| {
                    Error::with_reason(
                        ErrorKind::StorageError,
                        format!("Database query failed: {e}"),
                    )
                })?;
            }
            TrialStateValues::Running => {
                tx.execute(
                    "UPDATE trials SET state = ?, datetime_complete = NULL WHERE trial_id = ?",
                    params!["RUNNING", trial_id],
                )
                .map_err(|e| {
                    Error::with_reason(
                        ErrorKind::StorageError,
                        format!("Database query failed: {e}"),
                    )
                })?;
            }
            TrialStateValues::Waiting => {
                tx.execute(
                    "UPDATE trials SET state = ?, datetime_complete = NULL WHERE trial_id = ?",
                    params!["WAITING", trial_id],
                )
                .map_err(|e| {
                    Error::with_reason(
                        ErrorKind::StorageError,
                        format!("Database query failed: {e}"),
                    )
                })?;
            }
        }

        tx.commit().map_err(|e| {
            Error::with_reason(
                ErrorKind::StorageError,
                format!("Database query failed: {e}"),
            )
        })?;
        Ok(())
    }

    fn get_studies(&mut self) -> rustuna_core::Result<Vec<rustuna_core::study::PersistedStudy>> {
        let guard = self
            .conn
            .lock()
            .map_err(|_| Error::new(ErrorKind::StorageError))?;

        let mut studies = Vec::new();
        let mut stmt = guard
            .prepare("SELECT study_id, study_name FROM studies ORDER BY study_id")
            .map_err(|e| {
                Error::with_reason(
                    ErrorKind::StorageError,
                    format!("Database query failed: {e}"),
                )
            })?;
        let rows = stmt
            .query_map([], |row| {
                Ok((row.get::<_, u32>(0)?, row.get::<_, String>(1)?))
            })
            .map_err(|e| {
                Error::with_reason(
                    ErrorKind::StorageError,
                    format!("Database query failed: {e}"),
                )
            })?;
        for row in rows {
            let (study_id, study_name) = row.map_err(|e| {
                Error::with_reason(
                    ErrorKind::StorageError,
                    format!("Database query failed: {e}"),
                )
            })?;

            // Directions
            let mut directions_stmt = guard
                .prepare(
                    "SELECT direction FROM study_directions WHERE study_id = ? ORDER BY objective",
                )
                .map_err(|e| {
                    Error::with_reason(
                        ErrorKind::StorageError,
                        format!("Database query failed: {e}"),
                    )
                })?;
            let directions_rows = directions_stmt
                .query_map(params![study_id], |row| row.get::<_, String>(0))
                .map_err(|e| {
                    Error::with_reason(
                        ErrorKind::StorageError,
                        format!("Database query failed: {e}"),
                    )
                })?;
            let mut directions: Vec<Direction> = Vec::new();
            for d in directions_rows {
                let dir_str = d.map_err(|e| {
                    Error::with_reason(
                        ErrorKind::StorageError,
                        format!("Database query failed: {e}"),
                    )
                })?;
                let dir = match dir_str.as_str() {
                    "MINIMIZE" => Direction::Minimize,
                    "MAXIMIZE" => Direction::Maximize,
                    _ => return Err(Error::new(ErrorKind::StorageError)),
                };
                directions.push(dir);
            }

            // Attributes
            let mut attrs: Attrs = Attrs::new();

            let mut user_stmt = guard
                .prepare("SELECT key, value_json FROM study_user_attributes WHERE study_id = ?")
                .map_err(|e| {
                    Error::with_reason(
                        ErrorKind::StorageError,
                        format!("Database query failed: {e}"),
                    )
                })?;
            let user_rows = user_stmt
                .query_map(params![study_id], |row| {
                    Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
                })
                .map_err(|e| {
                    Error::with_reason(
                        ErrorKind::StorageError,
                        format!("Database query failed: {e}"),
                    )
                })?;
            for row in user_rows {
                let (key, value) = row.map_err(|e| {
                    Error::with_reason(
                        ErrorKind::StorageError,
                        format!("Database query failed: {e}"),
                    )
                })?;
                attrs.insert(AttrKey::User(key.into()), value);
            }

            let mut system_stmt = guard
                .prepare("SELECT key, value_json FROM study_system_attributes WHERE study_id = ?")
                .map_err(|e| {
                    Error::with_reason(
                        ErrorKind::StorageError,
                        format!("Database query failed: {e}"),
                    )
                })?;
            let system_rows = system_stmt
                .query_map(params![study_id], |row| {
                    Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
                })
                .map_err(|e| {
                    Error::with_reason(
                        ErrorKind::StorageError,
                        format!("Database query failed: {e}"),
                    )
                })?;
            for row in system_rows {
                let (key, value) = row.map_err(|e| {
                    Error::with_reason(
                        ErrorKind::StorageError,
                        format!("Database query failed: {e}"),
                    )
                })?;
                attrs.insert(AttrKey::System(key.into()), value);
            }

            // Optuna stores categorical labels in each distribution JSON, whereas Rustuna's
            // Storage API exposes them through study system attributes. Materialize that native
            // representation when an Optuna-created database is opened.
            let mut categorical_stmt = guard
                .prepare(
                    "SELECT DISTINCT tp.param_name, tp.distribution_json \
                     FROM trial_params AS tp \
                     JOIN trials AS t ON t.trial_id = tp.trial_id \
                     WHERE t.study_id = ?",
                )
                .map_err(|e| {
                    Error::with_reason(
                        ErrorKind::StorageError,
                        format!("Database query failed: {e}"),
                    )
                })?;
            let categorical_rows = categorical_stmt
                .query_map(params![study_id], |row| {
                    Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
                })
                .map_err(|e| {
                    Error::with_reason(
                        ErrorKind::StorageError,
                        format!("Database query failed: {e}"),
                    )
                })?;
            for row in categorical_rows {
                let (param_name, distribution_json) = row.map_err(|e| {
                    Error::with_reason(
                        ErrorKind::StorageError,
                        format!("Database query failed: {e}"),
                    )
                })?;
                let (_, labels) = json_to_distribution(&distribution_json)?;
                if let Some(labels) = labels {
                    for (key, value) in category_labels_to_attrs(&param_name, &labels) {
                        attrs.entry(key).or_insert(value);
                    }
                }
            }

            let study = PersistedStudy::new_with_attrs(study_id, study_name, directions, attrs);
            studies.push(study);
        }
        Ok(studies)
    }

    fn get_study(
        &mut self,
        study_id: u32,
    ) -> rustuna_core::Result<rustuna_core::study::PersistedStudy> {
        let studies = self.get_studies()?;
        studies
            .into_iter()
            .find(|s| s.id == study_id)
            .ok_or(Error::new(ErrorKind::StudyNotFound))
    }

    fn get_trial(
        &mut self,
        trial_id: u32,
    ) -> rustuna_core::Result<rustuna_core::trial::PersistedTrial> {
        let guard = self
            .conn
            .lock()
            .map_err(|_| Error::new(ErrorKind::StorageError))?;

        // Query to trials table.
        let trial_row: Option<TrialRow> = guard
            .query_row(
                "SELECT study_id, number, state, datetime_start, datetime_complete FROM trials WHERE trial_id = ?",
                params![trial_id],
                |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?, row.get(4)?)),
            )
            .optional()
            .map_err(|e| Error::with_reason(ErrorKind::StorageError, format!("Database query failed: {e}")))?;
        let (study_id, number, state_str, datetime_start, datetime_complete) =
            trial_row.ok_or(Error::new(ErrorKind::TrialNotFound))?;
        let state_values = match state_str.as_str() {
            "RUNNING" => TrialStateValues::Running,
            "WAITING" => TrialStateValues::Waiting,
            "PRUNED" => TrialStateValues::Pruned,
            "FAIL" => TrialStateValues::Fail,
            "COMPLETE" => TrialStateValues::Complete(read_trial_values(&guard, trial_id)?),
            _ => return Err(Error::new(ErrorKind::StorageError)),
        };

        // Query to trial_params table.
        let mut distributions = HashMap::new();
        let mut internal_params = HashMap::new();
        let mut stmt = guard
            .prepare(
                "SELECT param_name, param_value, distribution_json FROM trial_params WHERE trial_id = ?",
            )
            .map_err(|e| Error::with_reason(ErrorKind::StorageError, format!("Database query failed: {e}")))?;
        let param_rows = stmt
            .query_map(params![trial_id], |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, f64>(1)?,
                    row.get::<_, String>(2)?,
                ))
            })
            .map_err(|e| {
                Error::with_reason(
                    ErrorKind::StorageError,
                    format!("Database query failed: {e}"),
                )
            })?;
        for row in param_rows {
            let (name, value, distribution_json) = row.map_err(|e| {
                Error::with_reason(
                    ErrorKind::StorageError,
                    format!("Database query failed: {e}"),
                )
            })?;
            let (distribution, _labels) = json_to_distribution(&distribution_json)?;
            distributions.insert(name.clone(), distribution);
            internal_params.insert(name, value);
        }

        // User attributes
        let mut attrs: Attrs = Attrs::new();
        let mut stmt = guard
            .prepare("SELECT key, value_json FROM trial_user_attributes WHERE trial_id = ?")
            .map_err(|e| {
                Error::with_reason(
                    ErrorKind::StorageError,
                    format!("Database query failed: {e}"),
                )
            })?;
        let user_attr_rows = stmt
            .query_map(params![trial_id], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
            })
            .map_err(|e| {
                Error::with_reason(
                    ErrorKind::StorageError,
                    format!("Database query failed: {e}"),
                )
            })?;
        for row in user_attr_rows {
            let (key, value) = row.map_err(|e| {
                Error::with_reason(
                    ErrorKind::StorageError,
                    format!("Database query failed: {e}"),
                )
            })?;
            attrs.insert(AttrKey::User(key.into()), value);
        }

        // System attributes
        let mut stmt = guard
            .prepare("SELECT key, value_json FROM trial_system_attributes WHERE trial_id = ?")
            .map_err(|e| {
                Error::with_reason(
                    ErrorKind::StorageError,
                    format!("Database query failed: {e}"),
                )
            })?;
        let system_attr_rows = stmt
            .query_map(params![trial_id], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
            })
            .map_err(|e| {
                Error::with_reason(
                    ErrorKind::StorageError,
                    format!("Database query failed: {e}"),
                )
            })?;
        for row in system_attr_rows {
            let (key, value) = row.map_err(|e| {
                Error::with_reason(
                    ErrorKind::StorageError,
                    format!("Database query failed: {e}"),
                )
            })?;
            attrs.insert(AttrKey::System(key.into()), value);
        }

        let intermediate_values = read_intermediate_values(&guard, trial_id)?;

        let mut trial = PersistedTrial::new(trial_id, study_id, number);
        trial.state_values = state_values;
        trial.internal_params = internal_params;
        trial.distributions = distributions;
        trial.intermediate_values = intermediate_values;
        trial.attrs = attrs;
        trial.datetime_start = datetime_start;
        trial.datetime_complete = datetime_complete;
        Ok(trial)
    }

    fn get_study_attr(
        &mut self,
        study_id: u32,
        key: rustuna_core::attr::AttrKey,
    ) -> rustuna_core::Result<String> {
        let guard = self
            .conn
            .lock()
            .map_err(|_| Error::new(ErrorKind::StorageError))?;
        let (table, key_str) = match &key {
            AttrKey::User(k) => ("study_user_attributes", k.as_str()),
            AttrKey::System(k) => ("study_system_attributes", k.as_str()),
        };
        let sql = format!("SELECT value_json FROM {table} WHERE study_id = ? AND key = ?");
        guard
            .query_row(&sql, params![study_id, key_str], |row| row.get(0))
            .optional()
            .map_err(|e| {
                Error::with_reason(
                    ErrorKind::StorageError,
                    format!("Database query failed: {e}"),
                )
            })?
            .ok_or(Error::new(ErrorKind::AttrNotFound))
    }

    fn set_study_attrs(
        &mut self,
        study_id: u32,
        attrs: rustuna_core::attr::Attrs,
        error_on_overwrite: bool,
    ) -> rustuna_core::Result<()> {
        self.validate_study_id(study_id)?;

        let mut user_attrs = Vec::new();
        let mut system_attrs = Vec::new();
        for (key, value) in attrs {
            match key {
                AttrKey::User(key_str) => user_attrs.push((key_str.to_string(), value)),
                AttrKey::System(key_str) => system_attrs.push((key_str.to_string(), value)),
            }
        }

        let mut guard = self
            .conn
            .lock()
            .map_err(|_| Error::new(ErrorKind::StorageError))?;
        let tx = guard.transaction().map_err(|e| {
            Error::with_reason(
                ErrorKind::StorageError,
                format!("Database query failed: {e}"),
            )
        })?;

        if !user_attrs.is_empty() {
            let placeholders = user_attrs
                .iter()
                .map(|_| "(?, ?, ?)")
                .collect::<Vec<_>>()
                .join(", ");
            let sql = if error_on_overwrite {
                format!(
                    "INSERT INTO study_user_attributes (study_id, key, value_json) VALUES {placeholders}"
                )
            } else {
                format!(
                    "INSERT INTO study_user_attributes (study_id, key, value_json) VALUES {placeholders} \
                 ON CONFLICT(study_id, key) DO UPDATE SET value_json=excluded.value_json"
                )
            };
            let mut params: Vec<&dyn rusqlite::ToSql> = Vec::new();
            for (key, value) in &user_attrs {
                params.push(&study_id);
                params.push(key);
                params.push(value);
            }
            let res = tx.execute(&sql, params.as_slice());
            if let Err(RusqliteError::SqliteFailure(_, _)) = res {
                if error_on_overwrite {
                    return Err(Error::new(ErrorKind::AttrOverwriteNotAllowed));
                }
                return Err(Error::with_reason(
                    ErrorKind::StorageError,
                    "Database query failed".to_string(),
                ));
            } else if res.is_err() {
                return Err(Error::with_reason(
                    ErrorKind::StorageError,
                    "Database query failed".to_string(),
                ));
            }
        }

        if !system_attrs.is_empty() {
            let placeholders = system_attrs
                .iter()
                .map(|_| "(?, ?, ?)")
                .collect::<Vec<_>>()
                .join(", ");
            let sql = if error_on_overwrite {
                format!(
                    "INSERT INTO study_system_attributes (study_id, key, value_json) VALUES {placeholders}"
                )
            } else {
                format!(
                    "INSERT INTO study_system_attributes (study_id, key, value_json) VALUES {placeholders} \
                 ON CONFLICT(study_id, key) DO UPDATE SET value_json=excluded.value_json"
                )
            };
            let mut params: Vec<&dyn rusqlite::ToSql> = Vec::new();
            for (key, value) in &system_attrs {
                params.push(&study_id);
                params.push(key);
                params.push(value);
            }
            let res = tx.execute(&sql, params.as_slice());
            if let Err(RusqliteError::SqliteFailure(_, _)) = res {
                if error_on_overwrite {
                    return Err(Error::new(ErrorKind::AttrOverwriteNotAllowed));
                }
                return Err(Error::with_reason(
                    ErrorKind::StorageError,
                    "Database query failed".to_string(),
                ));
            } else if res.is_err() {
                return Err(Error::with_reason(
                    ErrorKind::StorageError,
                    "Database query failed".to_string(),
                ));
            }
        }

        tx.commit().map_err(|e| {
            Error::with_reason(
                ErrorKind::StorageError,
                format!("Database query failed: {e}"),
            )
        })?;
        Ok(())
    }

    fn set_trial_attrs(
        &mut self,
        trial_id: u32,
        attrs: rustuna_core::attr::Attrs,
        error_on_overwrite: bool,
    ) -> rustuna_core::Result<()> {
        let mut user_attrs = Vec::new();
        let mut system_attrs = Vec::new();
        for (key, value) in attrs {
            match key {
                AttrKey::User(key_str) => user_attrs.push((key_str.to_string(), value)),
                AttrKey::System(key_str) => system_attrs.push((key_str.to_string(), value)),
            }
        }

        let mut guard = self
            .conn
            .lock()
            .map_err(|_| Error::new(ErrorKind::StorageError))?;

        let tx = guard.transaction().map_err(|e| {
            Error::with_reason(
                ErrorKind::StorageError,
                format!("Database query failed: {e}"),
            )
        })?;

        if !user_attrs.is_empty() {
            let placeholders = user_attrs
                .iter()
                .map(|_| "(?, ?, ?)")
                .collect::<Vec<_>>()
                .join(", ");
            let sql = if error_on_overwrite {
                format!(
                    "INSERT INTO trial_user_attributes (trial_id, key, value_json) VALUES {placeholders}"
                )
            } else {
                format!(
                    "INSERT INTO trial_user_attributes (trial_id, key, value_json) VALUES {placeholders} \
                 ON CONFLICT(trial_id, key) DO UPDATE SET value_json=excluded.value_json"
                )
            };
            let mut params: Vec<&dyn rusqlite::ToSql> = Vec::new();
            for (key, value) in &user_attrs {
                params.push(&trial_id);
                params.push(key);
                params.push(value);
            }
            let res = tx.execute(&sql, params.as_slice());
            if let Err(RusqliteError::SqliteFailure(_, _)) = res {
                if error_on_overwrite {
                    return Err(Error::new(ErrorKind::AttrOverwriteNotAllowed));
                }
                return Err(Error::with_reason(
                    ErrorKind::StorageError,
                    "Database query failed".to_string(),
                ));
            } else if res.is_err() {
                return Err(Error::with_reason(
                    ErrorKind::StorageError,
                    "Database query failed".to_string(),
                ));
            }
        }

        if !system_attrs.is_empty() {
            let placeholders = system_attrs
                .iter()
                .map(|_| "(?, ?, ?)")
                .collect::<Vec<_>>()
                .join(", ");
            let sql = if error_on_overwrite {
                format!(
                    "INSERT INTO trial_system_attributes (trial_id, key, value_json) VALUES {placeholders}"
                )
            } else {
                format!(
                    "INSERT INTO trial_system_attributes (trial_id, key, value_json) VALUES {placeholders} \
                 ON CONFLICT(trial_id, key) DO UPDATE SET value_json=excluded.value_json"
                )
            };
            let mut params: Vec<&dyn rusqlite::ToSql> = Vec::new();
            for (key, value) in &system_attrs {
                params.push(&trial_id);
                params.push(key);
                params.push(value);
            }
            let res = tx.execute(&sql, params.as_slice());
            if let Err(RusqliteError::SqliteFailure(_, _)) = res {
                if error_on_overwrite {
                    return Err(Error::new(ErrorKind::AttrOverwriteNotAllowed));
                }
                return Err(Error::with_reason(
                    ErrorKind::StorageError,
                    "Database query failed".to_string(),
                ));
            } else if res.is_err() {
                return Err(Error::with_reason(
                    ErrorKind::StorageError,
                    "Database query failed".to_string(),
                ));
            }
        }

        tx.commit().map_err(|e| {
            Error::with_reason(
                ErrorKind::StorageError,
                format!("Database query failed: {e}"),
            )
        })?;
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

        let guard = self
            .conn
            .lock()
            .map_err(|_| Error::new(ErrorKind::StorageError))?;

        // TODO(c-bata): Check if Optuna enables PRAGMA foreign_keys and if we can skip this check
        // Explicitly check trial existence and state since the schema might be created by Optuna
        let trial_state: Option<String> = guard
            .query_row(
                "SELECT state FROM trials WHERE trial_id = ?",
                params![trial_id],
                |row| row.get(0),
            )
            .optional()
            .map_err(|e| {
                Error::with_reason(
                    ErrorKind::StorageError,
                    format!("Database query failed: {e}"),
                )
            })?;

        let state = trial_state.ok_or_else(|| Error::new(ErrorKind::TrialNotFound))?;

        if matches!(state.as_str(), "COMPLETE" | "FAIL" | "PRUNED") {
            return Err(Error::new(ErrorKind::TrialAlreadyFinished));
        }

        let placeholders = intermediate_values
            .iter()
            .map(|_| "(?, ?, ?, ?)")
            .collect::<Vec<_>>()
            .join(", ");
        let sql = format!(
            "INSERT INTO trial_intermediate_values (trial_id, step, intermediate_value, intermediate_value_type) VALUES {placeholders} \
             ON CONFLICT(trial_id, step) DO UPDATE SET intermediate_value=excluded.intermediate_value, intermediate_value_type=excluded.intermediate_value_type"
        );

        let mut params: Vec<Box<dyn rusqlite::ToSql>> = Vec::new();
        for (step, value) in &intermediate_values {
            let (stored_value, value_type) = if value.is_nan() {
                (None, "NAN")
            } else if value.is_infinite() {
                if value.is_sign_positive() {
                    (None, "INF_POS")
                } else {
                    (None, "INF_NEG")
                }
            } else {
                (Some(*value), "FINITE")
            };

            params.push(Box::new(trial_id));
            params.push(Box::new(*step));
            params.push(Box::new(stored_value));
            params.push(Box::new(value_type.to_string()));
        }

        let param_refs: Vec<&dyn rusqlite::ToSql> = params.iter().map(|p| p.as_ref()).collect();
        guard.execute(&sql, param_refs.as_slice()).map_err(|e| {
            Error::with_reason(
                ErrorKind::StorageError,
                format!("Database query failed: {e}"),
            )
        })?;

        Ok(())
    }

    fn get_trials_diff(
        &mut self,
        study_id: u32,
        included_numbers: &[u32],
        trial_number_greater_than: i32,
    ) -> rustuna_core::Result<Vec<rustuna_core::trial::PersistedTrial>> {
        let guard = self
            .conn
            .lock()
            .map_err(|_| Error::new(ErrorKind::StorageError))?;

        let select_columns =
            "SELECT trial_id, number, state, datetime_start, datetime_complete FROM trials";
        let discard_condition =
            if self.options.apply_discard && self.has_discarded_at_column.load(Ordering::Acquire) {
                " AND discarded_at IS NULL"
            } else {
                ""
            };
        // Numbers above the threshold are already returned by the range query. Filtering them
        // out here avoids an unnecessary second scan in the usual case where the current trial
        // is the only unfinished trial.
        let included_numbers: Vec<u32> = if trial_number_greater_than < 0 {
            Vec::new()
        } else {
            included_numbers
                .iter()
                .copied()
                .filter(|number| *number <= trial_number_greater_than as u32)
                .collect()
        };
        let (sql, params): (String, Vec<Box<dyn rusqlite::ToSql>>) = if included_numbers.is_empty()
        {
            (
                format!(
                    "{select_columns} WHERE study_id = ? AND number > ?{discard_condition} ORDER BY trial_id"
                ),
                vec![Box::new(study_id), Box::new(trial_number_greater_than)],
            )
        } else {
            let placeholders = included_numbers
                .iter()
                .map(|_| "?")
                .collect::<Vec<_>>()
                .join(", ");
            let mut params: Vec<Box<dyn rusqlite::ToSql>> = vec![
                Box::new(study_id),
                Box::new(trial_number_greater_than),
                Box::new(study_id),
            ];
            params.extend(
                included_numbers
                    .iter()
                    .map(|number| Box::new(*number) as Box<dyn rusqlite::ToSql>),
            );
            (
                format!(
                    "{select_columns} WHERE study_id = ? AND number > ?{discard_condition} \
                         UNION ALL \
                         {select_columns} WHERE study_id = ? AND number IN ({placeholders}){discard_condition} \
                         ORDER BY trial_id"
                ),
                params,
            )
        };

        let mut stmt = guard.prepare(&sql).map_err(|e| {
            Error::with_reason(
                ErrorKind::StorageError,
                format!("Database query failed: {e}"),
            )
        })?;

        let param_refs: Vec<&dyn rusqlite::ToSql> = params.iter().map(|p| p.as_ref()).collect();
        let trial_rows = stmt
            .query_map(param_refs.as_slice(), |row| {
                Ok((
                    row.get::<_, u32>(0)?,
                    row.get::<_, u32>(1)?,
                    row.get::<_, String>(2)?,
                    row.get::<_, Option<String>>(3)?,
                    row.get::<_, Option<String>>(4)?,
                ))
            })
            .map_err(|e| {
                Error::with_reason(
                    ErrorKind::StorageError,
                    format!("Database query failed: {e}"),
                )
            })?;

        let mut trials = Vec::new();
        for row in trial_rows {
            let (trial_id, number, state_str, datetime_start, datetime_complete) =
                row.map_err(|e| {
                    Error::with_reason(
                        ErrorKind::StorageError,
                        format!("Database query failed: {e}"),
                    )
                })?;

            // Parse state and get values if COMPLETE
            let state_values = match state_str.as_str() {
                "RUNNING" => TrialStateValues::Running,
                "WAITING" => TrialStateValues::Waiting,
                "PRUNED" => TrialStateValues::Pruned,
                "FAIL" => TrialStateValues::Fail,
                "COMPLETE" => TrialStateValues::Complete(read_trial_values(&guard, trial_id)?),
                _ => return Err(Error::new(ErrorKind::StorageError)),
            };

            // Get distributions and params
            let mut distributions = HashMap::new();
            let mut internal_params = HashMap::new();
            let mut params_stmt = guard
                .prepare("SELECT param_name, param_value, distribution_json FROM trial_params WHERE trial_id = ?")
                .map_err(|e| Error::with_reason(ErrorKind::StorageError, format!("Database query failed: {e}")))?;
            let param_rows = params_stmt
                .query_map(params![trial_id], |row| {
                    Ok((
                        row.get::<_, String>(0)?,
                        row.get::<_, f64>(1)?,
                        row.get::<_, String>(2)?,
                    ))
                })
                .map_err(|e| {
                    Error::with_reason(
                        ErrorKind::StorageError,
                        format!("Database query failed: {e}"),
                    )
                })?;
            for row in param_rows {
                let (name, value, distribution_json) = row.map_err(|e| {
                    Error::with_reason(
                        ErrorKind::StorageError,
                        format!("Database query failed: {e}"),
                    )
                })?;
                let (distribution, _labels) = json_to_distribution(&distribution_json)?;
                distributions.insert(name.clone(), distribution);
                internal_params.insert(name, value);
            }

            // Get user attributes
            let mut attrs: Attrs = Attrs::new();
            let mut user_attrs_stmt = guard
                .prepare("SELECT key, value_json FROM trial_user_attributes WHERE trial_id = ?")
                .map_err(|e| {
                    Error::with_reason(
                        ErrorKind::StorageError,
                        format!("Database query failed: {e}"),
                    )
                })?;
            let user_attr_rows = user_attrs_stmt
                .query_map(params![trial_id], |row| {
                    Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
                })
                .map_err(|e| {
                    Error::with_reason(
                        ErrorKind::StorageError,
                        format!("Database query failed: {e}"),
                    )
                })?;
            for row in user_attr_rows {
                let (key, value) = row.map_err(|e| {
                    Error::with_reason(
                        ErrorKind::StorageError,
                        format!("Database query failed: {e}"),
                    )
                })?;
                attrs.insert(AttrKey::User(key.into()), value);
            }

            // Get system attributes
            let mut system_attrs_stmt = guard
                .prepare("SELECT key, value_json FROM trial_system_attributes WHERE trial_id = ?")
                .map_err(|e| {
                    Error::with_reason(
                        ErrorKind::StorageError,
                        format!("Database query failed: {e}"),
                    )
                })?;
            let system_attr_rows = system_attrs_stmt
                .query_map(params![trial_id], |row| {
                    Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
                })
                .map_err(|e| {
                    Error::with_reason(
                        ErrorKind::StorageError,
                        format!("Database query failed: {e}"),
                    )
                })?;
            for row in system_attr_rows {
                let (key, value) = row.map_err(|e| {
                    Error::with_reason(
                        ErrorKind::StorageError,
                        format!("Database query failed: {e}"),
                    )
                })?;
                attrs.insert(AttrKey::System(key.into()), value);
            }

            let intermediate_values = read_intermediate_values(&guard, trial_id)?;

            let mut trial = PersistedTrial::new(trial_id, study_id, number);
            trial.state_values = state_values;
            trial.internal_params = internal_params;
            trial.distributions = distributions;
            trial.intermediate_values = intermediate_values;
            trial.attrs = attrs;
            trial.datetime_start = datetime_start;
            trial.datetime_complete = datetime_complete;
            trials.push(trial);
        }

        Ok(trials)
    }

    fn delete_study(&mut self, study_id: u32) -> Result<()> {
        let guard = self
            .conn
            .lock()
            .map_err(|_| Error::new(ErrorKind::StorageError))?;

        // Delete trial-related records using subquery
        guard
            .execute(
                "DELETE FROM trial_values WHERE trial_id IN (SELECT trial_id FROM trials WHERE study_id = ?)",
                params![study_id]
            )
            .map_err(|e| Error::with_reason(ErrorKind::StorageError, format!("Database query failed: {e}")))?;
        guard
            .execute(
                "DELETE FROM trial_intermediate_values WHERE trial_id IN (SELECT trial_id FROM trials WHERE study_id = ?)",
                params![study_id]
            )
            .map_err(|e| Error::with_reason(ErrorKind::StorageError, format!("Database query failed: {e}")))?;
        guard
            .execute(
                "DELETE FROM trial_params WHERE trial_id IN (SELECT trial_id FROM trials WHERE study_id = ?)",
                params![study_id]
            )
            .map_err(|e| Error::with_reason(ErrorKind::StorageError, format!("Database query failed: {e}")))?;
        guard
            .execute(
                "DELETE FROM trial_system_attributes WHERE trial_id IN (SELECT trial_id FROM trials WHERE study_id = ?)",
                params![study_id]
            )
            .map_err(|e| Error::with_reason(ErrorKind::StorageError, format!("Database query failed: {e}")))?;
        guard
            .execute(
                "DELETE FROM trial_user_attributes WHERE trial_id IN (SELECT trial_id FROM trials WHERE study_id = ?)",
                params![study_id]
            )
            .map_err(|e| Error::with_reason(ErrorKind::StorageError, format!("Database query failed: {e}")))?;
        guard
            .execute(
                "DELETE FROM trial_heartbeats WHERE trial_id IN (SELECT trial_id FROM trials WHERE study_id = ?)",
                params![study_id]
            )
            .map_err(|e| Error::with_reason(ErrorKind::StorageError, format!("Database query failed: {e}")))?;

        // Delete trials
        guard
            .execute("DELETE FROM trials WHERE study_id = ?", params![study_id])
            .map_err(|e| {
                Error::with_reason(
                    ErrorKind::StorageError,
                    format!("Database query failed: {e}"),
                )
            })?;

        // Delete study-related records
        guard
            .execute(
                "DELETE FROM study_system_attributes WHERE study_id = ?",
                params![study_id],
            )
            .map_err(|e| {
                Error::with_reason(
                    ErrorKind::StorageError,
                    format!("Database query failed: {e}"),
                )
            })?;
        guard
            .execute(
                "DELETE FROM study_user_attributes WHERE study_id = ?",
                params![study_id],
            )
            .map_err(|e| {
                Error::with_reason(
                    ErrorKind::StorageError,
                    format!("Database query failed: {e}"),
                )
            })?;
        guard
            .execute(
                "DELETE FROM study_directions WHERE study_id = ?",
                params![study_id],
            )
            .map_err(|e| {
                Error::with_reason(
                    ErrorKind::StorageError,
                    format!("Database query failed: {e}"),
                )
            })?;

        // Finally delete the study
        guard
            .execute("DELETE FROM studies WHERE study_id = ?", params![study_id])
            .map_err(|e| {
                Error::with_reason(
                    ErrorKind::StorageError,
                    format!("Database query failed: {e}"),
                )
            })?;

        Ok(())
    }
}

fn distribution_to_json(distribution: &Distribution, labels: Option<&[CategoryLabel]>) -> String {
    let (name, attributes) = match distribution {
        Distribution::Float {
            low,
            high,
            step,
            log,
        } => (
            "FloatDistribution",
            json!({
                "low": low,
                "high": high,
                "step": step,
                "log": log
            }),
        ),
        Distribution::Int {
            low,
            high,
            step,
            log,
        } => (
            "IntDistribution",
            json!({
                "low": low,
                "high": high,
                "step": step,
                "log": log
            }),
        ),
        Distribution::Categorical { cardinality } => {
            let choices = labels
                .map(|ls| ls.iter().map(category_label_to_value).collect::<Vec<_>>())
                .unwrap_or_else(|| {
                    (0..*cardinality as u32)
                        .map(|i| serde_json::Value::Number(i.into()))
                        .collect::<Vec<_>>()
                });
            (
                "CategoricalDistribution",
                json!({
                    "choices": choices,
                }),
            )
        }
    };

    json!({
        "name": name,
        "attributes": attributes,
    })
    .to_string()
}

fn read_category_labels(
    conn: &Connection,
    study_id: u32,
    param_name: &str,
    cardinality: usize,
) -> Result<Option<Vec<CategoryLabel>>> {
    if cardinality == 0 {
        return Ok(Some(Vec::new()));
    }

    let mut stmt = conn
        .prepare(
            "SELECT key, value_json FROM study_system_attributes \
             WHERE study_id = ? AND key LIKE 'category_labels:%'",
        )
        .map_err(|e| {
            Error::with_reason(
                ErrorKind::StorageError,
                format!("Database query failed: {e}"),
            )
        })?;
    let rows = stmt
        .query_map(params![study_id], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
        })
        .map_err(|e| {
            Error::with_reason(
                ErrorKind::StorageError,
                format!("Database query failed: {e}"),
            )
        })?;
    let mut attrs = Attrs::new();
    for row in rows {
        let (key, value) = row.map_err(|e| {
            Error::with_reason(
                ErrorKind::StorageError,
                format!("Database query failed: {e}"),
            )
        })?;
        attrs.insert(AttrKey::System(key.into()), value);
    }
    Ok(get_category_labels(&attrs, param_name, cardinality))
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

fn json_to_distribution(
    distribution_json: &str,
) -> Result<(Distribution, Option<Vec<CategoryLabel>>)> {
    let value: Value = serde_json::from_str(distribution_json).map_err(|e| {
        Error::with_reason(
            ErrorKind::StorageError,
            format!("JSON serialization failed: {e}"),
        )
    })?;
    let name = value.get("name").and_then(|v| v.as_str()).ok_or_else(|| {
        Error::with_reason(
            ErrorKind::StorageError,
            "JSON serialization failed: missing 'name' field".to_string(),
        )
    })?;
    let attributes = value
        .get("attributes")
        .and_then(|v| v.as_object())
        .ok_or_else(|| {
            Error::with_reason(
                ErrorKind::StorageError,
                "JSON serialization failed: missing 'attributes' field".to_string(),
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
                        "JSON serialization failed: invalid 'low' value".to_string(),
                    )
                })?;
            let high = attributes
                .get("high")
                .and_then(|v| v.as_f64())
                .ok_or_else(|| {
                    Error::with_reason(
                        ErrorKind::StorageError,
                        "JSON serialization failed: invalid 'high' value".to_string(),
                    )
                })?;
            let log = attributes
                .get("log")
                .and_then(|v| v.as_bool())
                .ok_or_else(|| {
                    Error::with_reason(
                        ErrorKind::StorageError,
                        "JSON serialization failed: invalid 'log' value".to_string(),
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
                        "JSON serialization failed: invalid 'low' value".to_string(),
                    )
                })?;
            let high = attributes
                .get("high")
                .and_then(|v| v.as_i64())
                .ok_or_else(|| {
                    Error::with_reason(
                        ErrorKind::StorageError,
                        "JSON serialization failed: invalid 'high' value".to_string(),
                    )
                })?;
            let log = attributes
                .get("log")
                .and_then(|v| v.as_bool())
                .ok_or_else(|| {
                    Error::with_reason(
                        ErrorKind::StorageError,
                        "JSON serialization failed: invalid 'log' value".to_string(),
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
                    "JSON serialization failed: invalid 'size' or 'choices'".to_string(),
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
            format!("JSON serialization failed: unknown distribution name '{name}'"),
        )),
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

fn read_trial_values(conn: &Connection, trial_id: u32) -> Result<Vec<f64>> {
    let mut stmt = conn
        .prepare("SELECT value FROM trial_values WHERE trial_id = ? ORDER BY objective")
        .map_err(|e| {
            Error::with_reason(
                ErrorKind::StorageError,
                format!("Database query failed: {e}"),
            )
        })?;
    let values = stmt
        .query_map(params![trial_id], |row| row.get(0))
        .map_err(|e| {
            Error::with_reason(
                ErrorKind::StorageError,
                format!("Database query failed: {e}"),
            )
        })?
        .collect::<std::result::Result<Vec<f64>, _>>()
        .map_err(|e| {
            Error::with_reason(
                ErrorKind::StorageError,
                format!("Database query failed: {e}"),
            )
        })?;
    if values.is_empty() {
        return Err(Error::with_reason(
            ErrorKind::StorageError,
            format!("Trial {trial_id} is COMPLETE but has no objective values"),
        ));
    }
    Ok(values)
}

fn read_intermediate_values(conn: &Connection, trial_id: u32) -> Result<HashMap<u32, f64>> {
    let mut stmt = conn
        .prepare(
            "SELECT step, intermediate_value, intermediate_value_type \
             FROM trial_intermediate_values WHERE trial_id = ? ORDER BY step",
        )
        .map_err(|e| {
            Error::with_reason(
                ErrorKind::StorageError,
                format!("Database query failed: {e}"),
            )
        })?;
    let rows = stmt
        .query_map(params![trial_id], |row| {
            Ok((
                row.get::<_, u32>(0)?,
                row.get::<_, Option<f64>>(1)?,
                row.get::<_, String>(2)?,
            ))
        })
        .map_err(|e| {
            Error::with_reason(
                ErrorKind::StorageError,
                format!("Database query failed: {e}"),
            )
        })?;

    let mut values = HashMap::new();
    for row in rows {
        let (step, stored_value, value_type) = row.map_err(|e| {
            Error::with_reason(
                ErrorKind::StorageError,
                format!("Database query failed: {e}"),
            )
        })?;
        values.insert(step, decode_intermediate_value(stored_value, &value_type)?);
    }
    Ok(values)
}

fn decode_intermediate_value(stored_value: Option<f64>, value_type: &str) -> Result<f64> {
    match value_type {
        "FINITE" => stored_value.ok_or_else(|| {
            Error::with_reason(ErrorKind::StorageError, "Finite intermediate value is NULL")
        }),
        "NAN" => Ok(f64::NAN),
        "INF_POS" => Ok(f64::INFINITY),
        "INF_NEG" => Ok(f64::NEG_INFINITY),
        _ => Err(Error::with_reason(
            ErrorKind::StorageError,
            format!("Invalid intermediate value type: {value_type}"),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cache::CachedStorage;
    use crate::test_utils::TempDir;
    use rustuna_core::sampler::RandomSampler;
    use rustuna_core::storage::Storage;
    use rustuna_core::study::{create_study, Direction};

    fn init_storage() -> Result<SQLite3Storage> {
        init_storage_with_option(SQLite3StorageOptions::default())
    }

    /// Reads SQLite's own idea of the current UTC time, truncated to whole seconds.
    ///
    /// It is an independent reference for the timestamps Rustuna binds from Rust: a column holding
    /// local time would sit an offset away from it on any machine that is not on UTC. Seconds are
    /// the finest common precision, since `strftime` stops at milliseconds.
    fn database_utc_second(storage: &SQLite3Storage) -> Result<String> {
        let guard = storage
            .conn
            .lock()
            .map_err(|_| Error::new(ErrorKind::StorageError))?;
        let now: String = guard
            .query_row("SELECT strftime('%Y-%m-%d %H:%M:%f', 'now')", [], |row| {
                row.get(0)
            })
            .map_err(|e| Error::with_reason(ErrorKind::StorageError, e.to_string()))?;
        Ok(now[..19].to_string())
    }

    #[test]
    fn trial_datetimes_are_stored_as_naive_utc() -> Result<()> {
        let mut storage = init_storage()?;
        let study_id = storage
            .create_new_study("example", vec![Direction::Minimize])?
            .id;
        let before = database_utc_second(&storage)?;
        let trial = storage.create_new_trial(study_id)?;
        let trial_id = trial.id;
        let reported_start = trial.datetime_start.clone().expect("datetime_start is set");
        storage.set_trial_state_values(trial_id, TrialStateValues::Complete(vec![1.0]))?;
        let trial = storage.get_trial(trial_id)?;
        let reported_complete = trial
            .datetime_complete
            .clone()
            .expect("datetime_complete is set");
        assert_eq!(
            trial.datetime_start.as_deref(),
            Some(reported_start.as_str())
        );

        let (stored_start, stored_complete): (String, String) = {
            let guard = storage
                .conn
                .lock()
                .map_err(|_| Error::new(ErrorKind::StorageError))?;
            guard
                .query_row(
                    "SELECT datetime_start, datetime_complete FROM trials WHERE trial_id = ?",
                    params![trial_id],
                    |row| Ok((row.get(0)?, row.get(1)?)),
                )
                .map_err(|e| Error::with_reason(ErrorKind::StorageError, e.to_string()))?
        };

        // PersistedTrial and the columns hold the same naive UTC value, with no conversion.
        assert_eq!(stored_start, reported_start);
        assert_eq!(stored_complete, reported_complete);

        // Naive UTC of a fixed width sorts chronologically, so bracketing the stored values
        // between two readings of the database clock needs no date arithmetic.
        let after = database_utc_second(&storage)?;
        for value in [&stored_start, &stored_complete] {
            let second = &value[..19];
            assert!(
                before.as_str() <= second && second <= after.as_str(),
                "{value} is not UTC (database clock went {before} -> {after})"
            );
        }
        Ok(())
    }

    fn init_storage_with_option(options: SQLite3StorageOptions) -> Result<SQLite3Storage> {
        let storage = SQLite3Storage::new_with_option(":memory:", options)?;
        storage.create_database()?;
        storage.validate_discard_support()?;
        Ok(storage)
    }

    #[test]
    fn create_new_study_inserts_rows() -> Result<()> {
        let mut storage = init_storage()?;
        assert_eq!(storage.get_studies()?.len(), 0);

        let study =
            storage.create_new_study("example", vec![Direction::Minimize, Direction::Maximize])?;
        assert_eq!(study.name, "example");
        assert_eq!(
            study.directions,
            vec![Direction::Minimize, Direction::Maximize]
        );
        assert_eq!(storage.get_studies()?.len(), 1);
        Ok(())
    }

    #[test]
    fn create_database_is_idempotent() -> Result<()> {
        let dir = TempDir::new().map_err(|_| Error::new(ErrorKind::Unexpected))?;
        let path = dir.path().join("storage.sqlite3");
        let storage = SQLite3Storage::new(path.to_string_lossy().as_ref())?;

        storage.create_database()?;
        storage.create_database()?;

        let mut storage = storage;
        let study = storage.create_new_study("example", vec![Direction::Minimize])?;
        assert_eq!(study.name, "example");
        assert_eq!(storage.get_studies()?.len(), 1);
        Ok(())
    }

    #[test]
    fn create_database_restores_trial_lookup_index() -> Result<()> {
        let storage = SQLite3Storage::new(":memory:")?;
        storage.create_database()?;

        {
            let conn = storage
                .conn
                .lock()
                .map_err(|_| Error::new(ErrorKind::StorageError))?;
            conn.execute_batch("DROP INDEX trials_study_id_number_key")
                .map_err(|e| Error::with_reason(ErrorKind::StorageError, e.to_string()))?;
        }

        // Existing databases skip SCHEMA_SQL, but the performance-critical index must still be
        // created when the storage is opened again.
        storage.create_database()?;

        let conn = storage
            .conn
            .lock()
            .map_err(|_| Error::new(ErrorKind::StorageError))?;
        let index_name: Option<String> = conn
            .query_row(
                "SELECT name FROM sqlite_master WHERE type = 'index' AND name = ?",
                params!["trials_study_id_number_key"],
                |row| row.get(0),
            )
            .optional()
            .map_err(|e| Error::with_reason(ErrorKind::StorageError, e.to_string()))?;
        assert_eq!(index_name.as_deref(), Some("trials_study_id_number_key"));
        Ok(())
    }

    #[test]
    fn discard_trials_are_omitted_by_cached_storage() -> Result<()> {
        let backend = init_storage_with_option(SQLite3StorageOptions {
            apply_discard: true,
        })?;
        let mut storage = CachedStorage::new(Box::new(backend));
        assert!(storage.may_omit_trials());

        let study_id = storage
            .create_new_study("example", vec![Direction::Minimize])?
            .id;
        let trial0_id = storage.create_new_trial(study_id)?.id;
        let trial1_id = storage.create_new_trial(study_id)?.id;
        storage.discard_trials(&[trial0_id])?;

        let trials = storage.get_trials(study_id)?;
        assert!(trials[0].is_none());
        assert_eq!(trials[1].as_ref().unwrap().id, trial1_id);
        assert!(matches!(
            storage.get_trial(trial0_id).unwrap_err().kind,
            ErrorKind::TrialDiscarded
        ));
        Ok(())
    }

    fn open_file_storage(path: &str, apply_discard: bool) -> Result<CachedStorage> {
        let backend =
            SQLite3Storage::new_with_option(path, SQLite3StorageOptions { apply_discard })?;
        backend.create_database()?;
        backend.validate_discard_support()?;
        Ok(CachedStorage::new(Box::new(backend)))
    }

    #[test]
    fn get_n_trials_counts_states_including_discarded_trials() -> Result<()> {
        let mut storage = init_storage()?;
        let study_id = storage
            .create_new_study("example", vec![Direction::Minimize])?
            .id;
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
    fn discarded_trials_stay_discarded_after_reopening() -> Result<()> {
        let dir = TempDir::new().map_err(|_| Error::new(ErrorKind::Unexpected))?;
        let path = dir.path().join("storage.sqlite3");
        let path = path.to_string_lossy().to_string();

        let (study_id, trial0_id, trial1_id) = {
            let mut storage = open_file_storage(&path, true)?;
            let study_id = storage
                .create_new_study("example", vec![Direction::Minimize])?
                .id;
            let trial0_id = storage.create_new_trial(study_id)?.id;
            storage.set_trial_state_values(trial0_id, TrialStateValues::Complete(vec![1.0]))?;
            let trial1_id = storage.create_new_trial(study_id)?.id;
            storage.set_trial_state_values(trial1_id, TrialStateValues::Complete(vec![2.0]))?;
            storage.discard_trials(&[trial0_id])?;
            (study_id, trial0_id, trial1_id)
        };

        // A fresh cache has no trial_id -> location mapping, so resolving the discarded trial
        // goes through the backend. It must not resurrect it into the cache.
        let mut storage = open_file_storage(&path, true)?;
        assert!(matches!(
            storage.get_trial(trial0_id).unwrap_err().kind,
            ErrorKind::TrialDiscarded
        ));
        assert_eq!(storage.get_trial(trial1_id)?.id, trial1_id);
        let trials = storage.get_trials(study_id)?;
        assert!(trials[0].is_none());
        assert_eq!(trials[1].as_ref().unwrap().id, trial1_id);

        // The same, but asking for the discarded trial before anything else is cached.
        let mut storage = open_file_storage(&path, true)?;
        assert!(matches!(
            storage.get_trial(trial0_id).unwrap_err().kind,
            ErrorKind::TrialDiscarded
        ));
        Ok(())
    }

    #[test]
    fn discards_are_persisted_when_apply_discard_is_disabled() -> Result<()> {
        let dir = TempDir::new().map_err(|_| Error::new(ErrorKind::Unexpected))?;
        let path = dir.path().join("storage.sqlite3");
        let path = path.to_string_lossy().to_string();

        let (study_id, trial_id) = {
            // As in JournalStorage, the discard is written even though this storage does not
            // apply it when reading.
            let mut storage = open_file_storage(&path, false)?;
            let study_id = storage
                .create_new_study("example", vec![Direction::Minimize])?
                .id;
            let trial_id = storage.create_new_trial(study_id)?.id;
            storage.create_new_trial(study_id)?;
            storage.discard_trials(&[trial_id])?;
            assert_eq!(storage.get_trial(trial_id)?.id, trial_id);
            assert!(!storage.may_omit_trials());
            (study_id, trial_id)
        };

        let mut storage = open_file_storage(&path, true)?;
        let trials = storage.get_trials(study_id)?;
        assert!(trials[0].is_none());
        assert!(matches!(
            storage.get_trial(trial_id).unwrap_err().kind,
            ErrorKind::TrialDiscarded
        ));
        Ok(())
    }

    #[test]
    fn discards_by_another_process_are_synchronized() -> Result<()> {
        let dir = TempDir::new().map_err(|_| Error::new(ErrorKind::Unexpected))?;
        let path = dir.path().join("storage.sqlite3");
        let path = path.to_string_lossy().to_string();

        let mut reader = open_file_storage(&path, true)?;
        let study_id = reader
            .create_new_study("example", vec![Direction::Minimize])?
            .id;
        let trial0_id = reader.create_new_trial(study_id)?.id;
        reader.set_trial_state_values(trial0_id, TrialStateValues::Complete(vec![1.0]))?;
        let trial1_id = reader.create_new_trial(study_id)?.id;
        reader.set_trial_state_values(trial1_id, TrialStateValues::Complete(vec![2.0]))?;
        assert_eq!(reader.get_trials(study_id)?.iter().flatten().count(), 2);

        // trial0 is already finished, so get_trials_diff will never revisit it. Only the
        // dedicated discard synchronization can notice this.
        let mut writer = open_file_storage(&path, true)?;
        writer.discard_trials(&[trial0_id])?;

        let trials = reader.get_trials(study_id)?;
        assert!(trials[0].is_none());
        assert_eq!(trials[1].as_ref().unwrap().id, trial1_id);
        Ok(())
    }

    #[test]
    fn rejected_writes_on_discarded_trials_do_not_reach_the_database() -> Result<()> {
        let dir = TempDir::new().map_err(|_| Error::new(ErrorKind::Unexpected))?;
        let path = dir.path().join("storage.sqlite3");
        let path = path.to_string_lossy().to_string();

        let (study_id, trial_id) = {
            let mut storage = open_file_storage(&path, true)?;
            let study_id = storage
                .create_new_study("example", vec![Direction::Minimize])?
                .id;
            let trial_id = storage.create_new_trial(study_id)?.id;
            storage.discard_trials(&[trial_id])?;

            assert!(matches!(
                storage
                    .set_trial_state_values(trial_id, TrialStateValues::Complete(vec![42.0]))
                    .unwrap_err()
                    .kind,
                ErrorKind::TrialDiscarded
            ));
            assert!(matches!(
                storage
                    .set_trial_intermediate_values(trial_id, HashMap::from([(0, 42.0)]))
                    .unwrap_err()
                    .kind,
                ErrorKind::TrialDiscarded
            ));
            (study_id, trial_id)
        };

        // Reopen without applying discards to see what actually landed in the database.
        let mut storage = open_file_storage(&path, false)?;
        let trials = storage.get_trials(study_id)?;
        let trial = trials[0].as_ref().expect("trial should still be readable");
        assert_eq!(trial.id, trial_id);
        assert!(matches!(trial.state_values, TrialStateValues::Running));
        assert!(trial.intermediate_values.is_empty());
        Ok(())
    }

    #[test]
    fn create_new_study_rejects_duplicate_name() -> Result<()> {
        let mut storage = init_storage()?;
        storage.create_new_study("dup", vec![Direction::Minimize])?;
        let err = storage
            .create_new_study("dup", vec![Direction::Minimize])
            .err()
            .expect("Expected DuplicatedStudy error");
        assert!(matches!(err.kind, ErrorKind::DuplicatedStudy));
        Ok(())
    }

    #[test]
    fn waiting_trials_round_trip_as_waiting() -> Result<()> {
        let mut storage = init_storage()?;
        let study_id = storage
            .create_new_study("example", vec![Direction::Minimize])?
            .id;
        let trial_id = storage.create_new_trial(study_id)?.id;
        storage.set_trial_state_values(trial_id, TrialStateValues::Waiting)?;

        let trial = storage.get_trial(trial_id)?;
        assert!(matches!(trial.state_values, TrialStateValues::Waiting));

        let trials = storage.get_trials_diff(study_id, &[trial.number], -1)?;
        assert_eq!(trials.len(), 1);
        assert!(matches!(trials[0].state_values, TrialStateValues::Waiting));

        Ok(())
    }

    // TODO(c-bata): Pass following test case by adding `AUTOINCREMENT` attribute to study_id field.
    // See the following comment in Optuna
    // https://github.com/optuna/optuna/blob/af238ea2/tests/storages_tests/test_storages.py#L95-L98
    // #[test]
    // fn create_new_study_unique_id() -> Result<()> {
    //     let mut storage = init_storage()?;
    //     assert_eq!(storage.get_studies()?.len(), 0);
    //     let study1 = storage.create_new_study("study-1", vec![Direction::Minimize])?;
    //     let study2 =
    //         storage.create_new_study("study-2", vec![Direction::Minimize])?;
    //     storage.delete_study(study2.id)?;
    //     let study3 = storage.create_new_study("study-3", vec![Direction::Minimize])?;
    //     assert_ne!(study1.id, study2.id);
    //     assert_ne!(study1.id, study3.id);
    //     assert_ne!(study2.id, study3.id);
    //     assert_eq!(storage.get_studies()?.len(), 2);
    //     Ok(())
    // }

    #[test]
    fn create_new_trial_inserts_row() -> Result<()> {
        let mut storage = init_storage()?;
        let study_id = storage
            .create_new_study("example", vec![Direction::Minimize])?
            .id;

        let trial = storage.create_new_trial(study_id)?;
        assert_eq!(trial.number, 0);
        assert_eq!(trial.state_values, TrialStateValues::Running);

        let trial = storage.create_new_trial(study_id)?;
        assert_eq!(trial.number, 1);
        Ok(())
    }

    #[test]
    fn create_new_trial_rolls_back_insert_when_number_assignment_fails() -> Result<()> {
        let mut storage = init_storage()?;
        let study_id = storage
            .create_new_study("example", vec![Direction::Minimize])?
            .id;
        {
            let guard = storage
                .conn
                .lock()
                .map_err(|_| Error::new(ErrorKind::StorageError))?;
            guard
                .execute_batch(
                    "CREATE TRIGGER fail_trial_number_update \
                     BEFORE UPDATE OF number ON trials \
                     WHEN NEW.number IS NOT NULL \
                     BEGIN \
                         SELECT RAISE(ABORT, 'forced number update failure'); \
                     END;",
                )
                .map_err(|e| Error::with_reason(ErrorKind::StorageError, e.to_string()))?;
        }

        let err = storage
            .create_new_trial(study_id)
            .expect_err("number assignment should fail");
        assert!(matches!(err.kind, ErrorKind::StorageError));

        let row_count: u32 = {
            let guard = storage
                .conn
                .lock()
                .map_err(|_| Error::new(ErrorKind::StorageError))?;
            guard
                .query_row("SELECT COUNT(*) FROM trials", [], |row| row.get(0))
                .map_err(|e| Error::with_reason(ErrorKind::StorageError, e.to_string()))?
        };
        assert_eq!(row_count, 0, "the INSERT must roll back with the UPDATE");
        Ok(())
    }

    #[test]
    fn create_new_trial_from_template_preserves_datetime() -> Result<()> {
        let mut storage = init_storage()?;
        let study_id = storage
            .create_new_study("example", vec![Direction::Minimize])?
            .id;

        let mut template = PersistedTrial::new(999, 998, 997);
        template.state_values = TrialStateValues::Complete(vec![0.5]);
        template.datetime_start = Some("2024-01-02 03:04:05.678".to_string());
        template.datetime_complete = Some("2024-01-02 03:14:15.678".to_string());
        template.internal_params.insert("x".to_string(), 0.4);
        template.distributions.insert(
            "x".to_string(),
            Distribution::new_float(0.0, 1.0, None, false),
        );

        let trial = storage.create_new_trial_from_template(study_id, &template)?;
        assert_eq!(trial.number, 0);
        assert_eq!(trial.datetime_start, template.datetime_start);
        assert_eq!(trial.datetime_complete, template.datetime_complete);
        assert_eq!(trial.state_values, template.state_values);
        Ok(())
    }

    #[test]
    fn set_trial_param() -> Result<()> {
        let mut storage = init_storage()?;
        let study_id = storage
            .create_new_study("example", vec![Direction::Minimize])?
            .id;
        let trial = storage.create_new_trial(study_id)?;

        // FloatDistribution
        let float_dist = Distribution::new_float(0.0, 1.0, None, false);
        storage.set_trial_param(trial.id, "float", &float_dist, 0.5)?;

        // IntDistribution
        let int_dist = Distribution::new_int(0, 10, 1, false);
        storage.set_trial_param(trial.id, "int", &int_dist, 5.0)?;

        // CategoricalDistribution
        let categorical_dist = Distribution::new_categorical(3);
        storage.set_trial_param(trial.id, "cat", &categorical_dist, 1.0)?;

        // Check distributions
        let trial = storage.get_trial(trial.id)?;
        assert_eq!(trial.distributions.len(), 3);
        assert_eq!(trial.distributions["float"], float_dist);
        assert_eq!(trial.distributions["int"], int_dist);
        assert_eq!(trial.distributions["cat"], categorical_dist);

        // Check params
        assert_eq!(trial.internal_params.len(), 3);
        assert_eq!(trial.internal_params["float"], 0.5);
        assert_eq!(trial.internal_params["int"], 5.0);
        assert_eq!(trial.internal_params["cat"], 1.0);
        Ok(())
    }

    #[test]
    fn suggest_categorical_enum_persists_categorical_choices() -> Result<()> {
        let dir = TempDir::new().map_err(|_| Error::new(ErrorKind::Unexpected))?;
        let path = dir.path().join("storage.sqlite3");
        let path = path.to_string_lossy().to_string();
        let storage = open_file_storage(&path, false)?;
        let study = create_study(
            "example",
            storage,
            RandomSampler::new(),
            vec![Direction::Minimize],
        )?;
        let choices = vec![
            CategoryLabel::String("red".to_string()),
            CategoryLabel::Int(2),
            CategoryLabel::Bool(true),
            CategoryLabel::None,
            CategoryLabel::Float(1.5),
        ];
        let mut trial = study.ask()?;
        trial.suggest_categorical_enum("color", &choices)?;

        let storage = SQLite3Storage::new(&path)?;
        let distribution_json: String = storage
            .conn
            .lock()
            .map_err(|_| Error::new(ErrorKind::StorageError))?
            .query_row(
                "SELECT distribution_json FROM trial_params WHERE trial_id = ?",
                params![trial.id],
                |row| row.get(0),
            )
            .map_err(|e| Error::with_reason(ErrorKind::StorageError, e.to_string()))?;
        let distribution: Value = serde_json::from_str(&distribution_json)
            .map_err(|e| Error::with_reason(ErrorKind::StorageError, e.to_string()))?;
        assert_eq!(
            distribution["attributes"]["choices"],
            json!(["red", 2, true, Value::Null, 1.5])
        );
        Ok(())
    }

    #[test]
    fn optuna_categorical_labels_are_exposed_as_study_attrs() -> Result<()> {
        let mut storage = SQLite3Storage::new(":memory:")?;
        storage.create_database()?;
        let study_id = storage
            .create_new_study("example", vec![Direction::Minimize])?
            .id;
        let trial_id = storage.create_new_trial(study_id)?.id;
        let labels = vec![
            CategoryLabel::String("red".to_string()),
            CategoryLabel::Int(2),
        ];
        let distribution = Distribution::new_categorical(labels.len());
        let distribution_json = distribution_to_json(&distribution, Some(&labels));
        storage
            .conn
            .lock()
            .map_err(|_| Error::new(ErrorKind::StorageError))?
            .execute(
                "INSERT INTO trial_params (trial_id, param_name, param_value, distribution_json) \
                 VALUES (?, 'color', 0, ?)",
                params![trial_id, distribution_json],
            )
            .map_err(|e| Error::with_reason(ErrorKind::StorageError, e.to_string()))?;

        let study = storage.get_study(study_id)?;
        assert_eq!(
            rustuna_core::attr::get_category_labels(&study.attrs, "color", 2),
            Some(labels)
        );
        Ok(())
    }

    #[test]
    fn set_study_attrs() -> Result<()> {
        let mut storage = init_storage()?;
        let study_id = storage
            .create_new_study("example", vec![Direction::Minimize])?
            .id;

        let mut attrs = Attrs::new();
        attrs.insert(AttrKey::User("user_key".into()), "user_value".to_string());
        attrs.insert(
            AttrKey::System("system_key".into()),
            "system_value".to_string(),
        );

        storage.set_study_attrs(study_id, attrs, false)?;

        let study = storage.get_study(study_id)?;
        assert_eq!(study.attrs.len(), 2);
        assert_eq!(
            study.attrs.get(&AttrKey::User("user_key".into())),
            Some(&"user_value".to_string())
        );
        assert_eq!(
            study.attrs.get(&AttrKey::System("system_key".into())),
            Some(&"system_value".to_string())
        );
        Ok(())
    }

    #[test]
    fn set_study_attrs_error_on_overwrite_rollback() -> Result<()> {
        let mut storage = init_storage()?;
        let study_id = storage
            .create_new_study("example", vec![Direction::Minimize])?
            .id;

        let mut attrs = Attrs::new();
        attrs.insert(AttrKey::User("user_key".into()), "user_value".to_string());
        storage.set_study_attrs(study_id, attrs, false)?;

        let mut overwrite = Attrs::new();
        overwrite.insert(
            AttrKey::User("user_key".into()),
            "user_value_overwrite".to_string(),
        );
        overwrite.insert(
            AttrKey::User("another_key".into()),
            "another_value".to_string(),
        );
        let err = storage
            .set_study_attrs(study_id, overwrite, true)
            .expect_err("Expected AttrOverwriteNotAllowed error");
        assert!(matches!(err.kind, ErrorKind::AttrOverwriteNotAllowed));

        let study = storage.get_study(study_id)?;
        assert_eq!(
            study.attrs.get(&AttrKey::User("user_key".into())),
            Some(&"user_value".to_string())
        );
        assert!(!study
            .attrs
            .contains_key(&AttrKey::User("another_key".into())));

        Ok(())
    }

    #[test]
    fn set_trial_attrs() -> Result<()> {
        let mut storage = init_storage()?;
        let study_id = storage
            .create_new_study("example", vec![Direction::Minimize])?
            .id;
        let trial = storage.create_new_trial(study_id)?;

        let mut attrs = Attrs::new();
        attrs.insert(
            AttrKey::User("trial_user_key".into()),
            "trial_user_value".to_string(),
        );
        attrs.insert(
            AttrKey::System("trial_system_key".into()),
            "trial_system_value".to_string(),
        );

        storage.set_trial_attrs(trial.id, attrs, false)?;

        let trial = storage.get_trial(trial.id)?;
        assert_eq!(
            trial.attrs.get(&AttrKey::User("trial_user_key".into())),
            Some(&"trial_user_value".to_string())
        );
        assert_eq!(
            trial.attrs.get(&AttrKey::System("trial_system_key".into())),
            Some(&"trial_system_value".to_string())
        );
        Ok(())
    }

    #[test]
    fn set_trial_attrs_error_on_overwrite_rollback() -> Result<()> {
        let mut storage = init_storage()?;
        let study_id = storage
            .create_new_study("example", vec![Direction::Minimize])?
            .id;
        let trial = storage.create_new_trial(study_id)?;

        let mut attrs = Attrs::new();
        attrs.insert(
            AttrKey::User("trial_user_key".into()),
            "trial_user_value".to_string(),
        );
        storage.set_trial_attrs(trial.id, attrs, false)?;

        let mut overwrite = Attrs::new();
        overwrite.insert(
            AttrKey::User("trial_user_key".into()),
            "overwritten".to_string(),
        );
        overwrite.insert(
            AttrKey::User("new_user_key".into()),
            "new_value".to_string(),
        );
        let err = storage
            .set_trial_attrs(trial.id, overwrite, true)
            .expect_err("Expected AttrOverwriteNotAllowed error");
        assert!(matches!(err.kind, ErrorKind::AttrOverwriteNotAllowed));

        let trial = storage.get_trial(trial.id)?;
        assert_eq!(
            trial.attrs.get(&AttrKey::User("trial_user_key".into())),
            Some(&"trial_user_value".to_string())
        );
        assert!(!trial
            .attrs
            .contains_key(&AttrKey::User("new_user_key".into())));

        Ok(())
    }

    #[test]
    fn set_trial_state_values_complete() -> Result<()> {
        let mut storage = init_storage()?;
        let study_id = storage
            .create_new_study("example", vec![Direction::Minimize, Direction::Maximize])?
            .id;
        let trial = storage.create_new_trial(study_id)?;

        assert_eq!(trial.state_values, TrialStateValues::Running);

        storage.set_trial_state_values(trial.id, TrialStateValues::Complete(vec![1.5, 2.5]))?;

        let trial = storage.get_trial(trial.id)?;
        assert_eq!(
            trial.state_values,
            TrialStateValues::Complete(vec![1.5, 2.5])
        );
        Ok(())
    }

    #[test]
    fn set_trial_state_values_rejects_empty_complete_values() -> Result<()> {
        let mut storage = init_storage()?;
        let study_id = storage
            .create_new_study("example", vec![Direction::Minimize])?
            .id;
        let trial_id = storage.create_new_trial(study_id)?.id;

        let err = storage
            .set_trial_state_values(trial_id, TrialStateValues::Complete(vec![]))
            .expect_err("COMPLETE requires at least one objective value");
        assert!(matches!(err.kind, ErrorKind::InvalidObjectiveValues));
        assert_eq!(
            storage.get_trial(trial_id)?.state_values,
            TrialStateValues::Running
        );
        Ok(())
    }

    #[test]
    fn set_trial_state_values_rolls_back_state_when_values_write_fails() -> Result<()> {
        let mut storage = init_storage()?;
        let study_id = storage
            .create_new_study("example", vec![Direction::Minimize])?
            .id;
        let trial_id = storage.create_new_trial(study_id)?.id;
        {
            let guard = storage
                .conn
                .lock()
                .map_err(|_| Error::new(ErrorKind::StorageError))?;
            guard
                .execute_batch(
                    "CREATE TRIGGER fail_trial_values_insert \
                     BEFORE INSERT ON trial_values \
                     BEGIN \
                         SELECT RAISE(ABORT, 'forced trial values failure'); \
                     END;",
                )
                .map_err(|e| Error::with_reason(ErrorKind::StorageError, e.to_string()))?;
        }

        let err = storage
            .set_trial_state_values(trial_id, TrialStateValues::Complete(vec![1.0]))
            .expect_err("trial values INSERT should fail");
        assert!(matches!(err.kind, ErrorKind::StorageError));

        let trial = storage.get_trial(trial_id)?;
        assert_eq!(trial.state_values, TrialStateValues::Running);
        assert_eq!(trial.datetime_complete, None);
        Ok(())
    }

    #[test]
    fn reads_reject_complete_trial_without_objective_values() -> Result<()> {
        let mut storage = init_storage()?;
        let study_id = storage
            .create_new_study("example", vec![Direction::Minimize])?
            .id;
        let trial_id = storage.create_new_trial(study_id)?.id;
        {
            let guard = storage
                .conn
                .lock()
                .map_err(|_| Error::new(ErrorKind::StorageError))?;
            guard
                .execute(
                    "UPDATE trials SET state = 'COMPLETE', datetime_complete = ? \
                     WHERE trial_id = ?",
                    params![now_naive_utc(), trial_id],
                )
                .map_err(|e| Error::with_reason(ErrorKind::StorageError, e.to_string()))?;
        }

        let get_trial_err = storage
            .get_trial(trial_id)
            .expect_err("get_trial must reject an incomplete COMPLETE row");
        assert!(matches!(get_trial_err.kind, ErrorKind::StorageError));
        assert!(get_trial_err.reason.contains(&format!("Trial {trial_id}")));

        let get_trials_diff_err = storage
            .get_trials_diff(study_id, &[], -1)
            .expect_err("get_trials_diff must reject an incomplete COMPLETE row");
        assert!(matches!(get_trials_diff_err.kind, ErrorKind::StorageError));
        assert!(get_trials_diff_err
            .reason
            .contains(&format!("Trial {trial_id}")));
        Ok(())
    }

    #[test]
    fn set_trial_state_values_fail() -> Result<()> {
        let mut storage = init_storage()?;
        let study_id = storage
            .create_new_study("example", vec![Direction::Minimize])?
            .id;
        let trial = storage.create_new_trial(study_id)?;

        storage.set_trial_state_values(trial.id, TrialStateValues::Fail)?;

        let trial = storage.get_trial(trial.id)?;
        assert_eq!(trial.state_values, TrialStateValues::Fail);
        Ok(())
    }

    #[test]
    fn get_trials_diff() -> Result<()> {
        let mut storage = init_storage()?;
        let study_id = storage
            .create_new_study("example", vec![Direction::Minimize])?
            .id;

        // Create 5 trials
        for i in 0..5 {
            let trial = storage.create_new_trial(study_id)?;
            storage.set_trial_state_values(trial.id, TrialStateValues::Complete(vec![i as f64]))?;
        }

        // Get all trials with number > 2
        let trials = storage.get_trials_diff(study_id, &[], 2)?;
        assert_eq!(trials.len(), 2);
        assert_eq!(trials[0].number, 3);
        assert_eq!(trials[1].number, 4);

        // Get specific trials by number
        let trials = storage.get_trials_diff(study_id, &[0, 2], -1)?;
        assert_eq!(trials.len(), 5); // All trials + included ones

        // Get trials with number > 3 OR in [0, 1]
        let trials = storage.get_trials_diff(study_id, &[0, 1], 3)?;
        assert_eq!(trials.len(), 3); // trials 0, 1, 4
        Ok(())
    }

    #[test]
    fn get_trials_diff_with_large_trial_number_greater_than() -> Result<()> {
        let mut storage = init_storage()?;
        let study_id = storage
            .create_new_study("example", vec![Direction::Minimize])?
            .id;

        storage.create_new_trial(study_id)?;

        // trial_number_greater_than is much larger than existing trial numbers
        let trials = storage.get_trials_diff(study_id, &[], 500000)?;
        assert_eq!(trials.len(), 0);

        Ok(())
    }

    // TODO(c-bata): Fix this test case
    // #[test]
    // fn get_trials_diff_with_large_included_numbers() -> Result<()> {
    //     let mut storage = init_storage()?;
    //     let study_id = storage
    //         .create_new_study("example", vec![Direction::Minimize])?
    //         .id;

    //     storage.create_new_trial(study_id)?;

    //     // A large inclusion list used to raise errors in some implementations.
    //     // Check that it is not an issue. See https://github.com/optuna/optuna/issues/1457.
    //     let large_numbers: Vec<u32> = (0..500000).collect();
    //     let trials = storage.get_trials_diff(study_id, &large_numbers, 500000)?;
    //     assert_eq!(trials.len(), 1);

    //     Ok(())
    // }

    #[test]
    fn get_trials_diff_with_negative_trial_number_greater_than() -> Result<()> {
        let mut storage = init_storage()?;
        let study_id = storage
            .create_new_study("example", vec![Direction::Minimize])?
            .id;

        storage.create_new_trial(study_id)?;

        // trial_number_greater_than = -1 should return all trials
        let trials = storage.get_trials_diff(study_id, &[], -1)?;
        assert_eq!(trials.len(), 1);

        // trial_number_greater_than much larger than existing trials
        let trials = storage.get_trials_diff(study_id, &[], 500001)?;
        assert_eq!(trials.len(), 0);

        Ok(())
    }

    #[test]
    fn run_optimization() -> Result<()> {
        let storage = SQLite3Storage::new(":memory:")?;
        storage.create_database()?;
        let storage = CachedStorage::new(Box::new(storage));

        let study = create_study(
            "simple-quadratic",
            storage,
            RandomSampler::new(),
            vec![Direction::Minimize],
        )?;
        study.optimize(
            |mut t| {
                let x = t.suggest_float("x", 0.0, 10.0)?;
                let y = t.suggest_float("y", 0.0, 10.0)?;
                let value = (x - 3.0).powi(2) + (y - 5.0).powi(2);
                println!("{:2} x: {}, y: {}, value: {}", t.number, x, y, value);
                Ok(vec![value])
            },
            100,
        )?;
        assert_eq!(study.get_trials()?.len(), 100);
        Ok(())
    }

    #[test]
    fn set_trial_intermediate_values() -> Result<()> {
        let mut storage = init_storage()?;
        let study_id = storage
            .create_new_study("example", vec![Direction::Minimize])?
            .id;
        let trial1 = storage.create_new_trial(study_id)?;
        let trial2 = storage.create_new_trial(study_id)?;
        let study_id2 = storage
            .create_new_study("example2", vec![Direction::Minimize])?
            .id;
        let trial3 = storage.create_new_trial(study_id2)?;
        let trial4 = storage.create_new_trial(study_id)?;

        // Test setting new values
        let mut values1 = HashMap::new();
        values1.insert(0, 0.3);
        values1.insert(2, 0.4);
        storage.set_trial_intermediate_values(trial1.id, values1)?;

        let mut values3 = HashMap::new();
        values3.insert(0, 0.1);
        values3.insert(1, 0.4);
        values3.insert(2, 0.5);
        values3.insert(3, f64::INFINITY);
        storage.set_trial_intermediate_values(trial3.id, values3)?;

        let mut values4 = HashMap::new();
        values4.insert(0, f64::NAN);
        storage.set_trial_intermediate_values(trial4.id, values4)?;

        // Verify trial 1
        let trial1_result = storage.get_trial(trial1.id)?;
        assert_eq!(trial1_result.intermediate_values.len(), 2);
        assert_eq!(trial1_result.intermediate_values[&0], 0.3);
        assert_eq!(trial1_result.intermediate_values[&2], 0.4);
        assert!(!trial1_result
            .attrs
            .contains_key(&AttrKey::System("intermediate_values".into())));

        // Verify trial 2 (no intermediate values)
        let trial2_result = storage.get_trial(trial2.id)?;
        assert!(trial2_result.intermediate_values.is_empty());
        assert!(!trial2_result
            .attrs
            .contains_key(&AttrKey::System("intermediate_values".into())));

        // Verify trial 3
        let trial3_result = storage.get_trial(trial3.id)?;
        assert_eq!(trial3_result.intermediate_values.len(), 4);
        assert_eq!(trial3_result.intermediate_values[&0], 0.1);
        assert_eq!(trial3_result.intermediate_values[&1], 0.4);
        assert_eq!(trial3_result.intermediate_values[&2], 0.5);
        assert_eq!(trial3_result.intermediate_values[&3], f64::INFINITY);

        // Verify trial 4 (NaN value)
        let trial4_result = storage.get_trial(trial4.id)?;
        assert_eq!(trial4_result.intermediate_values.len(), 1);
        assert!(trial4_result.intermediate_values[&0].is_nan());

        // Test overwriting existing step
        let mut values1_update = HashMap::new();
        values1_update.insert(0, 0.2);
        storage.set_trial_intermediate_values(trial1.id, values1_update)?;

        let trial1_updated = storage.get_trial(trial1.id)?;
        assert_eq!(trial1_updated.intermediate_values.len(), 2);
        assert_eq!(trial1_updated.intermediate_values[&0], 0.2);
        assert_eq!(trial1_updated.intermediate_values[&2], 0.4);

        // Test non-existent trial
        let non_existent_trial_id = trial4.id + 1000;
        let mut invalid_values = HashMap::new();
        invalid_values.insert(0, 0.5);
        let err = storage
            .set_trial_intermediate_values(non_existent_trial_id, invalid_values)
            .expect_err("Expected TrialNotFound error");
        assert!(matches!(err.kind, ErrorKind::TrialNotFound));

        Ok(())
    }
}
