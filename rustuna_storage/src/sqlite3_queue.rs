use rusqlite::{params, Connection};
use rustuna_core::trial_queue::TrialQueue;
use rustuna_core::{Error, ErrorKind, Result};
use std::path::Path;
use std::sync::Mutex;

/// SQLite-backed trial queue.
///
/// Multiple logical queues can share the same database file by using different namespaces.
pub struct SQLite3TrialQueue {
    conn: Mutex<Connection>,
    namespace: String,
}

impl SQLite3TrialQueue {
    /// Creates a new SQLite3TrialQueue with the specified database path and namespace.
    ///
    /// Multiple queues can share the same database file, with namespace used for isolation.
    /// The `trial_queue` table will be created if it doesn't exist.
    ///
    /// # Arguments
    ///
    /// * `db_path` - Path to the SQLite database file
    /// * `namespace` - Namespace to isolate trials for this queue
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// use rustuna_storage::sqlite3_queue::SQLite3TrialQueue;
    /// let queue = SQLite3TrialQueue::new("/path/to/queue.db", "study-1").unwrap();
    /// ```
    pub fn new(db_path: impl AsRef<Path>, namespace: impl Into<String>) -> Result<Self> {
        let conn = Connection::open(db_path).map_err(|e| {
            Error::with_reason(
                ErrorKind::StorageError,
                format!("Failed to open SQLite database: {e}"),
            )
        })?;

        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS trial_queue (
                rowid INTEGER PRIMARY KEY AUTOINCREMENT,
                namespace TEXT NOT NULL,
                trial_id INTEGER NOT NULL,
                enqueued_at INTEGER NOT NULL,
                sequence INTEGER NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_trial_queue_namespace_order 
                ON trial_queue(namespace, enqueued_at, sequence);",
        )
        .map_err(|e| {
            Error::with_reason(
                ErrorKind::StorageError,
                format!("Failed to create trial_queue table: {e}"),
            )
        })?;

        Ok(Self {
            conn: Mutex::new(conn),
            namespace: namespace.into(),
        })
    }
}

impl TrialQueue for SQLite3TrialQueue {
    fn enqueue(&self, trial_id: u32) -> Result<()> {
        let conn = self.conn.lock().expect("SQLite trial queue mutex poisoned");

        // Use current timestamp in microseconds for ordering
        let enqueued_at = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_micros() as i64;

        // Get next sequence number for this timestamp to handle concurrent inserts
        let sequence: i64 = conn
            .query_row(
                "SELECT COALESCE(MAX(sequence), -1) + 1 
                 FROM trial_queue 
                 WHERE namespace = ? AND enqueued_at = ?",
                params![self.namespace, enqueued_at],
                |row| row.get(0),
            )
            .map_err(|e| {
                Error::with_reason(
                    ErrorKind::StorageError,
                    format!("Failed to determine trial queue sequence: {e}"),
                )
            })?;

        conn.execute(
            "INSERT INTO trial_queue (namespace, trial_id, enqueued_at, sequence) 
             VALUES (?, ?, ?, ?)",
            params![self.namespace, trial_id, enqueued_at, sequence],
        )
        .map_err(|e| {
            Error::with_reason(
                ErrorKind::StorageError,
                format!("Failed to enqueue trial: {e}"),
            )
        })?;

        Ok(())
    }

    fn dequeue(&self) -> Result<u32> {
        let conn = self.conn.lock().expect("SQLite trial queue mutex poisoned");

        // Use a transaction with IMMEDIATE to ensure exclusive access
        conn.execute("BEGIN IMMEDIATE", []).map_err(|e| {
            Error::with_reason(
                ErrorKind::StorageError,
                format!("Failed to begin transaction: {e}"),
            )
        })?;

        let result: std::result::Result<u32, rusqlite::Error> = (|| {
            // Find and delete the oldest enqueued entry.
            let trial_id: u32 = conn.query_row(
                "DELETE FROM trial_queue
                 WHERE rowid = (
                     SELECT rowid FROM trial_queue 
                     WHERE namespace = ? 
                     ORDER BY enqueued_at ASC, sequence ASC
                     LIMIT 1
                 )
                 RETURNING trial_id",
                params![self.namespace],
                |row| row.get(0),
            )?;
            Ok(trial_id)
        })();

        match result {
            Ok(trial_id) => {
                conn.execute("COMMIT", []).map_err(|e| {
                    Error::with_reason(
                        ErrorKind::StorageError,
                        format!("Failed to commit transaction: {e}"),
                    )
                })?;
                Ok(trial_id)
            }
            Err(rusqlite::Error::QueryReturnedNoRows) => {
                conn.execute("ROLLBACK", []).map_err(|e| {
                    Error::with_reason(
                        ErrorKind::StorageError,
                        format!("Failed to roll back empty dequeue transaction: {e}"),
                    )
                })?;
                Err(Error::new(ErrorKind::TrialQueueEmpty))
            }
            Err(e) => {
                let rollback_result = conn.execute("ROLLBACK", []);
                let reason = match rollback_result {
                    Ok(_) => format!("Failed to dequeue trial: {e}"),
                    Err(rollback_error) => format!(
                        "Failed to dequeue trial: {e}; failed to roll back transaction: {rollback_error}"
                    ),
                };
                Err(Error::with_reason(ErrorKind::StorageError, reason))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::TempDir;

    #[test]
    fn test_enqueue_and_dequeue() {
        let temp_dir = TempDir::new().unwrap();
        let queue =
            SQLite3TrialQueue::new(temp_dir.path().join("queue.sqlite3"), "study-1").unwrap();

        queue.enqueue(10).unwrap();
        queue.enqueue(20).unwrap();
        queue.enqueue(30).unwrap();

        assert_eq!(queue.dequeue().unwrap(), 10);
        assert_eq!(queue.dequeue().unwrap(), 20);
        assert_eq!(queue.dequeue().unwrap(), 30);
    }

    #[test]
    fn test_dequeue_empty_queue() {
        let temp_dir = TempDir::new().unwrap();
        let queue =
            SQLite3TrialQueue::new(temp_dir.path().join("queue.sqlite3"), "study-1").unwrap();

        assert!(queue.dequeue().is_err());
    }

    #[test]
    fn test_fifo_ordering() {
        let temp_dir = TempDir::new().unwrap();
        let queue =
            SQLite3TrialQueue::new(temp_dir.path().join("queue.sqlite3"), "study-1").unwrap();

        for i in 1..=100 {
            queue.enqueue(i).unwrap();
        }

        for i in 1..=100 {
            assert_eq!(queue.dequeue().unwrap(), i);
        }
    }

    #[test]
    fn test_namespace_isolation() {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("queue.sqlite3");
        let queue1 = SQLite3TrialQueue::new(&db_path, "study-1").unwrap();
        let queue2 = SQLite3TrialQueue::new(&db_path, "study-2").unwrap();

        queue1.enqueue(10).unwrap();
        queue1.enqueue(20).unwrap();
        queue2.enqueue(30).unwrap();
        queue2.enqueue(40).unwrap();

        assert_eq!(queue1.dequeue().unwrap(), 10);
        assert_eq!(queue2.dequeue().unwrap(), 30);
        assert_eq!(queue1.dequeue().unwrap(), 20);
        assert_eq!(queue2.dequeue().unwrap(), 40);

        assert!(queue1.dequeue().is_err());
        assert!(queue2.dequeue().is_err());
    }

    #[test]
    fn test_push_pop_interleaved() {
        let temp_dir = TempDir::new().unwrap();
        let queue =
            SQLite3TrialQueue::new(temp_dir.path().join("queue.sqlite3"), "study-1").unwrap();

        queue.enqueue(1).unwrap();
        queue.enqueue(2).unwrap();
        assert_eq!(queue.dequeue().unwrap(), 1);

        queue.enqueue(3).unwrap();
        assert_eq!(queue.dequeue().unwrap(), 2);
        assert_eq!(queue.dequeue().unwrap(), 3);

        assert!(queue.dequeue().is_err());
    }

    #[test]
    fn test_reopen_queue() {
        let temp_dir = TempDir::new().unwrap();
        let db_path = temp_dir.path().join("queue.sqlite3");

        {
            let queue = SQLite3TrialQueue::new(&db_path, "study-1").unwrap();
            queue.enqueue(10).unwrap();
            queue.enqueue(20).unwrap();
        }

        {
            let queue = SQLite3TrialQueue::new(&db_path, "study-1").unwrap();
            assert_eq!(queue.dequeue().unwrap(), 10);
            assert_eq!(queue.dequeue().unwrap(), 20);
            assert!(queue.dequeue().is_err());
        }
    }
}
