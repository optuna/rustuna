use std::collections::VecDeque;
use std::sync::Mutex;

use crate::{Error, ErrorKind, Result};

/// Queue abstraction used by [`crate::study::Study::enqueue_trial`].
///
/// Rustuna keeps queuing outside the storage layer so that users who do not use queued trials do
/// not pay coordination overhead on every trial creation.
pub trait TrialQueue: Send + Sync {
    /// Pushes a trial ID into the queue.
    fn enqueue(&self, trial_id: u32) -> Result<()>;
    /// Pops the next queued trial ID.
    fn dequeue(&self) -> Result<u32>;
}

/// In-memory queue implementation used by default.
#[derive(Default)]
pub struct InMemoryTrialQueue {
    queue: Mutex<VecDeque<u32>>,
}

impl InMemoryTrialQueue {
    /// Creates an empty queue.
    pub fn new() -> Self {
        Self {
            queue: Mutex::new(VecDeque::new()),
        }
    }
}

impl TrialQueue for InMemoryTrialQueue {
    fn enqueue(&self, trial_id: u32) -> Result<()> {
        let mut guard = self.queue.lock().expect("queue guard poisoned");
        guard.push_back(trial_id);
        Ok(())
    }

    fn dequeue(&self) -> Result<u32> {
        let mut guard = self.queue.lock().expect("queue guard poisoned");
        guard
            .pop_front()
            .ok_or_else(|| Error::new(ErrorKind::TrialQueueEmpty))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_enqueue_and_dequeue() {
        let queue = InMemoryTrialQueue::new();

        assert!(queue.enqueue(1).is_ok());
        assert!(queue.enqueue(2).is_ok());
        assert!(queue.enqueue(3).is_ok());

        assert_eq!(queue.dequeue().unwrap(), 1);
        assert_eq!(queue.dequeue().unwrap(), 2);
        assert_eq!(queue.dequeue().unwrap(), 3);
    }

    #[test]
    fn test_dequeue_empty_queue() {
        let queue = InMemoryTrialQueue::new();
        assert!(queue.dequeue().is_err());
    }
}
