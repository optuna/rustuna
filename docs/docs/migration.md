# Migrating from Optuna

## When to Use Rustuna

We recommend using Rustuna if you fall into the following categories:

* **Low-cost Objective Functions**: You need to run many trials, and each objective evaluation is relatively cheap. In such cases, the overhead of parameter suggestion, storage access, and Python object handling can become a bottleneck, and Rustuna can reduce that overhead.
* **Minimal Dependencies**: You prefer a lightweight software stack without Python dependencies (e.g., to mitigate version conflicts or supply chain risks).
* **Multi-Language Support**: You want to leverage optimization in languages other than Python (currently supporting Rust and JavaScript/TypeScript).

Conversely, Rustuna might not be the best fit (at least for now) if:

* **Expensive Objective Functions**: If model training dominates the runtime of each trial, the performance difference between Rustuna and Optuna often becomes less important in practice.
* **Proven Maturity**: You require a stable library with a long history of production use and fewer bugs.
* **Cutting-Edge Optimization Algorithms**: Rustuna is not the best choice if you want to use advanced samplers such as Optuna's `GPSampler` or OptunaHub's `AutoSampler`. While it is technically possible to use them together with Rustuna, the cost of parameter suggestion in these samplers tends to dominate execution time, leaving little practical benefit in choosing Rustuna.

## Differences in concepts between Rustuna and Optuna

### FrozenTrial vs. PersistedTrial

`FrozenTrial` is a container class that holds the values of a trial stored in Optuna's storage.
My understanding is that the name “Frozen” was originally intended to indicate that this container object was immutable. However, as Optuna evolved and adapted to a wider range of use cases, the implementation changed, and `FrozenTrial` is no longer immutable in practice. Because of this historical background, the name `FrozenTrial` has remained even though renaming it has been discussed multiple times among Optuna developers. Since `FrozenTrial` is one of Optuna's core components and is used throughout the codebase, changing its name would have a very broad impact.

In Rustuna, the corresponding storage-side trial type is called `PersistedTrial`. The name is different, but its role is essentially the same. The same naming idea also applies at the study level: Optuna has `FrozenStudy`, while Rustuna uses `PersistedStudy` for the corresponding storage-side type.

### `study.enqueue_trial()` and the TrialQueue

In Optuna, [`study.enqueue_trial()`](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.study.Study.html#optuna.study.Study.enqueue_trial) allows users to manually specify parameter sets to be evaluated next. This is useful when you have prior knowledge of promising parameters. However, Optuna's implementation faces design challenges:

* **Performance Overhead**: Optuna checks the queue every time an objective function is evaluated. This process incurs a performance degradation even for users who do not use the queuing feature.
* **Storage Limitations (SQLite3)**: In distributed environments, queuing requires atomic "pop" operations. While `RDBStorage` handles this via `SELECT ... FOR UPDATE`, SQLite3 lacks this support, leading to a long-standing issue where duplicate trials are popped by different workers.

Since Rustuna is built for high performance, we decoupled the queue from the storage layer. Instead, we introduced the TrialQueue component.

```python
import rustuna
from rustuna.trial_queue import SQLite3TrialQueue

def objective(trial: rustuna.Trial) -> float:
    return trial.suggest_float("x", -10, 10)

study_name = "example"
trial_queue = SQLite3TrialQueue("optuna_queue.sqlite3", namespace=study_name)
study = rustuna.create_study(trial_queue=trial_queue, ...)

study.enqueue_trial({"x": 5})
study.optimize(objective, n_trials=10)
```

By moving the queue logic out of the storage layer, Rustuna avoids unnecessary overhead.
This also allows for flexible backends; for example, a Redis-based queue (planned) could handle distributed optimization more efficiently.
However, there are two important behavioral differences:

1. **Default In-Memory Queue**: By default, Rustuna uses an in-memory queue. In multi-process optimization, the queue state is not shared. A trial enqueued in Worker A will only be popped by Worker A. To share state across processes, use `TrialQueue.sqlite3` or a custom backend.
2. **add_trial() Behavior**: In Optuna, you can technically inject a trial into the queue by calling `study.add_trial()` with a WAITING state. In Rustuna, trials added via `add_trial` are not automatically queued. You must use `enqueue_trial()` or explicitly push the ID via `study.trial_queue.push(trial_id)`.


### User and Study attributes

Both Optuna and Rustuna allow users to store `user_attrs` and `system_attrs` on `Trial` and `Study` objects, making them usable as a simple key-value store for trial- or study-specific metadata.

Like Optuna, Rustuna's Python API stores any JSON-serializable object as an attribute value. Both user and system attributes are stored in Optuna's schema, so Optuna and Rustuna can read each other's attributes in journal files and SQLite3 databases. To keep `get_trials()` fast, decoded values are cached, so mutable values such as lists and dicts returned by `user_attrs` must not be modified in place.

Limitations: journal logs must be strict JSON, so `JournalFileStorage` cannot store `NaN` or `Infinity` in attributes.

Separately, Rustuna also provides bulk insert APIs such as `trial.set_user_attrs({"key1": "...", "key2": "..."})`.

### No Pruner Support (Currently)

Rustuna does not currently support pruners.

Pruning has long been one of Optuna’s flagship features and remains valuable in many use cases. At the same time, it has not become as widely adopted as we initially expected. Meanwhile, multi-objective optimization, which was introduced later, has become widely used in many scenarios, yet it still cannot be combined satisfactorily with pruning.

For this reason, we chose to prioritize multi-objective optimization first in Rustuna. We still consider pruning a powerful feature and believe it may well be introduced in the future, but if we add it, it should be designed from the outset with multi-objective optimization in mind, both at the algorithmic level and in the API design.
