use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyString};
use rustuna_core::attr::{
    category_labels_to_attrs, get_category_labels, AttrKey, Attrs, CategoryLabel,
};
use rustuna_core::distribution::Distribution;
use rustuna_core::storage::Storage;
use rustuna_core::trial::{PersistedTrial, Trial, TrialState, TrialStateValues};
use rustuna_core::ErrorKind;

use crate::attrs::{pyobj_to_attrs, AttrKind, AttrsDictView};
use crate::distribution::{
    category_label_to_pyobject, py_to_external_repr, pyobject_to_category_label, PyDistribution,
};
use crate::exception::err_to_exceptions;

fn state_values_from_py(
    state: &PyTrialState,
    value: Option<f64>,
    values: Option<Vec<f64>>,
) -> PyResult<TrialStateValues> {
    if value.is_some() && values.is_some() {
        return Err(PyValueError::new_err(
            "value and values must not be specified at the same time",
        ));
    }

    let values = match (value, values) {
        (Some(value), None) => Some(vec![value]),
        (None, values) => values,
        (Some(_), Some(_)) => unreachable!(),
    };

    match state {
        PyTrialState::RUNNING => Ok(TrialStateValues::Running),
        PyTrialState::COMPLETE => Ok(TrialStateValues::Complete(values.ok_or(
            PyValueError::new_err("value or values must be specified when state is COMPLETE"),
        )?)),
        PyTrialState::PRUNED => Ok(TrialStateValues::Pruned),
        PyTrialState::FAIL => Ok(TrialStateValues::Fail),
        PyTrialState::WAITING => Ok(TrialStateValues::Waiting),
    }
}

fn internal_param_from_py(obj: &Bound<'_, PyAny>, distribution: &PyDistribution) -> PyResult<f64> {
    match &distribution.distribution {
        Distribution::Float { .. } => obj.extract::<f64>(),
        Distribution::Int { .. } => obj.extract::<i64>().map(|v| v as f64),
        Distribution::Categorical { .. } => {
            let label = pyobject_to_category_label(obj)?;
            let labels = distribution
                .category_labels
                .as_ref()
                .ok_or(PyValueError::new_err(
                    "categorical distributions must include choices",
                ))?;
            labels
                .iter()
                .position(|candidate| candidate == &label)
                .map(|index| index as f64)
                .ok_or(PyValueError::new_err(
                    "parameter value is not contained in the categorical distribution",
                ))
        }
    }
    .map_err(|e| PyValueError::new_err(format!("Failed to convert parameter value: {e}")))
}

fn build_trial_attrs(
    user_attrs: HashMap<String, String>,
    system_attrs: HashMap<String, String>,
) -> Attrs {
    let mut trial_attrs = Attrs::with_capacity(user_attrs.len() + system_attrs.len());
    for (key, value) in user_attrs {
        trial_attrs.insert(AttrKey::User(key.into()), value);
    }
    for (key, value) in system_attrs {
        trial_attrs.insert(AttrKey::System(key.into()), value);
    }
    trial_attrs
}

fn build_params(
    params: Option<&Bound<'_, PyDict>>,
    distributions: &HashMap<String, PyDistribution>,
) -> PyResult<HashMap<String, f64>> {
    params
        .map(|params| -> PyResult<HashMap<String, f64>> {
            let mut internal_params = HashMap::with_capacity(params.len());
            for (key, value) in params.iter() {
                let name = key.extract::<String>()?;
                let distribution =
                    distributions
                        .get(&name)
                        .ok_or(PyValueError::new_err(format!(
                            "Parameter '{name}' is not found in distributions."
                        )))?;
                let internal_value = internal_param_from_py(&value, distribution)?;
                internal_params.insert(name, internal_value);
            }
            Ok(internal_params)
        })
        .transpose()
        .map(|opt| opt.unwrap_or_default())
}

fn study_attrs_from_distributions(distributions: &HashMap<String, PyDistribution>) -> Attrs {
    let mut attrs = Attrs::new();
    for (name, distribution) in distributions {
        if let Some(labels) = &distribution.category_labels {
            attrs.extend(category_labels_to_attrs(name, labels));
        }
    }
    attrs
}

#[pyfunction]
#[pyo3(
    name = "create_trial",
    signature = (
        *,
        state = PyTrialState::COMPLETE,
        value = None,
        values = None,
        params = None,
        distributions = None,
        user_attrs = None,
        system_attrs = None,
        constraints = None,
        intermediate_values = None,
    )
)]
#[allow(clippy::too_many_arguments)]
pub fn py_create_trial(
    state: PyTrialState,
    value: Option<f64>,
    values: Option<Vec<f64>>,
    params: Option<&Bound<'_, PyDict>>,
    distributions: Option<HashMap<String, PyDistribution>>,
    user_attrs: Option<HashMap<String, String>>,
    system_attrs: Option<HashMap<String, String>>,
    constraints: Option<HashMap<String, f64>>,
    intermediate_values: Option<HashMap<u32, f64>>,
) -> PyResult<PyPersistedTrial> {
    let mut trial = PersistedTrial::new(0, 0, 0);
    trial.state_values = state_values_from_py(&state, value, values)?;

    let distributions = distributions.unwrap_or_default();
    let study_attrs = study_attrs_from_distributions(&distributions);
    trial.internal_params = build_params(params, &distributions)?;
    trial.distributions = distributions
        .into_iter()
        .map(|(name, distribution)| (name, distribution.distribution))
        .collect();
    trial.intermediate_values = intermediate_values.unwrap_or_default();
    trial.attrs = build_trial_attrs(
        user_attrs.unwrap_or_default(),
        system_attrs.unwrap_or_default(),
    );
    trial.constraints = constraints.unwrap_or_default();

    let now = rustuna_core::datetime::now_naive_utc();
    if matches!(state, PyTrialState::WAITING) {
        trial.datetime_start = None;
        trial.datetime_complete = None;
    } else {
        trial.datetime_start = Some(now.clone());
        trial.datetime_complete = if state.is_finished() { Some(now) } else { None };
    }

    trial.validate().map_err(err_to_exceptions)?;
    Ok(PyPersistedTrial::new(trial, study_attrs))
}

#[derive(Clone, Debug, PartialEq)]
#[pyclass(name = "TrialState", eq, eq_int)]
#[pyo3(module = "rustuna")]
#[allow(clippy::upper_case_acronyms)]
pub enum PyTrialState {
    RUNNING = 0,
    COMPLETE = 1,
    PRUNED = 2,
    WAITING = 3,
    FAIL = 4,
}
impl From<TrialStateValues> for PyTrialState {
    fn from(item: TrialStateValues) -> Self {
        match item {
            TrialStateValues::Running => PyTrialState::RUNNING,
            TrialStateValues::Complete(_) => PyTrialState::COMPLETE,
            TrialStateValues::Pruned => PyTrialState::PRUNED,
            TrialStateValues::Fail => PyTrialState::FAIL,
            TrialStateValues::Waiting => PyTrialState::WAITING,
        }
    }
}

impl From<rustuna_core::trial::TrialState> for PyTrialState {
    fn from(item: rustuna_core::trial::TrialState) -> Self {
        match item {
            rustuna_core::trial::TrialState::Running => PyTrialState::RUNNING,
            rustuna_core::trial::TrialState::Complete => PyTrialState::COMPLETE,
            rustuna_core::trial::TrialState::Pruned => PyTrialState::PRUNED,
            rustuna_core::trial::TrialState::Fail => PyTrialState::FAIL,
            rustuna_core::trial::TrialState::Waiting => PyTrialState::WAITING,
        }
    }
}

impl From<PyTrialState> for TrialState {
    fn from(item: PyTrialState) -> Self {
        match item {
            PyTrialState::RUNNING => TrialState::Running,
            PyTrialState::COMPLETE => TrialState::Complete,
            PyTrialState::PRUNED => TrialState::Pruned,
            PyTrialState::FAIL => TrialState::Fail,
            PyTrialState::WAITING => TrialState::Waiting,
        }
    }
}
#[pymethods]
impl PyTrialState {
    fn __hash__(&self) -> isize {
        match self {
            PyTrialState::RUNNING => 0,
            PyTrialState::COMPLETE => 1,
            PyTrialState::PRUNED => 2,
            PyTrialState::WAITING => 3,
            PyTrialState::FAIL => 4,
        }
    }

    #[getter]
    pub fn name(&self) -> &'static str {
        match self {
            PyTrialState::RUNNING => "RUNNING",
            PyTrialState::COMPLETE => "COMPLETE",
            PyTrialState::PRUNED => "PRUNED",
            PyTrialState::WAITING => "WAITING",
            PyTrialState::FAIL => "FAIL",
        }
    }

    pub fn is_finished(&self) -> bool {
        matches!(
            self,
            PyTrialState::COMPLETE | PyTrialState::PRUNED | PyTrialState::FAIL
        )
    }
}

#[pyclass(name = "Trial")]
#[pyo3(module = "rustuna")]
pub struct PyTrial {
    trial: Trial,
    storage_pyobj: Py<PyAny>,
}
impl PyTrial {
    pub fn new(trial: Trial, storage_pyobj: Py<PyAny>) -> Self {
        PyTrial {
            trial,
            storage_pyobj,
        }
    }
}
#[pymethods]
impl PyTrial {
    #[getter(_trial_id)]
    pub fn id(&self) -> PyResult<u32> {
        Ok(self.trial.id)
    }
    #[getter]
    pub fn study_id(&self) -> PyResult<u32> {
        Ok(self.trial.study_id)
    }
    #[getter]
    pub fn number(&self) -> PyResult<u32> {
        Ok(self.trial.number)
    }
    #[getter]
    pub fn storage<'py>(&self, py: Python<'py>) -> Py<PyAny> {
        self.storage_pyobj.clone_ref(py)
    }
    #[pyo3(signature = (name, low, high, step=None, log=false))]
    pub fn suggest_float(
        &mut self,
        name: &str,
        low: f64,
        high: f64,
        step: Option<f64>,
        log: bool,
    ) -> PyResult<f64> {
        let dist = Distribution::new_float(low, high, step, log);
        let value = self.trial.suggest(name, &dist).map_err(|e| match e.kind {
            rustuna_core::ErrorKind::UnsupportedMultiObjective => PyRuntimeError::new_err(
                "The TPE sampler of rustuna currently only supports single objective study.",
            ),
            _ => PyRuntimeError::new_err(format!("Failed to suggest float: {e:?}")),
        })?;
        Ok(value)
    }
    #[pyo3(signature = (name, low, high, step=1, log=false))]
    pub fn suggest_int(
        &mut self,
        name: &str,
        low: i64,
        high: i64,
        step: i64,
        log: bool,
    ) -> PyResult<i64> {
        let dist = Distribution::new_int(low, high, step, log);
        let value = self.trial.suggest(name, &dist).map_err(|e| match e.kind {
            rustuna_core::ErrorKind::UnsupportedMultiObjective => PyRuntimeError::new_err(
                "The TPE sampler of rustuna currently only supports single objective study.",
            ),
            _ => PyRuntimeError::new_err(format!("Failed to suggest int: {e:?}")),
        })?;
        Ok(value as i64)
    }
    #[pyo3(signature = (name, choices))]
    pub fn suggest_categorical(
        &mut self,
        name: &str,
        choices: Vec<Py<PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        let mut category_labels: Vec<CategoryLabel> = Vec::with_capacity(choices.len());
        let category_labels = Python::attach(|py| {
            for choice in choices {
                match pyobject_to_category_label(choice.bind(py)) {
                    Ok(label) => category_labels.push(label),
                    Err(e) => return Err(e),
                }
            }
            Ok(category_labels)
        })?;
        let label = self
            .trial
            .suggest_categorical_enum(name, &category_labels)
            .map_err(|e| match e.kind {
                rustuna_core::ErrorKind::UnsupportedMultiObjective => PyRuntimeError::new_err(
                    "The TPE sampler of rustuna currently only supports single objective study.",
                ),
                _ => PyRuntimeError::new_err(format!("Failed to suggest categorical: {e:?}")),
            })?;

        Python::attach(|py| category_label_to_pyobject(py, label).map(|b| b.unbind()))
    }
    #[pyo3(signature = (key, value))]
    pub fn set_user_attr(&mut self, key: &str, value: String) -> PyResult<()> {
        self.trial.set_user_attr(key, value).map_err(|e| {
            PyRuntimeError::new_err(format!("Failed to set user attr: {:?}", e.kind))
        })?;
        Ok(())
    }

    fn set_user_attrs(&mut self, attrs: Py<PyAny>) -> PyResult<()> {
        let user_attrs: HashMap<String, String> = Python::attach(|py| attrs.bind(py).extract())?;

        self.trial.set_user_attrs(user_attrs).map_err(|e| {
            PyRuntimeError::new_err(format!("Failed to set user attrs: {:?}", e.kind))
        })?;
        Ok(())
    }
    #[pyo3(signature = (constraints))]
    pub fn set_constraints(&mut self, constraints: Py<PyAny>) -> PyResult<()> {
        let constraints: HashMap<String, f64> =
            Python::attach(|py| constraints.bind(py).extract())?;

        self.trial.set_constraints(constraints).map_err(|e| {
            PyRuntimeError::new_err(format!("Fialed to set constraints: {:?}", e.kind))
        })?;

        Ok(())
    }

    #[getter]
    pub fn user_attrs(&self) -> PyResult<HashMap<String, String>> {
        Ok(self.trial.get_user_attrs())
    }
}

enum PyPersistedTrialSource {
    Owned(Box<PersistedTrial>),
    StorageBacked {
        storage: Arc<RwLock<dyn Storage>>,
        trial_id: u32,
        // Eagerly cached lightweight fields
        study_id: u32,
        number: u32,
        state: PyTrialState,
    },
}

type TrialParams = Vec<(String, f64, Distribution)>;

#[pyclass(name = "PersistedTrial")]
#[pyo3(module = "rustuna")]
pub struct PyPersistedTrial {
    source: PyPersistedTrialSource,
    study_attrs: Option<Arc<Attrs>>,
}
impl PyPersistedTrial {
    /// Create a PyPersistedTrial that owns all trial data including study_attrs.
    ///
    /// Use this constructor when:
    /// - Constructing from Python via `__new__`
    /// - Retrieving a single trial where study_attrs are already available
    pub fn new(trial: PersistedTrial, study_attrs: Attrs) -> Self {
        PyPersistedTrial {
            source: PyPersistedTrialSource::Owned(Box::new(trial)),
            study_attrs: Some(Arc::new(study_attrs)),
        }
    }
    /// Create a lightweight PyPersistedTrial that caches frequently accessed fields
    /// but defers heavy fields (user_attrs, system_attrs) to storage lookup.
    ///
    /// Use this constructor when:
    /// - Retrieving multiple trials (e.g., `get_trials`, `best_trials`)
    /// - The caller has already fetched study_attrs and can share them via Arc
    ///
    /// This avoids cloning study_attrs for each trial and defers expensive
    /// attribute lookups until actually needed.
    pub fn from_storage(storage: Arc<RwLock<dyn Storage>>, trial: &PersistedTrial) -> Self {
        let state = trial_state_from_ref(&trial.state_values);
        PyPersistedTrial {
            source: PyPersistedTrialSource::StorageBacked {
                storage,
                trial_id: trial.id,
                study_id: trial.study_id,
                number: trial.number,
                state,
            },
            study_attrs: None,
        }
    }

    pub fn with_trial<R>(&self, f: impl FnOnce(&PersistedTrial) -> PyResult<R>) -> PyResult<R> {
        match &self.source {
            PyPersistedTrialSource::Owned(trial) => f(trial.as_ref()),
            PyPersistedTrialSource::StorageBacked {
                storage, trial_id, ..
            } => {
                let guard = storage
                    .read()
                    .map_err(|_| PyRuntimeError::new_err("Failed to acquire the storage guard"))?;
                let trial = guard
                    .get_cached_trial(*trial_id)
                    .map_err(err_to_exceptions)?;
                f(trial)
            }
        }
    }

    fn collect_params(&self) -> PyResult<(u32, TrialParams)> {
        self.with_trial(|trial| {
            let mut params: Vec<(String, f64, Distribution)> =
                Vec::with_capacity(trial.internal_params.len());
            for (name, internal_repr) in &trial.internal_params {
                let dist = trial
                    .distributions
                    .get(name)
                    .ok_or(PyValueError::new_err(format!("No distribution for {name}")))?;
                params.push((name.clone(), *internal_repr, dist.clone()));
            }
            Ok((trial.study_id, params))
        })
    }

    fn collect_distributions(&self) -> PyResult<(u32, Vec<(String, Distribution)>)> {
        self.with_trial(|trial| {
            let mut distributions: Vec<(String, Distribution)> =
                Vec::with_capacity(trial.distributions.len());
            for (name, dist) in &trial.distributions {
                distributions.push((name.clone(), dist.clone()));
            }
            Ok((trial.study_id, distributions))
        })
    }

    fn category_labels_for_param(
        &self,
        study_id: u32,
        param_name: &str,
        cardinality: usize,
    ) -> PyResult<Option<Vec<CategoryLabel>>> {
        if let Some(attrs) = &self.study_attrs {
            return Ok(get_category_labels(attrs, param_name, cardinality));
        }
        match &self.source {
            PyPersistedTrialSource::StorageBacked { storage, .. } => {
                let mut guard = storage
                    .write()
                    .map_err(|_| PyRuntimeError::new_err("Failed to acquire the storage guard"))?;
                guard
                    .get_category_labels(study_id, param_name, cardinality)
                    .map_err(err_to_exceptions)
            }
            PyPersistedTrialSource::Owned(_) => Ok(None),
        }
    }
}
#[allow(non_local_definitions)]
#[pymethods]
impl PyPersistedTrial {
    #[new]
    #[pyo3(signature = (*, trial_id, study_id, number, state, value=None, values=None, params=None, distributions=None, user_attrs=None, system_attrs=None, constraints=None, intermediate_values=None, datetime_start=None, datetime_complete=None))]
    #[allow(clippy::too_many_arguments)]
    pub fn py_new(
        trial_id: u32,
        study_id: u32,
        number: u32,
        state: PyTrialState,
        value: Option<f64>,
        values: Option<Vec<f64>>,
        params: Option<&Bound<'_, PyDict>>,
        distributions: Option<HashMap<String, PyDistribution>>,
        user_attrs: Option<HashMap<String, String>>,
        system_attrs: Option<HashMap<String, String>>,
        constraints: Option<HashMap<String, f64>>,
        intermediate_values: Option<HashMap<u32, f64>>,
        datetime_start: Option<Bound<'_, PyAny>>,
        datetime_complete: Option<Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let mut trial = PersistedTrial::new(trial_id, study_id, number);
        trial.state_values = state_values_from_py(&state, value, values)?;

        let distributions = distributions.unwrap_or_default();
        let study_attrs = study_attrs_from_distributions(&distributions);
        trial.internal_params = build_params(params, &distributions)?;
        trial.distributions = distributions
            .into_iter()
            .map(|(name, dist)| (name, dist.distribution))
            .collect();
        trial.intermediate_values = intermediate_values.unwrap_or_default();
        trial.attrs = build_trial_attrs(
            user_attrs.unwrap_or_default(),
            system_attrs.unwrap_or_default(),
        );
        trial.constraints = constraints.unwrap_or_default();
        trial.datetime_start = datetime_start
            .as_ref()
            .map(py_datetime_to_naive_utc)
            .transpose()?;
        trial.datetime_complete = datetime_complete
            .as_ref()
            .map(py_datetime_to_naive_utc)
            .transpose()?;

        trial.validate().map_err(err_to_exceptions)?;
        Ok(PyPersistedTrial::new(trial, study_attrs))
    }

    #[getter(_trial_id)]
    fn id(&self) -> PyResult<u32> {
        match &self.source {
            PyPersistedTrialSource::Owned(trial) => Ok(trial.id),
            PyPersistedTrialSource::StorageBacked { trial_id, .. } => Ok(*trial_id),
        }
    }

    #[getter]
    fn study_id(&self) -> PyResult<u32> {
        match &self.source {
            PyPersistedTrialSource::Owned(trial) => Ok(trial.study_id),
            PyPersistedTrialSource::StorageBacked { study_id, .. } => Ok(*study_id),
        }
    }

    #[getter]
    fn number(&self) -> PyResult<u32> {
        match &self.source {
            PyPersistedTrialSource::Owned(trial) => Ok(trial.number),
            PyPersistedTrialSource::StorageBacked { number, .. } => Ok(*number),
        }
    }

    #[getter]
    fn state(&self) -> PyResult<PyTrialState> {
        match &self.source {
            PyPersistedTrialSource::Owned(trial) => Ok(trial_state_from_ref(&trial.state_values)),
            PyPersistedTrialSource::StorageBacked { state, .. } => Ok(state.clone()),
        }
    }

    #[getter]
    fn values(&self) -> PyResult<Option<Vec<f64>>> {
        self.with_trial(|trial| match &trial.state_values {
            TrialStateValues::Complete(values) => Ok(Some(values.clone())),
            _ => Ok(None),
        })
    }

    #[getter]
    fn value(&self) -> PyResult<Option<f64>> {
        self.with_trial(|trial| match &trial.state_values {
            TrialStateValues::Complete(values) => Ok(values.first().copied()),
            _ => Ok(None),
        })
    }

    #[getter]
    fn distributions(&self) -> PyResult<HashMap<String, PyDistribution>> {
        let (study_id, distributions) = self.collect_distributions()?;
        let mut result = HashMap::with_capacity(distributions.len());
        for (name, dist) in distributions {
            let labels = match dist {
                Distribution::Categorical { cardinality } => {
                    self.category_labels_for_param(study_id, &name, cardinality)?
                }
                _ => None,
            };
            let distribution = PyDistribution::new_with_category_labels(dist, labels);
            result.insert(name, distribution);
        }
        Ok(result)
    }

    #[getter]
    fn internal_params(&self) -> PyResult<HashMap<String, f64>> {
        self.with_trial(|trial| Ok(trial.internal_params.clone()))
    }

    #[getter]
    fn datetime_start<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyAny>>> {
        // Read the stored value out first: the conversion below calls back into Python and must
        // not run while the storage guard is held.
        let stored = self.with_trial(|trial| Ok(trial.datetime_start.clone()))?;
        stored
            .as_deref()
            .map(|raw| naive_utc_to_py_local(py, raw))
            .transpose()
    }

    #[getter]
    fn datetime_complete<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyAny>>> {
        let stored = self.with_trial(|trial| Ok(trial.datetime_complete.clone()))?;
        stored
            .as_deref()
            .map(|raw| naive_utc_to_py_local(py, raw))
            .transpose()
    }

    #[getter]
    fn intermediate_values(&self) -> PyResult<HashMap<u32, f64>> {
        self.with_trial(|trial| Ok(trial.intermediate_values.clone()))
    }

    #[getter]
    fn params(&self) -> PyResult<HashMap<String, Py<PyAny>>> {
        let (study_id, params) = self.collect_params()?;
        let mut labels_map: HashMap<String, Option<Vec<CategoryLabel>>> = HashMap::new();
        for (name, _, dist) in &params {
            if let Distribution::Categorical { cardinality } = dist {
                let labels = self.category_labels_for_param(study_id, name, *cardinality)?;
                labels_map.insert(name.clone(), labels);
            }
        }
        Python::attach(|py| {
            params
                .into_iter()
                .map(|(name, internal_repr, dist)| {
                    let labels = labels_map.get(&name).and_then(|labels| labels.as_deref());
                    let maybe_pyobj: PyResult<Py<PyAny>> =
                        py_to_external_repr(py, &dist, internal_repr, labels).map(|b| b.unbind());
                    maybe_pyobj.map(|v| (name, v))
                })
                .collect()
        })
    }

    #[getter]
    fn user_attrs(&self) -> PyResult<AttrsDictView> {
        match &self.source {
            PyPersistedTrialSource::Owned(trial) => {
                Ok(AttrsDictView::from_trial(trial.as_ref(), AttrKind::User))
            }
            PyPersistedTrialSource::StorageBacked {
                storage, trial_id, ..
            } => Ok(AttrsDictView::from_storage(
                storage.clone(),
                *trial_id,
                AttrKind::User,
            )),
        }
    }

    #[getter]
    fn system_attrs(&self) -> PyResult<AttrsDictView> {
        match &self.source {
            PyPersistedTrialSource::Owned(trial) => {
                Ok(AttrsDictView::from_trial(trial.as_ref(), AttrKind::System))
            }
            PyPersistedTrialSource::StorageBacked {
                storage, trial_id, ..
            } => Ok(AttrsDictView::from_storage(
                storage.clone(),
                *trial_id,
                AttrKind::System,
            )),
        }
    }

    #[getter]
    fn constraints(&self) -> PyResult<HashMap<String, f64>> {
        self.with_trial(|trial| Ok(trial.constraints().clone()))
    }

    #[pyo3(name = "get_user_attr", signature = (key, *, decoder = None, default = None))]
    pub fn get_user_attr<'py>(
        &self,
        py: Python<'py>,
        key: String,
        decoder: Option<Py<PyAny>>,
        default: Option<Py<PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        let result = match &self.source {
            PyPersistedTrialSource::Owned(trial) => trial
                .attrs
                .get(&AttrKey::User(key.into()))
                .cloned()
                .ok_or(rustuna_core::Error::new(ErrorKind::AttrNotFound)),
            PyPersistedTrialSource::StorageBacked {
                storage, trial_id, ..
            } => {
                let guard = storage.read().map_err(|e| {
                    PyRuntimeError::new_err(format!("Failed to acquire the storage guard: {e:?}"))
                })?;
                guard.get_cached_trial(*trial_id).and_then(|trial| {
                    trial
                        .attrs
                        .get(&AttrKey::User(key.into()))
                        .cloned()
                        .ok_or(rustuna_core::Error::new(ErrorKind::AttrNotFound))
                })
            }
        };
        match result {
            Ok(value) => match decoder {
                Some(decoder) => {
                    let decoded = decoder.call1(py, (&value,))?;
                    Ok(decoded)
                }
                None => Ok(PyString::new(py, &value).into_any().unbind()),
            },
            Err(e) if matches!(e.kind, ErrorKind::AttrNotFound) => {
                Ok(default.unwrap_or_else(|| py.None()))
            }
            Err(e) => Err(err_to_exceptions(e)),
        }
    }

    fn __repr__(slf: &Bound<'_, Self>) -> PyResult<String> {
        let type_obj = slf.get_type();
        let class_name = type_obj.name()?.to_string_lossy().into_owned();
        Ok(format!("{}({})", class_name, slf.borrow().__str__()?))
    }

    fn __str__(&self) -> PyResult<String> {
        let params = Python::attach(|py| -> PyResult<String> {
            let py_params = self.params()?.into_pyobject(py)?;
            Ok(py_params.str()?.to_str()?.to_owned())
        })?;
        let user_attrs = self.user_attrs()?.format_as_dict()?;
        let system_attrs = self.system_attrs()?.format_as_dict()?;
        Ok(format!(
            "number={} state={:?} values={:?} params={} distributions={:?} user_attrs={} system_attrs={}",
            self.number()?,
            self.state()?,
            self.values()?,
            params,
            self.distributions()?,
            user_attrs,
            system_attrs,
        ))
    }
}

fn trial_state_from_ref(state_values: &TrialStateValues) -> PyTrialState {
    match state_values {
        TrialStateValues::Running => PyTrialState::RUNNING,
        TrialStateValues::Complete(_) => PyTrialState::COMPLETE,
        TrialStateValues::Pruned => PyTrialState::PRUNED,
        TrialStateValues::Fail => PyTrialState::FAIL,
        TrialStateValues::Waiting => PyTrialState::WAITING,
    }
}

// Design Note:
// PersistedTrial carries timezone-naive UTC, while FrozenTrial exposes timezone-naive local time.
// The two conversions below are done through Python's `datetime` module rather than a Rust
// datetime crate, for two reasons: the timezone database is then guaranteed to be the same one
// Optuna uses, and the storage layer stays free of timezone handling. Note that the storage calls
// in `storage::binding` run under `Python::detach`, so this conversion has to live here at the
// boundary, where the GIL is held, rather than inside the storage layer.
//
// These mirror `optuna.storages._rdb.models.TrialModel.datetime_start`, which is
//     stored.replace(tzinfo=timezone.utc).astimezone().replace(tzinfo=None)
// on the way out and
//     value.astimezone(timezone.utc).replace(tzinfo=None)
// on the way in.

fn naive_replace_tzinfo<'py>(
    value: &Bound<'py, PyAny>,
    tzinfo: Bound<'py, PyAny>,
) -> PyResult<Bound<'py, PyAny>> {
    let kwargs = PyDict::new(value.py());
    kwargs.set_item("tzinfo", tzinfo)?;
    value.call_method("replace", (), Some(&kwargs))
}

fn utc_timezone(py: Python<'_>) -> PyResult<Bound<'_, PyAny>> {
    py.import("datetime")?.getattr("timezone")?.getattr("utc")
}

/// Converts a timezone-naive UTC timestamp into the timezone-naive local datetime users see.
fn naive_utc_to_py_local<'py>(py: Python<'py>, value: &str) -> PyResult<Bound<'py, PyAny>> {
    let parsed = py
        .import("datetime")?
        .getattr("datetime")?
        .call_method1("fromisoformat", (value,))
        .map_err(|e| PyValueError::new_err(format!("Failed to parse datetime {value:?}: {e}")))?;
    let aware = naive_replace_tzinfo(&parsed, utc_timezone(py)?)?;
    let local = aware.call_method0("astimezone")?;
    naive_replace_tzinfo(&local, py.None().into_bound(py))
}

/// Converts a datetime coming from Python into the timezone-naive UTC timestamp storages hold.
///
/// A naive input is read as local time, which is what `datetime.astimezone` does and therefore
/// what Optuna does.
fn py_datetime_to_naive_utc(value: &Bound<'_, PyAny>) -> PyResult<String> {
    let py = value.py();
    let aware = value.call_method1("astimezone", (utc_timezone(py)?,))?;
    let naive = naive_replace_tzinfo(&aware, py.None().into_bound(py))?;
    naive
        .call_method1("isoformat", (" ", "microseconds"))?
        .extract()
}

pub fn pyobject_to_persisted_trial_with_category_labels(
    trial: &Bound<'_, PyAny>,
    study_id: u32,
) -> PyResult<(PersistedTrial, Attrs)> {
    let trial_id = match trial.getattr("id") {
        Ok(attr) => attr.extract::<u32>()?,
        Err(_) => trial.getattr("_trial_id")?.extract::<u32>()?,
    };
    let number = trial.getattr("number")?.extract::<u32>()?;
    let mut persisted_trial = PersistedTrial::new(trial_id, study_id, number);

    let state = trial.getattr("state")?.extract::<PyTrialState>()?;
    let values = trial.getattr("values")?.extract::<Option<Vec<f64>>>()?;
    persisted_trial.state_values = match state {
        PyTrialState::COMPLETE => {
            let values = values.ok_or(PyValueError::new_err(
                "values must be specified when state is COMPLETE",
            ))?;
            TrialStateValues::Complete(values)
        }
        PyTrialState::RUNNING => TrialStateValues::Running,
        PyTrialState::PRUNED => TrialStateValues::Pruned,
        PyTrialState::WAITING => TrialStateValues::Waiting,
        PyTrialState::FAIL => TrialStateValues::Fail,
    };
    let datetime_start = match trial.getattr("datetime_start") {
        Ok(value) => value.extract::<Option<Bound<'_, PyAny>>>()?,
        Err(_) => None,
    };
    let datetime_complete = match trial.getattr("datetime_complete") {
        Ok(value) => value.extract::<Option<Bound<'_, PyAny>>>()?,
        Err(_) => None,
    };
    persisted_trial.datetime_start = datetime_start
        .as_ref()
        .map(py_datetime_to_naive_utc)
        .transpose()?;
    persisted_trial.datetime_complete = datetime_complete
        .as_ref()
        .map(py_datetime_to_naive_utc)
        .transpose()?;

    let intermediate_values = match trial.getattr("intermediate_values") {
        Ok(value) => value.extract::<HashMap<u32, f64>>()?,
        Err(_) => HashMap::new(),
    };
    persisted_trial.intermediate_values = intermediate_values;

    let src_internal_params = trial.getattr("internal_params")?;
    if !src_internal_params.is_instance_of::<PyDict>() {
        return Err(PyRuntimeError::new_err("internal_params must be a dict"));
    }
    let src_internal_params = src_internal_params.cast::<PyDict>()?;
    let mut internal_params: HashMap<String, f64> =
        HashMap::with_capacity(src_internal_params.len());
    for (key, value) in src_internal_params.iter() {
        let key = key.extract::<String>()?;
        let value = value.extract::<f64>()?;
        internal_params.insert(key, value);
    }
    persisted_trial.internal_params = internal_params;

    let src_distributions = trial.getattr("distributions")?;
    if !src_distributions.is_instance_of::<PyDict>() {
        return Err(PyRuntimeError::new_err("distributions must be a dict"));
    }
    let src_distributions = src_distributions.cast::<PyDict>()?;
    let mut distributions: HashMap<String, Distribution> =
        HashMap::with_capacity(src_distributions.len());
    let mut category_attrs = Attrs::new();
    for (key, value) in src_distributions.iter() {
        let key = key.extract::<String>()?;
        let value = value.extract::<PyDistribution>()?;
        if let Some(labels) = &value.category_labels {
            category_attrs.extend(category_labels_to_attrs(&key, labels));
        }
        distributions.insert(key, value.into());
    }
    persisted_trial.distributions = distributions;

    let user_attrs = trial.getattr("user_attrs")?;
    let system_attrs = trial.getattr("system_attrs")?;
    persisted_trial.attrs = pyobj_to_attrs(&user_attrs, &system_attrs)?;
    persisted_trial.constraints = trial.getattr("constraints")?.extract()?;
    Ok((persisted_trial, category_attrs))
}
