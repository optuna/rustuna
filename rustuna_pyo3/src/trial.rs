use std::collections::HashMap;
use std::sync::{Arc, RwLock};

use chrono::NaiveDateTime;
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rustuna_core::attr::{get_category_labels, AttrKey, Attrs, CategoryLabel};
use rustuna_core::distribution::Distribution;
use rustuna_core::storage::Storage;
use rustuna_core::trial::{PersistedTrial, Trial, TrialStateValues};

use crate::attrs::pyobj_to_attrs;
use crate::distribution::{
    category_label_to_pyobject, py_to_external_repr, pyobject_to_category_label, PyDistribution,
};
use crate::exception::err_to_exceptions;

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
#[pymethods]
impl PyTrialState {
    pub fn is_finished(&self) -> bool {
        matches!(
            self,
            PyTrialState::COMPLETE | PyTrialState::PRUNED | PyTrialState::FAIL
        )
    }
}

#[pyclass(name = "Trial")]
#[pyo3(module = "rustuna")]
pub struct PyTrial(Trial);
impl From<Trial> for PyTrial {
    fn from(item: Trial) -> Self {
        PyTrial(item)
    }
}
#[pymethods]
impl PyTrial {
    #[getter]
    pub fn id(&self) -> PyResult<u32> {
        Ok(self.0.id)
    }
    #[getter]
    pub fn study_id(&self) -> PyResult<u32> {
        Ok(self.0.study_id)
    }
    #[getter]
    pub fn number(&self) -> PyResult<u32> {
        Ok(self.0.number)
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
        let dist = Distribution::Float {
            low,
            high,
            step,
            log,
        };
        let value = self.0.suggest(name, &dist).map_err(|e| match e.kind {
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
        let dist = Distribution::Int {
            low,
            high,
            step,
            log,
        };
        let value = self.0.suggest(name, &dist).map_err(|e| match e.kind {
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
            .0
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
        self.0.set_user_attr(key, value).map_err(|e| {
            PyRuntimeError::new_err(format!("Failed to set user attr: {:?}", e.kind))
        })?;
        Ok(())
    }
}

enum PyPersistedTrialSource {
    Owned(PersistedTrial),
    StorageBacked {
        storage: Arc<RwLock<dyn Storage>>,
        trial_id: u32,
        // Eagerly cached lightweight fields
        study_id: u32,
        number: u32,
        state: PyTrialState,
        values: Option<Vec<f64>>,
        internal_params: HashMap<String, f64>,
        distributions: HashMap<String, Distribution>,
        intermediate_values: HashMap<u32, f64>,
        datetime_start: Option<String>,
        datetime_complete: Option<String>,
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
            source: PyPersistedTrialSource::Owned(trial),
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
        let values = match &trial.state_values {
            TrialStateValues::Complete(v) => Some(v.clone()),
            _ => None,
        };
        PyPersistedTrial {
            source: PyPersistedTrialSource::StorageBacked {
                storage,
                trial_id: trial.id,
                study_id: trial.study_id,
                number: trial.number,
                state: PyTrialState::from(trial.state_values.clone()),
                values,
                internal_params: trial.internal_params.clone(),
                distributions: trial.distributions.clone(),
                intermediate_values: trial.intermediate_values.clone(),
                datetime_start: trial.datetime_start.clone(),
                datetime_complete: trial.datetime_complete.clone(),
            },
            study_attrs: None,
        }
    }

    fn with_trial<R>(&self, f: impl FnOnce(&PersistedTrial) -> PyResult<R>) -> PyResult<R> {
        match &self.source {
            PyPersistedTrialSource::Owned(trial) => f(trial),
            PyPersistedTrialSource::StorageBacked {
                storage, trial_id, ..
            } => {
                let mut guard = storage
                    .write()
                    .map_err(|_| PyRuntimeError::new_err("Failed to acquire the storage guard"))?;
                let trial = guard.get_trial(*trial_id).map_err(err_to_exceptions)?;
                f(trial)
            }
        }
    }

    fn collect_params(&self) -> PyResult<(u32, TrialParams)> {
        match &self.source {
            PyPersistedTrialSource::Owned(trial) => {
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
            }
            PyPersistedTrialSource::StorageBacked {
                study_id,
                internal_params,
                distributions,
                ..
            } => {
                let mut params: Vec<(String, f64, Distribution)> =
                    Vec::with_capacity(internal_params.len());
                for (name, internal_repr) in internal_params {
                    let dist = distributions
                        .get(name)
                        .ok_or(PyValueError::new_err(format!("No distribution for {name}")))?;
                    params.push((name.clone(), *internal_repr, dist.clone()));
                }
                Ok((*study_id, params))
            }
        }
    }

    fn collect_distributions(&self) -> PyResult<(u32, Vec<(String, Distribution)>)> {
        match &self.source {
            PyPersistedTrialSource::Owned(trial) => {
                let mut distributions: Vec<(String, Distribution)> =
                    Vec::with_capacity(trial.distributions.len());
                for (name, dist) in &trial.distributions {
                    distributions.push((name.clone(), dist.clone()));
                }
                Ok((trial.study_id, distributions))
            }
            PyPersistedTrialSource::StorageBacked {
                study_id,
                distributions,
                ..
            } => {
                let mut dists: Vec<(String, Distribution)> =
                    Vec::with_capacity(distributions.len());
                for (name, dist) in distributions {
                    dists.push((name.clone(), dist.clone()));
                }
                Ok((*study_id, dists))
            }
        }
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
    #[pyo3(signature = (trial_id, study_id, number, state, values=None, internal_params=None, distributions=None, user_attrs=None, system_attrs=None, datetime_start=None, datetime_complete=None, intermediate_values=None))]
    #[allow(clippy::too_many_arguments)]
    pub fn py_new(
        trial_id: u32,
        study_id: u32,
        number: u32,
        state: PyTrialState,
        values: Option<Vec<f64>>,
        internal_params: Option<HashMap<String, f64>>,
        distributions: Option<HashMap<String, PyDistribution>>,
        user_attrs: Option<HashMap<String, String>>,
        system_attrs: Option<HashMap<String, String>>,
        datetime_start: Option<NaiveDateTime>,
        datetime_complete: Option<NaiveDateTime>,
        intermediate_values: Option<HashMap<u32, f64>>,
    ) -> PyResult<Self> {
        if matches!(state, PyTrialState::COMPLETE) && values.is_none() {
            Err(PyValueError::new_err(
                "values must be specified when state is COMPLETE",
            ))?;
        }
        let mut trial = PersistedTrial::new(trial_id, study_id, number);
        trial.state_values = match state {
            PyTrialState::RUNNING => TrialStateValues::Running,
            PyTrialState::COMPLETE => TrialStateValues::Complete(values.ok_or(
                PyValueError::new_err("values must be specified when state is COMPLETE"),
            )?),
            PyTrialState::PRUNED => TrialStateValues::Pruned,
            PyTrialState::FAIL => TrialStateValues::Fail,
            PyTrialState::WAITING => TrialStateValues::Waiting,
        };

        trial.internal_params = internal_params.unwrap_or_default();
        trial.distributions = HashMap::with_capacity(match &distributions {
            Some(d) => d.len(),
            None => 0,
        });
        for (name, dist) in distributions.unwrap_or_default() {
            trial.distributions.insert(name, dist.distribution);
        }

        let user_attrs = user_attrs.unwrap_or_default();
        let system_attrs = system_attrs.unwrap_or_default();
        let n_user_attrs = user_attrs.len();
        let n_system_attrs = user_attrs.len();
        let mut trial_attrs = Attrs::with_capacity(n_user_attrs + n_system_attrs);
        for (key, value) in user_attrs {
            trial_attrs.insert(AttrKey::User(key.into()), value);
        }
        for (key, value) in system_attrs {
            trial_attrs.insert(AttrKey::System(key.into()), value);
        }
        trial.attrs = trial_attrs;
        trial.intermediate_values = intermediate_values.unwrap_or_default();
        trial.datetime_start = datetime_start.map(|dt| dt.to_string());
        trial.datetime_complete = datetime_complete.map(|dt| dt.to_string());

        let study_attrs = Attrs::new();
        Ok(PyPersistedTrial::new(trial, study_attrs))
    }

    #[getter]
    fn id(&self) -> PyResult<u32> {
        match &self.source {
            PyPersistedTrialSource::Owned(trial) => Ok(trial.id),
            PyPersistedTrialSource::StorageBacked { trial_id, .. } => Ok(*trial_id),
        }
    }

    #[getter]
    fn _trial_id(&self) -> PyResult<u32> {
        self.id()
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
            PyPersistedTrialSource::Owned(trial) => {
                Ok(PyTrialState::from(trial.state_values.clone()))
            }
            PyPersistedTrialSource::StorageBacked { state, .. } => Ok(state.clone()),
        }
    }

    #[getter]
    fn values(&self) -> PyResult<Option<Vec<f64>>> {
        match &self.source {
            PyPersistedTrialSource::Owned(trial) => match &trial.state_values {
                TrialStateValues::Complete(values) => Ok(Some(values.clone())),
                _ => Ok(None),
            },
            PyPersistedTrialSource::StorageBacked { values, .. } => Ok(values.clone()),
        }
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
        match &self.source {
            PyPersistedTrialSource::Owned(trial) => Ok(trial.internal_params.clone()),
            PyPersistedTrialSource::StorageBacked {
                internal_params, ..
            } => Ok(internal_params.clone()),
        }
    }

    #[getter]
    fn intermediate_values(&self) -> PyResult<HashMap<u32, f64>> {
        match &self.source {
            PyPersistedTrialSource::Owned(trial) => Ok(trial.intermediate_values.clone()),
            PyPersistedTrialSource::StorageBacked {
                intermediate_values,
                ..
            } => Ok(intermediate_values.clone()),
        }
    }

    #[getter]
    fn datetime_start(&self) -> PyResult<Option<NaiveDateTime>> {
        let value = match &self.source {
            PyPersistedTrialSource::Owned(trial) => trial.datetime_start.as_ref(),
            PyPersistedTrialSource::StorageBacked { datetime_start, .. } => datetime_start.as_ref(),
        };
        match value {
            Some(raw) => Ok(Some(parse_naive_datetime(raw)?)),
            None => Ok(None),
        }
    }

    #[getter]
    fn datetime_complete(&self) -> PyResult<Option<NaiveDateTime>> {
        let value = match &self.source {
            PyPersistedTrialSource::Owned(trial) => trial.datetime_complete.as_ref(),
            PyPersistedTrialSource::StorageBacked {
                datetime_complete, ..
            } => datetime_complete.as_ref(),
        };
        match value {
            Some(raw) => Ok(Some(parse_naive_datetime(raw)?)),
            None => Ok(None),
        }
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
    fn user_attrs(&self) -> PyResult<HashMap<String, String>> {
        self.with_trial(|trial| {
            let user_attrs = trial
                .attrs
                .iter()
                .filter_map(|(key, value)| match key {
                    AttrKey::User(k) => Some((k.to_string(), value.clone())),
                    _ => None,
                })
                .collect();
            Ok(user_attrs)
        })
    }

    #[getter]
    fn system_attrs(&self) -> PyResult<HashMap<String, String>> {
        self.with_trial(|trial| {
            let system_attrs = trial
                .attrs
                .iter()
                .filter_map(|(key, value)| match key {
                    AttrKey::System(k) => Some((k.to_string(), value.clone())),
                    _ => None,
                })
                .collect();
            Ok(system_attrs)
        })
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
        Ok(format!(
            "number={} state={:?} values={:?} params={} distributions={:?} user_attrs={:?} system_attrs={:?}",
            self.number()?,
            self.state()?,
            self.values()?,
            params,
            self.distributions()?,
            self.user_attrs()?,
            self.system_attrs()?,
        ))
    }
}

fn parse_naive_datetime(value: &str) -> PyResult<NaiveDateTime> {
    NaiveDateTime::parse_from_str(value, "%Y-%m-%d %H:%M:%S%.f")
        .or_else(|_| NaiveDateTime::parse_from_str(value, "%Y-%m-%d %H:%M:%S"))
        .map_err(|e| PyValueError::new_err(format!("Failed to parse datetime: {e}")))
}

pub fn pyobject_to_persisted_trial(
    trial: &Bound<'_, PyAny>,
    study_id: u32,
) -> PyResult<PersistedTrial> {
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
        Ok(value) => value.extract::<Option<NaiveDateTime>>()?,
        Err(_) => None,
    };
    let datetime_complete = match trial.getattr("datetime_complete") {
        Ok(value) => value.extract::<Option<NaiveDateTime>>()?,
        Err(_) => None,
    };
    persisted_trial.datetime_start = datetime_start.map(|dt| dt.to_string());
    persisted_trial.datetime_complete = datetime_complete.map(|dt| dt.to_string());

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
    for (key, value) in src_distributions.iter() {
        let key = key.extract::<String>()?;
        let value = value.extract::<PyDistribution>()?;
        distributions.insert(key, value.into());
    }
    persisted_trial.distributions = distributions;

    let user_attrs = trial.getattr("user_attrs")?;
    let system_attrs = trial.getattr("system_attrs")?;
    if !user_attrs.is_instance_of::<PyDict>() || !system_attrs.is_instance_of::<PyDict>() {
        return Err(PyRuntimeError::new_err(
            "user_attrs and system_attrs must be a dict",
        ));
    }
    persisted_trial.attrs = pyobj_to_attrs(&user_attrs, &system_attrs)?;

    if let Ok(intermediate_values) = trial.getattr("intermediate_values") {
        if !intermediate_values.is_instance_of::<PyDict>() {
            return Err(PyRuntimeError::new_err(
                "intermediate_values must be a dict",
            ));
        }
        let intermediate_values = intermediate_values.cast::<PyDict>()?;
        let mut values: HashMap<u32, f64> = HashMap::with_capacity(intermediate_values.len());
        for (key, value) in intermediate_values.iter() {
            let step = key.extract::<u32>()?;
            let value = value.extract::<f64>()?;
            values.insert(step, value);
        }
        persisted_trial.intermediate_values = values;
    }
    Ok(persisted_trial)
}
