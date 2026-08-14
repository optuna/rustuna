use std::collections::HashMap;

use rustuna_core::{Error, ErrorKind, Result};
use serde_json::{Map, Number, Value};

pub(crate) const OPTUNA_CONSTRAINTS_KEY: &str = "constraints";
pub(crate) const RUSTUNA_CONSTRAINTS_KEY: &str = "rustuna:constraints";

pub(crate) fn constraints_to_values(constraints: &HashMap<String, f64>) -> Vec<Value> {
    let mut constraints = constraints.iter().collect::<Vec<_>>();
    constraints.sort_unstable_by_key(|(name0, _)| *name0);
    constraints
        .into_iter()
        .map(|(_, value)| f64_to_value(*value))
        .collect()
}

pub(crate) fn constraints_to_map(constraints: &HashMap<String, f64>) -> Map<String, Value> {
    constraints
        .iter()
        .map(|(name, value)| (name.clone(), f64_to_value(*value)))
        .collect()
}

pub(crate) fn constraints_from_value(value: &Value) -> Result<HashMap<String, f64>> {
    match value {
        Value::Array(values) => values
            .iter()
            .enumerate()
            .map(|(index, value)| Ok((index.to_string(), value_to_f64(value)?)))
            .collect(),
        Value::Object(values) => values
            .iter()
            .map(|(name, value)| Ok((name.clone(), value_to_f64(value)?)))
            .collect(),
        Value::Null => Ok(HashMap::new()),
        _ => Err(Error::with_reason(
            ErrorKind::StorageError,
            "Constraints must be stored as a JSON array or object.".to_string(),
        )),
    }
}

pub(crate) fn constraints_from_json(value: &str) -> Result<HashMap<String, f64>> {
    let value = serde_json::from_str(value).map_err(|e| {
        Error::with_reason(
            ErrorKind::StorageError,
            format!("Failed to parse constraints as JSON: {e}"),
        )
    })?;
    constraints_from_value(&value)
}

fn f64_to_value(value: f64) -> Value {
    if value.is_nan() {
        return Value::String("NaN".to_string());
    }
    if value == f64::INFINITY {
        return Value::String("Infinity".to_string());
    }
    if value == f64::NEG_INFINITY {
        return Value::String("-Infinity".to_string());
    }
    Number::from_f64(value)
        .map(Value::Number)
        .unwrap_or(Value::Null)
}

fn value_to_f64(value: &Value) -> Result<f64> {
    match value {
        Value::Number(value) => value.as_f64().ok_or_else(|| {
            Error::with_reason(
                ErrorKind::StorageError,
                "Constraint is not representable as f64.".to_string(),
            )
        }),
        Value::String(value) => match value.as_str() {
            "NaN" => Ok(f64::NAN),
            "Infinity" => Ok(f64::INFINITY),
            "-Infinity" => Ok(f64::NEG_INFINITY),
            _ => value.parse::<f64>().map_err(|e| {
                Error::with_reason(
                    ErrorKind::StorageError,
                    format!("Failed to parse constraint as f64: {e}"),
                )
            }),
        },
        _ => Err(Error::with_reason(
            ErrorKind::StorageError,
            "Constraint must be a JSON number or string.".to_string(),
        )),
    }
}
