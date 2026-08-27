use std::collections::HashMap;

use crate::distribution::Distribution;
use crate::{Error, ErrorKind, Result};

/// Transforms numerical parameter configurations into a unit hypercube.
///
/// The transformation always applies log scaling, half-step boundary expansion, and unit-cube
/// normalization. Categorical distributions are not supported.
#[derive(Debug)]
pub struct SearchSpaceTransform {
    names: Vec<String>,
    distributions: Vec<Distribution>,
    raw_bounds: Vec<[f64; 2]>,
    bounds: Vec<[f64; 2]>,
}

impl SearchSpaceTransform {
    /// Creates a transformation for a numerical search space.
    pub fn new(search_space: &HashMap<String, Distribution>) -> Result<Self> {
        if search_space.is_empty() {
            return Err(Error::with_reason(
                ErrorKind::UnsupportedSearchSpace,
                "Cannot transform an empty search space",
            ));
        }

        let mut names: Vec<_> = search_space.keys().cloned().collect();
        names.sort();

        let mut distributions = Vec::with_capacity(names.len());
        let mut raw_bounds = Vec::with_capacity(names.len());
        for name in &names {
            let distribution = &search_space[name];
            if matches!(distribution, Distribution::Categorical { .. }) {
                return Err(Error::with_reason(
                    ErrorKind::UnsupportedSearchSpace,
                    format!("Categorical distribution is not supported: {name}"),
                ));
            }
            distributions.push(distribution.clone());
            raw_bounds.push(transformed_bounds(distribution));
        }

        let bounds = vec![[0.0, 1.0]; names.len()];
        Ok(Self {
            names,
            distributions,
            raw_bounds,
            bounds,
        })
    }

    /// Returns the parameter names in the order used by transformed vectors.
    pub fn parameter_names(&self) -> &[String] {
        &self.names
    }

    /// Returns the unit-cube bounds for each transformed parameter.
    pub fn bounds(&self) -> &[[f64; 2]] {
        &self.bounds
    }

    /// Transforms internal parameter values into the unit hypercube.
    pub fn transform(&self, params: &HashMap<String, f64>) -> Result<Vec<f64>> {
        self.names
            .iter()
            .zip(&self.distributions)
            .zip(&self.raw_bounds)
            .map(|((name, distribution), bounds)| {
                let param = params.get(name).ok_or_else(|| {
                    Error::with_reason(
                        ErrorKind::Unexpected,
                        format!("Parameter configuration is missing: {name}"),
                    )
                })?;
                Ok(normalize(
                    transform_numerical_param(*param, distribution),
                    *bounds,
                ))
            })
            .collect()
    }

    /// Converts unit-hypercube values back to internal parameter values.
    pub fn untransform(&self, transformed_params: &[f64]) -> Result<HashMap<String, f64>> {
        if transformed_params.len() != self.names.len() {
            return Err(Error::with_reason(
                ErrorKind::Unexpected,
                format!(
                    "Expected {} transformed parameters, got {}",
                    self.names.len(),
                    transformed_params.len()
                ),
            ));
        }

        Ok(self
            .names
            .iter()
            .zip(&self.distributions)
            .zip(&self.raw_bounds)
            .zip(transformed_params)
            .map(|(((name, distribution), bounds), transformed_param)| {
                (
                    name.clone(),
                    untransform_numerical_param(
                        unnormalize(*transformed_param, *bounds),
                        distribution,
                    ),
                )
            })
            .collect())
    }
}

fn transformed_bounds(distribution: &Distribution) -> [f64; 2] {
    match distribution {
        Distribution::Float {
            low,
            high,
            step,
            log,
        } => {
            let low = transform_value(*low, *log);
            let high = transform_value(*high, *log);
            let half_step = step.unwrap_or(0.0) / 2.0;
            [low - half_step, high + half_step]
        }
        Distribution::Int {
            low,
            high,
            step,
            log,
        } => {
            let half_step = *step as f64 / 2.0;
            if *log {
                [
                    transform_value(*low as f64 - half_step, true),
                    transform_value(*high as f64 + half_step, true),
                ]
            } else {
                [*low as f64 - half_step, *high as f64 + half_step]
            }
        }
        Distribution::Categorical { .. } => {
            unreachable!("Categorical distributions are rejected during construction")
        }
    }
}

fn transform_numerical_param(param: f64, distribution: &Distribution) -> f64 {
    match distribution {
        Distribution::Float { log, .. } | Distribution::Int { log, .. } => {
            transform_value(param, *log)
        }
        Distribution::Categorical { .. } => {
            unreachable!("Categorical distributions are rejected during construction")
        }
    }
}

fn transform_value(value: f64, log: bool) -> f64 {
    if log {
        value.ln()
    } else {
        value
    }
}

fn normalize(value: f64, bounds: [f64; 2]) -> f64 {
    if bounds[0] == bounds[1] {
        0.5
    } else {
        (value - bounds[0]) / (bounds[1] - bounds[0])
    }
}

fn unnormalize(value: f64, bounds: [f64; 2]) -> f64 {
    bounds[0] + value * (bounds[1] - bounds[0])
}

fn untransform_numerical_param(transformed_param: f64, distribution: &Distribution) -> f64 {
    match distribution {
        Distribution::Float {
            low,
            high,
            step,
            log,
        } => {
            if *log {
                let param = transformed_param.exp();
                if distribution.is_single() {
                    param
                } else {
                    // `exp` does not round-trip `ln` exactly, so a transformed value sitting on
                    // the lower bound can come back just below `low`. Clamping keeps every
                    // suggestion inside the range the user asked for.
                    param.clamp(*low, high.next_down())
                }
            } else if let Some(step) = step {
                round_to_step(transformed_param, *low, *high, *step)
            } else if distribution.is_single() {
                transformed_param
            } else {
                transformed_param.clamp(*low, high.next_down())
            }
        }
        Distribution::Int {
            low,
            high,
            step,
            log,
        } => {
            let param = if *log {
                transformed_param.exp().round_ties_even()
            } else {
                *low as f64
                    + ((transformed_param - *low as f64) / *step as f64).round_ties_even()
                        * *step as f64
            };
            param.clamp(*low as f64, *high as f64)
        }
        Distribution::Categorical { .. } => {
            unreachable!("Categorical distributions are rejected during construction")
        }
    }
}

fn round_to_step(value: f64, low: f64, high: f64, step: f64) -> f64 {
    (low + ((value - low) / step).round_ties_even() * step).clamp(low, high)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() < 1e-12,
            "actual: {actual}, expected: {expected}"
        );
    }

    #[test]
    fn test_transform_and_untransform_numerical_search_space() -> Result<()> {
        let search_space = HashMap::from([
            (
                "continuous".to_string(),
                Distribution::new_float(-2.0, 2.0, None, false),
            ),
            (
                "discrete".to_string(),
                Distribution::new_float(0.0, 10.0, Some(2.0), false),
            ),
            ("integer".to_string(), Distribution::new_int(1, 5, 2, false)),
            (
                "log_float".to_string(),
                Distribution::new_float(1.0, std::f64::consts::E.powi(2), None, true),
            ),
            (
                "log_integer".to_string(),
                Distribution::new_int(1, 9, 1, true),
            ),
        ]);
        let transform = SearchSpaceTransform::new(&search_space)?;
        let params = HashMap::from([
            ("continuous".to_string(), 0.0),
            ("discrete".to_string(), 2.0),
            ("integer".to_string(), 3.0),
            ("log_float".to_string(), std::f64::consts::E),
            ("log_integer".to_string(), 3.0),
        ]);

        assert_eq!(
            transform.parameter_names(),
            [
                "continuous",
                "discrete",
                "integer",
                "log_float",
                "log_integer"
            ]
        );
        assert_eq!(transform.bounds(), &[[0.0, 1.0]; 5]);

        let transformed = transform.transform(&params)?;
        assert_close(transformed[0], 0.5);
        assert_close(transformed[1], 0.25);
        assert_close(transformed[2], 0.5);
        assert_close(transformed[3], 0.5);
        assert_close(
            transformed[4],
            (3.0_f64.ln() - 0.5_f64.ln()) / (9.5_f64.ln() - 0.5_f64.ln()),
        );

        let untransformed = transform.untransform(&transformed)?;
        for (name, value) in params {
            assert_close(untransformed[&name], value);
        }
        Ok(())
    }

    #[test]
    fn test_untransform_keeps_log_parameters_at_or_above_low() {
        // `exp` does not round-trip `ln`: for this range `(1e-7f64).ln().exp()` lands just below
        // 1e-7. Without the clamp in `untransform_numerical_param` the lower end of the range
        // would leak out as a suggestion below `low`. A QMC sampler hits it on every study,
        // because the first point of its sequence is the origin of the unit hypercube.
        let low: f64 = 1e-7;
        assert!(
            low.ln().exp() < low,
            "the round trip is expected to undershoot"
        );

        let search_space = HashMap::from([(
            "x".to_string(),
            Distribution::new_float(low, 1.0, None, true),
        )]);
        let transform = SearchSpaceTransform::new(&search_space).unwrap();

        assert_eq!(transform.untransform(&[0.0]).unwrap()["x"], low);
        assert!(transform.untransform(&[1.0]).unwrap()["x"] < 1.0);
    }

    #[test]
    fn test_untransform_rounds_to_step() -> Result<()> {
        let search_space = HashMap::from([
            (
                "float".to_string(),
                Distribution::new_float(0.0, 10.0, Some(2.0), false),
            ),
            ("int".to_string(), Distribution::new_int(0, 10, 2, false)),
        ]);
        let transform = SearchSpaceTransform::new(&search_space)?;

        let params = transform.untransform(&[0.41, 0.59])?;
        assert_eq!(params["float"], 4.0);
        assert_eq!(params["int"], 6.0);
        Ok(())
    }

    #[test]
    fn test_new_rejects_categorical_distribution() {
        let search_space =
            HashMap::from([("category".to_string(), Distribution::new_categorical(2))]);

        let error = SearchSpaceTransform::new(&search_space).unwrap_err();
        assert!(matches!(error.kind, ErrorKind::UnsupportedSearchSpace));
    }

    #[test]
    fn test_transform_rejects_missing_parameter() -> Result<()> {
        let search_space = HashMap::from([(
            "x".to_string(),
            Distribution::new_float(0.0, 1.0, None, false),
        )]);
        let transform = SearchSpaceTransform::new(&search_space)?;

        assert!(transform.transform(&HashMap::new()).is_err());
        assert!(transform.untransform(&[]).is_err());
        Ok(())
    }
}
