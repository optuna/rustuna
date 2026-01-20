use serde::{Deserialize, Serialize};

/// Intermediate value entry for JSON serialization.
///
/// This structure is used to serialize intermediate values with their type information,
/// preserving special float values (NaN, Infinity, -Infinity) that cannot be represented
/// in standard JSON format.
///
/// # Fields
/// * `step` - The step number (epoch, iteration, etc.) for this intermediate value
/// * `value` - The actual f64 value. None for special values (NaN, Infinity, -Infinity)
/// * `value_type` - Type discriminator. One of:
///   - "FINITE": Normal floating-point value (value is Some)
///   - "NAN": Not a Number (value is None)
///   - "INF_POS": Positive infinity (value is None)
///   - "INF_NEG": Negative infinity (value is None)
#[derive(Debug, Serialize, Deserialize)]
pub struct IntermediateValueEntry {
    pub step: u32,
    pub value: Option<f64>,
    pub value_type: String,
}
