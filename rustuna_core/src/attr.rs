use crate::string_interner::InternedString;
use std::collections::HashMap;

/// Attribute map used by studies and trials.
///
/// In Optuna, user and system attributes are exposed as separate dictionaries. Rustuna stores
/// both in a single map and distinguishes them with [`AttrKey::User`] and [`AttrKey::System`].
/// Unlike Optuna, which accepts arbitrary JSON-serializable values, Rustuna stores attribute
/// values as strings.
pub type Attrs = HashMap<AttrKey, String>;

/// Distinguishes between user and system attributes.
#[derive(Eq, Hash, Clone, Debug, PartialEq)]
pub enum AttrKey {
    /// User-defined metadata.
    User(InternedString),
    /// Internal metadata managed by Rustuna.
    System(InternedString),
}

/// Label used for categorical choices and fixed queued parameters.
///
/// This matches Optuna's `CategoricalChoiceType`.
///
/// In Optuna, categorical choices are stored directly in each `CategoricalDistribution` object.
/// Rustuna's categorical distribution stores only its cardinality so that trials do not have to
/// carry heap allocated choice lists repeatedly. The actual choice labels are stored separately in
/// study system attributes and encoded with `CategoryLabel`.
#[derive(PartialEq, Debug, Clone)]
pub enum CategoryLabel {
    Float(f64),
    Int(i64),
    String(String),
    Bool(bool),
    None,
}
impl CategoryLabel {
    /// Serializes the label to a stable string representation.
    pub fn serialize(&self) -> String {
        match self {
            CategoryLabel::Float(f) => format!("f:0x{:016x}", f.to_bits()),
            CategoryLabel::Int(i) => format!("i:{i}"),
            CategoryLabel::String(s) => format!("s:{s}"),
            CategoryLabel::Bool(b) => {
                if *b {
                    String::from("true")
                } else {
                    String::from("false")
                }
            }
            CategoryLabel::None => "None".to_string(),
        }
    }
    /// Deserializes a value produced by [`CategoryLabel::serialize`].
    pub fn deserialize(s: &str) -> Option<CategoryLabel> {
        if s == "None" {
            return Some(CategoryLabel::None);
        }
        if s == "true" {
            return Some(CategoryLabel::Bool(true));
        }
        if s == "false" {
            return Some(CategoryLabel::Bool(false));
        }
        if let Some(f) = s.strip_prefix("f:") {
            if let Some(hex) = f.strip_prefix("0x") {
                if let Ok(bits) = u64::from_str_radix(hex, 16) {
                    return Some(CategoryLabel::Float(f64::from_bits(bits)));
                }
            }
            if let Ok(f) = f.parse::<f64>() {
                return Some(CategoryLabel::Float(f));
            }
            return None;
        }
        if let Some(i) = s.strip_prefix("i:") {
            let i = i.parse::<i64>().ok()?;
            return Some(CategoryLabel::Int(i));
        }
        if let Some(s) = s.strip_prefix("s:") {
            return Some(CategoryLabel::String(s.to_string()));
        }
        None // Must be unreachable.
    }
}

/// Returns the internal system-attribute key used to store a categorical label.
pub(crate) fn system_key_category_label(param_name: &str, choice_idx: usize) -> AttrKey {
    AttrKey::System(format!("category_labels:{param_name}:{choice_idx}").into())
}

/// Encodes categorical labels into system attributes.
pub fn category_labels_to_attrs(param_name: &str, labels: &[CategoryLabel]) -> Attrs {
    let mut attrs = Attrs::new();
    for (i, label) in labels.iter().enumerate() {
        let key = system_key_category_label(param_name, i);
        attrs.insert(key, label.serialize().clone());
    }
    attrs
}

/// Decodes categorical labels from system attributes.
pub fn get_category_labels(
    attrs: &Attrs,
    param_name: &str,
    len: usize,
) -> Option<Vec<CategoryLabel>> {
    let mut labels: Vec<CategoryLabel> = Vec::with_capacity(len);
    for i in 0..len {
        let key = system_key_category_label(param_name, i);
        {
            let label = attrs.get(&key)?;
            let label = CategoryLabel::deserialize(label)?;
            labels.push(label);
        }
    }
    Some(labels)
}

/// Returns the internal system-attribute key used for queued fixed parameters.
pub(crate) fn system_key_fixed_param(param_name: &str) -> AttrKey {
    AttrKey::System(format!("fixed_params:{param_name}").into())
}

/// Encodes fixed parameter values into trial attributes.
pub(crate) fn fixed_params_to_attrs(params: &HashMap<String, CategoryLabel>) -> Attrs {
    let mut attrs = Attrs::new();
    for (name, value) in params {
        let key = system_key_fixed_param(name);
        attrs.insert(key, value.serialize());
    }
    attrs
}

/// Extracts all fixed parameters stored in trial attributes.
pub(crate) fn extract_fixed_params(attrs: &Attrs) -> HashMap<String, CategoryLabel> {
    let mut params = HashMap::new();
    for (key, value) in attrs {
        if let AttrKey::System(s) = key {
            if let Some(param_name) = s.as_str().strip_prefix("fixed_params:") {
                if let Some(label) = CategoryLabel::deserialize(value) {
                    params.insert(param_name.to_string(), label);
                }
            }
        }
    }
    params
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_category_label() {
        let categories = vec![
            CategoryLabel::Float(1.0),
            CategoryLabel::Float(f64::from_bits(1)),
            CategoryLabel::Int(2),
            CategoryLabel::String("3".to_string()),
            CategoryLabel::Bool(true),
            CategoryLabel::Bool(false),
            CategoryLabel::None,
        ];

        for c in categories {
            let s = c.serialize();
            let c2 = CategoryLabel::deserialize(&s).expect("Failed to deserialize category label");
            assert_eq!(c, c2);
        }
    }

    #[test]
    fn test_category_label_deserialize_legacy_float_format() {
        let value = 2.2250738585072014e-308_f64;
        let serialized = format!("f:{value}");
        let deserialized = CategoryLabel::deserialize(&serialized).unwrap();
        assert_eq!(deserialized, CategoryLabel::Float(value));
    }
}
