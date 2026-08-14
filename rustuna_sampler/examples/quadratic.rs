// How to run this example:
// $ cargo run --example quadratic

use rustuna_core::storage::InMemoryStorage;
use rustuna_core::study::{create_study, Direction};
use rustuna_core::Result;
use rustuna_sampler::nsgaii::NSGAIISampler;
use std::collections::HashMap;

fn main() -> Result<()> {
    let storage = InMemoryStorage::new();
    let directions = vec![Direction::Minimize, Direction::Minimize];
    let study = create_study("simple-quadratic", storage, NSGAIISampler::seed_from_u64(1,50, None, 0.9, 0.1), directions)?;

    study.optimize(
        |mut t| {
            let x = t.suggest_float("x", -15.0, 30.0)?;
            let y = t.suggest_float("y", -15.0, 30.0)?;

            let v0 = 4.0 * x.powi(2) + 4.0 * y.powi(2) as f64;
            let v1 = (x - 5.0).powi(2) + 4.0 * (y - 5.0).powi(2) as f64;
            // t.set_constraints(HashMap::from([(String::from("c0"), 1000.0 - v0)]))?;

            Ok(vec![v0, v1])
        },
        1000,
    )?;

    Ok(())
}
