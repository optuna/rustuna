#![feature(test)]

extern crate test;

#[cfg(test)]
mod tests {
    use test::Bencher;

    use rustuna_core::storage::InMemoryStorage;
    use rustuna_core::study::{create_study, Direction};
    use rustuna_sampler::nsgaii::NsgaiiSampler;

    #[bench]
    fn bench_nsgaii(b: &mut Bencher) {
        b.iter(|| {
            let directions = vec![Direction::Minimize, Direction::Minimize];
            let storage = InMemoryStorage::new();
            let study =
                create_study("dummy", storage, NsgaiiSampler::default(), directions).unwrap();
            study
                .optimize(
                    |mut t| {
                        let x = t.suggest_float("x", 0.0, 5.0)?;
                        let y = t.suggest_float("y", 0.0, 3.0)?;

                        let v0 = 4.0 * x.powi(2) + 4.0 * y.powi(2);
                        let v1 = (x - 5.0).powi(2) + (y - 5.0).powi(2);
                        Ok(vec![v0, v1])
                    },
                    200,
                )
                .unwrap();
        })
    }
}
