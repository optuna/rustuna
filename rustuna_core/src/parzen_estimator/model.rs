use rand::rngs::StdRng;
use std::{borrow::Cow, collections::HashMap, vec};

use super::probability_distributions::CategoricalDistributions;
use super::probability_distributions::{
    DiscreteTruncLogNormDistributions, DiscreteTruncNormDistributions, Distributions,
    MixtureOfProductDistribution, TruncLogNormDistributions, TruncNormDistributions,
};
use crate::distribution::Distribution;
use crate::parzen_estimator::scott::ScottNumericalDistributionBuilder;

pub struct ParzenEstimator {
    mixuture_distribution: MixtureOfProductDistribution,
}

enum ObservationValues<'a, I> {
    Borrowed(&'a [f64]),
    Selected(I),
}

impl<'a, I: Iterator<Item = f64>> ObservationValues<'a, I> {
    fn into_slice(self) -> Cow<'a, [f64]> {
        match self {
            Self::Borrowed(values) => Cow::Borrowed(values),
            Self::Selected(values) => Cow::Owned(values.collect()),
        }
    }
}

impl<I: Iterator<Item = f64>> Iterator for ObservationValues<'_, I> {
    type Item = f64;

    fn next(&mut self) -> Option<f64> {
        match self {
            Self::Borrowed(values) => {
                let (value, rest) = values.split_first()?;
                *values = rest;
                Some(*value)
            }
            Self::Selected(values) => values.next(),
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        match self {
            Self::Borrowed(values) => (values.len(), Some(values.len())),
            Self::Selected(values) => values.size_hint(),
        }
    }
}

impl ParzenEstimator {
    pub fn new(
        observations: &HashMap<String, Vec<f64>>,
        search_space: &HashMap<String, Distribution>,
        weights: &[f64],
        prior_weight: f64,
    ) -> Self {
        Self::with_builder(
            observations,
            search_space,
            weights,
            prior_weight,
            &DefaultNumericalDistributionBuilder,
            &DefaultCategoricalDistributionBuilder,
        )
    }

    pub fn with_scott(
        observations: &HashMap<String, Vec<f64>>,
        search_space: &HashMap<String, Distribution>,
        weights: &[f64],
        prior_weight: f64,
    ) -> Self {
        Self::with_builder(
            observations,
            search_space,
            weights,
            prior_weight,
            &ScottNumericalDistributionBuilder::new(weights),
            &DefaultCategoricalDistributionBuilder,
        )
    }

    pub(crate) fn with_builder(
        observations: &HashMap<String, Vec<f64>>,
        search_space: &HashMap<String, Distribution>,
        weights: &[f64],
        prior_weight: f64,
        num_builder: &impl NumericalDistributionBuilder,
        cat_builder: &impl CategoricalDistributionBuilder,
    ) -> Self {
        let n_observations = observations.values().next().map_or(0, |v| {
            assert!(
                observations.values().all(|w| w.len() == v.len()),
                "Observations have inconsistent lengths"
            );
            v.len()
        });
        assert_eq!(
            n_observations,
            weights.len(),
            "Number of observations and length of weights must be equal"
        );

        let mut keys: Vec<_> = search_space.keys().collect();
        keys.sort();

        Self::with_ordered_builder(
            keys.into_iter().map(|key| {
                (
                    key.as_str(),
                    &search_space[key],
                    ObservationValues::<std::iter::Empty<f64>>::Borrowed(
                        observations.get(key).map(Vec::as_slice).unwrap_or(&[]),
                    ),
                )
            }),
            weights,
            prior_weight,
            num_builder,
            cat_builder,
        )
    }

    /// Builds a model from columns in strictly increasing parameter-name order.
    /// Each column must provide one value for each observation weight.
    pub fn from_ordered_observations<'a, I, V>(
        parameters: I,
        weights: &[f64],
        prior_weight: f64,
    ) -> Self
    where
        I: IntoIterator<Item = (&'a str, &'a Distribution, V)>,
        V: IntoIterator<Item = f64>,
    {
        let model = Self::with_ordered_builder(
            parameters.into_iter().map(|(name, distribution, values)| {
                (
                    name,
                    distribution,
                    ObservationValues::Selected(values.into_iter()),
                )
            }),
            weights,
            prior_weight,
            &DefaultNumericalDistributionBuilder,
            &DefaultCategoricalDistributionBuilder,
        );
        assert!(
            model
                .mixuture_distribution
                .distributions
                .iter()
                .all(|distribution| distribution.n_kernels() == weights.len() + 1),
            "Observation columns and weights differ in length"
        );
        model
    }

    fn with_ordered_builder<'a, I, V>(
        parameters: I,
        weights: &[f64],
        prior_weight: f64,
        num_builder: &impl NumericalDistributionBuilder,
        cat_builder: &impl CategoricalDistributionBuilder,
    ) -> Self
    where
        I: IntoIterator<Item = (&'a str, &'a Distribution, ObservationValues<'a, V>)>,
        V: Iterator<Item = f64>,
    {
        let parameters = parameters.into_iter();
        let mut distributions: Vec<(String, Distributions)> =
            Vec::with_capacity(parameters.size_hint().0);
        for (name, search_space, values) in parameters {
            if let Some((previous, _)) = distributions.last() {
                assert!(
                    previous.as_str() < name,
                    "Parameter names must be sorted and unique"
                );
            }
            let distribution = match search_space {
                Distribution::Float { .. } | Distribution::Int { .. } => {
                    let values = values.into_slice();
                    num_builder.calculate_numerical_distribution(&values, search_space)
                }
                Distribution::Categorical { .. } => {
                    cat_builder.calculate_categorical_distribution(values.into_iter(), search_space)
                }
            };
            distributions.push((name.to_owned(), distribution));
        }

        let weights_sum = {
            let s = weights.iter().sum::<f64>() + prior_weight;
            if s == 0.0 {
                (weights.len() + 1) as f64
            } else {
                s
            }
        };

        let weights_with_prior_weight = weights
            .iter()
            .chain(std::iter::once(&prior_weight))
            .map(|w| w / weights_sum)
            .collect();

        Self {
            mixuture_distribution: MixtureOfProductDistribution::new(
                distributions,
                weights_with_prior_weight,
            ),
        }
    }

    pub fn sample(&self, rng: &mut StdRng, size: usize) -> Vec<HashMap<String, f64>> {
        self.mixuture_distribution.sample(rng, size)
    }

    /// Returns the parameter names used by ordered samples and scores.
    pub fn parameter_names(&self) -> &[String] {
        &self.mixuture_distribution.param_names
    }

    /// Samples values in [`Self::parameter_names`] order.
    pub fn sample_ordered(&self, rng: &mut StdRng, size: usize) -> Vec<Vec<f64>> {
        self.mixuture_distribution.sample_ordered(rng, size)
    }

    pub fn log_pdf(&self, x: &HashMap<String, f64>) -> f64 {
        self.mixuture_distribution.log_pdf(x)
    }

    /// Scores values in [`Self::parameter_names`] order.
    /// Returns negative infinity if the value count does not match the model.
    pub fn log_pdf_ordered(&self, values: &[f64]) -> f64 {
        self.mixuture_distribution.log_pdf_ordered(values)
    }
}

pub(crate) trait NumericalDistributionBuilder {
    fn calculate_numerical_distribution(
        &self,
        observations: &[f64],
        search_space: &Distribution,
    ) -> Distributions;
}

pub(crate) trait CategoricalDistributionBuilder {
    fn calculate_categorical_distribution(
        &self,
        observations: impl Iterator<Item = f64>,
        search_space: &Distribution,
    ) -> Distributions;
}

pub(crate) struct DefaultNumericalDistributionBuilder;
pub(crate) struct DefaultCategoricalDistributionBuilder;

/// Each observation's larger distance to its two sorted neighbors, floored at `minsigma`, or
/// `None` if fewer than two observations are given or some of them are outside the search space.
///
/// A neighbor nearer than `minsigma` cannot lift that floor, and binning at that width hides
/// exactly those: within a bin only the extremes can have a farther neighbor, and it is the
/// adjacent non-empty bin's extreme. The per-bin extremes therefore give every bandwidth.
fn binned_sigmas(obs: &[f64], low: f64, high: f64) -> Option<Vec<f64>> {
    if obs.len() < 2 {
        // No inter-observation neighbor exists; the caller falls back to endpoint distances.
        return None;
    }

    let n_bins = 100.min(obs.len() + 2);
    let minsigma = (high - low) / n_bins as f64;

    let mut bins = vec![(f64::INFINITY, 0_usize, f64::NEG_INFINITY, 0_usize); n_bins];
    for (i, &v) in obs.iter().enumerate() {
        if !(low..=high).contains(&v) {
            // The observation is outside the search space.
            return None;
        }
        let bin = &mut bins[(((v - low) / minsigma) as usize).min(n_bins - 1)];
        if v < bin.0 {
            (bin.0, bin.1) = (v, i);
        }
        if v >= bin.2 {
            (bin.2, bin.3) = (v, i);
        }
    }

    let mut sigmas = vec![minsigma; obs.len()];
    let (mut prev_max, mut next_min) = (f64::NAN, f64::NAN);
    for bin in bins.iter().filter(|b| b.0 <= b.2) {
        sigmas[bin.1] = sigmas[bin.1].max(bin.0 - prev_max);
        prev_max = bin.2;
    }
    for bin in bins.iter().rev().filter(|b| b.0 <= b.2) {
        sigmas[bin.3] = sigmas[bin.3].max(next_min - bin.2);
        next_min = bin.0;
    }
    Some(sigmas)
}

impl NumericalDistributionBuilder for DefaultNumericalDistributionBuilder {
    fn calculate_numerical_distribution(
        &self,
        observations: &[f64],
        search_space: &Distribution,
    ) -> Distributions {
        // Currently, we assume consider_prior=True, consider_endpoints=False, and consider_magic_clip=True.
        let (low, high, step_opt, log) = match search_space {
            Distribution::Float {
                low,
                high,
                step,
                log,
            } => (*low, *high, *step, *log),
            Distribution::Int {
                low,
                high,
                step,
                log,
            } => (*low as f64, *high as f64, Some(*step as f64), *log),
            _ => unreachable!("Invalid distribution type for numerical calculation"),
        };

        let (mut adj_low, mut adj_high) = match step_opt {
            Some(s) => (low - s / 2.0, high + s / 2.0),
            None => (low, high),
        };

        if log {
            adj_low = adj_low.ln();
            adj_high = adj_high.ln();
        }

        let mus = observations
            .iter()
            .map(|&m| if log { m.ln() } else { m })
            .chain(std::iter::once((adj_low + adj_high) / 2.0)) // Add prior
            .collect::<Vec<_>>();

        let mut sigmas = Vec::with_capacity(mus.len() + 1); // +1 for prior
        if mus.len() == 1 {
            // Case: prior only
            sigmas.push(adj_high - adj_low);
        } else if let Some(binned) = binned_sigmas(&mus[..mus.len() - 1], adj_low, adj_high) {
            sigmas = binned;
            // Sigma for prior
            sigmas.push(adj_high - adj_low);
        } else {
            let m = mus.len() - 1; // exclude prior
            let mut idx_vals: Vec<(usize, f64)> = (0..m).map(|i| (i, mus[i])).collect();
            idx_vals.sort_by(|a, b| a.1.total_cmp(&b.1));
            let sorted_obs: Vec<f64> = idx_vals.iter().map(|&(_, v)| v).collect();

            // consider_endpoints=False: boundary observations use only the neighbor distance.
            // When m == 1, no inter-observation neighbor exists, so fall back to endpoint distances.
            sigmas.resize(m, 0.0);
            for (j, &(orig_idx, _)) in idx_vals.iter().enumerate() {
                sigmas[orig_idx] = if m == 1 {
                    (sorted_obs[0] - adj_low).max(adj_high - sorted_obs[0])
                } else if j == 0 {
                    sorted_obs[1] - sorted_obs[0]
                } else if j == m - 1 {
                    sorted_obs[m - 1] - sorted_obs[m - 2]
                } else {
                    (sorted_obs[j] - sorted_obs[j - 1]).max(sorted_obs[j + 1] - sorted_obs[j])
                };
            }
            // Sigma for prior
            sigmas.push(adj_high - adj_low);

            // Clamp (minsigma, maxsigma)
            let maxsigma = adj_high - adj_low;
            let minsigma = (adj_high - adj_low) / (100.0_f64.min(1.0 + sigmas.len() as f64));
            for s in sigmas.iter_mut() {
                *s = s.clamp(minsigma, maxsigma);
            }
        }

        match (step_opt, log) {
            (None, false) => {
                Distributions::TruncNorm(TruncNormDistributions::new(mus, sigmas, low, high))
            }
            (None, true) => {
                Distributions::TruncLogNorm(TruncLogNormDistributions::new(mus, sigmas, low, high))
            }
            (Some(step), false) => Distributions::DiscreteTruncNorm(
                DiscreteTruncNormDistributions::new(mus, sigmas, low, high, step),
            ),
            (Some(step), true) => Distributions::DiscreteTruncLogNorm(
                DiscreteTruncLogNormDistributions::new(mus, sigmas, low, high, step),
            ),
        }
    }
}

impl CategoricalDistributionBuilder for DefaultCategoricalDistributionBuilder {
    fn calculate_categorical_distribution(
        &self,
        observations: impl Iterator<Item = f64>,
        search_space: &Distribution,
    ) -> Distributions {
        let cardinality = match search_space {
            Distribution::Categorical { cardinality } => *cardinality,
            _ => unreachable!("Invalid distribution type for categorical calculation"),
        };

        Distributions::Categorical(CategoricalDistributions::new(observations, cardinality))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{Rng, SeedableRng};

    #[test]
    fn binned_sigmas_matches_a_naive_scan() {
        // The larger distance to the two neighbors in sorted order, floored.
        let naive = |obs: &[f64], minsigma: f64| {
            let n = obs.len();
            let mut order: Vec<usize> = (0..n).collect();
            order.sort_by(|&a, &b| obs[a].total_cmp(&obs[b]));
            let mut want = vec![0.0; n];
            for (j, &i) in order.iter().enumerate() {
                // A missing neighbor leaves NaN, which `max` discards.
                let gap = |k: usize| (obs[order[k]] - obs[i]).abs();
                let left = if j > 0 { gap(j - 1) } else { f64::NAN };
                let right = if j + 1 < n { gap(j + 1) } else { f64::NAN };
                want[i] = left.max(right).max(minsigma);
            }
            want
        };

        let (low, high) = (-10.0, 10.0);
        let mut rng = rand::rngs::StdRng::seed_from_u64(0);
        for m in [2, 3, 7, 64, 500] {
            for spread in [1.0, 1e-4, 1e-9] {
                // Endpoints, a duplicate, and a cluster far tighter than one bin, so that the
                // neighbor rule and its tie-breaking are both exercised.
                let mut obs: Vec<f64> = (0..m).map(|_| rng.gen_range(-spread..spread)).collect();
                (obs[0], obs[m - 1], obs[m / 2]) = (low, high, obs[m / 3]);

                let minsigma = (high - low) / (100.0_f64.min(2.0 + m as f64));
                let binned = binned_sigmas(&obs, low, high).unwrap();
                assert_eq!(binned, naive(&obs, minsigma), "m = {m}, spread = {spread}");
            }
        }
    }

    #[test]
    fn build_parzen_estimator() {
        let mut observations = HashMap::<String, Vec<f64>>::new();
        observations.insert("a".to_string(), vec![0.1, 0.4, 0.35]);
        observations.insert("c".to_string(), vec![0.05, 0.3, 0.7]);
        observations.insert("b".to_string(), vec![1.0, 3.0, 4.0]);
        observations.insert("d".to_string(), vec![2.0, 4.0, 5.0]);
        observations.insert("e".to_string(), vec![0.0, 1.0, 2.0]);

        let mut search_space = HashMap::<String, Distribution>::new();
        search_space.insert(
            "a".to_string(),
            Distribution::new_float(0.01, 1.0, None, false),
        );
        search_space.insert(
            "b".to_string(),
            Distribution::new_float(0.01, 1.0, None, true),
        );
        search_space.insert("c".to_string(), Distribution::new_int(1, 5, 1, false));
        search_space.insert("d".to_string(), Distribution::new_int(1, 5, 1, true));
        search_space.insert("e".to_string(), Distribution::new_categorical(3));

        let parzen_estimator =
            ParzenEstimator::new(&observations, &search_space, &[0.2, 0.5, 0.3], 1.0);
        let streamed = ParzenEstimator::from_ordered_observations(
            ["a", "b", "c", "d", "e"].map(|name| {
                (
                    name,
                    &search_space[name],
                    observations[name].iter().copied(),
                )
            }),
            &[0.2, 0.5, 0.3],
            1.0,
        );
        let mut rng = StdRng::seed_from_u64(42);
        let samples = parzen_estimator.sample(&mut rng, 10);
        let mut ordered_rng = StdRng::seed_from_u64(42);
        let ordered = parzen_estimator.sample_ordered(&mut ordered_rng, 10);
        let mut streamed_rng = StdRng::seed_from_u64(42);
        assert_eq!(ordered, streamed.sample_ordered(&mut streamed_rng, 10));
        assert_eq!(
            parzen_estimator.parameter_names(),
            ["a", "b", "c", "d", "e"]
        );
        for (sample, values) in samples.iter().zip(&ordered) {
            assert_eq!(
                *values,
                ["a", "b", "c", "d", "e"].map(|name| sample[name]).to_vec()
            );
            assert_eq!(
                parzen_estimator.log_pdf(sample).to_bits(),
                parzen_estimator.log_pdf_ordered(values).to_bits()
            );
            assert_eq!(
                parzen_estimator.log_pdf_ordered(values).to_bits(),
                streamed.log_pdf_ordered(values).to_bits()
            );
        }
        assert_eq!(
            rand::RngCore::next_u64(&mut rng),
            rand::RngCore::next_u64(&mut ordered_rng)
        );
        assert_eq!(parzen_estimator.log_pdf_ordered(&[]), f64::NEG_INFINITY);
        assert_eq!(
            parzen_estimator.log_pdf_ordered(&[0.0; 6]),
            f64::NEG_INFINITY
        );
        let mut invalid = ordered[0].clone();
        invalid[0] = -1.0;
        assert_eq!(
            parzen_estimator.log_pdf_ordered(&invalid),
            f64::NEG_INFINITY
        );
        assert_eq!(samples.len(), 10);
        for sample in samples.iter() {
            let a = sample.get("a").unwrap();
            assert!(*a >= 0.01 && *a <= 1.0);
            let b = sample.get("b").unwrap();
            assert!(*b >= 0.01 && *b <= 1.0);
            let c = sample.get("c").unwrap();
            assert!(*c == 1.0 || *c == 2.0 || *c == 3.0 || *c == 4.0 || *c == 5.0);
            let d = sample.get("d").unwrap();
            assert!(*d == 1.0 || *d == 2.0 || *d == 3.0 || *d == 4.0 || *d == 5.0);
            let e = sample.get("e").unwrap();
            assert!(*e == 0.0 || *e == 1.0 || *e == 2.0);
        }
    }

    #[test]
    fn ordered_observations_keep_the_uniform_categorical_prior() {
        let distribution = Distribution::new_categorical(3);
        let model = ParzenEstimator::from_ordered_observations(
            [("x", &distribution, std::iter::empty())],
            &[],
            1.0,
        );
        for category in 0..3 {
            assert_eq!(
                model.log_pdf_ordered(&[category as f64]),
                (1.0_f64 / 3.0).ln()
            );
        }
    }

    #[test]
    #[should_panic(expected = "Observation columns and weights differ in length")]
    fn ordered_observations_reject_inconsistent_weights() {
        ParzenEstimator::from_ordered_observations(
            [("x", &Distribution::new_categorical(3), [1.0, 2.0])],
            &[1.0],
            1.0,
        );
    }

    #[test]
    #[should_panic(expected = "Parameter names must be sorted and unique")]
    fn ordered_observations_reject_unsorted_parameters() {
        let distribution = Distribution::new_categorical(3);
        ParzenEstimator::from_ordered_observations(
            [("b", &distribution, [0.0]), ("a", &distribution, [1.0])],
            &[1.0],
            1.0,
        );
    }
}
