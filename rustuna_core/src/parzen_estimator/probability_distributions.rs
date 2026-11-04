use super::truncnorm;
use rand::rngs::StdRng;
use rand::Rng;
use rand_distr::{Distribution as RandDistribution, WeightedAliasIndex};
use std::collections::HashMap;

use super::truncnorm::NEG_HALF_LOG_2PI;

#[derive(Debug, Clone)]
pub(crate) struct TruncNormDistributions {
    pub mus: Vec<f64>,
    pub sigmas: Vec<f64>,
    pub low: f64,
    pub high: f64,
    /// Precomputed: NEG_HALF_LOG_2PI - ln(sigma_k) - ln_mass_k, i.e. every term that does not
    /// depend on x. `-inf` for a kernel with no probability mass.
    log_consts: Vec<f64>,
}

impl TruncNormDistributions {
    pub(crate) fn new(mus: Vec<f64>, sigmas: Vec<f64>, low: f64, high: f64) -> Self {
        let log_consts = mus
            .iter()
            .zip(sigmas.iter())
            .map(|(&mu, &sigma)| {
                let ln_mass = truncnorm::log_diff_cdf((low - mu) / sigma, (high - mu) / sigma)
                    .unwrap_or(f64::NEG_INFINITY);
                if ln_mass == f64::NEG_INFINITY {
                    f64::NEG_INFINITY
                } else {
                    NEG_HALF_LOG_2PI - sigma.ln() - ln_mass
                }
            })
            .collect();
        Self {
            mus,
            sigmas,
            low,
            high,
            log_consts,
        }
    }

    /// Adds each kernel's log PDF at x into `acc`.
    pub(crate) fn accumulate_log_pdf(&self, x: f64, acc: &mut [f64]) {
        for (k, a) in acc.iter_mut().enumerate() {
            let z = (x - self.mus[k]) / self.sigmas[k];
            *a += self.log_consts[k] - 0.5 * z * z;
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct TruncLogNormDistributions {
    pub mus: Vec<f64>,
    pub sigmas: Vec<f64>,
    pub low: f64,
    pub high: f64,
    /// Precomputed: NEG_HALF_LOG_2PI - ln(sigma_k) - ln_mass_k, i.e. every term that does not
    /// depend on x. `-inf` for a kernel with no probability mass.
    log_consts: Vec<f64>,
}

impl TruncLogNormDistributions {
    pub(crate) fn new(mus: Vec<f64>, sigmas: Vec<f64>, low: f64, high: f64) -> Self {
        let ln_low = low.ln();
        let ln_high = high.ln();
        let log_consts = mus
            .iter()
            .zip(sigmas.iter())
            .map(|(&mu, &sigma)| {
                let ln_mass =
                    truncnorm::log_diff_cdf((ln_low - mu) / sigma, (ln_high - mu) / sigma)
                        .unwrap_or(f64::NEG_INFINITY);
                if ln_mass == f64::NEG_INFINITY {
                    f64::NEG_INFINITY
                } else {
                    NEG_HALF_LOG_2PI - sigma.ln() - ln_mass
                }
            })
            .collect();
        Self {
            mus,
            sigmas,
            low,
            high,
            log_consts,
        }
    }

    /// Adds each kernel's log PDF at ln_x into `acc`.
    pub(crate) fn accumulate_log_pdf(&self, ln_x: f64, acc: &mut [f64]) {
        for (k, a) in acc.iter_mut().enumerate() {
            let z = (ln_x - self.mus[k]) / self.sigmas[k];
            *a += self.log_consts[k] - 0.5 * z * z - ln_x;
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct DiscreteTruncNormDistributions {
    pub mus: Vec<f64>,
    pub sigmas: Vec<f64>,
    pub low: f64,
    pub high: f64,
    pub step: f64,
    /// Precomputed: (low - mu_k) / sigma_k
    low_truncs: Vec<f64>,
    /// Precomputed: (high - mu_k) / sigma_k
    high_truncs: Vec<f64>,
    /// Precomputed: step / (2 * sigma_k)
    half_steps: Vec<f64>,
    /// Precomputed: log_diff_cdf(low_trunc_k, high_trunc_k)
    ln_denoms: Vec<f64>,
}

impl DiscreteTruncNormDistributions {
    pub(crate) fn new(mus: Vec<f64>, sigmas: Vec<f64>, low: f64, high: f64, step: f64) -> Self {
        let low_truncs: Vec<_> = mus
            .iter()
            .zip(sigmas.iter())
            .map(|(&mu, &sigma)| (low - mu) / sigma)
            .collect();
        let high_truncs: Vec<_> = mus
            .iter()
            .zip(sigmas.iter())
            .map(|(&mu, &sigma)| (high - mu) / sigma)
            .collect();
        let half_steps = sigmas.iter().map(|&sigma| step / (2.0 * sigma)).collect();
        let ln_denoms = low_truncs
            .iter()
            .zip(high_truncs.iter())
            .map(|(&a, &b)| truncnorm::log_diff_cdf(a, b).unwrap_or(f64::NEG_INFINITY))
            .collect();
        Self {
            mus,
            sigmas,
            low,
            high,
            step,
            low_truncs,
            high_truncs,
            half_steps,
            ln_denoms,
        }
    }

    /// Log PDF of the k-th kernel at x_val (without bounds check).
    pub(crate) fn log_pdf(&self, x_val: f64, k: usize) -> f64 {
        let ln_denom = self.ln_denoms[k];
        if ln_denom == f64::NEG_INFINITY {
            return f64::NEG_INFINITY;
        }
        let center = (x_val - self.mus[k]) / self.sigmas[k];
        let a = if x_val <= self.low {
            f64::NEG_INFINITY
        } else {
            center - self.half_steps[k]
        };
        let b = if x_val >= self.high {
            f64::INFINITY
        } else {
            center + self.half_steps[k]
        };
        if b <= self.low_truncs[k] || a >= self.high_truncs[k] {
            return f64::NEG_INFINITY;
        }
        let a_adj = a.max(self.low_truncs[k]);
        let b_adj = b.min(self.high_truncs[k]);
        if a_adj >= b_adj {
            return f64::NEG_INFINITY;
        }
        match truncnorm::log_diff_cdf(a_adj, b_adj) {
            Ok(ln_numer) => ln_numer - ln_denom,
            Err(_) => f64::NEG_INFINITY,
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct DiscreteTruncLogNormDistributions {
    pub mus: Vec<f64>,
    pub sigmas: Vec<f64>,
    pub low: f64,
    pub high: f64,
    pub step: f64,
    /// Precomputed: (ln(low) - mu_k) / sigma_k
    low_truncs: Vec<f64>,
    /// Precomputed: (ln(high) - mu_k) / sigma_k
    high_truncs: Vec<f64>,
    /// Precomputed: log_diff_cdf(low_trunc_k, high_trunc_k)
    ln_denoms: Vec<f64>,
}

impl DiscreteTruncLogNormDistributions {
    pub(crate) fn new(mus: Vec<f64>, sigmas: Vec<f64>, low: f64, high: f64, step: f64) -> Self {
        let ln_low = low.ln();
        let ln_high = high.ln();
        let low_truncs: Vec<_> = mus
            .iter()
            .zip(sigmas.iter())
            .map(|(&mu, &sigma)| (ln_low - mu) / sigma)
            .collect();
        let high_truncs: Vec<_> = mus
            .iter()
            .zip(sigmas.iter())
            .map(|(&mu, &sigma)| (ln_high - mu) / sigma)
            .collect();
        let ln_denoms = low_truncs
            .iter()
            .zip(high_truncs.iter())
            .map(|(&a, &b)| truncnorm::log_diff_cdf(a, b).unwrap_or(f64::NEG_INFINITY))
            .collect();
        Self {
            mus,
            sigmas,
            low,
            high,
            step,
            low_truncs,
            high_truncs,
            ln_denoms,
        }
    }

    /// Log PDF of the k-th kernel at x_val (without bounds check).
    pub(crate) fn log_pdf(&self, x_val: f64, k: usize) -> f64 {
        let ln_denom = self.ln_denoms[k];
        if ln_denom == f64::NEG_INFINITY {
            return f64::NEG_INFINITY;
        }
        let low_bound = (x_val - self.step / 2.0).max(f64::MIN_POSITIVE);
        let high_bound = x_val + self.step / 2.0;
        let a = if x_val <= self.low {
            f64::NEG_INFINITY
        } else {
            (low_bound.ln() - self.mus[k]) / self.sigmas[k]
        };
        let b = if x_val >= self.high {
            f64::INFINITY
        } else {
            (high_bound.ln() - self.mus[k]) / self.sigmas[k]
        };
        if b <= self.low_truncs[k] || a >= self.high_truncs[k] {
            return f64::NEG_INFINITY;
        }
        let a_adj = a.max(self.low_truncs[k]);
        let b_adj = b.min(self.high_truncs[k]);
        if a_adj >= b_adj {
            return f64::NEG_INFINITY;
        }
        match truncnorm::log_diff_cdf(a_adj, b_adj) {
            Ok(ln_numer) => ln_numer - ln_denom,
            Err(_) => f64::NEG_INFINITY,
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct CategoricalDistributions {
    // Each observation contributes a uniform mass plus a spike at its category.
    // The final kernel is the uniform prior. Joint mixtures retain row identity.
    observed_categories: Vec<usize>,
    cardinality: usize,
    matching: f64,
    other: f64,
}

impl CategoricalDistributions {
    pub(crate) fn new(observations: impl IntoIterator<Item = f64>, cardinality: usize) -> Self {
        assert!(cardinality > 0, "Categorical cardinality must be positive");
        let observations = observations.into_iter();
        let (lower, upper) = observations.size_hint();
        let mut observed_categories = Vec::with_capacity(upper.unwrap_or(lower));
        for value in observations {
            let category = value as usize;
            assert!(category < cardinality, "Observed category is out of range");
            observed_categories.push(category);
        }
        let prior_mass = 1.0 / (observed_categories.len() + 1) as f64;
        let denominator = 1.0 + cardinality as f64 * prior_mass;
        let matching = (1.0 + prior_mass) / denominator;
        let other = prior_mass / denominator;
        Self {
            observed_categories,
            cardinality,
            matching,
            other,
        }
    }

    fn probability(&self, kernel: usize, category: usize) -> f64 {
        match self.observed_categories.get(kernel) {
            Some(&observed) if observed == category => self.matching,
            Some(_) => self.other,
            None => 1.0 / self.cardinality as f64,
        }
    }

    fn accumulate_log_pdf(&self, category: usize, scores: &mut [f64]) {
        debug_assert_eq!(scores.len(), self.observed_categories.len() + 1);
        let log_matching = self.matching.ln();
        let log_other = self.other.ln();
        for (&observed, score) in self.observed_categories.iter().zip(scores.iter_mut()) {
            *score += if observed == category {
                log_matching
            } else {
                log_other
            };
        }
        *scores.last_mut().unwrap() += (1.0 / self.cardinality as f64).ln();
    }
}

#[derive(Debug, Clone)]
pub(crate) enum Distributions {
    TruncNorm(TruncNormDistributions),
    TruncLogNorm(TruncLogNormDistributions),
    DiscreteTruncNorm(DiscreteTruncNormDistributions),
    DiscreteTruncLogNorm(DiscreteTruncLogNormDistributions),
    Categorical(CategoricalDistributions),
}

impl Distributions {
    pub(crate) fn n_kernels(&self) -> usize {
        match self {
            Self::TruncNorm(d) => d.mus.len(),
            Self::TruncLogNorm(d) => d.mus.len(),
            Self::DiscreteTruncNorm(d) => d.mus.len(),
            Self::DiscreteTruncLogNorm(d) => d.mus.len(),
            Self::Categorical(d) => d.observed_categories.len() + 1,
        }
    }
}

#[derive(Debug)]
pub(crate) struct MixtureOfProductDistribution {
    pub param_names: Vec<String>,          // Sorted param names
    pub distributions: Vec<Distributions>, // Sorted distributions
    pub log_weights: Vec<f64>,             // ln(w_i)
    pub log_sum_weights: f64,              // ln(sum weights)
    pub alias: WeightedAliasIndex<f64>,
    pub n_kernels: usize,
}

impl MixtureOfProductDistribution {
    pub fn new(entries: Vec<(String, Distributions)>, weights: Vec<f64>) -> Self {
        let sum_w = weights.iter().sum::<f64>();
        let log_sum_weights = if sum_w > 0.0 {
            sum_w.ln()
        } else {
            0.0_f64.ln()
        };
        let log_weights = weights
            .iter()
            .map(|&w| if w > 0.0 { w.ln() } else { f64::NEG_INFINITY })
            .collect::<Vec<_>>();

        let n_kernels = weights.len();
        let alias =
            WeightedAliasIndex::new(weights).expect("weights must be non-empty and non-negative");

        let (param_names, distributions) = entries.into_iter().unzip();

        MixtureOfProductDistribution {
            param_names,
            distributions,
            log_weights,
            log_sum_weights,
            alias,
            n_kernels,
        }
    }

    pub fn sample(&self, rng: &mut StdRng, size: usize) -> Vec<HashMap<String, f64>> {
        self.sample_ordered(rng, size)
            .into_iter()
            .map(|values| self.param_names.iter().cloned().zip(values).collect())
            .collect()
    }

    pub fn sample_ordered(&self, rng: &mut StdRng, size: usize) -> Vec<Vec<f64>> {
        let mut samples: Vec<Vec<f64>> = Vec::with_capacity(size);

        for _ in 0..size {
            let k = self.alias.sample(rng); // Active kernel index
            let mut sample = Vec::with_capacity(self.param_names.len());
            for (param, dist) in self.param_names.iter().zip(self.distributions.iter()) {
                match dist {
                    Distributions::TruncNorm(d) => {
                        let mu = d.mus[k];
                        let sigma = d.sigmas[k];
                        let value = truncnorm::rvs(
                            rng,
                            (d.low - mu) / sigma,
                            (d.high - mu) / sigma,
                            mu,
                            sigma,
                        )
                        .unwrap();
                        sample.push(value);
                    }
                    Distributions::TruncLogNorm(d) => {
                        let mu = d.mus[k];
                        let sigma = d.sigmas[k];
                        let log_value = truncnorm::rvs(
                            rng,
                            (d.low.ln() - mu) / sigma,
                            (d.high.ln() - mu) / sigma,
                            mu,
                            sigma,
                        )
                        .unwrap();
                        sample.push(log_value.exp());
                    }
                    Distributions::DiscreteTruncNorm(d) => {
                        let mu = d.mus[k];
                        let sigma = d.sigmas[k];
                        let value = truncnorm::rvs(
                            rng,
                            (d.low - d.step / 2.0 - mu) / sigma,
                            (d.high + d.step / 2.0 - mu) / sigma,
                            mu,
                            sigma,
                        )
                        .unwrap();
                        let discrete_value = (d.low + ((value - d.low) / d.step).round() * d.step)
                            .max(d.low)
                            .min(d.high);
                        sample.push(discrete_value);
                    }
                    Distributions::DiscreteTruncLogNorm(d) => {
                        let mu = d.mus[k];
                        let sigma = d.sigmas[k];
                        let log_value = truncnorm::rvs(
                            rng,
                            ((d.low - d.step / 2.0).max(f64::MIN_POSITIVE).ln() - mu) / sigma,
                            ((d.high + d.step / 2.0).max(f64::MIN_POSITIVE).ln() - mu) / sigma,
                            mu,
                            sigma,
                        )
                        .unwrap();
                        let original = log_value.exp();
                        let discrete_value = (d.low
                            + ((original - d.low) / d.step).round() * d.step)
                            .max(d.low)
                            .min(d.high);
                        sample.push(discrete_value);
                    }
                    Distributions::Categorical(d) => {
                        let sum: f64 = (0..d.cardinality)
                            .map(|category| d.probability(k, category))
                            .sum();
                        assert!(
                            sum > 0.0,
                            "Categorical distribution has non-positive total probability for param {param}"
                        );

                        let u = rng.gen::<f64>() * sum;
                        let mut cum = 0.0;
                        let mut chosen = (d.cardinality - 1) as f64; // fallback
                        for category in 0..d.cardinality {
                            cum += d.probability(k, category);
                            if u <= cum {
                                chosen = category as f64;
                                break;
                            }
                        }
                        sample.push(chosen);
                    }
                }
            }
            samples.push(sample);
        }

        samples
    }

    pub fn log_pdf(&self, x: &HashMap<String, f64>) -> f64 {
        self.log_pdf_values(self.param_names.iter().map(|name| x.get(name).copied()))
    }

    pub fn log_pdf_ordered(&self, values: &[f64]) -> f64 {
        if values.len() != self.distributions.len() {
            return f64::NEG_INFINITY;
        }
        self.log_pdf_values(values.iter().copied().map(Some))
    }

    fn log_pdf_values(&self, values: impl Iterator<Item = Option<f64>>) -> f64 {
        let n = self.n_kernels;
        let mut weighted_log_pdf = vec![0.0_f64; n];

        for (dist, value) in self.distributions.iter().zip(values) {
            let Some(x_val) = value else {
                return f64::NEG_INFINITY;
            };

            match dist {
                Distributions::TruncNorm(d) => {
                    if x_val < d.low || x_val > d.high {
                        return f64::NEG_INFINITY;
                    }
                    d.accumulate_log_pdf(x_val, &mut weighted_log_pdf);
                }
                Distributions::TruncLogNorm(d) => {
                    if x_val <= 0.0 || x_val < d.low || x_val > d.high {
                        return f64::NEG_INFINITY;
                    }
                    d.accumulate_log_pdf(x_val.ln(), &mut weighted_log_pdf);
                }
                Distributions::DiscreteTruncNorm(d) => {
                    for (k, weight) in weighted_log_pdf.iter_mut().enumerate().take(n) {
                        if *weight == f64::NEG_INFINITY {
                            continue;
                        }
                        let lp = d.log_pdf(x_val, k);
                        if lp == f64::NEG_INFINITY {
                            *weight = f64::NEG_INFINITY;
                        } else {
                            *weight += lp;
                        }
                    }
                }
                Distributions::DiscreteTruncLogNorm(d) => {
                    if x_val <= 0.0 {
                        return f64::NEG_INFINITY;
                    }
                    for (k, weight) in weighted_log_pdf.iter_mut().enumerate().take(n) {
                        if *weight == f64::NEG_INFINITY {
                            continue;
                        }
                        let lp = d.log_pdf(x_val, k);
                        if lp == f64::NEG_INFINITY {
                            *weight = f64::NEG_INFINITY;
                        } else {
                            *weight += lp;
                        }
                    }
                }
                Distributions::Categorical(d) => {
                    let xi = x_val as usize;
                    if xi >= d.cardinality {
                        return f64::NEG_INFINITY;
                    }
                    d.accumulate_log_pdf(xi, &mut weighted_log_pdf);
                }
            }
        }

        // Add log weights
        for (k, weight) in weighted_log_pdf.iter_mut().enumerate().take(n) {
            let lw = self.log_weights[k];
            if lw.is_infinite() && lw.is_sign_negative() {
                *weight = f64::NEG_INFINITY;
            } else {
                *weight += lw;
            }
        }

        // Log-sum-exp across kernels
        let max = weighted_log_pdf
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        if max.is_infinite() && max.is_sign_negative() {
            // All -inf -> return -inf

            return f64::NEG_INFINITY;
        }
        let sum_exp: f64 = weighted_log_pdf.iter().map(|v| (v - max).exp()).sum();
        // Weights are basically normalized, but sum to 1 may not hold due to float rounding error
        (max + sum_exp.ln()) - self.log_sum_weights // Normalize by total weight to avoid float rounding error
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    #[test]
    fn test_mixture_of_product_distribution() {
        let truncnorm_dist = TruncNormDistributions::new(
            vec![-0.5, 0.0], // mus[-1] is prior
            vec![2.0, 1.0],  // sigma[-1] is prior
            -1.0,
            1.0,
        );
        let trunclognorm_dist = TruncLogNormDistributions::new(
            vec![2.0, 3.0], // ditto
            vec![2.0, 1.0], // ditto
            1.0,
            5.0,
        );
        let discrete_truncnorm_dist = DiscreteTruncNormDistributions::new(
            vec![-0.5, 0.0], // ditto
            vec![1.0, 1.0],  // ditto
            -1.0,
            1.0,
            1.0,
        );
        let discrete_trunclognorm_dist = DiscreteTruncLogNormDistributions::new(
            vec![2.0, 3.0], // ditto
            vec![1.0, 1.0], // ditto
            1.0,
            5.0,
            1.0,
        );
        let categorical_dist = CategoricalDistributions::new([0.0], 2);
        let mut distributions = vec![
            (
                "param_truncnorm".to_string(),
                Distributions::TruncNorm(truncnorm_dist),
            ),
            (
                "param_trunclognorm".to_string(),
                Distributions::TruncLogNorm(trunclognorm_dist),
            ),
            (
                "param_discretetruncnorm".to_string(),
                Distributions::DiscreteTruncNorm(discrete_truncnorm_dist),
            ),
            (
                "param_discretetrunclognorm".to_string(),
                Distributions::DiscreteTruncLogNorm(discrete_trunclognorm_dist),
            ),
            (
                "param_categorical".to_string(),
                Distributions::Categorical(categorical_dist),
            ),
        ];
        distributions.sort_by(|a, b| a.0.cmp(&b.0));
        let mixture = MixtureOfProductDistribution::new(
            distributions,
            vec![0.5, 0.5], // weights.len() == mus.len()
        );
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let samples = mixture.sample(&mut rng, 10);
        for sample in samples.iter() {
            let val_truncnorm = sample.get("param_truncnorm").unwrap();
            assert!(*val_truncnorm >= -1.0 && *val_truncnorm <= 1.0);
            let val_trunclognorm = sample.get("param_trunclognorm").unwrap();
            assert!(*val_trunclognorm >= 1.0 && *val_trunclognorm <= 5.0);
            let val_discretetruncnorm = sample.get("param_discretetruncnorm").unwrap();
            assert!(*val_discretetruncnorm >= -1.0 && *val_discretetruncnorm <= 1.0);
            let val_discretetrunclognorm = sample.get("param_discretetrunclognorm").unwrap();
            assert!(*val_discretetrunclognorm >= 1.0 && *val_discretetrunclognorm <= 5.0);
            let val_categorical = sample.get("param_categorical").unwrap();
            assert!(*val_categorical == 0.0 || *val_categorical == 1.0);
        }
    }

    #[test]
    fn categorical_kernels_keep_smoothed_probabilities_and_uniform_prior() {
        let distribution = CategoricalDistributions::new([1.0, 2.0], 3);
        let expected = [
            [1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0],
            [1.0 / 6.0, 1.0 / 6.0, 2.0 / 3.0],
            [1.0 / 3.0; 3],
        ];
        for (category, _) in expected[0].iter().enumerate() {
            let mut scores = vec![0.0; 3];
            distribution.accumulate_log_pdf(category, &mut scores);
            for (kernel, (row, score)) in expected.iter().zip(scores).enumerate() {
                assert!((distribution.probability(kernel, category) - row[category]).abs() < 1e-15);
                assert!((score - row[category].ln()).abs() < 1e-15);
            }
        }
        for cardinality in [1, 3, 257] {
            let prior = CategoricalDistributions::new([], cardinality);
            let singleton = CategoricalDistributions::new([0.0; 10], 1);
            let mut scores = vec![0.0; 1];
            prior.accumulate_log_pdf(0, &mut scores);
            assert_eq!(prior.probability(0, 0), 1.0 / cardinality as f64);
            assert_eq!(scores[0], (1.0 / cardinality as f64).ln());
            assert_eq!(singleton.probability(0, 0), 1.0);
        }
    }

    #[test]
    fn categorical_joint_density_keeps_observation_correlations_and_weights() {
        let mixture = MixtureOfProductDistribution::new(
            vec![
                (
                    "x".into(),
                    Distributions::Categorical(CategoricalDistributions::new([0.0, 1.0], 2)),
                ),
                (
                    "y".into(),
                    Distributions::Categorical(CategoricalDistributions::new([0.0, 1.0], 2)),
                ),
            ],
            vec![1.0, 2.0, 0.5],
        );
        let matching = 4.0_f64 / 5.0;
        let other = 1.0_f64 / 5.0;
        for (x, y) in [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)] {
            let likelihood = (if x == 0.0 { matching } else { other })
                * (if y == 0.0 { matching } else { other })
                + 2.0
                    * (if x == 1.0 { matching } else { other })
                    * (if y == 1.0 { matching } else { other })
                + 0.5 * 0.25;
            let actual = mixture.log_pdf(&HashMap::from([("x".into(), x), ("y".into(), y)]));
            assert!((actual - (likelihood / 3.5).ln()).abs() < 1e-14);
        }
        assert_eq!(
            mixture.log_pdf(&HashMap::from([("x".into(), 2.0), ("y".into(), 0.0)])),
            f64::NEG_INFINITY
        );
        assert_eq!(
            mixture.log_pdf(&HashMap::from([("x".into(), 0.0)])),
            f64::NEG_INFINITY
        );
        let mut rng = StdRng::seed_from_u64(42);
        let samples = mixture.sample(&mut rng, 20_000);
        let equal = samples
            .iter()
            .filter(|sample| sample["x"] == sample["y"])
            .count() as f64
            / samples.len() as f64;
        // Both observed kernels give P(x=y)=0.8^2+0.2^2; the prior gives 0.5.
        let expected_equal = (3.0 * (matching * matching + other * other) + 0.5 * 0.5) / 3.5;
        assert!((equal - expected_equal).abs() < 0.015);
    }
}
