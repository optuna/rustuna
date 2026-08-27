//! Generator for the Sobol' sequence. See [`super`] for what the sequence guarantees.

use std::sync::LazyLock;

use rustuna_core::{Error, ErrorKind, Result};

use super::direction_numbers::{self, MAX_DIM};

/// Number of fixed-point bits behind each coordinate, matching `scipy.stats.qmc.Sobol`.
const BITS: usize = 30;

/// Number of points the sequence holds before it would start repeating.
const CAPACITY: u64 = 1 << BITS;

/// Direction numbers for every supported dimension, built once per process.
///
/// Row `d` depends only on dimension `d`, so the first `dim` rows are what a `dim`-dimensional
/// sequence needs. Building all of them costs 120 KiB once, which beats rebuilding a smaller
/// table whenever the dimension changes, and the table needs no synchronization because it never
/// changes after initialization.
static DIRECTIONS: LazyLock<Vec<[u32; BITS]>> =
    LazyLock::new(|| initialize_direction_numbers(MAX_DIM));

/// Returns the point at index `n` of the `dim`-dimensional Sobol' sequence, where index 0 is the
/// origin.
///
/// This evaluates the Gray code definition of the sequence directly rather than iterating the
/// recurrence, so the cost does not grow with `n`.
pub fn nth_point(dim: usize, n: u64) -> Result<Vec<f64>> {
    if dim == 0 || dim > MAX_DIM {
        return Err(Error::with_reason(
            ErrorKind::SamplerError,
            format!("Sobol' dimension must be in 1..={MAX_DIM}, got {dim}"),
        ));
    }
    if n >= CAPACITY {
        return Err(Error::with_reason(
            ErrorKind::SamplerError,
            format!("Sobol' point index must be below 2**{BITS}={CAPACITY}, got {n}"),
        ));
    }

    Ok(DIRECTIONS[..dim]
        .iter()
        .map(|row| scale(gray_code_xor(row, n)))
        .collect())
}

/// Exclusive-ors the direction numbers selected by the set bits of the Gray code of `n`.
fn gray_code_xor(row: &[u32; BITS], n: u64) -> u32 {
    // The Gray code of an index below `CAPACITY` is also below it, so `j` stays within `row`.
    debug_assert!(n < CAPACITY);

    let mut quasi = 0;
    let mut gray = n ^ (n >> 1);
    let mut j = 0;
    while gray != 0 {
        if gray & 1 == 1 {
            quasi ^= row[j];
        }
        gray >>= 1;
        j += 1;
    }
    quasi
}

/// Turns a `BITS`-wide fixed-point fraction into a `f64` in `[0, 1)`.
fn scale(quasi: u32) -> f64 {
    f64::from(quasi) * (1.0 / CAPACITY as f64)
}

/// Builds the direction numbers for `dim` dimensions as `BITS`-wide fixed-point fractions.
///
/// This mirrors SciPy's `_initialize_v`. The initial values `m_1..m_s` come from the Joe-Kuo
/// table; the rest follow the recurrence induced by the dimension's primitive polynomial
/// `x^s + a_1 x^(s-1) + ... + a_(s-1) x + 1`:
///
/// ```text
/// m_i = 2 a_1 m_(i-1) XOR 4 a_2 m_(i-2) XOR ... XOR 2^(s-1) a_(s-1) m_(i-s+1)
///       XOR 2^s m_(i-s) XOR m_(i-s)
/// ```
fn initialize_direction_numbers(dim: usize) -> Vec<[u32; BITS]> {
    let mut sv = vec![[0u32; BITS]; dim];

    // Dimension 0 is the van der Corput sequence, which uses m_i = 1 for every i and needs no
    // primitive polynomial.
    sv[0].fill(1);

    for (index, entry) in direction_numbers::decode(dim).enumerate() {
        let row = &mut sv[index + 1];
        let degree = entry.degree;

        for (j, slot) in row.iter_mut().enumerate().take(degree.min(BITS)) {
            *slot = entry.m[j];
        }
        for j in degree..BITS {
            let mut value = row[j - degree];
            let mut pow2 = 1;
            for k in 0..degree {
                pow2 <<= 1;
                if (entry.poly >> (degree - 1 - k)) & 1 == 1 {
                    value ^= pow2 * row[j - k - 1];
                }
            }
            row[j] = value;
        }
    }

    // Left-align every m_j so that the stored integer is the direction number times 2^BITS.
    for row in &mut sv {
        for (j, value) in row.iter_mut().enumerate() {
            *value <<= BITS - 1 - j;
        }
    }
    sv
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Returns the points at indices `0..n`.
    fn first_points(dim: usize, n: u64) -> Result<Vec<Vec<f64>>> {
        (0..n).map(|index| nth_point(dim, index)).collect()
    }

    #[test]
    fn matches_scipy_in_two_dimensions() -> Result<()> {
        // scipy.stats.qmc.Sobol(d=2, scramble=False).random(16)
        let expected = [
            [0.0, 0.0],
            [0.5, 0.5],
            [0.75, 0.25],
            [0.25, 0.75],
            [0.375, 0.375],
            [0.875, 0.875],
            [0.625, 0.125],
            [0.125, 0.625],
            [0.1875, 0.3125],
            [0.6875, 0.8125],
            [0.9375, 0.0625],
            [0.4375, 0.5625],
            [0.3125, 0.1875],
            [0.8125, 0.6875],
            [0.5625, 0.4375],
            [0.0625, 0.9375],
        ];
        let points = first_points(2, expected.len() as u64)?;
        assert_eq!(points, expected.map(|point| point.to_vec()));
        Ok(())
    }

    #[test]
    fn rejects_unsupported_dimensions() {
        assert!(nth_point(0, 0).is_err());
        assert!(nth_point(MAX_DIM + 1, 0).is_err());
        assert!(nth_point(MAX_DIM, 0).is_ok());
    }

    #[test]
    fn refuses_indices_past_the_end_of_the_sequence() -> Result<()> {
        assert_eq!(nth_point(2, CAPACITY - 1)?.len(), 2);
        assert!(nth_point(2, CAPACITY).is_err());
        Ok(())
    }
}
