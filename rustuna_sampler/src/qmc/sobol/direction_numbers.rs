//! Joe-Kuo direction numbers for the Sobol' sequence.
//!
//! The table is the leading part of the `new-joe-kuo-6.21201` set published by Joe and Kuo,
//! selected with their search criterion 6. SciPy embeds all 21201 dimensions; Rustuna keeps the
//! first [`MAX_DIM`], where the numbers are identical to SciPy's.
//!
//! See <https://web.maths.unsw.edu.au/~fkuo/sobol/> and
//! S. Joe and F. Y. Kuo, "Constructing Sobol sequences with better two-dimensional projections",
//! SIAM Journal on Scientific Computing, 30(5):2635-2654, 2008.

/// Number of dimensions the embedded table supports.
pub const MAX_DIM: usize = 1024;

/// Largest primitive polynomial degree in the embedded table.
pub const MAX_DEGREE: usize = 13;

/// Byte width of the packed degree field.
const DEGREE_WIDTH: usize = 1;

/// Byte width of the packed polynomial field.
const POLY_WIDTH: usize = 4;

/// Bytes the `i`-th initial value occupies, which is `ceil(i / 8)` because `m_i < 2^i`.
fn m_width(i: usize) -> usize {
    i.div_ceil(8)
}

/// Packed table generated from the published `new-joe-kuo-6.21201` text file, truncated to
/// [`MAX_DIM`] dimensions. The script that produced it is at
/// <https://github.com/optuna/rustuna/pull/221#issuecomment-5353189930>.
///
/// Fields are little-endian and byte-aligned, with dimensions in ascending order and no padding
/// between them. For a dimension of degree `s` the layout is
///
/// ```text
/// s    : 1 byte
/// poly : 4 bytes
/// m_1  : 1 byte
/// m_2  : 1 byte
/// ...
/// m_9  : 2 bytes
/// ...
/// m_s  : ceil(s / 8) bytes
/// ```
///
/// Every `m_i` is below `2^i`, so narrowing it to `ceil(i / 8)` bytes keeps the table at 21 KiB
/// rather than the 76 KiB a fixed `u32` per value would need, without giving up byte alignment.
static PACKED: &[u8] = include_bytes!("joe_kuo_6.bin");

/// Direction numbers for a single dimension.
pub struct Entry {
    /// Primitive polynomial, with bit `i` set when the term `x^i` is present. This is the
    /// encoding used by Bratley and Fox and by SciPy.
    pub poly: u32,
    /// Degree of [`Entry::poly`], i.e. the number of valid entries in [`Entry::m`].
    pub degree: usize,
    /// Initial values `m_1..m_degree`; entries beyond `degree` are unspecified.
    pub m: [u32; MAX_DEGREE],
}

/// Reads little-endian integers of `width` bytes out of the packed table.
struct ByteReader {
    pos: usize,
}

impl ByteReader {
    fn read(&mut self, width: usize) -> u32 {
        let mut value = 0;
        for k in 0..width {
            value |= u32::from(PACKED[self.pos]) << (8 * k);
            self.pos += 1;
        }
        value
    }
}

/// Decodes the direction numbers for dimensions `2..=dim`, in ascending order.
///
/// Dimension 1 is not covered because it uses `m_i = 1` for every `i` and needs no table lookup.
/// The iterator therefore yields `dim - 1` entries. `dim` must not exceed [`MAX_DIM`].
pub fn decode(dim: usize) -> impl Iterator<Item = Entry> {
    debug_assert!(dim <= MAX_DIM);
    let mut reader = ByteReader { pos: 0 };

    (1..dim).map(move |_| {
        let degree = reader.read(DEGREE_WIDTH) as usize;
        let poly = reader.read(POLY_WIDTH);
        debug_assert!((1..=MAX_DEGREE).contains(&degree));

        let mut m = [0; MAX_DEGREE];
        for (i, value) in m.iter_mut().enumerate().take(degree) {
            *value = reader.read(m_width(i + 1));
        }
        Entry { poly, degree, m }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decodes_known_entries() {
        // Expected values are the rows of `new-joe-kuo-6.21201`, which are also what
        // `scipy.stats._sobol` loads from `_sobol_direction_numbers.npz`. The indices below are
        // zero-based dimension indices, so entry `i` of the iterator is dimension index `i + 1`.
        let entries: Vec<_> = decode(12).collect();

        // Dimension index 1: x + 1, m = [1].
        assert_eq!(entries[0].poly, 0b11);
        assert_eq!(entries[0].degree, 1);
        assert_eq!(&entries[0].m[..1], &[1]);

        // Dimension index 3: x^3 + x + 1, m = [1, 3, 1].
        assert_eq!(entries[2].poly, 0b1011);
        assert_eq!(entries[2].degree, 3);
        assert_eq!(&entries[2].m[..3], &[1, 3, 1]);

        // Dimension index 9: x^5 + x^3 + x^2 + x + 1, m = [1, 1, 7, 11, 19].
        assert_eq!(entries[8].poly, 0b101111);
        assert_eq!(entries[8].degree, 5);
        assert_eq!(&entries[8].m[..5], &[1, 1, 7, 11, 19]);
    }

    #[test]
    fn decodes_every_dimension_within_the_packed_table() {
        let mut count = 0;
        let mut previous_degree = 0;
        for entry in decode(MAX_DIM) {
            assert!(entry.degree >= 1 && entry.degree <= MAX_DEGREE);
            // The published table lists dimensions in order of non-decreasing degree. Checking
            // it here catches a bit stream that has drifted out of alignment.
            assert!(entry.degree >= previous_degree, "degrees must not decrease");
            previous_degree = entry.degree;

            for (i, &value) in entry.m[..entry.degree].iter().enumerate() {
                let i = i + 1;
                assert_eq!(value % 2, 1, "m_{i} must be odd");
                assert!(value < (1 << i), "m_{i} must be below 2^{i}");
            }
            count += 1;
        }
        assert_eq!(count, MAX_DIM - 1);
    }
}
