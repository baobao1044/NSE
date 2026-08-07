//! Locality-Sensitive Hashing (random hyperplane LSH) for sparse weight
//! selection during training.
//!
//! LSH maps a vector `v` to a `b`-bit hash by taking the sign of `v . h_i` for
//! `b` random hyperplanes `h_i`. Vectors with high inner product tend to share
//! the same bucket (with probability `1 - θ/π` for angle `θ`). Used by the
//! LSH-sparse trainer to identify, for each training step, the weight rows
//! whose *input activation* falls in the same bucket as the activation seen
//! by a matmul — those are the "relevant" rows to update; the rest stay frozen.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// A random-hyperplane LSH family.
pub struct LshIndex {
    /// Hyperplanes: `num_bits` vectors of length `dim`.
    hyperplanes: Vec<Vec<f32>>,
    num_bits: usize,
}

impl LshIndex {
    /// Build `num_bits` random hyperplanes in `dim` dimensions, seeded by
    /// `seed`. Each entry is Gaussian-ish via sum of uniforms (no `rand_distr`
    /// dependency needed for the POC; a sum of 4 uniforms is a fine
    /// approximation of a Gaussian for hashing).
    pub fn new(dim: usize, num_bits: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let hyperplanes: Vec<Vec<f32>> = (0..num_bits)
            .map(|_| {
                (0..dim)
                    .map(|_| {
                        // Sum of 4 uniforms in [-1,1] approximates N(0, 4/3).
                        let mut s = 0.0;
                        for _ in 0..4 {
                            s += rng.gen::<f32>() * 2.0 - 1.0;
                        }
                        s
                    })
                    .collect()
            })
            .collect();
        Self { hyperplanes, num_bits }
    }

    /// Hash a vector to `num_bits` bits, packed into the low bits of a `u32`.
    pub fn hash(&self, v: &[f32]) -> u32 {
        let mut h = 0u32;
        for (i, hp) in self.hyperplanes.iter().enumerate() {
            let dot: f32 = v.iter().zip(hp.iter()).map(|(a, b)| a * b).sum();
            if dot >= 0.0 {
                h |= 1 << i;
            }
        }
        h
    }

    /// Number of hash bits.
    pub fn num_bits(&self) -> usize {
        self.num_bits
    }

    /// Expected number of buckets = `2^num_bits`. The trainer picks `num_bits`
    /// so that `2^num_bits ≈ 1 / sparse_fraction` (so each bucket holds about
    /// `sparse_fraction` of the rows on average).
    pub fn num_buckets(&self) -> usize {
        1 << self.num_bits
    }
}
