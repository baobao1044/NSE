//! Product Quantization (PQ) codebook (performance scaffold).
//!
//! PQ compresses each weight vector into a short code of sub-vector indices,
//! each pointing into a shared codebook of < 1 MB (per spec, kept on L3
//! cache). At inference, `_mm256_shuffle_epi8` looks up the codebook values
//! directly from L1 — no RAM access needed.
//!
//! The POC uses ternary quantization ([`crate::quantize`]) which is simpler
//! and sufficient for the PPL measurement. PQ offers tighter compression at
//! a small accuracy cost; it's the production path for the spec's 2.7T model.
//!
//! Status: scaffold (M6). Implementation deferred to the production phase.

#![allow(dead_code)]

/// A trained PQ codebook: `M` sub-codebooks, each with `2^nbits` centroids.
pub struct PqCodebook {
    /// Number of sub-vectors.
    pub num_sub_vectors: usize,
    /// Bits per code (typically 8 → 256 entries per sub-codebook).
    pub nbits: usize,
    /// Sub-vector dimensionality (= in_dim / num_sub_vectors).
    pub sub_dim: usize,
    /// Codebook data: `num_sub_vectors * (2^nbits) * sub_dim` floats.
    pub codebook: Vec<f32>,
}

/// Encode a weight vector into PQ codes (one byte per sub-vector for nbits=8).
///
/// (Stub — implemented in the production phase.)
pub fn encode_pq(_row: &[f32], _codebook: &PqCodebook) -> Vec<u8> {
    todo!("production phase: PQ encode")
}

/// Decode PQ codes back to an approximate weight vector.
///
/// (Stub — implemented in the production phase.)
pub fn decode_pq(_codes: &[u8], _codebook: &PqCodebook) -> Vec<f32> {
    todo!("production phase: PQ decode")
}

/// Train a PQ codebook from a set of weight vectors using k-means per
/// sub-vector.
///
/// (Stub — implemented in the production phase.)
pub fn train_pq(_weights: &[Vec<f32>], num_sub_vectors: usize, nbits: usize) -> PqCodebook {
    let _ = num_sub_vectors;
    let _ = nbits;
    todo!("production phase: PQ codebook training via sub-vector k-means")
}
