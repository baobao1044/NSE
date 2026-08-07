//! Maximum Inner Product Search (MIPS) index.
//!
//! The POC uses exact brute-force MIPS (correct, O(N)). An HNSW/LSH index with
//! O(log N) lookup is scaffolded for the scale-out phase.
//!
//! Status: skeleton (M0). Brute-force index lands in M4.

use nse_core::tensor::Matrix;

/// A query against the index returns the top-K experts by inner product.
#[derive(Debug, Clone)]
pub struct Hit {
    pub expert_id: u32,
    pub score: f32,
}

/// Brute-force exact MIPS index over a centroid matrix `[N, dim]`.
pub struct MipsIndex {
    pub centroids: Matrix,
}

impl MipsIndex {
    pub fn new(centroids: Matrix) -> Self {
        Self { centroids }
    }

    /// Query the top-K experts by inner product with `query`. (Stub — M4.)
    pub fn query(&self, _query: &[f32], _k: usize) -> anyhow::Result<Vec<Hit>> {
        todo!("M4: brute-force MIPS query")
    }
}
