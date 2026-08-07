//! Micro-expert clustering (ZSTM stage 2).
//!
//! Partitions the residual weight columns into cache-sized micro-experts via
//! spherical k-means (or SVD). Clustering only permutes column indices; the
//! numeric values are untouched, so the transform is exact up to permutation.
//!
//! Status: skeleton (M0). Real spherical k-means lands in M3.

use nse_core::tensor::Matrix;

/// Target size of one micro-expert, chosen to fit L1/L2 cache (per spec).
pub const MICRO_EXPERT_TARGET_BYTES: usize = 64 * 1024; // 64 KB

#[derive(Debug, Clone)]
pub struct ClusterConfig {
    /// Number of micro-experts (`N`). If 0, derived from target size.
    pub num_experts: usize,
}

impl Default for ClusterConfig {
    fn default() -> Self {
        Self { num_experts: 0 }
    }
}

#[derive(Debug, Clone)]
pub struct ClusterResult {
    /// Assignment of each input column to a micro-expert id.
    pub assignment: Vec<u32>,
    /// Centroid vector per micro-expert, shape `[num_experts, dim]`.
    pub centroids: Matrix,
}

/// Cluster `weights` columns into micro-experts. (Stub — M3.)
pub fn cluster(_weights: &Matrix, _cfg: &ClusterConfig) -> anyhow::Result<ClusterResult> {
    todo!("M3: spherical k-means micro-expert clustering")
}
