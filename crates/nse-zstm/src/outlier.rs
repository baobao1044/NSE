//! Outlier channel extraction (ZSTM stage 1).
//!
//! Extracts high-amplitude "activation outlier" channels into a fixed dense
//! core that stays resident on L1 cache at inference time, preserving the
//! zero-shot accuracy of the source model.
//!
//! Status: skeleton (M0). Real extraction (top-k by magnitude) lands in M3.

use nse_core::tensor::Matrix;

/// Configuration for outlier extraction.
#[derive(Debug, Clone)]
pub struct OutlierConfig {
    /// Fraction of channels to keep in the dense core (e.g. 0.001 = 0.1%).
    pub fraction: f32,
}

impl Default for OutlierConfig {
    fn default() -> Self {
        Self { fraction: 0.001 }
    }
}

/// Result of outlier extraction: the dense core and the residual matrix.
#[derive(Debug, Clone)]
pub struct OutlierResult {
    /// Packed dense core bytes (FP16/INT8 in the real impl).
    pub dense_core: Vec<u8>,
    /// Residual weights after removing outlier channels.
    pub residual: Matrix,
}

/// Extract outlier channels from `weights`. (Stub — M3.)
pub fn extract(_weights: &Matrix, _cfg: &OutlierConfig) -> anyhow::Result<OutlierResult> {
    todo!("M3: outlier extraction by magnitude")
}
