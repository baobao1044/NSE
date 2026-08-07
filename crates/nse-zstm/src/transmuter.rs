//! High-level driver that runs the full ZSTM pipeline and writes a `.nse`
//! file. Status: skeleton (M0). Wired up in M3.

use std::path::Path;

use nse_core::tensor::Matrix;

use crate::{cluster::ClusterConfig, outlier::OutlierConfig};

/// Full transmutation configuration.
#[derive(Debug, Clone, Default)]
pub struct TransmuteConfig {
    pub outlier: OutlierConfig,
    pub cluster: ClusterConfig,
}

/// Run the offline transmutation of `weights` and write a `.nse` file at
/// `out_path`. (Stub — M3.)
pub fn transmute(
    _weights: &Matrix,
    _cfg: &TransmuteConfig,
    _out_path: impl AsRef<Path>,
) -> anyhow::Result<()> {
    todo!("M3: full ZSTM transmutation -> .nse")
}
