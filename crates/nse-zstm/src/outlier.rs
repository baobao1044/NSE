//! Outlier channel extraction (ZSTM stage 1).
//!
//! Extracts high-amplitude output rows of a weight matrix `W [out, in]` into
//! a fixed dense core that stays resident (the "L1 cache" path per the spec)
//! and is always active during inference. The residual rows feed stage 2
//! (clustering) and stage 3 (quantization).
//!
//! Selection is by per-row L2 norm: the top `fraction` of rows become the
//! core. Core weights are stored in FP32 for the POC (the spec suggests
//! FP16/INT8; that's a size optimization with no accuracy impact here).

use nse_core::tensor::Matrix;

/// Configuration for outlier extraction.
#[derive(Debug, Clone)]
pub struct OutlierConfig {
    /// Fraction of output rows to keep in the dense core (e.g. 0.05 = 5%).
    /// 0 means no core (everything is sparse-routed).
    pub fraction: f32,
}

impl Default for OutlierConfig {
    fn default() -> Self {
        Self { fraction: 0.05 }
    }
}

/// Result of outlier extraction.
#[derive(Debug, Clone)]
pub struct OutlierResult {
    /// Dense core matrix `[n_core, in]`.
    pub dense_core: Matrix,
    /// Original output-row ids of the core rows.
    pub core_row_ids: Vec<u32>,
    /// Residual matrix `[out - n_core, in]` (the non-outlier rows).
    pub residual: Matrix,
    /// Original output-row ids of the residual rows (order matches `residual`).
    pub residual_row_ids: Vec<u32>,
}

/// Extract outlier rows from `weights [out, in]` by per-row L2 norm.
pub fn extract(weights: &Matrix, cfg: &OutlierConfig) -> anyhow::Result<OutlierResult> {
    let out = weights.rows;
    let in_dim = weights.cols;
    if out == 0 {
        return Ok(OutlierResult {
            dense_core: Matrix::zeros(0, in_dim),
            core_row_ids: vec![],
            residual: Matrix::zeros(0, in_dim),
            residual_row_ids: vec![],
        });
    }

    let n_core = ((out as f32) * cfg.fraction).round() as usize;
    // Keep at least 1 and at most out-1 (so clustering has something to do),
    // unless the layer is tiny.
    let n_core = n_core.clamp(0, out.saturating_sub(1).max(0));

    // Per-row L2 norm.
    let norms: Vec<f32> = (0..out)
        .map(|r| {
            let row = &weights.data[r * in_dim..(r + 1) * in_dim];
            row.iter().map(|v| v * v).sum::<f32>().sqrt()
        })
        .collect();

    // Sort row indices by norm descending; take top n_core as core.
    let mut order: Vec<usize> = (0..out).collect();
    order.sort_by(|&a, &b| norms[b].partial_cmp(&norms[a]).unwrap_or(std::cmp::Ordering::Equal));

    let mut core_rows = order[..n_core].to_vec();
    core_rows.sort_unstable(); // stable original order for core
    let residual_rows: Vec<usize> = order[n_core..].iter().cloned().collect();
    let mut residual_rows = residual_rows;
    residual_rows.sort_unstable();

    let mut dense_core = Matrix::zeros(n_core, in_dim);
    for (i, &r) in core_rows.iter().enumerate() {
        dense_core.data[i * in_dim..(i + 1) * in_dim]
            .copy_from_slice(&weights.data[r * in_dim..(r + 1) * in_dim]);
    }
    let n_res = residual_rows.len();
    let mut residual = Matrix::zeros(n_res, in_dim);
    for (i, &r) in residual_rows.iter().enumerate() {
        residual.data[i * in_dim..(i + 1) * in_dim]
            .copy_from_slice(&weights.data[r * in_dim..(r + 1) * in_dim]);
    }

    Ok(OutlierResult {
        dense_core,
        core_row_ids: core_rows.iter().map(|&r| r as u32).collect(),
        residual,
        residual_row_ids: residual_rows.iter().map(|&r| r as u32).collect(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_top_rows_by_norm() {
        // 4 rows, in_dim=2. Row 1 has the largest norm.
        let mut w = Matrix::zeros(4, 2);
        w.data[0..2].copy_from_slice(&[0.0, 0.0]);      // norm 0
        w.data[2..4].copy_from_slice(&[3.0, 4.0]);      // norm 5
        w.data[4..6].copy_from_slice(&[0.1, 0.0]);      // norm 0.1
        w.data[6..8].copy_from_slice(&[0.0, 0.2]);      // norm 0.2
        let r = extract(&w, &OutlierConfig { fraction: 0.25 }).unwrap();
        assert_eq!(r.core_row_ids, vec![1]);
        assert_eq!(r.dense_core.data, vec![3.0, 4.0]);
        assert_eq!(r.residual_row_ids, vec![0, 2, 3]);
        assert_eq!(r.residual.rows, 3);
    }

    #[test]
    fn zero_fraction_no_core() {
        let w = Matrix::zeros(5, 3);
        let r = extract(&w, &OutlierConfig { fraction: 0.0 }).unwrap();
        assert!(r.core_row_ids.is_empty());
        assert_eq!(r.residual_row_ids, vec![0, 1, 2, 3, 4]);
    }
}
