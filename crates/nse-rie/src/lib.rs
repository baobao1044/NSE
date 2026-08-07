//! # nse-rie
//!
//! Routing & Indexing Engine (online).
//!
//! For each input activation, finds the relevant micro-experts without
//! scanning the whole model:
//!
//! - `index` — Maximum Inner Product Search (MIPS) over expert centroids.
//!   Brute-force exact (O(N)) for the POC; HNSW/LSH scaffolded for scale-out.
//! - `router` — adaptive threshold router: prune centroids below `θ(x)`,
//!   keep the dynamic top-K (or keep all as the upper-bound reference).
//! - `bias` — static bias compensator: add the precomputed `B[i]` to restore
//!   the expected contribution of pruned experts.
//!
//! The sparse-linear primitive that ties RIE + LLER together for one layer is
//! [`sparse_linear`]: given a [`nse_core::SparseLayer`] and an input vector,
//! produce the output vector using the dense core, activated experts, and
//! bias.

#![allow(dead_code)]

pub mod bias;
pub mod index;
pub mod router;

pub use bias::{apply as apply_bias, apply_layer};
pub use index::{Hit, MipsIndex};
pub use router::{route_all, route_by_ratio, RouterConfig};

use nse_core::sparse::SparseLayer;
use nse_ller::{
    apply_bias as ller_apply_bias, compute_dense_core, compute_ternary_micro_expert_scalar,
};

/// Sparse forward of one linear layer: `y = core(x) + sum_activated experts(x)
/// + bias`. `activated` is the set of expert indices (into `sl.experts`) to
/// compute; the rest are pruned and their average contribution is covered by
/// the static bias.
pub fn sparse_linear(
    sl: &SparseLayer,
    x: &[f32],
    activated: &[usize],
) -> Vec<f32> {
    let mut y = vec![0.0f32; sl.out_dim];
    // Dense core (always on).
    compute_dense_core(&sl.dense_core, &sl.core_row_ids, x, &mut y);
    // Activated micro-experts.
    for &eid in activated {
        let expert = &sl.experts[eid];
        compute_ternary_micro_expert_scalar(expert, x, &mut y);
    }
    // Static bias (compensates pruned experts on average).
    ller_apply_bias(&sl.bias, &mut y);
    y
}

/// Sparse forward with **all** experts activated — the upper-bound reference.
/// With all experts on, the only degradation vs dense is ternary
/// quantization error (the bias is then zero on average for the prunable
/// rows, since nothing is pruned).
pub fn sparse_linear_all(sl: &SparseLayer, x: &[f32]) -> Vec<f32> {
    let activated: Vec<usize> = (0..sl.experts.len()).collect();
    sparse_linear(sl, x, &activated)
}

#[cfg(test)]
mod tests {
    use super::*;
    use nse_core::sparse::MicroExpert;
    use nse_core::tensor::Matrix;

    fn mk_layer(core: &[[f32; 3]], core_ids: &[u32], expert_rows: &[(&[i8], f32, u32)]) -> SparseLayer {
        // expert_rows: (ternary row, scale, original row id)
        let in_dim = 3;
        let dense_core = Matrix {
            rows: core.len(),
            cols: in_dim,
            data: core.iter().flat_map(|r| r.iter().copied()).collect(),
        };
        let expert = MicroExpert {
            row_ids: expert_rows.iter().map(|(_, _, id)| *id).collect(),
            ternary: expert_rows.iter().flat_map(|(t, _, _)| t.iter().copied()).collect(),
            row_scales: expert_rows.iter().map(|(_, s, _)| *s).collect(),
            centroid: vec![1.0; in_dim],
            mean_input: vec![0.0; in_dim],
        };
        let out_dim = 1 + expert_rows.len();
        let bias = vec![0.0; out_dim];
        SparseLayer {
            out_dim,
            in_dim,
            dense_core,
            core_row_ids: core_ids.to_vec(),
            experts: vec![expert],
            bias,
            mean_input: vec![0.0; in_dim],
        }
    }

    #[test]
    fn sparse_linear_with_all_matches_reconstruction() {
        // core owns row 0: [1,2,3]; expert owns row 1: scale 2, ternary [1,-1,0]
        let sl = mk_layer(&[[1.0, 2.0, 3.0]], &[0], &[(vec![1i8, -1, 0].as_slice(), 2.0, 1u32)]);
        let x = vec![0.5, 1.0, 2.0];
        let y = sparse_linear_all(&sl, &x);
        // y[0] = 1*.5+2*1+3*2 = 8.5 ; y[1] = 2*(+0.5 -1.0 +0) = 2*(-0.5) = -1.0
        assert!((y[0] - 8.5).abs() < 1e-5);
        assert!((y[1] - (-1.0)).abs() < 1e-5);
    }

    #[test]
    fn sparse_linear_with_none_uses_bias_only() {
        // bias[i] = 0 for core rows, set a nonzero bias for the expert row.
        let mut sl = mk_layer(&[[1.0, 0.0, 0.0]], &[0], &[(vec![1i8, 0, 0].as_slice(), 1.0, 1u32)]);
        sl.bias[1] = 7.0;
        let x = vec![2.0, 0.0, 0.0];
        let y = sparse_linear(&sl, &x, &[]); // no experts activated
        // y[0] = core = 1*2 = 2 ; y[1] = bias only = 7 (expert pruned)
        assert!((y[0] - 2.0).abs() < 1e-5);
        assert!((y[1] - 7.0).abs() < 1e-5);
    }
}
