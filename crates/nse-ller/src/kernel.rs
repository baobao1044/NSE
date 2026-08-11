//! SIMD compute kernels.
//!
//! The **scalar reference kernel** is the mathematical ground truth used for
//! PPL — it must match the dense math exactly. AVX2 `_mm256_*` kernels are
//! scaffolded for performance work later; they must produce bit-identical
//! results to the scalar kernel.
//!
//! Two kernel families:
//! - [`compute_dense_core`]: a plain `f32` mat-vec for the always-on outlier
//!   rows (`y[i] = W[i] . x`).
//! - [`compute_ternary_micro_expert_scalar`]: ternary accumulation for a
//!   micro-expert (`y[row] += scale * sum_j ternary[j] * x[j]`), realized as
//!   add / subtract / skip to mirror the spec's AVX2 add/sub/skip scheme.

use nse_core::sparse::{MicroExpert, PqCodebook};

/// Dense core mat-vec: for each owned output row `i`, `y[i] += W[i] . x`.
/// `core` is `[n_core, in]`, `x` is `[in]`, `y` is `[out]` (the full output
/// space; `row_ids` maps core rows to their original positions).
pub fn compute_dense_core(core: &nse_core::tensor::Matrix, row_ids: &[u32], x: &[f32], y: &mut [f32]) {
    let in_dim = core.cols;
    for (i, &rid) in row_ids.iter().enumerate() {
        let row = &core.data[i * in_dim..(i + 1) * in_dim];
        let mut s = 0.0f32;
        for j in 0..in_dim {
            s += row[j] * x[j];
        }
        y[rid as usize] += s;
    }
}

/// Scalar reference ternary micro-expert kernel.
///
/// For each owned output row `i` (original id `row_ids[i]`), accumulate
/// `y[row] += row_scales[i] * sum_j ternary[i*in + j] * x[j]`, where the
/// multiply by `{−1,0,1}` is realized as add / subtract / skip — mirroring
/// the spec's AVX2 `_mm256_add_ps` / `_mm256_sub_ps` / skip scheme.
///
/// This is the POC ground-truth kernel; it must be numerically equivalent to
/// the dense mat-vec on the reconstructed weights `scale * ternary`.
pub fn compute_ternary_micro_expert_scalar(
    expert: &MicroExpert,
    x: &[f32],
    y: &mut [f32],
) {
    let in_dim = x.len();
    let n_rows = expert.row_ids.len();
    for i in 0..n_rows {
        let rid = expert.row_ids[i] as usize;
        let scale = expert.row_scales[i];
        let codes = &expert.ternary[i * in_dim..(i + 1) * in_dim];
        let mut acc = 0.0f32;
        for j in 0..in_dim {
            let c = codes[j];
            if c > 0 {
                acc += x[j];
            } else if c < 0 {
                acc -= x[j];
            }
            // c == 0 -> skip
        }
        y[rid] += scale * acc;
    }
}

/// Apply the static bias: `y[i] += bias[i]` for all output rows.
pub fn apply_bias(bias: &[f32], y: &mut [f32]) {
    for (i, b) in bias.iter().enumerate() {
        if i < y.len() {
            y[i] += b;
        }
    }
}

/// Scalar reference PQ micro-expert kernel.
///
/// For each owned output row `i` (original id `row_ids[i]`), decode its `M`
/// PQ codes against the shared `codebook` *inline* (no allocation), dot the
/// reconstruction with `x`, and accumulate `y[row] += row_scales[i] * dot`.
/// This is the canonical ground-truth PQ kernel; the AVX2 version must match
/// it within FP tolerance.
///
/// `expert.pq` must be `Some` and `codebook` must match the expert's
/// `codebook_idx` — the caller (`sparse_linear_with_kernel`) guarantees this.
pub fn compute_pq_micro_expert_scalar(
    expert: &MicroExpert,
    x: &[f32],
    y: &mut [f32],
    codebook: &PqCodebook,
) {
    let pq = match &expert.pq {
        Some(p) => p,
        None => return, // caller should dispatch ternary for pq == None
    };
    let m = pq.num_sub_vectors;
    let sub_dim = codebook.sub_dim;
    let n_entries = codebook.num_entries();
    let in_dim = x.len();
    debug_assert_eq!(m * sub_dim, in_dim, "PQ sub-vector geometry mismatch");
    for i in 0..pq.row_scales.len() {
        let rid = expert.row_ids[i] as usize;
        let scale = pq.row_scales[i];
        let codes = &pq.codes[i * m..(i + 1) * m];
        let mut acc = 0.0f32;
        for sm in 0..m {
            let c = (codes[sm] as usize).min(n_entries - 1);
            let base = sm * n_entries * sub_dim + c * sub_dim;
            let cent = &codebook.codebook[base..base + sub_dim];
            let xsub = &x[sm * sub_dim..(sm + 1) * sub_dim];
            for j in 0..sub_dim {
                acc += cent[j] * xsub[j];
            }
        }
        y[rid] += scale * acc;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nse_core::tensor::Matrix;

    #[test]
    fn dense_core_matches_dot() {
        // W [2, 3], x [3], y [4] (out_dim=4, core owns rows 1 and 3).
        let mut core = Matrix::zeros(2, 3);
        core.data[0..3].copy_from_slice(&[1.0, 2.0, 3.0]);
        core.data[3..6].copy_from_slice(&[4.0, 5.0, 6.0]);
        let row_ids = vec![1u32, 3];
        let x = vec![0.5, 1.0, 2.0];
        let mut y = vec![0.0f32; 4];
        compute_dense_core(&core, &row_ids, &x, &mut y);
        // y[1] = 1*.5+2*1+3*2 = 8.5 ; y[3] = 4*.5+5*1+6*2 = 19
        assert!((y[1] - 8.5).abs() < 1e-5);
        assert!((y[3] - 19.0).abs() < 1e-5);
        assert_eq!(y[0], 0.0);
    }

    #[test]
    fn ternary_kernel_matches_reconstructed_dense() {
        // Expert with 2 rows, in_dim=3.
        let expert = MicroExpert {
            row_ids: vec![0, 1],
            ternary: vec![1, -1, 0, 0, 1, -1],
            row_scales: vec![2.0, 0.5],
            centroid: vec![0.0; 3],
            mean_input: vec![0.0; 3],
            pq: None,
        };
        let x = vec![3.0, 4.0, 5.0];
        let mut y_sparse = vec![0.0f32; 2];
        compute_ternary_micro_expert_scalar(&expert, &x, &mut y_sparse);
        // Dense reconstruction: row0 = 2*(+3 -4 +0)=2*(-1)=-2 ; row1 = .5*(0 +4 -5)=.5*(-1)=-.5
        assert!((y_sparse[0] - (-2.0)).abs() < 1e-5);
        assert!((y_sparse[1] - (-0.5)).abs() < 1e-5);

        // Cross-check against explicit dense mat-vec on reconstructed weights.
        let recon = [
            2.0 * 1.0, 2.0 * -1.0, 2.0 * 0.0,
            0.5 * 0.0, 0.5 * 1.0, 0.5 * -1.0,
        ];
        let mut y_dense = vec![0.0f32; 2];
        for r in 0..2 {
            for j in 0..3 {
                y_dense[r] += recon[r * 3 + j] * x[j];
            }
        }
        assert_eq!(y_sparse, y_dense);
    }

    #[test]
    fn bias_adds_in_place() {
        let bias = vec![1.0, 2.0, 3.0];
        let mut y = vec![10.0, 20.0, 30.0];
        apply_bias(&bias, &mut y);
        assert_eq!(y, vec![11.0, 22.0, 33.0]);
    }

    /// PQ scalar kernel matches the explicit decode-then-dot math. The
    /// canonical ground truth for the PQ path (AVX2 must match within tol).
    #[test]
    fn pq_kernel_matches_decode() {
        // Build a tiny codebook: M=2 sub-vectors of dim 2, 2 entries each.
        let codebook = PqCodebook {
            num_sub_vectors: 2,
            nbits: 1,
            sub_dim: 2,
            // Sub-vector 0: centroids [1,1] and [-1,-1].
            // Sub-vector 1: centroids [2,0] and [0,2].
            codebook: vec![
                1.0, 1.0, -1.0, -1.0,
                2.0, 0.0, 0.0, 2.0,
            ],
        };
        // Expert with 2 rows, scale [1.0, 0.5], in_dim=4 (=2*2).
        // Row 0 codes [0,0] -> recon [1,1,2,0]; row 1 codes [1,1] -> recon
        // [-1,-1,0,2] scaled by 0.5 -> [-0.5,-0.5,0,1].
        let expert = MicroExpert {
            row_ids: vec![0, 1],
            ternary: vec![],
            row_scales: vec![], // unused for PQ
            centroid: vec![0.0; 4],
            mean_input: vec![0.0; 4],
            pq: Some(nse_core::sparse::PqExpertData {
                codes: vec![0, 0, 1, 1],
                row_scales: vec![1.0, 0.5],
                num_sub_vectors: 2,
            }),
        };
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let mut y = vec![0.0f32; 2];
        compute_pq_micro_expert_scalar(&expert, &x, &mut y, &codebook);
        // Row 0: scale 1.0 * ([1,1].[1,2] + [2,0].[3,4]) = (1+2) + (6+0) = 9.
        // Row 1: scale 0.5 * ([-1,-1].[1,2] + [0,2].[3,4]) = 0.5 * (-3 + 8) = 2.5.
        assert!((y[0] - 9.0).abs() < 1e-5, "row0: {}", y[0]);
        assert!((y[1] - 2.5).abs() < 1e-5, "row1: {}", y[1]);
    }
}
