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

use nse_core::sparse::MicroExpert;

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
}
