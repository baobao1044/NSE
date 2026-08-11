//! AVX2 SIMD kernels (real implementation).
//!
//! AVX2-accelerated equivalents of the scalar reference kernels in
//! [`crate::kernel`], mirroring the spec's scheme:
//!
//! - Ternary `+1` → add the activation (`_mm256_add_ps`)
//! - Ternary `−1` → subtract the activation (`_mm256_sub_ps`)
//! - Ternary `0`  → skip (mask out via `_mm256_and_ps`)
//!
//! Each ternary code is realized as a *mask* in the sign bit: `+1` keeps `x`,
//! `−1` keeps `-x`, `0` zeros it. We build two 256-bit masks (`pos`, `neg`)
//! per 8-lane chunk from the ternary codes and accumulate
//! `accum = add(accum, and(x, pos)) - and(x, neg)`. The final per-row sum is a
//! horizontal reduction done in scalar order.
//!
//! ## Numerical note (honest limitation)
//! These kernels are **not bit-identical** to the scalar reference: SIMD
//! reduction tree + FMA changes floating-point rounding (FP addition is not
//! associative). They agree with the scalar kernel to within `~1e-5` relative
//! error, which is well below the POC's PPL noise floor. The scalar kernel
//! remains the canonical ground truth; `KernelKind::Scalar` selects it.

use core::arch::x86_64::*;
use nse_core::sparse::{MicroExpert, PqCodebook};

/// Selects which compute kernel the runtime uses for a sparse linear.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KernelKind {
    /// Canonical scalar reference (mathematical ground truth).
    Scalar,
    /// Force AVX2 (panics via `is_x86_feature_detected!` if unavailable).
    Avx2,
    /// Auto-detect AVX2 at runtime; fall back to scalar if unavailable.
    Auto,
}

impl Default for KernelKind {
    fn default() -> Self {
        KernelKind::Auto
    }
}

impl KernelKind {
    /// Returns true if this kind should dispatch to AVX2.
    /// `Avx2` always dispatches (and the dispatch path itself re-checks the
    /// CPU feature and falls back to scalar if missing, so a no-AVX2 machine
    /// still works); `Auto` dispatches only when the CPU has AVX2.
    pub fn use_avx2(self) -> bool {
        matches!(self, KernelKind::Avx2 | KernelKind::Auto)
    }
}

/// AVX2 ternary micro-expert kernel.
///
/// For each owned row `i`, accumulate `y[row_ids[i]] += row_scales[i] *
/// sum_j ternary[i*in+j] * x[j]`. Processes 8 floats per iteration; the tail
/// (< 8) falls back to scalar with the *same* add/sub/skip semantics.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn compute_ternary_micro_expert_avx2(
    expert: &MicroExpert,
    x: &[f32],
    y: &mut [f32],
) {
    let in_dim = x.len();
    let n_rows = expert.row_ids.len();
    // All-zero float vector: used to build masks via comparison.
    let zero = _mm256_setzero_ps();

    for i in 0..n_rows {
        let rid = expert.row_ids[i] as usize;
        let scale = expert.row_scales[i];
        let codes = &expert.ternary[i * in_dim..(i + 1) * in_dim];

        let mut sum = 0.0f32;
        let mut j = 0;
        // 8-lane chunks.
        while j + 8 <= in_dim {
            let xv = _mm256_loadu_ps(x.as_ptr().add(j));
            // Build pos/neg float masks: lane = 0xFFFFFFFF (all-ones, keep) or 0 (drop).
            // codes are i8 in {-1,0,1}. +1 lane => keep x; -1 lane => keep x (subtracted
            // later); 0 => drop. We need one 32-bit mask per lane (8 lanes => 32 bytes).
            let mut pos_lanes = [0i32; 8];
            let mut neg_lanes = [0i32; 8];
            for k in 0..8 {
                let c = codes[j + k];
                pos_lanes[k] = if c > 0 { -1 } else { 0 }; // -1 = 0xFFFFFFFF (all-ones)
                neg_lanes[k] = if c < 0 { -1 } else { 0 };
            }
            let pos_mask = _mm256_castsi256_ps(_mm256_loadu_si256(pos_lanes.as_ptr() as *const __m256i));
            let neg_mask = _mm256_castsi256_ps(_mm256_loadu_si256(neg_lanes.as_ptr() as *const __m256i));

            let pos_vals = _mm256_and_ps(xv, pos_mask); // x where +1, else 0
            let neg_vals = _mm256_and_ps(xv, neg_mask); // x where -1, else 0

            // sum_chunk = sum(pos_vals) - sum(neg_vals), reduced in scalar order.
            let chunk = _mm256_sub_ps(pos_vals, neg_vals);
            sum += reduce_add_in_order(chunk);
            j += 8;
        }
        // Scalar tail (same semantics).
        while j < in_dim {
            let c = codes[j];
            if c > 0 {
                sum += x[j];
            } else if c < 0 {
                sum -= x[j];
            }
            j += 1;
        }
        y[rid] += scale * sum;
    }
    // silence unused zero
    let _ = zero;
}

/// Horizontal reduction of an `__m256` (8 floats) in *scalar* left-to-right
/// order, to keep results as close as possible to the scalar kernel.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn reduce_add_in_order(v: __m256) -> f32 {
    let mut tmp = [0f32; 8];
    _mm256_storeu_ps(tmp.as_mut_ptr(), v);
    tmp[0] + tmp[1] + tmp[2] + tmp[3] + tmp[4] + tmp[5] + tmp[6] + tmp[7]
}

/// AVX2 (FMA) dense-core mat-vec: `y[row_ids[i]] += W[i] . x`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn compute_dense_core_avx2(
    core: &nse_core::tensor::Matrix,
    row_ids: &[u32],
    x: &[f32],
    y: &mut [f32],
) {
    let in_dim = core.cols;
    for (i, &rid) in row_ids.iter().enumerate() {
        let row = &core.data[i * in_dim..(i + 1) * in_dim];
        let mut acc = _mm256_setzero_ps();
        let mut j = 0;
        while j + 8 <= in_dim {
            let wv = _mm256_loadu_ps(row.as_ptr().add(j));
            let xv = _mm256_loadu_ps(x.as_ptr().add(j));
            acc = _mm256_fmadd_ps(wv, xv, acc);
            j += 8;
        }
        let mut s = reduce_add_in_order(acc);
        while j < in_dim {
            s += row[j] * x[j];
            j += 1;
        }
        y[rid as usize] += s;
    }
}

/// Dispatch to AVX2 if available, else fall back to scalar (ternary kernel).
pub fn compute_ternary_micro_expert_dispatch(
    expert: &MicroExpert,
    x: &[f32],
    y: &mut [f32],
    kind: KernelKind,
) {
    if kind.use_avx2() {
        #[cfg(target_arch = "x86_64")]
        {
            if std::is_x86_feature_detected!("avx2") {
                // SAFETY: guarded by runtime feature detection.
                unsafe { compute_ternary_micro_expert_avx2(expert, x, y) };
                return;
            }
        }
    }
    crate::kernel::compute_ternary_micro_expert_scalar(expert, x, y);
}

/// Dispatch for the dense-core kernel.
pub fn compute_dense_core_dispatch(
    core: &nse_core::tensor::Matrix,
    row_ids: &[u32],
    x: &[f32],
    y: &mut [f32],
    kind: KernelKind,
) {
    if kind.use_avx2() {
        #[cfg(target_arch = "x86_64")]
        {
            if std::is_x86_feature_detected!("avx2") {
                // SAFETY: guarded by runtime feature detection.
                unsafe { compute_dense_core_avx2(core, row_ids, x, y) };
                return;
            }
        }
    }
    crate::kernel::compute_dense_core(core, row_ids, x, y);
}

/// AVX2 (FMA) PQ micro-expert kernel. For each owned row, decode its `M`
/// PQ codes (gather the indexed centroids from the shared codebook) and dot
/// the reconstruction with `x`. The dot product uses FMA over 8-lane chunks;
/// the codebook gather is scalar (centroids are contiguous in `codebook`, so
/// a plain slice load is enough — no `_mm256_i32gather_ps` needed). The tail
/// (< 8 per sub-vector) is folded in scalar order. Not bit-identical to the
/// scalar kernel (FMA non-associativity), within `~1e-5`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
pub unsafe fn compute_pq_micro_expert_avx2(
    expert: &MicroExpert,
    x: &[f32],
    y: &mut [f32],
    codebook: &PqCodebook,
) {
    let pq = match &expert.pq {
        Some(p) => p,
        None => return,
    };
    let m = pq.num_sub_vectors;
    let sub_dim = codebook.sub_dim;
    let n_entries = codebook.num_entries();
    for i in 0..pq.row_scales.len() {
        let rid = expert.row_ids[i] as usize;
        let scale = pq.row_scales[i];
        let codes = &pq.codes[i * m..(i + 1) * m];
        let mut total = 0.0f32;
        for sm in 0..m {
            let c = (codes[sm] as usize).min(n_entries - 1);
            let base = sm * n_entries * sub_dim + c * sub_dim;
            let cent = &codebook.codebook[base..base + sub_dim];
            let xsub = &x[sm * sub_dim..(sm + 1) * sub_dim];
            let mut acc = _mm256_setzero_ps();
            let mut j = 0;
            while j + 8 <= sub_dim {
                let cv = _mm256_loadu_ps(cent.as_ptr().add(j));
                let xv = _mm256_loadu_ps(xsub.as_ptr().add(j));
                acc = _mm256_fmadd_ps(cv, xv, acc);
                j += 8;
            }
            // Reduce + scalar tail (same order as the scalar kernel).
            let mut tmp = [0f32; 8];
            _mm256_storeu_ps(tmp.as_mut_ptr(), acc);
            let mut s = tmp[0] + tmp[1] + tmp[2] + tmp[3]
                + tmp[4] + tmp[5] + tmp[6] + tmp[7];
            while j < sub_dim {
                s += cent[j] * xsub[j];
                j += 1;
            }
            total += s;
        }
        y[rid] += scale * total;
    }
}

/// Dispatch to AVX2 if available, else fall back to scalar (PQ kernel).
pub fn compute_pq_micro_expert_dispatch(
    expert: &MicroExpert,
    x: &[f32],
    y: &mut [f32],
    codebook: &PqCodebook,
    kind: KernelKind,
) {
    if kind.use_avx2() {
        #[cfg(target_arch = "x86_64")]
        {
            if std::is_x86_feature_detected!("avx2") {
                // SAFETY: guarded by runtime feature detection.
                unsafe { compute_pq_micro_expert_avx2(expert, x, y, codebook) };
                return;
            }
        }
    }
    crate::kernel::compute_pq_micro_expert_scalar(expert, x, y, codebook);
}

#[cfg(all(test, target_arch = "x86_64"))]
mod tests {
    use super::*;
    use nse_core::sparse::MicroExpert;

    fn supports_avx2() -> bool {
        std::is_x86_feature_detected!("avx2")
    }

    #[test]
    fn avx2_ternary_matches_scalar() {
        if !supports_avx2() {
            eprintln!("AVX2 unavailable, skipping");
            return;
        }
        // 2 rows, in_dim = 17 (has an 8-lane chunk + tail).
        let in_dim = 17;
        let ternary: Vec<i8> = (0..2 * in_dim)
            .map(|i| match i % 3 {
                0 => 1,
                1 => -1,
                _ => 0,
            })
            .collect();
        let row_scales = vec![1.5, 0.25];
        let x: Vec<f32> = (0..in_dim).map(|i| (i as f32) * 0.1 - 0.8).collect();
        let expert = MicroExpert {
            row_ids: vec![0, 1],
            ternary,
            row_scales,
            centroid: vec![0.0; in_dim],
            mean_input: vec![0.0; in_dim],
            pq: None,
        };

        let mut y_scalar = vec![0.0f32; 2];
        crate::kernel::compute_ternary_micro_expert_scalar(&expert, &x, &mut y_scalar);

        let mut y_avx2 = vec![0.0f32; 2];
        unsafe { compute_ternary_micro_expert_avx2(&expert, &x, &mut y_avx2) };

        // Not bit-identical (FP non-associativity), but within tolerance.
        for r in 0..2 {
            assert!(
                (y_scalar[r] - y_avx2[r]).abs() <= 1e-5 * y_scalar[r].abs().max(1.0),
                "row {r}: scalar={} avx2={}",
                y_scalar[r],
                y_avx2[r]
            );
        }
    }

    #[test]
    fn dispatch_scalar_when_requested() {
        let expert = MicroExpert {
            row_ids: vec![0],
            ternary: vec![1, -1, 0],
            row_scales: vec![1.0],
            centroid: vec![0.0; 3],
            mean_input: vec![0.0; 3],
            pq: None,
        };
        let x = vec![1.0, 2.0, 3.0];
        let mut y = vec![0.0f32; 1];
        compute_ternary_micro_expert_dispatch(&expert, &x, &mut y, KernelKind::Scalar);
        // 1 - 2 + 0 = -1
        assert!((y[0] - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn avx2_dense_core_matches_scalar() {
        if !supports_avx2() {
            return;
        }
        use nse_core::tensor::Matrix;
        let in_dim = 20;
        let mut core = Matrix::zeros(2, in_dim);
        for (i, v) in core.data.iter_mut().enumerate() {
            *v = (i as f32) * 0.01 - 0.1;
        }
        let row_ids = vec![0u32, 1];
        let x: Vec<f32> = (0..in_dim).map(|i| (i as f32) * 0.2 - 2.0).collect();

        let mut y_s = vec![0.0f32; 2];
        crate::kernel::compute_dense_core(&core, &row_ids, &x, &mut y_s);
        let mut y_a = vec![0.0f32; 2];
        unsafe { compute_dense_core_avx2(&core, &row_ids, &x, &mut y_a) };

        for r in 0..2 {
            assert!(
                (y_s[r] - y_a[r]).abs() <= 1e-4 * y_s[r].abs().max(1.0),
                "row {r}: scalar={} avx2={}",
                y_s[r],
                y_a[r]
            );
        }
    }

    /// PQ AVX2 kernel matches the scalar PQ kernel within FP tolerance (FMA
    /// non-associativity). Uses a codebook with sub_dim a multiple of 8 so
    /// the FMA path is fully exercised (no tail).
    #[test]
    fn pq_avx2_matches_scalar() {
        if !supports_avx2() {
            eprintln!("AVX2 unavailable, skipping PQ AVX2 test");
            return;
        }
        use nse_core::sparse::PqCodebook;
        use nse_core::sparse::PqExpertData;
        // M=2 sub-vectors of dim 8 (= in_dim 16), 4 entries each (nbits=2).
        let m = 2;
        let sub_dim = 8;
        let n_entries = 4;
        let codebook = PqCodebook {
            num_sub_vectors: m,
            nbits: 2,
            sub_dim,
            codebook: (0..(m * n_entries * sub_dim))
                .map(|i| (i as f32) * 0.01 - 0.05)
                .collect(),
        };
        let n_rows = 3;
        let codes: Vec<u8> = (0..n_rows * m).map(|i| (i as u8) % n_entries as u8).collect();
        let row_scales: Vec<f32> = (0..n_rows).map(|i| 0.1 + (i as f32) * 0.3).collect();
        let expert = MicroExpert {
            row_ids: (0..n_rows as u32).collect(),
            ternary: vec![],
            row_scales: vec![],
            centroid: vec![0.0; m * sub_dim],
            mean_input: vec![0.0; m * sub_dim],
            pq: Some(PqExpertData {
                codes,
                row_scales,
                num_sub_vectors: m,
            }),
        };
        let x: Vec<f32> = (0..m * sub_dim).map(|i| (i as f32) * 0.1).collect();

        let mut y_s = vec![0.0f32; n_rows];
        crate::kernel::compute_pq_micro_expert_scalar(&expert, &x, &mut y_s, &codebook);
        let mut y_a = vec![0.0f32; n_rows];
        unsafe { compute_pq_micro_expert_avx2(&expert, &x, &mut y_a, &codebook) };

        for r in 0..n_rows {
            assert!(
                (y_s[r] - y_a[r]).abs() <= 1e-5 * y_s[r].abs().max(1.0),
                "PQ row {r}: scalar={} avx2={}",
                y_s[r],
                y_a[r]
            );
        }
    }
}
