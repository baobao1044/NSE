//! AVX2 SIMD kernel (performance scaffold).
//!
//! The scalar reference kernel in [`crate::kernel`] is the mathematical ground
//! truth. This module provides the AVX2-accelerated equivalent using
//! `#[target_feature(enable = "avx2")]`, mirroring the spec's scheme:
//!
//! - Ternary `+1` → `_mm256_add_ps` (add activation)
//! - Ternary `−1` → `_mm256_sub_ps` (subtract activation)
//! - Ternary `0`  → skip (no-op)
//!
//! The AVX2 kernel must produce **bit-identical** results to the scalar kernel
//! (same accumulation order, same rounding). It is gated behind `target_feature`
//! so it only compiles on x86-64 with AVX2 support; a runtime `is_x86_feature_detected!`
//! check dispatches to it, falling back to scalar otherwise.
//!
//! Status: scaffold (M6). Implementation deferred to the performance phase.
//! The scalar kernel is used for all POC PPL measurements.

#![allow(dead_code)]

use nse_core::sparse::MicroExpert;

/// AVX2-accelerated ternary micro-expert kernel.
///
/// Processes 8 floats per iteration with `_mm256_add_ps` / `_mm256_sub_ps`.
/// Must match [`crate::kernel::compute_ternary_micro_expert_scalar`] exactly.
///
/// (Stub — implemented in the performance phase post-M6.)
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
pub unsafe fn compute_ternary_micro_expert_avx2(
    _expert: &MicroExpert,
    _x: &[f32],
    _y: &mut [f32],
) {
    // Implementation plan:
    // 1. Load 8 activations into __m256 via _mm256_load_ps.
    // 2. Decode 2-bit ternary masks from packed bytes:
    //    - mask_pos: bits where ternary == +1
    //    - mask_neg: bits where ternary == -1
    // 3. accum = _mm256_add_ps(accum, _mm256_and_ps(x, mask_pos))
    // 4. accum = _mm256_sub_ps(accum, _mm256_and_ps(x, mask_neg))
    // 5. Store accum via _mm256_store_ps.
    todo!("performance phase: AVX2 ternary kernel")
}

/// Dispatch to AVX2 if available, else fall back to scalar.
pub fn compute_ternary_micro_expert_dispatch(
    expert: &MicroExpert,
    x: &[f32],
    y: &mut [f32],
) {
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") {
            // SAFETY: guarded by is_x86_feature_detected.
            unsafe { compute_ternary_micro_expert_avx2(expert, x, y) };
            return;
        }
    }
    // Scalar fallback.
    crate::kernel::compute_ternary_micro_expert_scalar(expert, x, y);
}
