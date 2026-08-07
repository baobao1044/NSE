//! SIMD compute kernels.
//!
//! The **scalar reference kernel** (`compute_ternary_micro_expert_scalar`) is
//! the mathematical ground truth used for PPL — it must match the dense math
//! exactly. AVX2 `_mm256_*` kernels (ternary add/sub via masks, PQ
//! `shuffle_epi8` codebook lookup) are scaffolded for performance work later;
//! they must produce bit-identical results to the scalar kernel.
//!
//! Status: skeleton (M0). Scalar ternary kernel lands in M4.

/// Scalar reference: accumulate one micro-expert's contribution.
///
/// `packed_weights` holds ternary values decoded to `{-1,0,1}`; for each
/// channel `i`, `output[i] += weight[i] * input[i]` where the multiply is
/// realized as add (weight=+1) / subtract (weight=-1) / skip (weight=0),
/// mirroring the AVX2 kernel in the spec.
///
/// (Stub — M4.)
pub fn compute_ternary_micro_expert_scalar(
    _input: &[f32],
    _weights_ternary: &[i8], // {-1, 0, 1}
    _output: &mut [f32],
) {
    todo!("M4: scalar ternary micro-expert kernel")
}
