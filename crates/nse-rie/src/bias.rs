//! Static bias compensator.
//!
//! Re-exports the LLER bias-apply kernel and provides a thin helper to apply
//! the [`nse_core::sparse::SparseLayer`] bias to an output vector. The bias
//! `B[i] = W[i] . mean_input` is precomputed by ZSTM for every output row;
//! the bias is added unconditionally to the sparse output, restoring the
//! expected contribution of pruned experts on average.
//!
//! Core rows have `B[i] = 0` (they're computed exactly by the dense path), so
//! adding the full bias never double-counts the core.

pub use nse_ller::apply_bias as apply_bias_kernel;

use nse_core::sparse::SparseLayer;

/// Add `sl.bias` to `output` in place (full bias for every output row).
pub fn apply(layer_bias: &[f32], output: &mut [f32]) {
    apply_bias_kernel(layer_bias, output);
}

/// Convenience: apply a [`SparseLayer`]'s bias.
pub fn apply_layer(sl: &SparseLayer, output: &mut [f32]) {
    apply_bias_kernel(&sl.bias, output);
}
