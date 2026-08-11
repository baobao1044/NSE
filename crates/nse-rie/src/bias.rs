//! Bias compensator — static (legacy/pruned-only) and adaptive (per-token).
//!
//! Three application policies, dispatched by `sparse_linear_with_kernel`:
//! - **Legacy** (`row_to_expert` empty): the fixed `B[i] = W[i] . mean_input`
//!   is added *unconditionally* to every output row (the M8-and-earlier
//!   behavior, kept verbatim for old `.nse` files).
//! - **Pruned-only** (`row_to_expert` non-empty, no `bias_table`): the same
//!   fixed bias, but added *only* to rows whose owning expert was NOT
//!   activated for the current token — the correctness fix for the M8
//!   double-count (`W_quant[i]·x + W[i]·mean_input`).
//! - **Adaptive** (`row_to_expert` + `bias_table` + `input_codebook`):
//!   the token's activation `x` is encoded against `input_codebook` into a
//!   single code `c`, and `bias_table[c*out_dim + i]` is added to each
//!   pruned row `i` — a per-token, codebook-keyed bias instead of the
//!   corpus mean. See [`apply_adaptive`].

use nse_core::sparse::PqCodebook;

pub use nse_ller::apply_bias as apply_bias_kernel;
pub use nse_ller::apply_bias_pruned_only as apply_bias_pruned_only_kernel;

use nse_core::sparse::SparseLayer;

/// Add `sl.bias` to `output` in place (full bias for every output row).
pub fn apply(layer_bias: &[f32], output: &mut [f32]) {
    apply_bias_kernel(layer_bias, output);
}

/// Convenience: apply a [`SparseLayer`]'s bias.
pub fn apply_layer(sl: &SparseLayer, output: &mut [f32]) {
    apply_bias_kernel(&sl.bias, output);
}

/// Adaptive per-token bias: encode the token's activation `x` against
/// `input_codebook` (a VQ codebook, `num_sub_vectors = 1`, `nbits = 8` → 256
/// centroids) into a single code `c`, then add `bias_table[c * out_dim + i]`
/// to every output row `i` whose owning expert was NOT activated (and is not
/// a core row). Activated and core rows get no bias — their contribution is
/// computed exactly.
///
/// The encode cost (`256 * in_dim` dot products) is paid once per token and
/// amortized over all pruned rows; each pruned row then costs a single
/// lookup. This is the "PQ là foundation" path: the activation codebook is
/// trained by `nse_zstm::pq::train_pq` with `num_sub_vectors = 1` (VQ via
/// the PQ machinery), and `bias_table[c][i]` is precomputed offline as
/// `W_quant[i] . decode_pq([c], input_codebook)`.
///
/// Returns whether the adaptive path was actually used (always `true` when
/// called — the caller uses the return value to skip the pruned-only
/// fallback).
pub fn apply_adaptive(
    bias_table: &[f32],
    output: &mut [f32],
    row_to_expert: &[i32],
    activated_set: &[bool],
    x: &[f32],
    input_codebook: &PqCodebook,
    out_dim: usize,
) -> bool {
    use nse_zstm::pq::encode_pq;
    // Encode `x` against the activation codebook → M codes (M=1 for VQ).
    let codes = encode_pq(x, input_codebook);
    // M=1 VQ → single code; collapse to one index. (For M>1 we'd need a
    // combined index; the M=1 design keeps `bias_table` to 256×out_dim.)
    let c = codes[0] as usize;
    let base = c * out_dim;
    for (i, &e) in row_to_expert.iter().enumerate() {
        if i >= output.len() || i >= out_dim {
            break;
        }
        if e < 0 {
            continue; // core row — computed exactly
        }
        let e = e as usize;
        if e < activated_set.len() && activated_set[e] {
            continue; // owning expert activated → no bias
        }
        // Pruned expert row → add the per-token adaptive bias.
        let off = base + i;
        if off < bias_table.len() {
            output[i] += bias_table[off];
        }
    }
    true
}
