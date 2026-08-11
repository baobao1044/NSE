//! Sparse layer representation shared by ZSTM (produces), RIE+LLER (consumes),
//! and eval (compares).
//!
//! A [`SparseLayer`] is the transmuted form of one dense weight matrix
//! `W [out, in]` mapping input `x [in] -> y [out]`. It decomposes `W` into:
//!
//! - a small **dense core** of outlier output rows (kept in FP32, always
//!   active, on the "L1 cache" path per the spec);
//! - `K` **micro-experts**, each holding a subset of the remaining output rows
//!   quantized to ternary `{−1, 0, 1}` plus a per-row scale;
//! - a **static bias** `B [out]` approximating the contribution of pruned
//!   (non-activated) experts under the mean activation.
//!
//! Sparse forward (see `nse_rier`/`nse_ller`):
//! ```text
//! y = W_core @ x
//!   + sum_{activated k} (ternary W_expert_k @ x, rescaled)
//!   + sum_{pruned k} B[rows_k]
//! ```
//! Degradation vs dense comes from (1) ternary quantization and (2) the bias
//! using the mean activation in place of the actual token's `x`.

use serde::{Deserialize, Serialize};

use crate::tensor::Matrix;

/// Minimal config mirror stored alongside a transmuted model so the sparse
/// forward can run without depending on `nse-models` (avoids a dependency
/// cycle). The full `nse_models::Config` converts to/from this via the
/// `From` impls below.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ConfigStub {
    pub vocab_size: usize,
    pub dim: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub max_seq_len: usize,
    pub ff_dim: usize,
}

/// A trained PQ codebook: `M` sub-codebooks, each with `2^nbits` centroids.
/// Lives at the layer level (one shared codebook per `SparseLayer`, per the
/// spec: "shared codebook < 1 MB, kept on L3 cache") and is referenced by
/// each expert's [`PqExpertData`] via `codebook_idx`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PqCodebook {
    /// Number of sub-vectors `M` (a row of length `in_dim` splits into `M`
    /// contiguous sub-vectors of length `sub_dim`).
    pub num_sub_vectors: usize,
    /// Bits per code (8 → 256 entries per sub-codebook).
    pub nbits: usize,
    /// Sub-vector dimensionality `= in_dim / num_sub_vectors`.
    pub sub_dim: usize,
    /// Codebook data: `num_sub_vectors * (2^nbits) * sub_dim` floats, laid out
    /// as `[subvec m][centroid c][dim j]` — so centroid `c` of sub-vector `m`
    /// sits at `codebook[m * (2^nbits) * sub_dim + c * sub_dim + j]`.
    pub codebook: Vec<f32>,
}

impl PqCodebook {
    /// Number of centroids per sub-codebook = `2^nbits`.
    pub fn num_entries(&self) -> usize {
        1usize << self.nbits
    }
}

/// Per-expert PQ data: the codes (one byte per sub-vector per owned row) plus
/// a per-row scale (BitNet-style, so decode = `scale * concat(reconstructed
/// sub-vectors)`). References the layer's shared [`PqCodebook`] via
/// `codebook_idx`. When `None` on [`MicroExpert`], the ternary path is used.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct PqExpertData {
    /// PQ codes: `[num_rows * num_sub_vectors]` bytes, one byte per
    /// sub-vector per owned row (row-major: row `i`'s codes are
    /// `codes[i * num_sub_vectors .. (i+1) * num_sub_vectors]`).
    pub codes: Vec<u8>,
    /// Per-row scale (analogous to ternary's `row_scales`); the decoded row is
    /// `row_scales[i] * decode(codes[i*..])`.
    pub row_scales: Vec<f32>,
    /// Number of sub-vectors (cached on the expert so the kernel can index
    /// codes without going through the shared codebook).
    pub num_sub_vectors: usize,
}

/// One micro-expert: a group of output rows of the source matrix, their
/// ternary codes, per-row scales, and the centroid used for routing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MicroExpert {
    /// Output-row indices this expert owns (into the original `[out]` space).
    pub row_ids: Vec<u32>,
    /// Ternary codes for each owned row, length `rows * in_dim`, values in
    /// `{-1, 0, 1}`. Stored row-major. Unused when [`Self::pq`] is `Some`.
    pub ternary: Vec<i8>,
    /// Per-row scale `s` so the reconstructed weight is `s * ternary`. Unused
    /// when [`Self::pq`] is `Some` (PQ has its own `row_scales`).
    pub row_scales: Vec<f32>,
    /// Centroid vector in input space (`in_dim`), used by the router
    /// (`score = x . centroid`).
    pub centroid: Vec<f32>,
    /// Mean activation norm cached for bias bookkeeping (see [`SparseLayer`]).
    pub mean_input: Vec<f32>,
    /// Product-Quantization codes for this expert's rows. When `None`, the
    /// ternary path is used; when `Some`, the PQ kernel decodes against the
    /// layer's shared [`PqCodebook`] (`SparseLayer::pq_codebook`). The field
    /// is `#[serde(default)]` so old `model.nse` files (ternary-only)
    /// deserialize as `None` (backward-compatible).
    #[serde(default)]
    pub pq: Option<PqExpertData>,
}

/// A fully transmuted linear layer `W [out, in] -> y [out]`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SparseLayer {
    pub out_dim: usize,
    pub in_dim: usize,
    /// Outlier rows kept dense (FP32), always active. Shape `[n_core, in]`.
    pub dense_core: Matrix,
    /// Original output-row ids of the dense core rows.
    pub core_row_ids: Vec<u32>,
    /// Micro-experts covering the non-core rows.
    pub experts: Vec<MicroExpert>,
    /// Static bias `B [out]`: `B[i] = W[i] . mean_input` for prunable rows;
    /// 0 for core rows (they're always computed exactly).
    pub bias: Vec<f32>,
    /// Mean activation over the transmutation corpus, in input space `[in]`.
    pub mean_input: Vec<f32>,
    /// Shared PQ codebook for this layer's experts (when any expert has
    /// `pq: Some`). `None` for ternary-only layers (backward-compatible).
    #[serde(default)]
    pub pq_codebook: Option<PqCodebook>,
}

impl SparseLayer {
    /// Number of micro-experts.
    pub fn num_experts(&self) -> usize {
        self.experts.len()
    }

    /// Total output rows covered (core + all experts), should equal `out_dim`.
    pub fn covered_rows(&self) -> usize {
        let mut n = self.core_row_ids.len();
        for e in &self.experts {
            n += e.row_ids.len();
        }
        n
    }

    /// Approximate fraction of parameters "activated" per token: core (always)
    /// plus the average number of expert rows activated, over `out_dim*in_dim`.
    pub fn active_fraction(&self, avg_experts_on: f32) -> f32 {
        let total = (self.out_dim * self.in_dim) as f32;
        if total == 0.0 {
            return 0.0;
        }
        let core = (self.core_row_ids.len() * self.in_dim) as f32;
        let avg_expert_rows: f32 = self
            .experts
            .iter()
            .map(|e| e.row_ids.len() as f32)
            .sum::<f32>()
            * (avg_experts_on / self.experts.len().max(1) as f32);
        (core + avg_expert_rows * self.in_dim as f32) / total
    }
}

/// A whole transmuted model: the token embedding (kept dense) plus a sparse
/// layer for each matmul weight in the Toy LM, indexed by layer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransmutedModel {
    /// Original config (vocab, dims, layers, heads, ...).
    pub config: ConfigStub,
    /// Dense token embedding `[vocab, dim]` (kept as-is).
    pub token_embed: Matrix,
    /// Per-layer sparse layers: qkv, attn_out, ff_up, ff_down (in that order).
    pub layers: Vec<[SparseLayer; 4]>,
    /// Layernorm gains kept dense (ln1, ln2 per layer, ln_f).
    pub ln1_gain: Vec<Vec<f32>>,
    pub ln2_gain: Vec<Vec<f32>>,
    pub ln_f_gain: Vec<f32>,
}

/// Indices into the per-layer `[SparseLayer; 4]` array.
pub const IDX_QKV: usize = 0;
pub const IDX_ATTN_OUT: usize = 1;
pub const IDX_FF_UP: usize = 2;
pub const IDX_FF_DOWN: usize = 3;
