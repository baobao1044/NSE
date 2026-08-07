//! Sparse forward pass over a [`TransmutedModel`], mirroring the dense
//! `ToyLm::forward` but replacing each of the 4 per-layer matmuls (qkv,
//! attn_out, ff_up, ff_down) with a sparse linear via [`nse_rie::sparse_linear`].
//!
//! Everything else — token embedding, layer norm, causal self-attention,
//! GELU FFN, residual additions, tied output head — is identical to the dense
//! forward, so the *only* source of numerical difference is:
//!   1. ternary quantization of the expert weights, and
//!   2. expert pruning (mitigated by the static bias).
//!
//! With `route_all` (every expert activated), difference (2) vanishes and only
//! ternary quantization error remains — the upper bound on sparse quality.

use nse_core::sparse::{SparseLayer, TransmutedModel, IDX_ATTN_OUT, IDX_FF_DOWN, IDX_FF_UP, IDX_QKV};
use nse_rie::{route_all, route_by_ratio, RouterConfig};

const EPS_LN: f32 = 1e-5;

/// How experts are activated during the sparse forward.
#[derive(Debug, Clone, Copy)]
pub enum Activation {
    /// Activate every expert (upper bound; only ternary error remains).
    All,
    /// Adaptive threshold routing: keep experts with score >= max*ratio,
    /// capped at `max_k`. Pruned experts' contribution is covered by the bias.
    Threshold { ratio: f32, max_k: usize },
}

impl Default for Activation {
    fn default() -> Self {
        // POC default: keep all experts so the headline PPL reflects only the
        // quantization cost (the honest upper bound of zero-shot transmutation).
        Activation::All
    }
}

/// Runtime options for the sparse forward: compute kernel + MIPS index.
#[derive(Debug, Clone, Copy)]
pub struct SparseOptions {
    /// Compute kernel: scalar (canonical) or AVX2 (auto-detected).
    pub kernel: nse_rie::KernelKind,
    /// MIPS index backend: brute-force (canonical) or HNSW.
    pub index: nse_rie::IndexKind,
}

impl Default for SparseOptions {
    fn default() -> Self {
        Self {
            kernel: nse_rie::KernelKind::Scalar,
            index: nse_rie::IndexKind::Brute,
        }
    }
}

/// Sparse forward: returns logits `[seq, vocab]` (row-major), with default
/// options (scalar kernel, brute-force index). Backward-compatible entry point.
pub fn sparse_forward(tm: &TransmutedModel, tokens: &[u32], act: Activation) -> Vec<f32> {
    sparse_forward_with_options(tm, tokens, act, SparseOptions::default())
}

/// Sparse forward with explicit runtime options (kernel + index).
pub fn sparse_forward_with_options(
    tm: &TransmutedModel,
    tokens: &[u32],
    act: Activation,
    opts: SparseOptions,
) -> Vec<f32> {
    let cfg = &tm.config;
    let seq = tokens.len();
    let d = cfg.dim;
    let v = cfg.vocab_size;
    let nh = cfg.num_heads;

    // 1. Token embedding lookup (dense, unchanged).
    let mut x = vec![0.0f32; seq * d];
    for (t, &tok) in tokens.iter().enumerate() {
        let tok = (tok as usize).min(v - 1);
        for j in 0..d {
            x[t * d + j] = tm.token_embed.data[tok * d + j];
        }
    }

    // 2. Per-layer block.
    for layer in 0..cfg.num_layers {
        let block = &tm.layers[layer];
        let ln1 = &tm.ln1_gain[layer];
        let ln2 = &tm.ln2_gain[layer];

        // --- Attention sub-block ---
        let h = layernorm(&x, seq, d, ln1); // [seq, dim]
        let qkv_out = sparse_linear_seq(&block[IDX_QKV], &h, act, opts); // [seq, 3*dim]
        let (q, k, v_proj) = split_qkv(&qkv_out, seq, d);
        let attn = causal_self_attention(&q, &k, &v_proj, seq, d, nh);
        let attn_proj = sparse_linear_seq(&block[IDX_ATTN_OUT], &attn, act, opts); // [seq, dim]
        for i in 0..seq * d {
            x[i] += attn_proj[i];
        }

        // --- FFN sub-block ---
        let h2 = layernorm(&x, seq, d, ln2); // [seq, dim]
        let mut up = sparse_linear_seq(&block[IDX_FF_UP], &h2, act, opts); // [seq, ff_dim]
        gelu_inplace(&mut up);
        let down = sparse_linear_seq(&block[IDX_FF_DOWN], &up, act, opts); // [seq, dim]
        for i in 0..seq * d {
            x[i] += down[i];
        }
    }

    // 3. Final layernorm.
    let x = layernorm(&x, seq, d, &tm.ln_f_gain);

    // 4. Tied head (dense, unchanged): logits = x @ E^T.
    let mut logits = vec![0.0f32; seq * v];
    for t in 0..seq {
        for w in 0..v {
            let mut s = 0.0;
            for j in 0..d {
                s += x[t * d + j] * tm.token_embed.data[w * d + j];
            }
            logits[t * v + w] = s;
        }
    }
    logits
}

/// Run [`nse_rie::sparse_linear_with_kernel`] over each row of `h [seq, in]`,
/// stacking outputs into `[seq, out]`. Selects the index backend and kernel
/// from `opts`.
fn sparse_linear_seq(sl: &SparseLayer, h: &[f32], act: Activation, opts: SparseOptions) -> Vec<f32> {
    let in_dim = sl.in_dim;
    let out_dim = sl.out_dim;
    let mut out = vec![0.0f32; h.len() / in_dim * out_dim];
    // Build the index backend once for the layer (HNSW build is amortized over
    // all positions in the sequence).
    let hnsw = match opts.index {
        nse_rie::IndexKind::Hnsw => Some(nse_rie::build_hnsw_for_layer(sl)),
        nse_rie::IndexKind::Brute => None,
    };
    let brute = nse_rie::MipsIndex::new(&sl.experts);
    for t in 0..(h.len() / in_dim) {
        let x = &h[t * in_dim..(t + 1) * in_dim];
        // Query the chosen backend for all hits (sorted descending).
        let hits: Vec<nse_rie::Hit> = match &hnsw {
            Some(hi) => nse_rie::MipsQuery::query_all(hi, x),
            None => nse_rie::MipsQuery::query_all(&brute, x),
        };
        let activated = match act {
            Activation::All => route_all(&hits),
            Activation::Threshold { ratio, max_k } => {
                route_by_ratio(&hits, &RouterConfig { threshold_ratio: ratio, max_k })
            }
        };
        let ids: Vec<usize> = activated.iter().map(|h| h.expert_id).collect();
        let y = nse_rie::sparse_linear_with_kernel(sl, x, &ids, opts.kernel);
        out[t * out_dim..(t + 1) * out_dim].copy_from_slice(&y);
    }
    out
}

fn layernorm(x: &[f32], seq: usize, dim: usize, gain: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0f32; seq * dim];
    for t in 0..seq {
        let row = &x[t * dim..(t + 1) * dim];
        let mean = row.iter().sum::<f32>() / dim as f32;
        let var = row.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / dim as f32;
        let inv = 1.0 / (var + EPS_LN).sqrt();
        for j in 0..dim {
            out[t * dim + j] = gain[j] * (row[j] - mean) * inv;
        }
    }
    out
}

fn split_qkv(qkv: &[f32], seq: usize, dim: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut q = vec![0.0; seq * dim];
    let mut k = vec![0.0; seq * dim];
    let mut v = vec![0.0; seq * dim];
    for t in 0..seq {
        for j in 0..dim {
            q[t * dim + j] = qkv[t * 3 * dim + j];
            k[t * dim + j] = qkv[t * 3 * dim + dim + j];
            v[t * dim + j] = qkv[t * 3 * dim + 2 * dim + j];
        }
    }
    (q, k, v)
}

fn causal_self_attention(q: &[f32], k: &[f32], v: &[f32], seq: usize, dim: usize, num_heads: usize) -> Vec<f32> {
    let head_dim = dim / num_heads;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut out = vec![0.0f32; seq * dim];
    for h in 0..num_heads {
        let off = h * head_dim;
        for i in 0..seq {
            let mut weights = vec![0.0f32; seq];
            let mut max = f32::NEG_INFINITY;
            for j in 0..=i {
                let mut s = 0.0;
                for d2 in 0..head_dim {
                    s += q[i * dim + off + d2] * k[j * dim + off + d2];
                }
                s *= scale;
                weights[j] = s;
                if s > max {
                    max = s;
                }
            }
            let mut sum = 0.0;
            for j in 0..=i {
                weights[j] = (weights[j] - max).exp();
                sum += weights[j];
            }
            let inv = if sum > 0.0 { 1.0 / sum } else { 0.0 };
            for j in 0..=i {
                weights[j] *= inv;
            }
            for d2 in 0..head_dim {
                let mut acc = 0.0;
                for j in 0..=i {
                    acc += weights[j] * v[j * dim + off + d2];
                }
                out[i * dim + off + d2] = acc;
            }
        }
    }
    out
}

fn gelu_inplace(x: &mut [f32]) {
    const SQRT2_INV: f32 = 0.7071067811865475;
    for v in x.iter_mut() {
        let t = *v * SQRT2_INV;
        let inner = 1.0 + 0.044715 * t * t;
        let tanh = (2.0 * t * inner).tanh();
        *v = 0.5 * *v * (1.0 + tanh);
    }
}
