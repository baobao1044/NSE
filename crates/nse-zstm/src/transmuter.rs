//! High-level ZSTM driver: transmute a trained Toy LM into a [`TransmutedModel`]
//! and (optionally) serialize it. This assembles the three stages — outlier
//! extraction, k-means clustering, ternary quantization — and precomputes the
//! static bias `B[i] = W[i] . mean_input` for the prunable rows.
//!
//! `mean_input` is estimated from a corpus of input activations; if none is
//! supplied it defaults to zero, which makes the bias zero (the sparse path
//! then matches a model whose non-activated experts contribute nothing — a
//! valid, if lossy, baseline). A real run passes a corpus so the bias is
//! meaningful.

use std::path::Path;

use nse_core::sparse::{
    ConfigStub, MicroExpert, PqExpertData, SparseLayer, TransmutedModel,
};
use nse_core::tensor::Matrix;
use nse_models::{ToyLm, forward_cached};

use crate::cluster::{cluster, ClusterConfig};
use crate::outlier::{extract, OutlierConfig};
use crate::pq::{encode_pq, train_pq};
use crate::quantize::quantize_matrix;

/// Quantization scheme for ZSTM stage 3 (weight quantization).
///
/// Controls how each micro-expert's owned rows are compressed after
/// outlier extraction + clustering. The scheme is set on [`TransmuteConfig`]
/// and applies uniformly to every weight matrix of the model.
#[derive(Debug, Clone)]
pub enum QuantSchemeConfig {
    /// Ternary `{-1, 0, 1}` + per-row scale (BitNet-style). Default — matches
    /// all prior paper results and is backward-compatible with existing
    /// `model.nse` files.
    Ternary,
    /// Product Quantization: each row splits into `num_sub_vectors`
    /// sub-vectors, each quantized against a shared 8-bit codebook (256
    /// centroids) trained per `SparseLayer` on the normalized residual
    /// rows. A per-row scale (`mean(|w|)`) captures magnitude exactly, so
    /// the codebook only needs to represent shape — 256 levels per
    /// sub-vector vs ternary's 3. Targets the +82% sparse PPL degradation
    /// (paper 5.2) by replacing coarse ternary with a learned codebook.
    Pq {
        /// Number of sub-vectors `M`. `in_dim` must be divisible by `M`
        /// (otherwise the largest divisor `<= M` is used). `M=4` for
        /// `dim=64` gives `sub_dim=16` (plan default).
        num_sub_vectors: usize,
        /// Bits per code. `8` → 256 centroids per sub-codebook (plan default).
        nbits: usize,
        /// K-means iterations per sub-vector codebook.
        iters: usize,
        /// Deterministic centroid init seed.
        seed: u64,
    },
}

impl Default for QuantSchemeConfig {
    fn default() -> Self {
        QuantSchemeConfig::Ternary
    }
}

/// Full transmutation configuration.
#[derive(Debug, Clone, Default)]
pub struct TransmuteConfig {
    pub outlier: OutlierConfig,
    pub cluster: ClusterConfig,
    pub quant: QuantSchemeConfig,
}

impl TransmuteConfig {
    /// A small default good enough for the POC toy model (ternary scheme).
    pub fn poc() -> Self {
        Self {
            outlier: OutlierConfig { fraction: 0.1 },
            cluster: ClusterConfig { num_experts: 0, iters: 10, seed: 7 },
            quant: QuantSchemeConfig::Ternary,
        }
    }

    /// PQ variant of [`poc`]: same outlier/cluster defaults but with Product
    /// Quantization (M=4 sub-vectors, 8-bit codebook, 20 k-means iters).
    /// Used by the sparse-quality recovery experiments (Phase 7 / M8).
    pub fn pq() -> Self {
        Self {
            outlier: OutlierConfig { fraction: 0.1 },
            cluster: ClusterConfig { num_experts: 0, iters: 10, seed: 7 },
            quant: QuantSchemeConfig::Pq {
                num_sub_vectors: 4,
                nbits: 8,
                iters: 20,
                seed: 7,
            },
        }
    }
}

/// Transmute a trained Toy LM into a [`TransmutedModel`].
///
/// `corpus` (if given) is tokenized and run through the dense model to collect
/// input activations for each transmuted weight; the per-layer mean input is
/// used to precompute the static bias and to seed the expert centroids with
/// realistic routing targets. If `corpus` is `None`, means are zero.
pub fn transmute(
    lm: &ToyLm,
    corpus: Option<&[u8]>,
    cfg: &TransmuteConfig,
) -> anyhow::Result<TransmutedModel> {
    let config = ConfigStub {
        vocab_size: lm.config.vocab_size,
        dim: lm.config.dim,
        num_layers: lm.config.num_layers,
        num_heads: lm.config.num_heads,
        max_seq_len: lm.config.max_seq_len,
        ff_dim: lm.config.ff_dim,
    };

    // Collect mean input activations per transmuted weight, averaged over a
    // few forward windows of the corpus. The 4 matmuls per layer feed: qkv,
    // attn_out, ff_up, ff_down.
    let mean_inputs = mean_inputs_for(lm, corpus);

    let mut layers: Vec<[SparseLayer; 4]> = Vec::with_capacity(lm.config.num_layers);
    for l in 0..lm.config.num_layers {
        let qkv = transmute_matrix(&lm.weights.qkv[l], &mean_inputs.qkv[l], cfg)?;
        let attn_out = transmute_matrix(&lm.weights.attn_out[l], &mean_inputs.attn_out[l], cfg)?;
        let ff_up = transmute_matrix(&lm.weights.ff_up[l], &mean_inputs.ff_up[l], cfg)?;
        let ff_down = transmute_matrix(&lm.weights.ff_down[l], &mean_inputs.ff_down[l], cfg)?;
        layers.push([qkv, attn_out, ff_up, ff_down]);
    }

    Ok(TransmutedModel {
        config,
        token_embed: lm.weights.token_embed.clone(),
        layers,
        ln1_gain: lm.weights.ln1_gain.clone(),
        ln2_gain: lm.weights.ln2_gain.clone(),
        ln_f_gain: lm.weights.ln_f_gain.clone(),
    })
}

/// Transmute a single weight matrix `W [out, in]` into a [`SparseLayer`].
/// `mean_input` (length `in`) is the expected activation; it seeds the bias.
///
/// The quantization scheme is selected by `cfg.quant`:
/// - [`QuantSchemeConfig::Ternary`] (default): ternary `{-1,0,1}` + per-row
///   scale per expert (the original M1–M7 path).
/// - [`QuantSchemeConfig::Pq`]: trains one shared PQ codebook on the
///   normalized residual rows, encodes each expert's rows against it, and
///   stores the codebook on the returned `SparseLayer::pq_codebook`.
pub fn transmute_matrix(
    w: &Matrix,
    mean_input: &[f32],
    cfg: &TransmuteConfig,
) -> anyhow::Result<SparseLayer> {
    let out_dim = w.rows;
    let in_dim = w.cols;
    let mean = if mean_input.len() == in_dim {
        mean_input.to_vec()
    } else {
        vec![0.0; in_dim]
    };

    // Stage 1: outlier extraction.
    let outlier = extract(w, &cfg.outlier)?;
    let dense_core = outlier.dense_core.clone();
    let core_row_ids = outlier.core_row_ids.clone();
    let residual = &outlier.residual;
    let residual_row_ids = &outlier.residual_row_ids;

    // Stage 2: cluster residual rows into micro-experts.
    let cluster_res = cluster(residual, &cfg.cluster)?;
    let n_experts = cluster_res.centroids.rows;

    // Stage 3: quantize each expert's rows — branch on scheme.
    let (experts, pq_codebook) = match &cfg.quant {
        QuantSchemeConfig::Ternary => {
            let experts = build_ternary_experts(
                &cluster_res, residual, residual_row_ids, in_dim, &mean, n_experts,
            );
            (experts, None)
        }
        QuantSchemeConfig::Pq { num_sub_vectors, nbits, iters, seed } => {
            let (experts, codebook) = build_pq_experts(
                &cluster_res,
                residual,
                residual_row_ids,
                in_dim,
                &mean,
                n_experts,
                *num_sub_vectors,
                *nbits,
                *iters,
                *seed,
            )?;
            (experts, Some(codebook))
        }
    };

    // Static bias B[out]: for every output row i, B[i] = W[i] . mean_input.
    // Core rows are computed exactly at inference, so their bias entry is 0
    // (the dense core path replaces the bias). Expert rows get the full
    // contribution so the pruned-expert path is unbiased on average.
    // This is scheme-agnostic: it uses the *original* weights, not the
    // quantized form, so ternary and PQ layers share the same bias.
    let mut bias = vec![0.0f32; out_dim];
    for i in 0..out_dim {
        let row = &w.data[i * in_dim..(i + 1) * in_dim];
        bias[i] = row.iter().zip(mean.iter()).map(|(a, b)| a * b).sum();
    }
    // Zero out bias for core rows (handled by dense path).
    for &r in &core_row_ids {
        bias[r as usize] = 0.0;
    }

    Ok(SparseLayer {
        out_dim,
        in_dim,
        dense_core,
        core_row_ids,
        experts,
        bias,
        mean_input: mean,
        pq_codebook,
    })
}

/// Build micro-experts with ternary quantization (the default M1–M7 path).
/// Each expert's member rows are ternary-quantized (`{-1,0,1}` + per-row
/// scale); `pq` is `None` so the kernel dispatches to the ternary path.
fn build_ternary_experts(
    cluster_res: &crate::cluster::ClusterResult,
    residual: &Matrix,
    residual_row_ids: &[u32],
    in_dim: usize,
    mean: &[f32],
    n_experts: usize,
) -> Vec<MicroExpert> {
    let mut experts = Vec::with_capacity(n_experts);
    for (k, members) in cluster_res.members.iter().enumerate() {
        let row_ids: Vec<u32> = members
            .iter()
            .map(|&r| residual_row_ids[r])
            .collect();
        // Build a local matrix of the member rows for quantization.
        let mut block = Matrix::zeros(members.len(), in_dim);
        for (i, &r) in members.iter().enumerate() {
            block.data[i * in_dim..(i + 1) * in_dim]
                .copy_from_slice(&residual.data[r * in_dim..(r + 1) * in_dim]);
        }
        let (ternary, row_scales) = quantize_matrix(&block);
        // Centroid from k-means (already in input space).
        let centroid: Vec<f32> =
            cluster_res.centroids.data[k * in_dim..(k + 1) * in_dim].to_vec();
        experts.push(MicroExpert {
            row_ids,
            ternary,
            row_scales,
            centroid,
            mean_input: mean.to_vec(),
            pq: None,
        });
    }
    experts
}

/// Build micro-experts with Product Quantization + a shared layer-level
/// codebook (Phase 7 / M8 sparse-quality recovery path).
///
/// Per-row scale `s = mean(|w|)` captures magnitude exactly; the codebook is
/// trained on the normalized rows `w / s` so it only needs to represent
/// shape. Each expert stores its rows' PQ codes + scales; the shared
/// codebook is returned separately and stored on `SparseLayer::pq_codebook`.
fn build_pq_experts(
    cluster_res: &crate::cluster::ClusterResult,
    residual: &Matrix,
    residual_row_ids: &[u32],
    in_dim: usize,
    mean: &[f32],
    n_experts: usize,
    num_sub_vectors_req: usize,
    nbits: usize,
    iters: usize,
    seed: u64,
) -> anyhow::Result<(Vec<MicroExpert>, nse_core::sparse::PqCodebook)> {
    let n_residual = residual.rows;
    if n_residual == 0 {
        anyhow::bail!("PQ requires at least one residual row to train the codebook");
    }
    // `in_dim` must be divisible by `M`; fall back to the largest divisor
    // <= the request (or M=1 = plain VQ if `in_dim` is prime / no divisor).
    let m = (1..=num_sub_vectors_req)
        .rev()
        .find(|&m| in_dim % m == 0)
        .unwrap_or(1);

    // Per-row scale = mean(|w|); normalized rows = w / scale (codebook shape).
    let mut scales: Vec<f32> = Vec::with_capacity(n_residual);
    let mut normalized: Vec<Vec<f32>> = Vec::with_capacity(n_residual);
    for r in 0..n_residual {
        let row = &residual.data[r * in_dim..(r + 1) * in_dim];
        let scale = row.iter().map(|v| v.abs()).sum::<f32>() / in_dim.max(1) as f32;
        let s = scale.max(1e-8); // avoid div-by-zero on near-zero rows
        scales.push(scale);
        normalized.push(row.iter().map(|&v| v / s).collect());
    }

    // Train one shared codebook on the normalized residual rows.
    let codebook = train_pq(&normalized, m, nbits, iters, seed);

    // Encode each expert's member rows against the shared codebook.
    let mut experts = Vec::with_capacity(n_experts);
    for (k, members) in cluster_res.members.iter().enumerate() {
        let row_ids: Vec<u32> = members
            .iter()
            .map(|&r| residual_row_ids[r])
            .collect();
        let mut codes: Vec<u8> = Vec::with_capacity(members.len() * m);
        let mut row_scales: Vec<f32> = Vec::with_capacity(members.len());
        for &r in members {
            row_scales.push(scales[r]);
            let row_codes = encode_pq(&normalized[r], &codebook);
            codes.extend_from_slice(&row_codes);
        }
        let centroid: Vec<f32> =
            cluster_res.centroids.data[k * in_dim..(k + 1) * in_dim].to_vec();
        experts.push(MicroExpert {
            row_ids,
            ternary: vec![],     // unused on the PQ path
            row_scales: vec![],  // unused on the PQ path (PQ has its own)
            centroid,
            mean_input: mean.to_vec(),
            pq: Some(PqExpertData {
                codes,
                row_scales,
                num_sub_vectors: m,
            }),
        });
    }
    Ok((experts, codebook))
}

/// Collect mean input activations for each transmuted weight, averaged over a
/// few forward windows of the corpus. The 4 matmuls per layer feed: qkv, attn_out,
/// ff_up, ff_down.
struct LayerMeanInputs {
    qkv: Vec<Vec<f32>>,
    attn_out: Vec<Vec<f32>>,
    ff_up: Vec<Vec<f32>>,
    ff_down: Vec<Vec<f32>>,
}

fn mean_inputs_for(lm: &ToyLm, corpus: Option<&[u8]>) -> LayerMeanInputs {
    let cfg = &lm.config;
    let zero = vec![0.0f32; cfg.dim];
    let mut acc = LayerMeanInputs {
        qkv: vec![zero.clone(); cfg.num_layers],
        attn_out: vec![vec![0.0; cfg.dim]; cfg.num_layers],
        ff_up: vec![vec![0.0; cfg.dim]; cfg.num_layers],
        ff_down: vec![vec![0.0; cfg.ff_dim]; cfg.num_layers],
    };

    let corpus = match corpus {
        Some(c) if !c.is_empty() => c,
        _ => return acc, // zero means → zero bias
    };
    let tok = nse_models::Tokenizer::from_corpus(corpus);
    let ids = tok.encode(corpus);
    let seq = cfg.max_seq_len.min(ids.len().saturating_sub(1)).max(2);
    if ids.len() < seq + 1 {
        return acc;
    }

    let tokens: Vec<u32> = ids[..seq].to_vec();
    // Reuse the dense forward to get the layer caches; we only need the
    // normalized pre-activation vectors, so call the autograd forward.
    let (cache, _logits) = forward_cached(lm, &tokens);
    let t = tokens.len();

    for l in 0..cfg.num_layers {
        let lc = &cache.layers[l];
        // qkv input = ln1_out [t, dim]  → mean over t
        acc.qkv[l] = mean_over_rows(&lc.ln1_out, t, cfg.dim);
        // attn_out input = attn output [t, dim] (before projection)
        acc.attn_out[l] = mean_over_rows(&lc.attn_out, t, cfg.dim);
        // ff_up input = ln2_out [t, dim]
        acc.ff_up[l] = mean_over_rows(&lc.ln2_out, t, cfg.dim);
        // ff_down input = ff_up_act [t, ff_dim]
        acc.ff_down[l] = mean_over_rows(&lc.ff_up_act, t, cfg.ff_dim);
    }
    acc
}

fn mean_over_rows(flat: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; cols];
    if rows == 0 {
        return out;
    }
    for r in 0..rows {
        for j in 0..cols {
            out[j] += flat[r * cols + j];
        }
    }
    for v in out.iter_mut() {
        *v /= rows as f32;
    }
    out
}

/// Serialize a [`TransmutedModel`] to a JSON file (POC container; the binary
/// `.nse` format is wired in M4+ once the RIE/LLER read path lands).
pub fn save_transmuted(model: &TransmutedModel, path: impl AsRef<Path>) -> anyhow::Result<()> {
    let json = serde_json::to_vec(model)?;
    std::fs::write(path, json)?;
    Ok(())
}

/// Load a [`TransmutedModel`] from a JSON file written by [`save_transmuted`].
pub fn load_transmuted(path: impl AsRef<Path>) -> anyhow::Result<TransmutedModel> {
    let bytes = std::fs::read(path)?;
    let model = serde_json::from_slice(&bytes)?;
    Ok(model)
}

#[cfg(test)]
mod tests {
    use super::*;
    use nse_models::Config;

    #[test]
    fn transmute_covers_all_rows() {
        let cfg = Config {
            vocab_size: 8,
            dim: 4,
            num_layers: 1,
            num_heads: 2,
            max_seq_len: 8,
            ff_dim: 6,
        };
        let lm = ToyLm::init_random(cfg, 3);
        let tcfg = TransmuteConfig::poc();
        let tm = transmute(&lm, None, &tcfg).unwrap();
        assert_eq!(tm.config.dim, 4);
        for layer in &tm.layers {
            for sl in layer {
                assert_eq!(sl.covered_rows(), sl.out_dim,
                    "covered rows must equal out_dim");
            }
        }
        // bias zeroed for core rows.
        for layer in &tm.layers {
            for sl in layer {
                for &r in &sl.core_row_ids {
                    assert_eq!(sl.bias[r as usize], 0.0);
                }
            }
        }
    }

    #[test]
    fn save_load_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("tm.json");
        let cfg = Config {
            vocab_size: 8,
            dim: 4,
            num_layers: 1,
            num_heads: 2,
            max_seq_len: 8,
            ff_dim: 6,
        };
        let lm = ToyLm::init_random(cfg, 3);
        let tm = transmute(&lm, None, &TransmuteConfig::poc()).unwrap();
        save_transmuted(&tm, &path).unwrap();
        let tm2 = load_transmuted(&path).unwrap();
        assert_eq!(tm2.config, tm.config);
        assert_eq!(tm2.token_embed, tm.token_embed);
    }

    /// PQ path: `transmute` with `QuantSchemeConfig::Pq` produces a
    /// `TransmutedModel` where every non-empty layer has `pq_codebook: Some`
    /// and every expert has `pq: Some` (no expert left on the ternary path
    /// by accident). Covered-rows invariant still holds, and the bias is
    /// still zeroed on core rows (scheme-agnostic).
    #[test]
    fn transmute_pq_roundtrip() {
        let cfg = Config {
            vocab_size: 16,
            dim: 32,
            num_layers: 1,
            num_heads: 2,
            max_seq_len: 8,
            ff_dim: 64,
        };
        let lm = ToyLm::init_random(cfg, 7);
        let tcfg = TransmuteConfig::pq();
        let tm = transmute(&lm, None, &tcfg).unwrap();

        // dim=32 is divisible by M=4 → sub_dim=8 (no fallback).
        for layer in &tm.layers {
            for sl in layer {
                // Covered-rows invariant: scheme-agnostic.
                assert_eq!(sl.covered_rows(), sl.out_dim,
                    "PQ: covered rows must equal out_dim");
                // Bias still zeroed on core rows (uses original W, not quant).
                for &r in &sl.core_row_ids {
                    assert_eq!(sl.bias[r as usize], 0.0, "PQ: core-row bias must be 0");
                }
                // If the layer has any experts, the codebook must be present.
                if !sl.experts.is_empty() {
                    let cb = sl.pq_codebook.as_ref()
                        .expect("PQ layer must have a shared codebook");
                    assert_eq!(cb.num_sub_vectors, 4, "PQ M should be 4 for dim 32/ff 64");
                    assert_eq!(cb.nbits, 8, "PQ nbits should be 8");
                    // sub_dim = in_dim / M; the 4 matmuls have different
                    // in_dims (qkv/attn_out/ff_up feed `dim`, ff_down feeds
                    // `ff_dim`), so check consistency rather than hardcoding.
                    assert_eq!(cb.sub_dim, sl.in_dim / cb.num_sub_vectors,
                        "PQ sub_dim must equal in_dim / M");
                    assert_eq!(
                        cb.codebook.len(),
                        cb.num_sub_vectors * cb.num_entries() * cb.sub_dim,
                        "PQ codebook size"
                    );
                    for e in &sl.experts {
                        let pq = e.pq.as_ref()
                            .expect("PQ layer experts must have pq: Some");
                        assert_eq!(pq.num_sub_vectors, 4, "expert num_sub_vectors");
                        assert_eq!(
                            pq.codes.len(),
                            e.row_ids.len() * 4,
                            "PQ codes length = rows * M"
                        );
                        assert_eq!(
                            pq.row_scales.len(),
                            e.row_ids.len(),
                            "PQ row_scales length = rows"
                        );
                    }
                }
            }
        }
    }

    /// PQ save → load roundtrip preserves the codebook + expert codes
    /// (serde backward-compat: the `#[serde(default)] Option` fields survive
    /// the JSON roundtrip).
    #[test]
    fn transmute_pq_save_load_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("tm_pq.json");
        let cfg = Config {
            vocab_size: 16,
            dim: 32,
            num_layers: 1,
            num_heads: 2,
            max_seq_len: 8,
            ff_dim: 64,
        };
        let lm = ToyLm::init_random(cfg, 7);
        let tm = transmute(&lm, None, &TransmuteConfig::pq()).unwrap();
        save_transmuted(&tm, &path).unwrap();
        let tm2 = load_transmuted(&path).unwrap();

        // Spot-check: codebooks + a few experts match exactly.
        for (l, (la, lb)) in tm.layers.iter().zip(tm2.layers.iter()).enumerate() {
            for (m, (sa, sb)) in la.iter().zip(lb.iter()).enumerate() {
                assert_eq!(sa.pq_codebook, sb.pq_codebook,
                    "PQ codebook mismatch at layer {l} matmul {m}");
                assert_eq!(sa.experts.len(), sb.experts.len());
                for (e1, e2) in sa.experts.iter().zip(sb.experts.iter()) {
                    assert_eq!(e1.pq, e2.pq, "PQ expert data mismatch");
                }
            }
        }
    }

    /// `in_dim` not divisible by the requested `M` falls back to the largest
    /// divisor `<= M`. `dim=30` (divisors ≤4: {1,2,3}) → `M=3, sub_dim=10`;
    /// `ff_dim=60` (divisible by 4) → stays `M=4, sub_dim=15`. Tests the
    /// geometry fallback so the CLI doesn't panic on odd dims.
    #[test]
    fn transmute_pq_fallback_m_for_undivisible_in_dim() {
        let cfg = Config {
            vocab_size: 12,
            dim: 30,        // not divisible by 4; largest divisor ≤4 is 3
            num_layers: 1,
            num_heads: 2,
            max_seq_len: 8,
            ff_dim: 60,     // divisible by 4 → stays M=4
        };
        let lm = ToyLm::init_random(cfg, 7);
        let tcfg = TransmuteConfig {
            outlier: OutlierConfig { fraction: 0.1 },
            cluster: ClusterConfig { num_experts: 0, iters: 10, seed: 7 },
            quant: QuantSchemeConfig::Pq {
                num_sub_vectors: 4,
                nbits: 8,
                iters: 10,
                seed: 7,
            },
        };
        let tm = transmute(&lm, None, &tcfg).unwrap();
        // qkv/attn_out/ff_up feed dim=30 → fallback to M=3 (sub_dim=10).
        let sl_qkv = &tm.layers[0][0];
        if let Some(cb) = &sl_qkv.pq_codebook {
            assert_eq!(cb.num_sub_vectors, 3, "dim=30 should fall back to M=3");
            assert_eq!(cb.sub_dim, 10);
        }
        // ff_down feeds ff_dim=60 → stays M=4 (sub_dim=15).
        let sl_ff_down = &tm.layers[0][3];
        if let Some(cb) = &sl_ff_down.pq_codebook {
            assert_eq!(cb.num_sub_vectors, 4, "ff_dim=60 should keep M=4");
            assert_eq!(cb.sub_dim, 15);
        }
    }
}
