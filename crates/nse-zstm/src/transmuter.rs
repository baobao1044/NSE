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
    ConfigStub, MicroExpert, SparseLayer, TransmutedModel,
};
use nse_core::tensor::Matrix;
use nse_models::{Config, ToyLm, forward_cached};

use crate::cluster::{cluster, ClusterConfig};
use crate::outlier::{extract, OutlierConfig};
use crate::quantize::quantize_matrix;

/// Full transmutation configuration.
#[derive(Debug, Clone, Default)]
pub struct TransmuteConfig {
    pub outlier: OutlierConfig,
    pub cluster: ClusterConfig,
}

impl TransmuteConfig {
    /// A small default good enough for the POC toy model.
    pub fn poc() -> Self {
        Self {
            outlier: OutlierConfig { fraction: 0.1 },
            cluster: ClusterConfig { num_experts: 0, iters: 10, seed: 7 },
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

    // Stage 3: quantize each expert's rows to ternary.
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
        let centroid: Vec<f32> = cluster_res.centroids.data[k * in_dim..(k + 1) * in_dim].to_vec();
        experts.push(MicroExpert {
            row_ids,
            ternary,
            row_scales,
            centroid,
            mean_input: mean.clone(),
        });
    }

    // Static bias B[out]: for every output row i, B[i] = W[i] . mean_input.
    // Core rows are computed exactly at inference, so their bias entry is 0
    // (the dense core path replaces the bias). Expert rows get the full
    // contribution so the pruned-expert path is unbiased on average.
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
    })
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
}
