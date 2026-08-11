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

/// Bias application policy (Phase 8 / M9). Controls how the static bias is
/// computed offline and applied online:
///
/// - [`BiasMode::Mean`] (default, legacy): precompute `B[i] = W[i] . mean_input`
///   from a single calibration window. At inference the bias is applied
///   **pruned-only** (when `row_to_expert` is non-empty — the S1 correctness
///   fix) or **unconditionally** (legacy, empty `row_to_expert`). This
///   reproduces the M8 behavior for old `.nse` files.
/// - [`BiasMode::Adaptive`]: in addition to the mean bias, train a VQ
///   codebook (PQ machinery, `M=1`, 256 centroids) on calibration
///   activations per matmul and precompute a per-token `bias_table[c][i] =
///   W_quant[i] . centroid[c]`. At inference the token's `x` is encoded
///   against the activation codebook into a code `c`, and `bias_table[c][i]`
///   is added to pruned rows — a per-token, codebook-keyed bias instead of
///   the corpus mean. This is the "PQ là foundation cho cả 2" path.
#[derive(Debug, Clone)]
pub enum BiasMode {
    /// Fixed mean-input bias (`B[i] = W[i] . mean_input`), pruned-only when
    /// `row_to_expert` is set. Default — reproduces M8 PPL for old `.nse`.
    Mean,
    /// Per-token adaptive bias via an activation VQ codebook + precomputed
    /// `bias_table`. Targets the `W[i] . (mean_input - x)` error on the
    /// pruned path by replacing the corpus mean with a codebook-keyed
    /// per-token estimate.
    Adaptive {
        /// VQ codebook bits (8 → 256 centroids in input space).
        nbits: usize,
        /// K-means iterations for the activation codebook.
        iters: usize,
        /// Deterministic centroid init seed.
        seed: u64,
    },
}

impl Default for BiasMode {
    fn default() -> Self {
        BiasMode::Mean
    }
}

/// Full transmutation configuration.
#[derive(Debug, Clone, Default)]
pub struct TransmuteConfig {
    pub outlier: OutlierConfig,
    pub cluster: ClusterConfig,
    pub quant: QuantSchemeConfig,
    pub bias_mode: BiasMode,
}

impl TransmuteConfig {
    /// A small default good enough for the POC toy model (ternary scheme,
    /// mean bias — legacy).
    pub fn poc() -> Self {
        Self {
            outlier: OutlierConfig { fraction: 0.1 },
            cluster: ClusterConfig { num_experts: 0, iters: 10, seed: 7 },
            quant: QuantSchemeConfig::Ternary,
            bias_mode: BiasMode::Mean,
        }
    }

    /// PQ variant of [`poc`]: same outlier/cluster defaults but with Product
    /// Quantization (M=4 sub-vectors, 8-bit codebook, 20 k-means iters).
    /// Used by the sparse-quality recovery experiments (Phase 7 / M8).
    /// Bias mode stays `Mean` (pruned-only with the S1 fix) — pass
    /// [`TransmuteConfig::adaptive`] for the Phase 8 adaptive path.
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
            bias_mode: BiasMode::Mean,
        }
    }

    /// Adaptive-bias variant of [`pq`]: PQ weights (M=4, 8-bit) + an
    /// activation VQ codebook (M=1, 8-bit, 256 centroids) for the
    /// per-token bias table. The Phase 8 / M9 path — "PQ là foundation cho
    /// cả 2" (weight codebook + activation codebook).
    pub fn adaptive() -> Self {
        Self {
            outlier: OutlierConfig { fraction: 0.1 },
            cluster: ClusterConfig { num_experts: 0, iters: 10, seed: 7 },
            quant: QuantSchemeConfig::Pq {
                num_sub_vectors: 4,
                nbits: 8,
                iters: 20,
                seed: 7,
            },
            bias_mode: BiasMode::Adaptive {
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
///
/// `calibration_corpus` selects the activation source for bias + (if
/// `bias_mode == Adaptive`) the activation codebook. When `None`, it falls
/// back to `corpus` (the M8-and-earlier behavior — one corpus does triple
/// duty). When `Some`, the bias/codebook are calibrated on a distinct set
/// from the transmutation corpus, the Phase 8 calibration discipline.
pub fn transmute(
    lm: &ToyLm,
    corpus: Option<&[u8]>,
    calibration_corpus: Option<&[u8]>,
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

    // Collect per-token activation rows over (possibly multiple) sliding
    // windows of the calibration corpus. The 4 matmuls per layer feed:
    // qkv, attn_out, ff_up, ff_down. The mean-input bias is the column-wise
    // mean of these rows; the adaptive path additionally trains a VQ
    // codebook on them (handled per-matmul in `transmute_matrix`).
    let cal_corpus = calibration_corpus.or(corpus).filter(|c| !c.is_empty());
    let act_rows = collect_activations(lm, cal_corpus);

    let mut layers: Vec<[SparseLayer; 4]> = Vec::with_capacity(lm.config.num_layers);
    for l in 0..lm.config.num_layers {
        // Mean input per matmul = column-wise mean of the collected rows
        // (matches the M8 single-window mean when only one window fits).
        let mean_qkv = mean_of_rows(&act_rows.qkv[l], lm.config.dim);
        let mean_attn = mean_of_rows(&act_rows.attn_out[l], lm.config.dim);
        let mean_ff_up = mean_of_rows(&act_rows.ff_up[l], lm.config.dim);
        let mean_ff_down = mean_of_rows(&act_rows.ff_down[l], lm.config.ff_dim);

        let qkv = transmute_matrix_with_cal(
            &lm.weights.qkv[l], &mean_qkv, Some(&act_rows.qkv[l]), cfg,
        )?;
        let attn_out = transmute_matrix_with_cal(
            &lm.weights.attn_out[l], &mean_attn, Some(&act_rows.attn_out[l]), cfg,
        )?;
        let ff_up = transmute_matrix_with_cal(
            &lm.weights.ff_up[l], &mean_ff_up, Some(&act_rows.ff_up[l]), cfg,
        )?;
        let ff_down = transmute_matrix_with_cal(
            &lm.weights.ff_down[l], &mean_ff_down, Some(&act_rows.ff_down[l]), cfg,
        )?;
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

/// Backward-compatible entry: transmute with the calibration corpus equal
/// to `corpus` (the M8 behavior). Existing callers/tests use this.
pub fn transmute_matrix(
    w: &Matrix,
    mean_input: &[f32],
    cfg: &TransmuteConfig,
) -> anyhow::Result<SparseLayer> {
    transmute_matrix_with_cal(w, mean_input, None, cfg)
}

/// Transmute a single weight matrix `W [out, in]` into a [`SparseLayer`].
///
/// `mean_input` (length `in`) is the column-wise mean of the calibration
/// activations; it seeds the static bias `B[i] = W[i] . mean_input`.
/// `cal_rows` (when `Some`) are the per-token calibration activation rows
/// for this matmul — used only in [`BiasMode::Adaptive`] to train the
/// activation VQ codebook + precompute the per-token `bias_table`. When
/// `None` (legacy path / `transmute_matrix`) the adaptive artifacts are
/// left empty and the layer falls back to the mean-input pruned-only bias.
///
/// The quantization scheme is selected by `cfg.quant`:
/// - [`QuantSchemeConfig::Ternary`] (default): ternary `{-1,0,1}` + per-row
///   scale per expert (the original M1–M7 path).
/// - [`QuantSchemeConfig::Pq`]: trains one shared PQ codebook on the
///   normalized residual rows, encodes each expert's rows against it, and
///   stores the codebook on the returned `SparseLayer::pq_codebook`.
pub fn transmute_matrix_with_cal(
    w: &Matrix,
    mean_input: &[f32],
    cal_rows: Option<&[Vec<f32>]>,
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

    // Row→expert ownership map (length out_dim): -1 for core rows, k for
    // rows owned by expert k. Non-empty switches the bias application from
    // the legacy unconditional path to pruned-only (S1 correctness fix).
    let mut row_to_expert = vec![-1i32; out_dim];
    for (k, e) in experts.iter().enumerate() {
        for &r in &e.row_ids {
            row_to_expert[r as usize] = k as i32;
        }
    }

    // Adaptive bias artifacts (Phase 8 / M9): when `bias_mode == Adaptive`
    // and calibration rows are available, train a VQ codebook (PQ machinery,
    // M=1, 256 centroids) on the calibration activations and precompute a
    // per-token `bias_table[c][i] = W_quant[i] . centroid[c]` for every code
    // `c` and prunable row `i`. Looked up online after encoding the token's
    // `x` against the activation codebook.
    let (input_codebook, bias_table) = match (&cfg.bias_mode, cal_rows) {
        (BiasMode::Adaptive { nbits, iters, seed }, Some(rows)) if !rows.is_empty() => {
            // VQ via the PQ machinery: M=1 sub-vector (= the whole input),
            // 2^nbits centroids in input space. `train_pq` handles the
            // k-means + codebook layout; `encode_pq`/`decode_pq` reuse.
            let cb = train_pq(rows, 1, *nbits, *iters, *seed);
            let n_codes = cb.num_entries(); // 2^nbits
            // Reconstruct the quantized weight row for each prunable row
            // (so bias_table matches what the kernel computes exactly).
            // `wq[i]` = decoded weight for output row i (ternary scale·codes
            // or PQ scale·decode(codes)); core rows are skipped (bias 0).
            let wq = reconstruct_quantized_rows(&experts, &core_row_ids, &pq_codebook, out_dim, in_dim);
            // bias_table[c * out_dim + i] = wq[i] . centroid[c], for prunable
            // rows i (core rows left 0). Centroid c of the M=1 codebook is
            // at codebook[c * sub_dim .. c * sub_dim + sub_dim], sub_dim = in_dim.
            let sub_dim = cb.sub_dim;
            let mut table = vec![0.0f32; n_codes * out_dim];
            for c in 0..n_codes {
                let cent = &cb.codebook[c * sub_dim..c * sub_dim + sub_dim];
                let base = c * out_dim;
                for i in 0..out_dim {
                    if row_to_expert[i] < 0 {
                        continue; // core row — leave 0 (computed exactly)
                    }
                    let wrow = &wq[i * in_dim..(i + 1) * in_dim];
                    let dot: f32 = wrow.iter().zip(cent.iter()).map(|(a, b)| a * b).sum();
                    table[base + i] = dot;
                }
            }
            (Some(cb), Some(table))
        }
        _ => (None, None),
    };

    Ok(SparseLayer {
        out_dim,
        in_dim,
        dense_core,
        core_row_ids,
        experts,
        bias,
        mean_input: mean,
        pq_codebook,
        row_to_expert,
        input_codebook,
        bias_table,
    })
}

/// Reconstruct the quantized weight matrix `[out_dim, in_dim]` from a
/// `SparseLayer`'s experts (decoded ternary or PQ), with core rows copied
/// FP32. Used to precompute the adaptive `bias_table[c][i] = W_quant[i] .
/// centroid[c]` — the bias must match exactly what the kernel computes for
/// the pruned path, so we decode the quantized weights (not the originals).
/// Mirrors `reconstruct_dense` in `nse-eval/sparse_forward.rs` but lives here
/// (ZSTM) since it's a transmute-time helper.
fn reconstruct_quantized_rows(
    experts: &[MicroExpert],
    core_row_ids: &[u32],
    pq_codebook: &Option<nse_core::sparse::PqCodebook>,
    out_dim: usize,
    in_dim: usize,
) -> Vec<f32> {
    let mut w = vec![0.0f32; out_dim * in_dim];
    // Core rows: FP32 copy (not used by bias_table — core rows skipped —
    // but filled for completeness).
    let _ = core_row_ids; // core rows are skipped in the bias_table loop
    let cb = pq_codebook.as_ref();
    for e in experts {
        match (&e.pq, cb) {
            (Some(pq), Some(cb)) => {
                let m = pq.num_sub_vectors;
                let sub_dim = cb.sub_dim;
                let n_entries = cb.num_entries();
                for (j, &r) in e.row_ids.iter().enumerate() {
                    let scale = pq.row_scales[j];
                    let row = r as usize;
                    let codes = &pq.codes[j * m..(j + 1) * m];
                    for sm in 0..m {
                        let c = (codes[sm] as usize).min(n_entries - 1);
                        let base = sm * n_entries * sub_dim + c * sub_dim;
                        let cent = &cb.codebook[base..base + sub_dim];
                        let out_off = row * in_dim + sm * sub_dim;
                        for k in 0..sub_dim {
                            w[out_off + k] = scale * cent[k];
                        }
                    }
                }
            }
            _ => {
                for (j, &r) in e.row_ids.iter().enumerate() {
                    let scale = e.row_scales[j];
                    let row = r as usize;
                    for k in 0..in_dim {
                        w[row * in_dim + k] = scale * e.ternary[j * in_dim + k] as f32;
                    }
                }
            }
        }
    }
    w
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

/// Per-token calibration activation rows, per matmul. Each inner `Vec<f32>`
/// is one token's input to that matmul (`in_dim` floats). The mean-input
/// bias is the column-wise mean of these rows; the adaptive path trains a
/// VQ codebook on them (so the per-token distribution, not just the mean,
/// matters). Layout matches [`LayerMeanInputs`] from M8 but keeps rows
/// instead of collapsing to a single mean.
struct LayerActivationRows {
    qkv: Vec<Vec<Vec<f32>>>,      // [num_layers][num_tokens][dim]
    attn_out: Vec<Vec<Vec<f32>>>, // [num_layers][num_tokens][dim]
    ff_up: Vec<Vec<Vec<f32>>>,    // [num_layers][num_tokens][dim]
    ff_down: Vec<Vec<Vec<f32>>>,  // [num_layers][num_tokens][ff_dim]
}

/// Collect per-token activation rows over (possibly multiple) sliding
/// windows of the calibration corpus. The 4 matmuls per layer feed:
/// qkv ← `ln1_out`, attn_out ← `attn_out`, ff_up ← `ln2_out`, ff_down ←
/// `ff_up_act`. Windows step by `seq_len/2` (50% overlap) so a corpus of
/// `2*seq_len` tokens yields ~3 windows instead of the M8 single window —
/// the bigger the calibration set, the more representative the activation
/// distribution for the VQ codebook.
///
/// When `corpus` is `None`/empty, returns empty rows per layer → zero means
/// (the M8 `corpus=None` path).
fn collect_activations(lm: &ToyLm, corpus: Option<&[u8]>) -> LayerActivationRows {
    let cfg = &lm.config;
    let empty_rows: Vec<Vec<Vec<f32>>> = (0..cfg.num_layers).map(|_| Vec::new()).collect();
    let mut acc = LayerActivationRows {
        qkv: empty_rows.clone(),
        attn_out: empty_rows.clone(),
        ff_up: empty_rows.clone(),
        ff_down: empty_rows.clone(),
    };

    let corpus = match corpus {
        Some(c) if !c.is_empty() => c,
        _ => return acc, // empty → zero means (mean_of_rows of [] = zeros)
    };
    let tok = nse_models::Tokenizer::from_corpus(corpus);
    let ids = tok.encode(corpus);
    let seq = cfg.max_seq_len.min(ids.len().saturating_sub(1)).max(2);
    if ids.len() < seq + 1 {
        return acc;
    }

    // Sliding windows with 50% overlap. Step = seq/2 → ~2x the windows of a
    // non-overlapping stride. Each window contributes `seq` token rows per
    // matmul; rows accumulate (not averaged) so the VQ codebook sees the
    // distribution, not just the mean.
    let step = (seq / 2).max(1);
    let mut start = 0;
    while start + seq <= ids.len() {
        let tokens: Vec<u32> = ids[start..start + seq].to_vec();
        // Reuse the dense autograd forward to populate the layer caches.
        let (cache, _logits) = forward_cached(lm, &tokens);
        let t = tokens.len();
        for l in 0..cfg.num_layers {
            let lc = &cache.layers[l];
            append_rows(&mut acc.qkv[l], &lc.ln1_out, t, cfg.dim);
            append_rows(&mut acc.attn_out[l], &lc.attn_out, t, cfg.dim);
            append_rows(&mut acc.ff_up[l], &lc.ln2_out, t, cfg.dim);
            append_rows(&mut acc.ff_down[l], &lc.ff_up_act, t, cfg.ff_dim);
        }
        start += step;
    }
    acc
}

/// Append each of `rows` rows (length `cols`) from `flat` to `out` as
/// separate `Vec<f32>` entries (per-token rows, not collapsed).
fn append_rows(out: &mut Vec<Vec<f32>>, flat: &[f32], rows: usize, cols: usize) {
    for r in 0..rows {
        out.push(flat[r * cols..(r + 1) * cols].to_vec());
    }
}

/// Column-wise mean of a set of rows (each length `cols`). Empty input →
/// zeros. Replaces M8's `mean_over_rows` (which took a flat `[t*cols]`
/// slice); this takes `&[Vec<f32>]` since `collect_activations` keeps rows.
fn mean_of_rows(rows: &[Vec<f32>], cols: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; cols];
    if rows.is_empty() {
        return out;
    }
    for r in rows {
        for j in 0..cols {
            out[j] += r[j];
        }
    }
    for v in out.iter_mut() {
        *v /= rows.len() as f32;
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
        let tm = transmute(&lm, None, None, &tcfg).unwrap();
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
        let tm = transmute(&lm, None, None, &TransmuteConfig::poc()).unwrap();
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
        let tm = transmute(&lm, None, None, &tcfg).unwrap();

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
        let tm = transmute(&lm, None, None, &TransmuteConfig::pq()).unwrap();
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
            bias_mode: BiasMode::Mean,
        };
        let tm = transmute(&lm, None, None, &tcfg).unwrap();
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

    /// Adaptive bias: `transmute` with `BiasMode::Adaptive` produces a
    /// `TransmutedModel` where every non-empty layer has `input_codebook:
    /// Some` + `bias_table: Some` + non-empty `row_to_expert`, and every
    /// expert has `pq: Some`. Covered-rows + core-bias-zeroed invariants
    /// still hold.
    #[test]
    fn transmute_adaptive_roundtrip() {
        let cfg = Config {
            vocab_size: 32,
            dim: 64,
            num_layers: 1,
            num_heads: 2,
            max_seq_len: 32,
            ff_dim: 64,
        };
        let lm = ToyLm::init_random(cfg, 11);
        // A corpus long enough for >1 sliding window (seq=32, step=16).
        let corpus = b"to be or not to be that is the question whether tis nobler \
                       in the mind to suffer the slings and arrows of outrageous \
                       fortune or to take arms against a sea of troubles and by \
                       opposing end them to die to sleep no more and by a sleep";
        let tm = transmute(&lm, Some(corpus), None, &TransmuteConfig::adaptive()).unwrap();
        for layer in &tm.layers {
            for sl in layer {
                assert_eq!(sl.covered_rows(), sl.out_dim, "covered rows");
                for &r in &sl.core_row_ids {
                    assert_eq!(sl.bias[r as usize], 0.0, "core bias zeroed");
                }
                // Adaptive artifacts present on any layer with experts.
                if !sl.experts.is_empty() {
                    assert!(sl.row_to_expert.len() == sl.out_dim, "row_to_expert set");
                    let cb = sl.input_codebook.as_ref().expect("input_codebook");
                    assert_eq!(cb.num_sub_vectors, 1, "VQ M=1");
                    assert_eq!(cb.nbits, 8, "VQ 8-bit");
                    let table = sl.bias_table.as_ref().expect("bias_table");
                    assert_eq!(table.len(), cb.num_entries() * sl.out_dim, "bias_table size");
                    // Every expert is PQ (adaptive implies PQ quant).
                    for e in &sl.experts {
                        assert!(e.pq.is_some(), "expert pq present");
                    }
                }
            }
        }
    }

    /// `bias_table[c][i]` math: for a prunable row `i`, the stored value must
    /// equal `W_quant[i] . centroid[c]` (decoded weight dotted with the
    /// activation codebook centroid). Cross-checks against the PQ decode of
    /// the expert's weight codes + the activation codebook centroids.
    #[test]
    fn activation_pq_bias_table_math() {
        let cfg = Config {
            vocab_size: 24,
            dim: 32,
            num_layers: 1,
            num_heads: 2,
            max_seq_len: 32,
            ff_dim: 32,
        };
        let lm = ToyLm::init_random(cfg, 11);
        let corpus = b"to be or not to be that is the question whether tis nobler \
                       in the mind to suffer the slings and arrows of outrageous";
        let tm = transmute(&lm, Some(corpus), None, &TransmuteConfig::adaptive()).unwrap();
        let sl = &tm.layers[0][0]; // qkv
        let pq_cb = sl.pq_codebook.as_ref().expect("weight codebook");
        let act_cb = sl.input_codebook.as_ref().expect("activation codebook");
        let table = sl.bias_table.as_ref().expect("bias_table");
        let n_codes = act_cb.num_entries();
        let sub_dim = act_cb.sub_dim; // = in_dim (M=1)
        assert_eq!(sub_dim, sl.in_dim);
        // Pick a prunable row (owned by some expert) and check a few codes.
        let prunable: Vec<usize> = (0..sl.out_dim)
            .filter(|&i| sl.row_to_expert[i] >= 0)
            .collect();
        assert!(!prunable.is_empty(), "must have prunable rows");
        let i = prunable[0];
        // Reconstruct W_quant[i] from the owning expert's PQ codes.
        let (expert_idx, local_j) = {
            let e = sl.row_to_expert[i] as usize;
            let j = sl.experts[e].row_ids.iter().position(|&r| r as usize == i).unwrap();
            (e, j)
        };
        let expert = &sl.experts[expert_idx];
        let pq = expert.pq.as_ref().unwrap();
        let m = pq.num_sub_vectors;
        let scale = pq.row_scales[local_j];
        let codes = &pq.codes[local_j * m..(local_j + 1) * m];
        let n_entries = pq_cb.num_entries();
        let wq: Vec<f32> = {
            let mut row = vec![0.0f32; sl.in_dim];
            for sm in 0..m {
                let c = (codes[sm] as usize).min(n_entries - 1);
                let base = sm * n_entries * pq_cb.sub_dim + c * pq_cb.sub_dim;
                for k in 0..pq_cb.sub_dim {
                    row[sm * pq_cb.sub_dim + k] = scale * pq_cb.codebook[base + k];
                }
            }
            row
        };
        // For each code c: bias_table[c*out_dim + i] should == wq · centroid[c].
        for c in 0..n_codes {
            let cent = &act_cb.codebook[c * sub_dim..c * sub_dim + sub_dim];
            let dot: f32 = wq.iter().zip(cent.iter()).map(|(a, b)| a * b).sum();
            let stored = table[c * sl.out_dim + i];
            assert!(
                (stored - dot).abs() < 1e-3,
                "code {c}: stored {stored:.6} vs computed {dot:.6} (diff {})",
                (stored - dot).abs()
            );
        }
        // Core rows should be 0 in the bias_table.
        for &r in &sl.core_row_ids {
            for c in 0..n_codes {
                assert_eq!(table[c * sl.out_dim + r as usize], 0.0, "core row zero in table");
            }
        }
    }

    /// Calibration multi-window: a corpus long enough for >1 sliding window
    /// collects more activation rows than a single-window run. Verifies the
    /// sliding-window infra collects from multiple windows (the mean differs
    /// from a hypothetical single-window mean by virtue of more samples).
    #[test]
    fn calibration_multi_window() {
        use nse_models::Config;
        // dim=16 so the model is tiny; max_seq_len=8 → step=4. A corpus of
        // ~40 chars → ~5+ windows, vs the M8 single-window path.
        let cfg = Config {
            vocab_size: 16,
            dim: 16,
            num_layers: 1,
            num_heads: 2,
            max_seq_len: 8,
            ff_dim: 16,
        };
        let lm = ToyLm::init_random(cfg, 3);
        // Long corpus (multi-window).
        let long_corpus = b"to be or not to be that is the question whether \
                            tis nobler in the mind to suffer the slings and \
                            arrows of outrageous fortune or to take arms against \
                            a sea of troubles and by opposing end them to die";
        // Short corpus (single window only — fits one seq of 8).
        let short_corpus = b"to be or not to be that is the question";
        let long_rows = collect_activations(&lm, Some(long_corpus));
        let short_rows = collect_activations(&lm, Some(short_corpus));
        // The long corpus must collect strictly more token rows per matmul
        // than the short corpus (multi-window vs single-window).
        assert!(
            long_rows.qkv[0].len() > short_rows.qkv[0].len(),
            "multi-window should collect more rows: long={} short={}",
            long_rows.qkv[0].len(), short_rows.qkv[0].len()
        );
        // The mean over more rows should differ from a single-window mean
        // (the activation distribution is richer than one window's mean).
        let long_mean = mean_of_rows(&long_rows.qkv[0], 16);
        let short_mean = mean_of_rows(&short_rows.qkv[0], 16);
        let diff: f32 = long_mean.iter().zip(short_mean.iter())
            .map(|(a, b)| (a - b).abs()).sum();
        assert!(diff > 1e-4, "means should differ with more calibration data");
    }
}
