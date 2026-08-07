//! Perplexity (PPL) computation for both the dense and sparse models.
//!
//! PPL = exp(mean cross-entropy over tokens), where position `t` predicts
//! token `t+1`. Computed identically for both backends so the comparison is
//! apples-to-apples.

use nse_core::sparse::TransmutedModel;
use nse_models::ToyLm;

/// Per-token natural-log probabilities for a sequence, predicting t+1.
/// `targets[t]` is the token predicted by position `t`.
pub fn logprobs(logits: &[f32], targets: &[u32], vocab: usize) -> Vec<f32> {
    let seq = targets.len();
    let mut out = Vec::with_capacity(seq);
    for t in 0..seq {
        let row = &logits[t * vocab..(t + 1) * vocab];
        let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0;
        for &v in row {
            sum += (v - max).exp();
        }
        let tgt = targets[t] as usize;
        let p = ((row[tgt] - max).exp() / sum).max(1e-12);
        out.push(p.ln());
    }
    out
}

/// Perplexity from per-token log-probabilities.
pub fn perplexity_from_logprobs(logprobs: &[f32]) -> f32 {
    if logprobs.is_empty() {
        return f32::INFINITY;
    }
    let mean = logprobs.iter().sum::<f32>() / logprobs.len() as f32;
    (-mean).exp()
}

/// PPL of the dense [`ToyLm`] over a sliding window of length `seq_len`,
/// predicting t+1. Averages the loss over all windows.
pub fn dense_ppl(lm: &ToyLm, ids: &[u32], seq_len: usize) -> f32 {
    let vocab = lm.config.vocab_size;
    let mut total_lp = 0.0f32;
    let mut count = 0usize;
    for start in 0..ids.len().saturating_sub(seq_len + 1) {
        let tokens = &ids[start..start + seq_len];
        let targets = &ids[start + 1..start + 1 + seq_len];
        let logits = lm.forward(tokens);
        let lp = logprobs(&logits, targets, vocab);
        total_lp += lp.iter().sum::<f32>();
        count += lp.len();
    }
    if count == 0 {
        f32::INFINITY
    } else {
        (-total_lp / count as f32).exp()
    }
}

/// PPL of the sparse [`TransmutedModel`] over a sliding window of length
/// `seq_len`, predicting t+1, using the given [`Activation`](super::sparse_forward::Activation).
/// Uses default runtime options (scalar kernel, brute-force index).
pub fn sparse_ppl(
    tm: &TransmutedModel,
    ids: &[u32],
    seq_len: usize,
    act: super::sparse_forward::Activation,
) -> f32 {
    sparse_ppl_with_options(tm, ids, seq_len, act, super::sparse_forward::SparseOptions::default())
}

/// Sparse PPL with explicit runtime options (kernel + index).
pub fn sparse_ppl_with_options(
    tm: &TransmutedModel,
    ids: &[u32],
    seq_len: usize,
    act: super::sparse_forward::Activation,
    opts: super::sparse_forward::SparseOptions,
) -> f32 {
    let vocab = tm.config.vocab_size;
    let mut total_lp = 0.0f32;
    let mut count = 0usize;
    for start in 0..ids.len().saturating_sub(seq_len + 1) {
        let tokens = &ids[start..start + seq_len];
        let targets = &ids[start + 1..start + 1 + seq_len];
        let logits = super::sparse_forward::sparse_forward_with_options(tm, tokens, act, opts);
        let lp = logprobs(&logits, targets, vocab);
        total_lp += lp.iter().sum::<f32>();
        count += lp.len();
    }
    if count == 0 {
        f32::INFINITY
    } else {
        (-total_lp / count as f32).exp()
    }
}
