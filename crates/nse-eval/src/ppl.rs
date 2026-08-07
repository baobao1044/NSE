//! Perplexity (PPL) computation.
//!
//! PPL = exp(mean cross-entropy over tokens). Computed identically for both
//! the dense and sparse models so the comparison is apples-to-apples.
//!
//! Status: skeleton (M0). Real PPL runner lands in M5.

/// Perplexity of a sequence of per-token log-probabilities (natural log).
pub fn perplexity_from_logprobs(logprobs: &[f32]) -> f32 {
    if logprobs.is_empty() {
        return f32::INFINITY;
    }
    let mean = logprobs.iter().sum::<f32>() / logprobs.len() as f32;
    (-mean).exp()
}
