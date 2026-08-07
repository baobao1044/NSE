//! Adaptive threshold router.
//!
//! Given a token's similarity scores against all centroids, prune any below
//! a dynamic threshold `θ(x)` and keep the dynamic top-K micro-experts. Two
//! strategies are provided:
//!
//! - [`route_by_ratio`]: `θ = max_score * threshold_ratio`, keep up to `max_k`.
//!   Simpler; the spec's "if `Score_k < θ`, set 0 and prune".
//! - [`route_all`]: activate every expert (no pruning). Used as the
//!   upper-bound / correctness reference — with `route_all`, the sparse output
//!   equals the dense output up to ternary quantization error (the bias is
//!   then exactly zero on average since nothing is pruned).

use crate::index::Hit;

#[derive(Debug, Clone)]
pub struct RouterConfig {
    /// `θ = max_score * threshold_ratio`. 0 keeps everything (no prune).
    pub threshold_ratio: f32,
    /// Hard cap on the number of selected experts.
    pub max_k: usize,
}

impl Default for RouterConfig {
    fn default() -> Self {
        Self { threshold_ratio: 0.5, max_k: 64 }
    }
}

/// Route by ratio: keep hits with `score >= max_score * ratio`, capped at
/// `max_k`. Assumes `hits` is already sorted descending (as produced by
/// [`crate::index::MipsIndex::query_all`]).
pub fn route_by_ratio(hits: &[Hit], cfg: &RouterConfig) -> Vec<Hit> {
    if hits.is_empty() {
        return Vec::new();
    }
    let max_score = hits[0].score;
    let threshold = max_score * cfg.threshold_ratio;
    let mut out: Vec<Hit> = hits
        .iter()
        .copied()
        .filter(|h| h.score >= threshold)
        .collect();
    out.truncate(cfg.max_k);
    out
}

/// Activate every expert (no pruning) — the upper-bound reference. Bias is
/// then exactly zero on average (no pruned contribution to compensate).
pub fn route_all(hits: &[Hit]) -> Vec<Hit> {
    hits.to_vec()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn h(id: usize, s: f32) -> Hit {
        Hit { expert_id: id, score: s }
    }

    #[test]
    fn ratio_prunes_low_scores() {
        let hits = vec![h(0, 2.0), h(1, 1.5), h(2, 0.5), h(3, -1.0)];
        let out = route_by_ratio(&hits, &RouterConfig { threshold_ratio: 0.5, max_k: 10 });
        // threshold = 2*0.5 = 1.0 ; keep >= 1.0 -> ids 0,1
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].expert_id, 0);
        assert_eq!(out[1].expert_id, 1);
    }

    #[test]
    fn max_k_caps_results() {
        let hits = vec![h(0, 2.0), h(1, 1.9), h(2, 1.8), h(3, 1.7)];
        let out = route_by_ratio(&hits, &RouterConfig { threshold_ratio: 0.0, max_k: 2 });
        assert_eq!(out.len(), 2);
    }

    #[test]
    fn all_keeps_everything() {
        let hits = vec![h(0, 2.0), h(1, -5.0)];
        let out = route_all(&hits);
        assert_eq!(out.len(), 2);
    }
}
