//! Adaptive threshold router.
//!
//! For each token, computes similarity scores against centroids and prunes
//! any below a dynamic threshold `θ(x)`, keeping only the dynamic top-K
//! micro-experts.
//!
//! Status: skeleton (M0). Real router lands in M4.

use crate::index::Hit;

#[derive(Debug, Clone)]
pub struct RouterConfig {
    /// Fraction of max score used as the dynamic threshold.
    pub threshold_ratio: f32,
    /// Hard cap on the number of selected experts.
    pub max_k: usize,
}

impl Default for RouterConfig {
    fn default() -> Self {
        Self { threshold_ratio: 0.5, max_k: 64 }
    }
}

/// Filter `hits` by the dynamic threshold and cap at `max_k`. (Stub — M4.)
pub fn route(_hits: &[Hit], _cfg: &RouterConfig) -> anyhow::Result<Vec<Hit>> {
    todo!("M4: adaptive threshold routing")
}
