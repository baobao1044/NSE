//! LSH Sparse Weight Training (research scaffold).
//!
//! Locality-Sensitive Hashing identifies the ~0.01% of weights relevant to a
//! given training example; only those receive a gradient update while the rest
//! stays frozen. Shares the LSH index with the inference router (`nse-rie`).
//!
//! Status: skeleton (M0). Real implementation after M6.

use nse_models::ToyLm;

use crate::Trainer;

pub struct LshSparseTrainer {
    pub sparse_fraction: f32,
}

impl Default for LshSparseTrainer {
    fn default() -> Self {
        Self { sparse_fraction: 0.0001 }
    }
}

impl Trainer for LshSparseTrainer {
    fn name(&self) -> &'static str {
        "lsh-sparse"
    }

    fn train(&mut self, _model: &mut ToyLm, _corpus: &[u8]) -> anyhow::Result<()> {
        todo!("post-M6: LSH-indexed sparse weight updates")
    }
}
