//! Vanilla SGD trainer (real backprop baseline).
//!
//! Status: skeleton (M0). Real training loop lands in M2; this is what produces
//! a Toy LM with a reasonable perplexity for the dense baseline.

use nse_models::ToyLm;

use crate::Trainer;

/// Hyperparameters for the SGD baseline trainer.
#[derive(Debug, Clone)]
pub struct SgdConfig {
    pub learning_rate: f32,
    pub batch_tokens: usize,
    pub seq_len: usize,
    pub epochs: usize,
}

impl Default for SgdConfig {
    fn default() -> Self {
        Self {
            learning_rate: 3e-4,
            batch_tokens: 512,
            seq_len: 64,
            epochs: 3,
        }
    }
}

/// Plain SGD (with momentum) trainer — the dense baseline.
pub struct SgdTrainer {
    pub config: SgdConfig,
}

impl SgdTrainer {
    pub fn new(config: SgdConfig) -> Self {
        Self { config }
    }
}

impl Trainer for SgdTrainer {
    fn name(&self) -> &'static str {
        "sgd-baseline"
    }

    fn train(&mut self, _model: &mut ToyLm, _corpus: &[u8]) -> anyhow::Result<()> {
        todo!("M2: SGD training loop")
    }
}
