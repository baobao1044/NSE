//! Forward-Forward / Predictive Coding trainer (research scaffold).
//!
//! Hinton's Forward-Forward algorithm: two forward passes (positive + negative
//! data), local "goodness" updates per layer, no backprop, ~zero VRAM overhead.
//!
//! Status: skeleton (M0). Real implementation after M6.

use nse_models::ToyLm;

use crate::Trainer;

pub struct ForwardForwardTrainer {
    pub goodness_threshold: f32,
}

impl Default for ForwardForwardTrainer {
    fn default() -> Self {
        Self { goodness_threshold: 0.5 }
    }
}

impl Trainer for ForwardForwardTrainer {
    fn name(&self) -> &'static str {
        "forward-forward"
    }

    fn train(&mut self, _model: &mut ToyLm, _corpus: &[u8]) -> anyhow::Result<()> {
        todo!("post-M6: Forward-Forward local goodness updates")
    }
}
