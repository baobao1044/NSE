//! The `Trainer` trait abstracting training backends.

use nse_models::ToyLm;

/// Common interface implemented by every training backend (SGD baseline,
/// Forward-Forward, Hopfield, LSH-sparse).
pub trait Trainer {
    /// Human-readable name of the training algorithm.
    fn name(&self) -> &'static str;

    /// Train `model` in-place on the given byte corpus.
    fn train(&mut self, model: &mut ToyLm, corpus: &[u8]) -> anyhow::Result<()>;
}
