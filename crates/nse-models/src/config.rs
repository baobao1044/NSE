//! Model hyperparameter configuration.

use serde::{Deserialize, Serialize};

/// Hyperparameters for the Toy LM and a generic container for other models.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    /// Vocabulary size (char-level for the Toy LM POC).
    pub vocab_size: usize,
    /// Hidden / embedding dimension `d`.
    pub dim: usize,
    /// Number of transformer layers.
    pub num_layers: usize,
    /// Number of attention heads.
    pub num_heads: usize,
    /// Maximum sequence length (context).
    pub max_seq_len: usize,
    /// Intermediate width of the feed-forward block.
    pub ff_dim: usize,
}

impl Config {
    /// A tiny default config for the POC: ~1k vocab, d=128, 2 layers.
    pub fn toy_default(vocab_size: usize) -> Self {
        Self {
            vocab_size,
            dim: 128,
            num_layers: 2,
            num_heads: 4,
            max_seq_len: 256,
            ff_dim: 512,
        }
    }
}

impl Default for Config {
    fn default() -> Self {
        Self::toy_default(256)
    }
}
