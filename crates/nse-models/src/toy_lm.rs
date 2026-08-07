//! Toy language model — a minimal transformer-style LM for the NSE POC.
//!
//! Forward pass is implemented in M1; here we declare the weight layout so
//! other crates can reference it. Weights are stored as owned row-major
//! matrices (`Vec<f32>`), matching the [`nse_core::tensor::Matrix`] layout.

use nse_core::tensor::Matrix;
use serde::{Deserialize, Serialize};

use crate::Config;

/// Weights of the Toy LM. Layered blocks share the same shape.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToyLmWeights {
    /// `[vocab_size, dim]` token embedding (tied with the output head).
    pub token_embed: Matrix,
    /// Per-layer pre-attention layer-norm gain, length `dim`.
    pub ln1_gain: Vec<Vec<f32>>,
    /// Per-layer attention `Wqkv` fused, shape `[3*dim, dim]`.
    pub qkv: Vec<Matrix>,
    /// Per-layer attention output projection, shape `[dim, dim]`.
    pub attn_out: Vec<Matrix>,
    /// Per-layer pre-FFN layer-norm gain, length `dim`.
    pub ln2_gain: Vec<Vec<f32>>,
    /// Per-layer FFN up projection, shape `[ff_dim, dim]`.
    pub ff_up: Vec<Matrix>,
    /// Per-layer FFN down projection, shape `[dim, ff_dim]`.
    pub ff_down: Vec<Matrix>,
    /// Final layer-norm gain, length `dim`.
    pub ln_f_gain: Vec<f32>,
}

/// The Toy LM bundles its config and weights.
#[derive(Debug, Clone)]
pub struct ToyLm {
    pub config: Config,
    pub weights: ToyLmWeights,
}

impl ToyLm {
    /// Allocate a Toy LM with random-ish weights (zeros for M0; init in M1).
    pub fn new(config: Config) -> Self {
        let weights = ToyLmWeights {
            token_embed: Matrix::zeros(config.vocab_size, config.dim),
            ln1_gain: vec![vec![1.0; config.dim]; config.num_layers],
            qkv: vec![Matrix::zeros(3 * config.dim, config.dim); config.num_layers],
            attn_out: vec![Matrix::zeros(config.dim, config.dim); config.num_layers],
            ln2_gain: vec![vec![1.0; config.dim]; config.num_layers],
            ff_up: vec![Matrix::zeros(config.ff_dim, config.dim); config.num_layers],
            ff_down: vec![Matrix::zeros(config.dim, config.ff_dim); config.num_layers],
            ln_f_gain: vec![1.0; config.dim],
        };
        Self { config, weights }
    }

    /// Total parameter count (for the `.nse` header).
    pub fn num_params(&self) -> u64 {
        let c = &self.config;
        let embed = c.vocab_size * c.dim;
        let per_layer =
            3 * c.dim * c.dim + c.dim * c.dim + c.ff_dim * c.dim + c.dim * c.ff_dim;
        let ln = 2 * c.dim; // ln1 + ln2 per layer (bias omitted)
        let total = embed + c.num_layers * (per_layer + ln) + c.dim;
        total as u64
    }
}
