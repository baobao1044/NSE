//! Toy language model — a minimal transformer-style LM for the NSE POC.
//!
//! Forward pass implemented here: token embedding -> per-layer block
//! (layernorm -> causal self-attention -> residual; layernorm -> GELU FFN ->
//! residual) -> final layernorm -> logits (tied with the token embedding).
//!
//! Weights are stored as owned row-major matrices (`Vec<f32>`), matching the
//! [`nse_core::tensor::Matrix`] layout, so the ZSTM can transmute them
//! in-place without copying.

use nse_core::tensor::Matrix;
use serde::{Deserialize, Serialize};

use crate::Config;

/// Weights of the Toy LM. Each layer holds the same shapes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToyLmWeights {
    /// `[vocab_size, dim]` token embedding (tied with the output head).
    pub token_embed: Matrix,
    /// Per-layer pre-attention layer-norm gain, length `dim`.
    pub ln1_gain: Vec<Vec<f32>>,
    /// Per-layer attention fused `Wqkv`, shape `[3*dim, dim]` (row = output).
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
    /// Allocate a Toy LM with zero weights. Use [`ToyLm::init_random`] for a
    /// trainable random initialization.
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

    /// Allocate a Toy LM with Xavier/Glorot random initialization suitable for
    /// training. Uses a simple LCG seeded by `seed` so we don't pull a heavy
    /// RNG dependency into every call site.
    pub fn init_random(config: Config, seed: u64) -> Self {
        let mut lm = Self::new(config.clone());
        let mut rng = Lcg::new(seed);
        xavier(&mut lm.weights.token_embed.data, config.dim, &mut rng);
        for l in 0..config.num_layers {
            xavier(&mut lm.weights.qkv[l].data, config.dim, &mut rng);
            xavier(&mut lm.weights.attn_out[l].data, config.dim, &mut rng);
            xavier(&mut lm.weights.ff_up[l].data, config.dim, &mut rng);
            xavier(&mut lm.weights.ff_down[l].data, config.ff_dim, &mut rng);
        }
        lm
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

    /// Forward pass over a sequence of token ids.
    ///
    /// Returns logits of shape `[seq_len, vocab_size]` (row-major). Logits at
    /// position `t` predict token `t+1`; the caller handles the shift.
    pub fn forward(&self, tokens: &[u32]) -> Vec<f32> {
        let c = &self.config;
        let seq = tokens.len();
        let d = c.dim;
        let v = c.vocab_size;

        // 1. Token embedding lookup -> [seq, dim].
        let mut x = vec![0.0f32; seq * d];
        for (t, &tok) in tokens.iter().enumerate() {
            let tok = (tok as usize).min(v - 1);
            for j in 0..d {
                x[t * d + j] = self.weights.token_embed.data[tok * d + j];
            }
        }

        // 2. Per-layer transformer block.
        for layer in 0..c.num_layers {
            let ln1 = &self.weights.ln1_gain[layer];
            let qkv = &self.weights.qkv[layer];
            let attn_out = &self.weights.attn_out[layer];
            let ln2 = &self.weights.ln2_gain[layer];
            let ff_up = &self.weights.ff_up[layer];
            let ff_down = &self.weights.ff_down[layer];

            // --- Attention sub-block ---
            let h = layernorm(&x, seq, d, ln1); // [seq, dim]
            let qkv_out = matmul_rows(&h, qkv); // [seq, 3*dim]
            let (q, k, v_proj) = split_qkv(&qkv_out, seq, d);
            let attn = causal_self_attention(&q, &k, &v_proj, seq, d, c.num_heads);
            let attn_out_proj = matmul_rows(&attn, attn_out); // [seq, dim]
            // residual
            for i in 0..seq * d {
                x[i] += attn_out_proj[i];
            }

            // --- FFN sub-block ---
            let h2 = layernorm(&x, seq, d, ln2); // [seq, dim]
            let mut up = matmul_rows(&h2, ff_up); // [seq, ff_dim]
            gelu_inplace(&mut up);
            let down = matmul_rows(&up, ff_down); // [seq, dim]
            // residual
            for i in 0..seq * d {
                x[i] += down[i];
            }
        }

        // 3. Final layernorm.
        let x = layernorm(&x, seq, d, &self.weights.ln_f_gain);

        // 4. Tied output head: logits = x @ token_embed^T -> [seq, vocab].
        let mut logits = vec![0.0f32; seq * v];
        for t in 0..seq {
            for w in 0..v {
                let mut s = 0.0;
                for j in 0..d {
                    s += x[t * d + j] * self.weights.token_embed.data[w * d + j];
                }
                logits[t * v + w] = s;
            }
        }
        logits
    }

    /// Forward pass with the FFN replaced by a **Hopfield retrieval** rule:
    /// instead of `gelu(h2 @ ff_up) @ ff_down`, the FFN output is
    /// `ff_down · softmax(β · (ff_up · h2))` per position. This is the forward
    /// path the Hopfield trainer's writes were *designed* for (modern Hopfield
    /// retrieval); the standard [`forward`](Self::forward) uses GELU and so
    /// does not exercise the associative memory. `beta` is the retrieval
    /// sharpness. Used to test the architecture-mismatch hypothesis.
    pub fn forward_hopfield(&self, tokens: &[u32], beta: f32) -> Vec<f32> {
        let c = &self.config;
        let seq = tokens.len();
        let d = c.dim;
        let v = c.vocab_size;
        let fd = c.ff_dim;

        let mut x = vec![0.0f32; seq * d];
        for (t, &tok) in tokens.iter().enumerate() {
            let tok = (tok as usize).min(v - 1);
            for j in 0..d {
                x[t * d + j] = self.weights.token_embed.data[tok * d + j];
            }
        }

        for layer in 0..c.num_layers {
            let ln1 = &self.weights.ln1_gain[layer];
            let qkv = &self.weights.qkv[layer];
            let attn_out = &self.weights.attn_out[layer];
            let ln2 = &self.weights.ln2_gain[layer];
            let ff_up = &self.weights.ff_up[layer];
            let ff_down = &self.weights.ff_down[layer];

            // --- Attention (identical to forward) ---
            let h = layernorm(&x, seq, d, ln1);
            let qkv_out = matmul_rows(&h, qkv);
            let (q, k, vp) = split_qkv(&qkv_out, seq, d);
            let attn = causal_self_attention(&q, &k, &vp, seq, d, c.num_heads);
            let attn_out_proj = matmul_rows(&attn, attn_out);
            for i in 0..seq * d {
                x[i] += attn_out_proj[i];
            }

            // --- FFN: Hopfield retrieval (softmax) instead of GELU ---
            let h2 = layernorm(&x, seq, d, ln2); // [seq, dim] — the query
            // scores[i] = ff_up[i,:] · h2[t,:], i in 0..ff_dim
            let mut scores = vec![0.0f32; fd];
            for t in 0..seq {
                for i in 0..fd {
                    let mut s = 0.0;
                    for j in 0..d {
                        s += ff_up.data[i * d + j] * h2[t * d + j];
                    }
                    scores[i] = beta * s;
                }
                // softmax over slots
                let max = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0.0;
                for s in scores.iter_mut() {
                    *s = (*s - max).exp();
                    sum += *s;
                }
                let inv = if sum > 0.0 { 1.0 / sum } else { 0.0 };
                for s in scores.iter_mut() {
                    *s *= inv;
                }
                // down[t,j] = sum_i ff_down[j,i] * softmax_i
                for j in 0..d {
                    let mut acc = 0.0;
                    for i in 0..fd {
                        acc += ff_down.data[j * fd + i] * scores[i];
                    }
                    x[t * d + j] += acc;
                }
            }
        }

        let x = layernorm(&x, seq, d, &self.weights.ln_f_gain);
        let mut logits = vec![0.0f32; seq * v];
        for t in 0..seq {
            for w in 0..v {
                let mut s = 0.0;
                for j in 0..d {
                    s += x[t * d + j] * self.weights.token_embed.data[w * d + j];
                }
                logits[t * v + w] = s;
            }
        }
        logits
    }
}

/// LayerNorm without bias: `y = gain * (x - mean) / sqrt(var + eps)`.
fn layernorm(x: &[f32], seq: usize, dim: usize, gain: &[f32]) -> Vec<f32> {
    const EPS: f32 = 1e-5;
    let mut out = vec![0.0f32; seq * dim];
    for t in 0..seq {
        let row = &x[t * dim..(t + 1) * dim];
        let mean = row.iter().sum::<f32>() / dim as f32;
        let var = row.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / dim as f32;
        let inv = 1.0 / (var + EPS).sqrt();
        for j in 0..dim {
            out[t * dim + j] = gain[j] * (row[j] - mean) * inv;
        }
    }
    out
}

/// `out[i, j] = sum_k h[i,k] * w[j,k]` — i.e. `h @ w^T`.
/// `h` is `[seq, in]`, `w` is `[out, in]`, result is `[seq, out]`.
fn matmul_rows(h: &[f32], w: &Matrix) -> Vec<f32> {
    let seq = h.len() / w.cols;
    let out = w.rows;
    let mut result = vec![0.0f32; seq * out];
    for i in 0..seq {
        for j in 0..out {
            let mut s = 0.0;
            for k in 0..w.cols {
                s += h[i * w.cols + k] * w.data[j * w.cols + k];
            }
            result[i * out + j] = s;
        }
    }
    result
}

/// Split fused `[seq, 3*dim]` QKV into Q, K, V each `[seq, dim]`.
fn split_qkv(qkv: &[f32], seq: usize, dim: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut q = vec![0.0; seq * dim];
    let mut k = vec![0.0; seq * dim];
    let mut v = vec![0.0; seq * dim];
    for t in 0..seq {
        for j in 0..dim {
            q[t * dim + j] = qkv[t * 3 * dim + j];
            k[t * dim + j] = qkv[t * 3 * dim + dim + j];
            v[t * dim + j] = qkv[t * 3 * dim + 2 * dim + j];
        }
    }
    (q, k, v)
}

/// Causal multi-head self-attention. Returns `[seq, dim]`.
fn causal_self_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    seq: usize,
    dim: usize,
    num_heads: usize,
) -> Vec<f32> {
    let head_dim = dim / num_heads;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut out = vec![0.0f32; seq * dim];

    for h in 0..num_heads {
        let off = h * head_dim;
        // attention per head
        for i in 0..seq {
            // scores for j in 0..=i (causal)
            let mut weights = vec![0.0f32; seq];
            let mut max = f32::NEG_INFINITY;
            for j in 0..=i {
                let mut s = 0.0;
                for d2 in 0..head_dim {
                    s += q[i * dim + off + d2] * k[j * dim + off + d2];
                }
                s *= scale;
                weights[j] = s;
                if s > max {
                    max = s;
                }
            }
            // softmax over 0..=i
            let mut sum = 0.0;
            for j in 0..=i {
                weights[j] = (weights[j] - max).exp();
                sum += weights[j];
            }
            let inv = if sum > 0.0 { 1.0 / sum } else { 0.0 };
            for j in 0..=i {
                weights[j] *= inv;
            }
            // weighted sum of V
            for d2 in 0..head_dim {
                let mut acc = 0.0;
                for j in 0..=i {
                    acc += weights[j] * v[j * dim + off + d2];
                }
                out[i * dim + off + d2] = acc;
            }
        }
    }
    out
}

/// GELU (tanh approximation) applied in place.
fn gelu_inplace(x: &mut [f32]) {
    const SQRT2_INV: f32 = 0.7071067811865475; // 1/sqrt(2)
    for v in x.iter_mut() {
        let t = *v * SQRT2_INV;
        let inner = 1.0 + 0.044715 * t * t;
        let tanh = (2.0 * t * inner).tanh();
        *v = 0.5 * *v * (1.0 + tanh);
    }
}

/// Minimal xorshift RNG — deterministic, no external dependency.
struct Lcg {
    state: u64,
}

impl Lcg {
    fn new(seed: u64) -> Self {
        Self {
            state: if seed == 0 { 0x9e3779b97f4a7c15 } else { seed },
        }
    }
    fn next_u32(&mut self) -> u32 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        (x & 0xffff_ffff) as u32
    }
    /// Uniform float in [0, 1).
    fn next_f32(&mut self) -> f32 {
        (self.next_u32() as f32) / (u32::MAX as f32 + 1.0)
    }
}

/// Xavier/Glorot init: scale `sqrt(2 / fan_in)` centered at 0.
fn xavier(data: &mut [f32], fan_in: usize, rng: &mut Lcg) {
    let scale = (2.0 / fan_in.max(1) as f32).sqrt();
    for v in data.iter_mut() {
        *v = (rng.next_f32() * 2.0 - 1.0) * scale;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forward_produces_right_shape() {
        let cfg = Config { vocab_size: 16, dim: 8, num_layers: 2, num_heads: 2, max_seq_len: 32, ff_dim: 16 };
        let lm = ToyLm::new(cfg.clone());
        let tokens = vec![0u32, 1, 2, 3];
        let logits = lm.forward(&tokens);
        assert_eq!(logits.len(), tokens.len() * cfg.vocab_size);
        // With zero weights, logits are all zero.
        assert!(logits.iter().all(|&l| l == 0.0));
    }

    #[test]
    fn forward_with_random_weights_is_finite() {
        let cfg = Config { vocab_size: 16, dim: 8, num_layers: 1, num_heads: 2, max_seq_len: 32, ff_dim: 16 };
        let mut lm = ToyLm::new(cfg.clone());
        // Sprinkle small random-ish values.
        for w in lm.weights.token_embed.data.iter_mut() {
            *w = 0.01;
        }
        for m in lm.weights.qkv.iter_mut() {
            for v in m.data.iter_mut() {
                *v = 0.01;
            }
        }
        for m in lm.weights.ff_up.iter_mut() {
            for v in m.data.iter_mut() {
                *v = 0.01;
            }
        }
        for m in lm.weights.ff_down.iter_mut() {
            for v in m.data.iter_mut() {
                *v = 0.01;
            }
        }
        let logits = lm.forward(&[0u32, 1, 2]);
        assert_eq!(logits.len(), 3 * cfg.vocab_size);
        assert!(logits.iter().all(|&l| l.is_finite()));
    }

    #[test]
    fn num_params_matches_layout() {
        let cfg = Config { vocab_size: 10, dim: 4, num_layers: 2, num_heads: 2, max_seq_len: 16, ff_dim: 8 };
        let lm = ToyLm::new(cfg.clone());
        // embed: 10*4=40
        // per layer: qkv 3*4*4=48, attn_out 4*4=16, ff_up 8*4=32, ff_down 4*8=32, ln 2*4=8 => 136
        // 2 layers => 272
        // ln_f: 4
        // total: 40 + 272 + 4 = 316
        assert_eq!(lm.num_params(), 316);
    }
}
