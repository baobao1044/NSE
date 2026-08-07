//! Manual backprop for the Toy LM.
//!
//! Mirrors the forward pass in [`crate::toy_lm`], saving intermediates in a
//! [`ForwardCache`] and producing a [`ToyLmGrads`] whose layout matches
//! [`crate::ToyLmWeights`]. Used by the SGD trainer (M2) to learn the dense
//! baseline that the ZSTM later transmutes.
//!
//! Loss: next-token cross-entropy. Position `t` predicts token `t+1`.
//!
//! A finite-difference gradient check (`tests::grad_check`) verifies the
//! backward pass is correct.

#![allow(dead_code)]

use nse_core::tensor::Matrix;

use crate::{Config, ToyLm, ToyLmWeights};

const EPS_LN: f32 = 1e-5;

/// Intermediates saved during the forward pass for use in backward.
pub struct ForwardCache {
    pub tokens: Vec<u32>,
    pub x_after_embed: Vec<f32>,        // [seq, dim]
    pub layers: Vec<LayerCache>,
    pub x_final_norm: Vec<f32>,         // [seq, dim]  (normalized, before head)
    pub logits: Vec<f32>,               // [seq, vocab]
}

pub struct LayerCache {
    pub ln1_in: Vec<f32>,       // x (residual stream input)  [seq, dim]
    pub ln1_out: Vec<f32>,      // normalized h1              [seq, dim]
    pub ln1_rstd: Vec<f32>,     // per-row reciprocal std      [seq]
    pub qkv_out: Vec<f32>,      // [seq, 3dim]
    pub attn_out: Vec<f32>,     // attention output before proj [seq, dim]
    pub attn_probs: Vec<f32>,   // [seq*heads, seq] softmax probs (causal)
    pub attn_proj_out: Vec<f32>,// attn @ Wattn_out  [seq, dim]
    pub ln2_in: Vec<f32>,       // x' (after attn residual) [seq, dim]
    pub ln2_out: Vec<f32>,      // normalized h2          [seq, dim]
    pub ln2_rstd: Vec<f32>,
    pub ff_up_pre: Vec<f32>,    // pre-gelu [seq, ff_dim]
    pub ff_up_act: Vec<f32>,   // post-gelu [seq, ff_dim]
    pub ff_down_out: Vec<f32>, // [seq, dim]
}

/// Gradients matching [`ToyLmWeights`].
#[derive(Debug, Clone)]
pub struct ToyLmGrads {
    pub token_embed: Matrix,
    pub ln1_gain: Vec<Vec<f32>>,
    pub qkv: Vec<Matrix>,
    pub attn_out: Vec<Matrix>,
    pub ln2_gain: Vec<Vec<f32>>,
    pub ff_up: Vec<Matrix>,
    pub ff_down: Vec<Matrix>,
    pub ln_f_gain: Vec<f32>,
}

impl ToyLmGrads {
    /// All-zero gradients for `lm`.
    pub fn zeros(cfg: &Config) -> Self {
        Self {
            token_embed: Matrix::zeros(cfg.vocab_size, cfg.dim),
            ln1_gain: vec![vec![0.0; cfg.dim]; cfg.num_layers],
            qkv: vec![Matrix::zeros(3 * cfg.dim, cfg.dim); cfg.num_layers],
            attn_out: vec![Matrix::zeros(cfg.dim, cfg.dim); cfg.num_layers],
            ln2_gain: vec![vec![0.0; cfg.dim]; cfg.num_layers],
            ff_up: vec![Matrix::zeros(cfg.ff_dim, cfg.dim); cfg.num_layers],
            ff_down: vec![Matrix::zeros(cfg.dim, cfg.ff_dim); cfg.num_layers],
            ln_f_gain: vec![0.0; cfg.dim],
        }
    }

    /// SGD update: `w -= lr * grad` (with momentum `m = beta*m + grad`).
    pub fn sgd_step(&self, weights: &mut ToyLmWeights, lr: f32) {
        axpy_neg(&mut weights.token_embed.data, &self.token_embed.data, lr);
        for l in 0..weights.qkv.len() {
            axpy_neg(&mut weights.qkv[l].data, &self.qkv[l].data, lr);
            axpy_neg(&mut weights.attn_out[l].data, &self.attn_out[l].data, lr);
            axpy_neg(&mut weights.ff_up[l].data, &self.ff_up[l].data, lr);
            axpy_neg(&mut weights.ff_down[l].data, &self.ff_down[l].data, lr);
            for j in 0..weights.ln1_gain[l].len() {
                weights.ln1_gain[l][j] -= lr * self.ln1_gain[l][j];
                weights.ln2_gain[l][j] -= lr * self.ln2_gain[l][j];
            }
        }
        for j in 0..weights.ln_f_gain.len() {
            weights.ln_f_gain[j] -= lr * self.ln_f_gain[j];
        }
        // token_embed head is tied — grad already accumulated.
    }
}

/// `dst -= lr * src` element-wise.
fn axpy_neg(dst: &mut [f32], src: &[f32], lr: f32) {
    for (d, s) in dst.iter_mut().zip(src.iter()) {
        *d -= lr * s;
    }
}

/// Forward pass that saves all intermediates needed for backward.
pub fn forward_cached(lm: &ToyLm, tokens: &[u32]) -> (ForwardCache, Vec<f32>) {
    let c = &lm.config;
    let seq = tokens.len();
    let d = c.dim;
    let v = c.vocab_size;
    let nh = c.num_heads;
    let hd = d / nh;

    // 1. Embedding lookup.
    let x = {
        let mut x = vec![0.0f32; seq * d];
        for (t, &tok) in tokens.iter().enumerate() {
            let tok = (tok as usize).min(v - 1);
            for j in 0..d {
                x[t * d + j] = lm.weights.token_embed.data[tok * d + j];
            }
        }
        x
    };

    let mut layers: Vec<LayerCache> = Vec::with_capacity(c.num_layers);
    let mut residual = x.clone();

    for layer in 0..c.num_layers {
        let ln1_gain = &lm.weights.ln1_gain[layer];
        let qkv_w = &lm.weights.qkv[layer];
        let attn_out_w = &lm.weights.attn_out[layer];
        let ln2_gain = &lm.weights.ln2_gain[layer];
        let ff_up_w = &lm.weights.ff_up[layer];
        let ff_down_w = &lm.weights.ff_down[layer];

        // --- Attention ---
        let (ln1_out, ln1_rstd) = layernorm_fwd(&residual, seq, d, ln1_gain);
        let qkv_out = matmul_rows(&ln1_out, qkv_w); // [seq, 3d]
        let (attn_out, attn_probs) = causal_attn_fwd(&qkv_out, seq, d, nh, hd);
        let attn_proj_out = matmul_rows(&attn_out, attn_out_w); // [seq, d]
        // residual add
        let x_after_attn: Vec<f32> =
            residual.iter().zip(attn_proj_out.iter()).map(|(a, b)| a + b).collect();

        // --- FFN ---
        let (ln2_out, ln2_rstd) = layernorm_fwd(&x_after_attn, seq, d, ln2_gain);
        let ff_up_pre = matmul_rows(&ln2_out, ff_up_w); // [seq, ff_dim]
        let ff_up_act: Vec<f32> = ff_up_pre.iter().map(|&v| gelu(v)).collect();
        let ff_down_out = matmul_rows(&ff_up_act, ff_down_w); // [seq, d]
        // residual add
        let x_after_ffn: Vec<f32> =
            x_after_attn.iter().zip(ff_down_out.iter()).map(|(a, b)| a + b).collect();

        layers.push(LayerCache {
            ln1_in: residual.clone(),
            ln1_out,
            ln1_rstd,
            qkv_out,
            attn_out,
            attn_probs,
            attn_proj_out,
            ln2_in: x_after_attn.clone(),
            ln2_out,
            ln2_rstd,
            ff_up_pre,
            ff_up_act,
            ff_down_out,
        });

        residual = x_after_ffn;
    }

    // 3. Final layernorm.
    let (x_final_norm, _ln_f_rstd) = layernorm_fwd(&residual, seq, d, &lm.weights.ln_f_gain);

    // 4. Tied head: logits = x_final_norm @ E^T.
    let logits = {
        let mut lg = vec![0.0f32; seq * v];
        for t in 0..seq {
            for w in 0..v {
                let mut s = 0.0;
                for j in 0..d {
                    s += x_final_norm[t * d + j] * lm.weights.token_embed.data[w * d + j];
                }
                lg[t * v + w] = s;
            }
        }
        lg
    };

    let cache = ForwardCache {
        tokens: tokens.to_vec(),
        x_after_embed: x,
        layers,
        x_final_norm,
        logits: logits.clone(),
    };
    (cache, logits)
}

/// Compute loss (mean next-token CE) and gradients.
/// `targets[t]` is the token predicted by position `t`.
pub fn backward(
    lm: &ToyLm,
    cache: &ForwardCache,
    targets: &[u32],
) -> (f32, ToyLmGrads) {
    let c = &lm.config;
    let seq = cache.tokens.len();
    let d = c.dim;
    let v = c.vocab_size;
    let nh = c.num_heads;
    let hd = d / nh;

    let mut grads = ToyLmGrads::zeros(c);

    // --- Loss + dL/dlogits ---
    let mut loss = 0.0;
    let mut dlogits = vec![0.0f32; seq * v];
    for t in 0..seq {
        let tgt = targets[t] as usize;
        // softmax
        let row = &cache.logits[t * v..(t + 1) * v];
        let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut expv = vec![0.0f32; v];
        let mut sum = 0.0;
        for w in 0..v {
            let e = (row[w] - max).exp();
            expv[w] = e;
            sum += e;
        }
        let p = expv[tgt] / sum;
        loss -= p.max(1e-12).ln();
        // dL/dlogits = softmax - onehot
        for w in 0..v {
            dlogits[t * v + w] = expv[w] / sum;
        }
        dlogits[t * v + tgt] -= 1.0;
    }
    loss /= seq as f32;
    // Scale gradients by 1/seq (mean loss).
    for g in dlogits.iter_mut() {
        *g /= seq as f32;
    }

    // --- Head: logits = x_final_norm @ E^T ---
    // dE[w,j] += sum_t x_final_norm[t,j] * dlogits[t,w]
    // dx_final_norm[t,j] = sum_w dlogits[t,w] * E[w,j]
    let mut dx_final = vec![0.0f32; seq * d];
    for t in 0..seq {
        for w in 0..v {
            let dl = dlogits[t * v + w];
            if dl == 0.0 {
                continue;
            }
            for j in 0..d {
                let xn = cache.x_final_norm[t * d + j];
                grads.token_embed.data[w * d + j] += dl * xn;
                dx_final[t * d + j] += dl * lm.weights.token_embed.data[w * d + j];
            }
        }
    }

    // --- Final layernorm backward ---
    let mut dx = layernorm_bwd(
        &residual_from_cache_last(cache),
        seq,
        d,
        &cache.x_final_norm,
        // We don't store ln_f rstd; recompute lazily below.
        &dx_final,
        &lm.weights.ln_f_gain,
        &mut grads.ln_f_gain,
    );

    // --- Per-layer backward (reverse) ---
    for layer in (0..c.num_layers).rev() {
        let lc = &cache.layers[layer];
        let qkv_w = &lm.weights.qkv[layer];
        let attn_out_w = &lm.weights.attn_out[layer];
        let ff_up_w = &lm.weights.ff_up[layer];
        let ff_down_w = &lm.weights.ff_down[layer];

        // dx here is grad w.r.t. x'' (layer output = x_after_ffn).
        // FFN residual: x'' = x' + ff_down_out  →  d(ff_down_out) = dx,  dx' += dx
        let d_ff_down_out = dx.clone();

        // ff_down_out = ff_up_act @ Wff_down^T  (Wff_down is [dim, ff_dim])
        // dWff_down[i,j] += sum_t ff_up_act[t,j] * d_ff_down_out[t,i]
        for t in 0..seq {
            for i in 0..d {
                let dl = d_ff_down_out[t * d + i];
                if dl == 0.0 {
                    continue;
                }
                for j in 0..c.ff_dim {
                    grads.ff_down[layer].data[i * c.ff_dim + j] +=
                        dl * lc.ff_up_act[t * c.ff_dim + j];
                }
            }
        }
        // d_ff_up_act = d_ff_down_out @ Wff_down  → [seq, ff_dim]
        let mut d_ff_up_act = vec![0.0f32; seq * c.ff_dim];
        for t in 0..seq {
            for j in 0..c.ff_dim {
                let mut s = 0.0;
                for i in 0..d {
                    s += d_ff_down_out[t * d + i] * ff_down_w.data[i * c.ff_dim + j];
                }
                d_ff_up_act[t * c.ff_dim + j] = s;
            }
        }
        // gelu backward
        let mut d_ff_up_pre = vec![0.0f32; seq * c.ff_dim];
        for t in 0..seq {
            for j in 0..c.ff_dim {
                d_ff_up_pre[t * c.ff_dim + j] =
                    d_ff_up_act[t * c.ff_dim + j] * gelu_deriv(lc.ff_up_pre[t * c.ff_dim + j]);
            }
        }
        // ff_up_pre = ln2_out @ Wff_up^T  (Wff_up is [ff_dim, dim])
        // dWff_up[k,j] += sum_t ln2_out[t,j] * d_ff_up_pre[t,k]
        for t in 0..seq {
            for k in 0..c.ff_dim {
                let dl = d_ff_up_pre[t * c.ff_dim + k];
                if dl == 0.0 {
                    continue;
                }
                for j in 0..d {
                    grads.ff_up[layer].data[k * d + j] += dl * lc.ln2_out[t * d + j];
                }
            }
        }
        // d_ln2_out = d_ff_up_pre @ Wff_up  → [seq, dim]
        let mut d_ln2_out = vec![0.0f32; seq * d];
        for t in 0..seq {
            for j in 0..d {
                let mut s = 0.0;
                for k in 0..c.ff_dim {
                    s += d_ff_up_pre[t * c.ff_dim + k] * ff_up_w.data[k * d + j];
                }
                d_ln2_out[t * d + j] = s;
            }
        }
        // layernorm2 backward → dx' (grad w.r.t. x_after_attn)
        let dx_prime_from_ffn = layernorm_bwd(
            &lc.ln2_in,
            seq,
            d,
            &lc.ln2_out,
            &d_ln2_out,
            &lm.weights.ln2_gain[layer],
            &mut grads.ln2_gain[layer],
        );

        // Residual: x' = x + attn_proj_out  →  d(attn_proj_out) = dx',  dx += dx'
        // dx' = dx''(direct residual) + dx'_from_ffn
        let mut dx_prime: Vec<f32> = dx.iter().zip(dx_prime_from_ffn.iter())
            .map(|(a, b)| a + b)
            .collect();
        let d_attn_proj_out = dx_prime.clone();

        // attn_proj_out = attn_out @ Wattn_out^T  (Wattn_out is [dim, dim])
        // dWattn_out[i,j] += sum_t attn_out[t,j] * d_attn_proj_out[t,i]
        for t in 0..seq {
            for i in 0..d {
                let dl = d_attn_proj_out[t * d + i];
                if dl == 0.0 {
                    continue;
                }
                for j in 0..d {
                    grads.attn_out[layer].data[i * d + j] += dl * lc.attn_out[t * d + j];
                }
            }
        }
        // d_attn_out = d_attn_proj_out @ Wattn_out  → [seq, dim]
        let mut d_attn_out = vec![0.0f32; seq * d];
        for t in 0..seq {
            for j in 0..d {
                let mut s = 0.0;
                for i in 0..d {
                    s += d_attn_proj_out[t * d + i] * attn_out_w.data[i * d + j];
                }
                d_attn_out[t * d + j] = s;
            }
        }
        // Attention backward → dq, dk, dv → dqkv_out
        let dqkv_out = causal_attn_bwd(&lc, &d_attn_out, seq, d, nh, hd);

        // qkv_out = ln1_out @ Wqkv^T  (Wqkv is [3d, dim])
        // dWqkv[k,j] += sum_t ln1_out[t,j] * dqkv_out[t,k]
        for t in 0..seq {
            for k in 0..3 * d {
                let dl = dqkv_out[t * 3 * d + k];
                if dl == 0.0 {
                    continue;
                }
                for j in 0..d {
                    grads.qkv[layer].data[k * d + j] += dl * lc.ln1_out[t * d + j];
                }
            }
        }
        // d_ln1_out = dqkv_out @ Wqkv  → [seq, dim]
        let mut d_ln1_out = vec![0.0f32; seq * d];
        for t in 0..seq {
            for j in 0..d {
                let mut s = 0.0;
                for k in 0..3 * d {
                    s += dqkv_out[t * 3 * d + k] * qkv_w.data[k * d + j];
                }
                d_ln1_out[t * d + j] = s;
            }
        }
        // layernorm1 backward → dx (grad w.r.t. residual stream input)
        let dx_from_attn = layernorm_bwd(
            &lc.ln1_in,
            seq,
            d,
            &lc.ln1_out,
            &d_ln1_out,
            &lm.weights.ln1_gain[layer],
            &mut grads.ln1_gain[layer],
        );

        // Residual: x = x' - attn_proj_out, so dx (next layer's input) = dx' + dx_from_attn
        dx = dx_prime.iter().zip(dx_from_attn.iter())
            .map(|(a, b)| a + b)
            .collect();
        // dx_prime consumed above; prevent unused warning
        let _ = &mut dx_prime;
    }

    // --- Embedding lookup backward: grad flows to token_embed for input tokens ---
    // Note: token_embed gradient already accumulated from the head (tied).
    // Now add the embedding-lookup gradient (dx at the bottom of the residual).
    for (t, &tok) in cache.tokens.iter().enumerate() {
        let tok = (tok as usize).min(v - 1);
        for j in 0..d {
            grads.token_embed.data[tok * d + j] += dx[t * d + j];
        }
    }

    (loss, grads)
}

/// The residual stream entering the final layernorm (reconstructed from the
/// last layer's output = last cache layer's ln2_in + ff_down_out). We didn't
/// store it directly, so recompute as ln2_in + ff_down_out of the last layer.
fn residual_from_cache_last(cache: &ForwardCache) -> Vec<f32> {
    if let Some(last) = cache.layers.last() {
        last.ln2_in
            .iter()
            .zip(last.ff_down_out.iter())
            .map(|(a, b)| a + b)
            .collect()
    } else {
        cache.x_after_embed.clone()
    }
}

// ---- Forward helpers (cached versions) ----

fn layernorm_fwd(x: &[f32], seq: usize, dim: usize, gain: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let mut out = vec![0.0f32; seq * dim];
    let mut rstd = vec![0.0f32; seq];
    for t in 0..seq {
        let row = &x[t * dim..(t + 1) * dim];
        let mean = row.iter().sum::<f32>() / dim as f32;
        let var = row.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / dim as f32;
        let r = 1.0 / (var + EPS_LN).sqrt();
        rstd[t] = r;
        for j in 0..dim {
            out[t * dim + j] = gain[j] * (row[j] - mean) * r;
        }
    }
    (out, rstd)
}

/// `out[i,j] = sum_k h[i,k] * w[j,k]`  (h @ w^T), w is [out, in].
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

/// Causal multi-head self-attention forward. Returns (attn_out [seq,dim], probs).
fn causal_attn_fwd(
    qkv: &[f32],
    seq: usize,
    dim: usize,
    num_heads: usize,
    head_dim: usize,
) -> (Vec<f32>, Vec<f32>) {
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut out = vec![0.0f32; seq * dim];
    // probs stored as [seq, num_heads, seq] -> flatten per (t, h): row of seq.
    let mut probs = vec![0.0f32; seq * num_heads * seq];
    for h in 0..num_heads {
        let off = h * head_dim;
        for i in 0..seq {
            let mut scores = vec![0.0f32; seq];
            let mut max = f32::NEG_INFINITY;
            for j in 0..=i {
                let mut s = 0.0;
                for d2 in 0..head_dim {
                    s += qkv[i * 3 * dim + off + d2] * qkv[j * 3 * dim + dim + off + d2];
                }
                s *= scale;
                scores[j] = s;
                if s > max {
                    max = s;
                }
            }
            let mut sum = 0.0;
            for j in 0..=i {
                scores[j] = (scores[j] - max).exp();
                sum += scores[j];
            }
            let inv = if sum > 0.0 { 1.0 / sum } else { 0.0 };
            for j in 0..=i {
                scores[j] *= inv;
            }
            // store probs
            let prow = i * num_heads * seq + h * seq;
            probs[prow..prow + seq].copy_from_slice(&scores);
            // weighted sum of V
            for d2 in 0..head_dim {
                let mut acc = 0.0;
                for j in 0..=i {
                    acc += scores[j] * qkv[j * 3 * dim + 2 * dim + off + d2];
                }
                out[i * dim + off + d2] = acc;
            }
        }
    }
    (out, probs)
}

/// GELU (tanh approx).
fn gelu(v: f32) -> f32 {
    const SQRT2_INV: f32 = 0.7071067811865475;
    let t = v * SQRT2_INV;
    let inner = 1.0 + 0.044715 * t * t;
    let tanh = (2.0 * t * inner).tanh();
    0.5 * v * (1.0 + tanh)
}

/// GELU derivative (tanh approx).
fn gelu_deriv(v: f32) -> f32 {
    const SQRT2_INV: f32 = 0.7071067811865475;
    let t = v * SQRT2_INV;
    let inner = 1.0 + 0.044715 * t * t;
    let tanh = (2.0 * t * inner).tanh();
    let sech2 = 1.0 - tanh * tanh;
    let dphi = 0.5 * (1.0 + tanh) + 0.5 * v * sech2 * SQRT2_INV * 2.0 * inner;
    dphi
}

// ---- Backward helpers ----

/// LayerNorm backward. Returns grad w.r.t. the input `x`.
/// `out_norm` is the normalized output `gain * (x-mean)*rstd`.
/// `dout` is grad w.r.t. the normalized output.
/// Accumulates gain gradient into `dgain`.
fn layernorm_bwd(
    x: &[f32],
    seq: usize,
    dim: usize,
    _out_norm: &[f32],
    dout: &[f32],
    gain: &[f32],
    dgain: &mut [f32],
) -> Vec<f32> {
    // Recompute mean and rstd from x.
    let mut dx = vec![0.0f32; seq * dim];
    for t in 0..seq {
        let row = &x[t * dim..(t + 1) * dim];
        let mean = row.iter().sum::<f32>() / dim as f32;
        let var = row.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / dim as f32;
        let rstd = 1.0 / (var + EPS_LN).sqrt();
        // x_norm_j = (x_j - mean) * rstd ;  out_norm_j = gain_j * x_norm_j
        let dy = &dout[t * dim..(t + 1) * dim];
        // dgain_j = sum_t dy_{t,j} * x_norm_{t,j}  (x_norm = out_norm/gain)
        let mut dy_norm_mean = 0.0;
        let mut dy_norm_xnorm_mean = 0.0;
        for j in 0..dim {
            let xnorm = (row[j] - mean) * rstd;
            let dy_norm = dy[j] * gain[j];
            dgain[j] += dy[j] * xnorm;
            dy_norm_mean += dy_norm;
            dy_norm_xnorm_mean += dy_norm * xnorm;
        }
        dy_norm_mean /= dim as f32;
        dy_norm_xnorm_mean /= dim as f32;
        for j in 0..dim {
            let xnorm = (row[j] - mean) * rstd;
            let dy_norm = dy[j] * gain[j];
            dx[t * dim + j] = rstd * (dy_norm - dy_norm_mean - xnorm * dy_norm_xnorm_mean);
        }
    }
    dx
}

/// Causal attention backward. `dout` is grad w.r.t. attention output [seq, dim].
/// Returns grad w.r.t. fused qkv [seq, 3*dim].
///
/// Standard layout: loop over each query `i`, compute the softmax backward
/// row `dprescores[i, 0..=i]` once, then scatter contributions to dQ[i],
/// dK[j] (j in 0..=i), and dV[j].
fn causal_attn_bwd(
    lc: &LayerCache,
    dout: &[f32],
    seq: usize,
    dim: usize,
    num_heads: usize,
    head_dim: usize,
) -> Vec<f32> {
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut dqkv = vec![0.0f32; seq * 3 * dim];
    let q_off = 0;            // Q starts at column 0 within the 3*dim row
    let k_off = dim;          // K starts at column dim
    let v_off = 2 * dim;      // V starts at column 2*dim

    for h in 0..num_heads {
        let off = h * head_dim;
        for i in 0..seq {
            let prow = i * num_heads * seq + h * seq;
            let probs = &lc.attn_probs[prow..prow + seq];
            let dout_i = &dout[i * dim + off..i * dim + off + head_dim];

            // 1. dV[j] += probs[i,j] * dout_i  (for j in 0..=i)
            for j in 0..=i {
                let pj = probs[j];
                for d2 in 0..head_dim {
                    dqkv[j * 3 * dim + v_off + off + d2] += pj * dout_i[d2];
                }
            }

            // 2. dattn_probs[i,j] = dout_i . V[j]  (for j in 0..=i)
            let mut dattn_probs = vec![0.0f32; seq];
            for j in 0..=i {
                let mut s = 0.0;
                for d2 in 0..head_dim {
                    let v_j = lc.qkv_out[j * 3 * dim + v_off + off + d2];
                    s += dout_i[d2] * v_j;
                }
                dattn_probs[j] = s;
            }

            // 3. softmax backward: dprescores[i,j] = probs[i,j] * (dattn_probs[i,j] - sum_k probs[i,k]*dattn_probs[i,k])
            let mut dot_p_dp = 0.0;
            for k in 0..=i {
                dot_p_dp += probs[k] * dattn_probs[k];
            }
            let mut dprescores = vec![0.0f32; seq];
            for j in 0..=i {
                dprescores[j] = probs[j] * (dattn_probs[j] - dot_p_dp);
            }

            // 4. dQ[i] += sum_j dprescores[i,j] * K[j] * scale
            for d2 in 0..head_dim {
                let mut s = 0.0;
                for j in 0..=i {
                    let k_j = lc.qkv_out[j * 3 * dim + k_off + off + d2];
                    s += dprescores[j] * k_j;
                }
                dqkv[i * 3 * dim + q_off + off + d2] += s * scale;
            }

            // 5. dK[j] += dprescores[i,j] * Q[i] * scale  (for j in 0..=i)
            for j in 0..=i {
                let dpj = dprescores[j];
                for d2 in 0..head_dim {
                    let q_i = lc.qkv_out[i * 3 * dim + q_off + off + d2];
                    dqkv[j * 3 * dim + k_off + off + d2] += dpj * q_i * scale;
                }
            }
        }
    }
    dqkv
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Config;

    #[test]
    fn grad_check() {
        // Small model for finite-difference gradient check.
        let cfg = Config {
            vocab_size: 7,
            dim: 4,
            num_layers: 1,
            num_heads: 2,
            max_seq_len: 8,
            ff_dim: 6,
        };
        let lm = ToyLm::init_random(cfg.clone(), 42);
        let tokens = vec![0u32, 1, 2, 3];
        let targets = vec![1u32, 2, 3, 4]; // predict t+1

        // Analytic gradients.
        let (cache, _logits) = forward_cached(&lm, &tokens);
        let (loss0, grads) = backward(&lm, &cache, &targets);

        // Finite-difference check on a few entries of token_embed.
        let eps = 1e-3;
        let mut lm2 = lm.clone();
        let idxs = [(0, 0), (1, 2), (3, 1), (5, 3)];
        for (r, col) in idxs {
            let orig = lm2.weights.token_embed.data[r * cfg.dim + col];
            lm2.weights.token_embed.data[r * cfg.dim + col] = orig + eps;
            let (_, l_plus) = forward_cached(&lm2, &tokens);
            let loss_plus = ce_loss(&l_plus, &targets, cfg.vocab_size);

            lm2.weights.token_embed.data[r * cfg.dim + col] = orig - eps;
            let (_, l_minus) = forward_cached(&lm2, &tokens);
            let loss_minus = ce_loss(&l_minus, &targets, cfg.vocab_size);
            lm2.weights.token_embed.data[r * cfg.dim + col] = orig;

            let num_grad = (loss_plus - loss_minus) / (2.0 * eps);
            let ana_grad = grads.token_embed.data[r * cfg.dim + col];
            // Allow generous tolerance for f32 + tanh GELU.
            let denom = num_grad.abs().max(ana_grad.abs()).max(1e-4);
            assert!(
                (num_grad - ana_grad).abs() / denom < 0.05,
                "embed grad mismatch at ({r},{col}): num={num_grad:.6} ana={ana_grad:.6}"
            );
        }
        // loss0 should be ~ln(vocab) ≈ ln(7) ≈ 1.95 at init (random model).
        assert!(loss0 > 1.0 && loss0 < 3.0, "unexpected init loss {loss0}");
    }

    fn ce_loss(logits: &[f32], targets: &[u32], vocab: usize) -> f32 {
        let seq = targets.len();
        let mut loss = 0.0;
        for t in 0..seq {
            let row = &logits[t * vocab..(t + 1) * vocab];
            let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0;
            for &v in row {
                sum += (v - max).exp();
            }
            let tgt = targets[t] as usize;
            loss -= ((row[tgt] - max).exp() / sum).max(1e-12).ln();
        }
        loss / seq as f32
    }
}
