//! Shared SGD momentum + gradient-clipping helpers, used by both the dense
//! baseline ([`crate::sgd::SgdTrainer`]) and the LSH-sparse trainer
//! ([`crate::lsh_sparse::LshSparseTrainer`]).
//!
//! `apply_step` mutates a [`nse_models::ToyLm`] in place using a
//! [`nse_models::ToyLmGrads`]-shaped momentum velocity buffer, with global-norm
//! gradient clipping.

use nse_models::{ToyLm, ToyLmGrads};

/// Momentum SGD step with global-norm gradient clipping.
///
/// `v = momentum*v + scale*grad`; `w -= lr*v`, where `scale` clips the gradient
/// to `max_grad_norm` (no-op if below the threshold). The `grad` slice may
/// already be masked (for LSH): the masking reduces the effective update,
/// while clipping still uses the *masked* gradient's norm — so a sparse update
/// is also clipped, keeping training stable.
pub fn apply_step(
    model: &mut ToyLm,
    grads: &ToyLmGrads,
    vel: &mut ToyLmGrads,
    lr: f32,
    momentum: f32,
    max_grad_norm: f32,
) {
    let n_layers = model.config.num_layers;
    let norm = grad_norm_sq(grads).sqrt();
    let scale = if norm > max_grad_norm {
        max_grad_norm / norm.max(1e-8)
    } else {
        1.0
    };
    let w = &mut model.weights;

    apply(&mut w.token_embed.data, &grads.token_embed.data,
          &mut vel.token_embed.data, lr, momentum, scale);
    for l in 0..n_layers {
        apply(&mut w.qkv[l].data, &grads.qkv[l].data,
              &mut vel.qkv[l].data, lr, momentum, scale);
        apply(&mut w.attn_out[l].data, &grads.attn_out[l].data,
              &mut vel.attn_out[l].data, lr, momentum, scale);
        apply(&mut w.ff_up[l].data, &grads.ff_up[l].data,
              &mut vel.ff_up[l].data, lr, momentum, scale);
        apply(&mut w.ff_down[l].data, &grads.ff_down[l].data,
              &mut vel.ff_down[l].data, lr, momentum, scale);
        apply(&mut w.ln1_gain[l], &grads.ln1_gain[l],
              &mut vel.ln1_gain[l], lr, momentum, scale);
        apply(&mut w.ln2_gain[l], &grads.ln2_gain[l],
              &mut vel.ln2_gain[l], lr, momentum, scale);
    }
    apply(&mut w.ln_f_gain, &grads.ln_f_gain,
          &mut vel.ln_f_gain, lr, momentum, scale);
}

/// Sum of squares of all gradient entries (for global-norm clipping).
pub fn grad_norm_sq(g: &ToyLmGrads) -> f32 {
    let mut s = 0.0;
    s += g.token_embed.data.iter().map(|v| v * v).sum::<f32>();
    for l in 0..g.qkv.len() {
        s += g.qkv[l].data.iter().map(|v| v * v).sum::<f32>();
        s += g.attn_out[l].data.iter().map(|v| v * v).sum::<f32>();
        s += g.ff_up[l].data.iter().map(|v| v * v).sum::<f32>();
        s += g.ff_down[l].data.iter().map(|v| v * v).sum::<f32>();
        s += g.ln1_gain[l].iter().map(|v| v * v).sum::<f32>();
        s += g.ln2_gain[l].iter().map(|v| v * v).sum::<f32>();
    }
    s += g.ln_f_gain.iter().map(|v| v * v).sum::<f32>();
    s
}

/// Momentum SGD on one slice: `v = mom*v + scale*g`; `w -= lr*v`.
fn apply(w: &mut [f32], g: &[f32], v: &mut [f32], lr: f32, mom: f32, scale: f32) {
    debug_assert_eq!(w.len(), g.len());
    debug_assert_eq!(w.len(), v.len());
    for i in 0..w.len() {
        let gi = g[i] * scale;
        v[i] = mom * v[i] + gi;
        w[i] -= lr * v[i];
    }
}
