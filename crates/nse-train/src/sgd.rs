//! Vanilla SGD trainer (real backprop baseline).
//!
//! Trains the Toy LM via next-token cross-entropy using the manual backprop in
//! [`nse_models::autograd`]. Produces a dense model with a reasonable
//! perplexity — the baseline the ZSTM later transmutes. SGD with momentum,
//! applied slice-by-slice (no flattening/unsafe).

use nse_models::{ToyLm, Tokenizer, ToyLmGrads, forward_cached, backward};

use crate::Trainer;

/// Hyperparameters for the SGD baseline trainer.
#[derive(Debug, Clone)]
pub struct SgdConfig {
    pub learning_rate: f32,
    pub seq_len: usize,
    pub epochs: usize,
    /// Apply a simple LR decay: lr *= decay each epoch.
    pub lr_decay: f32,
    /// Print loss every N steps (0 = silent during steps; epochs still logged).
    pub log_every: usize,
    /// Random init seed.
    pub seed: u64,
}

impl Default for SgdConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.3,
            seq_len: 32,
            epochs: 200,
            lr_decay: 0.995,
            log_every: 0,
            seed: 1337,
        }
    }
}

/// SGD with momentum — the dense baseline. Velocity is kept in a
/// `ToyLmGrads`-shaped buffer, one slot per parameter, applied slice-by-slice.
pub struct SgdTrainer {
    pub config: SgdConfig,
    pub momentum: f32,
    /// Max global gradient norm; gradients are clipped to this before the step.
    pub max_grad_norm: f32,
    vel: Option<ToyLmGrads>,
}

impl SgdTrainer {
    pub fn new(config: SgdConfig) -> Self {
        Self { config, momentum: 0.9, max_grad_norm: 1.0, vel: None }
    }
}

impl Trainer for SgdTrainer {
    fn name(&self) -> &'static str {
        "sgd-baseline"
    }

    fn train(&mut self, model: &mut ToyLm, corpus: &[u8]) -> anyhow::Result<()> {
        let (seq, epochs, lr_decay, log_every) = (
            self.config.seq_len,
            self.config.epochs,
            self.config.lr_decay,
            self.config.log_every,
        );
        let tok = Tokenizer::from_corpus(corpus);
        let ids = tok.encode(corpus);

        if ids.len() < seq + 2 {
            anyhow::bail!("corpus too small for seq_len={seq}: {} tokens", ids.len());
        }

        let mut lr = self.config.learning_rate;
        let num_windows = ids.len().saturating_sub(seq + 1);

        for epoch in 0..epochs {
            let mut epoch_loss = 0.0f32;
            let mut n_steps = 0usize;
            for start in 0..num_windows {
                let tokens: Vec<u32> = ids[start..start + seq].to_vec();
                let targets: Vec<u32> = ids[start + 1..start + 1 + seq].to_vec();

                let (cache, _logits) = forward_cached(model, &tokens);
                let (loss, grads) = backward(model, &cache, &targets);

                self.apply_step(model, &grads, lr);

                epoch_loss += loss;
                n_steps += 1;
                if log_every > 0 && n_steps % log_every == 0 {
                    eprintln!("[epoch {epoch} step {n_steps}] loss={loss:.4}");
                }
            }
            let mean_loss = if n_steps > 0 { epoch_loss / n_steps as f32 } else { 0.0 };
            let ppl = mean_loss.exp();
            if log_every > 0 || epoch == 0 || epoch == epochs - 1 || epoch % 20 == 19 {
                eprintln!(
                    "[epoch {epoch}/{epochs}] mean_loss={mean_loss:.4} ppl={ppl:.2} lr={lr:.5}"
                );
            }
            lr *= lr_decay;
        }
        Ok(())
    }
}

impl SgdTrainer {
    /// Momentum update with global-norm gradient clipping:
    /// `g *= max(1, ||g||/clip)`; `v = momentum*v + g`; `w -= lr*v`.
    fn apply_step(&mut self, model: &mut ToyLm, grads: &ToyLmGrads, lr: f32) {
        let n_layers = model.config.num_layers;
        let mom = self.momentum;
        if self.vel.is_none() {
            self.vel = Some(ToyLmGrads::zeros(&model.config));
        }
        let vel = self.vel.as_mut().unwrap();

        // Global gradient norm clipping.
        let norm = grad_norm_sq(grads).sqrt();
        let scale = if norm > self.max_grad_norm {
            self.max_grad_norm / norm.max(1e-8)
        } else {
            1.0
        };

        apply(&mut model.weights.token_embed.data, &grads.token_embed.data,
              &mut vel.token_embed.data, lr, mom, scale);
        for l in 0..n_layers {
            apply(&mut model.weights.qkv[l].data, &grads.qkv[l].data,
                  &mut vel.qkv[l].data, lr, mom, scale);
            apply(&mut model.weights.attn_out[l].data, &grads.attn_out[l].data,
                  &mut vel.attn_out[l].data, lr, mom, scale);
            apply(&mut model.weights.ff_up[l].data, &grads.ff_up[l].data,
                  &mut vel.ff_up[l].data, lr, mom, scale);
            apply(&mut model.weights.ff_down[l].data, &grads.ff_down[l].data,
                  &mut vel.ff_down[l].data, lr, mom, scale);
            apply(&mut model.weights.ln1_gain[l], &grads.ln1_gain[l],
                  &mut vel.ln1_gain[l], lr, mom, scale);
            apply(&mut model.weights.ln2_gain[l], &grads.ln2_gain[l],
                  &mut vel.ln2_gain[l], lr, mom, scale);
        }
        apply(&mut model.weights.ln_f_gain, &grads.ln_f_gain,
              &mut vel.ln_f_gain, lr, mom, scale);
    }
}

/// Sum of squares of all gradient entries (for global-norm clipping).
fn grad_norm_sq(g: &ToyLmGrads) -> f32 {
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

#[cfg(test)]
mod tests {
    use super::*;
    use nse_models::Config;

    #[test]
    fn training_reduces_ppl() {
        // Train a tiny model on a short corpus and verify perplexity drops well
        // below the uniform baseline (vocab_size).
        let corpus = b"to be or not to be that is the question whether tis nobler \
                        in the mind to suffer the slings and arrows of outrageous \
                        fortune or to take arms against a sea of troubles and by \
                        opposing end them to die to sleep no more and by a sleep \
                        to say we end the heartache and the thousand natural shocks";
        let tok = Tokenizer::from_corpus(corpus);
        let ids = tok.encode(corpus);
        let cfg = Config {
            vocab_size: tok.vocab_size,
            dim: 32,
            num_layers: 1,
            num_heads: 2,
            max_seq_len: 64,
            ff_dim: 64,
        };
        let mut lm = ToyLm::init_random(cfg, 7);

        // Baseline PPL before training (averaged over the same sliding windows
        // the trainer uses, so train/eval are comparable).
        let base_ppl = window_ppl(&lm, &ids, 16, tok.vocab_size);

        let mut trainer = SgdTrainer::new(SgdConfig {
            learning_rate: 0.05,
            seq_len: 16,
            epochs: 30,
            lr_decay: 1.0,
            log_every: 0,
            seed: 7,
        });
        trainer.max_grad_norm = 1.0;
        trainer.train(&mut lm, corpus).unwrap();

        // Trained PPL over the same sliding windows.
        let trained_ppl = window_ppl(&lm, &ids, 16, tok.vocab_size);

        assert!(
            trained_ppl < base_ppl * 0.5,
            "PPL did not drop enough: base={base_ppl:.2} trained={trained_ppl:.2}"
        );
        assert!(
            trained_ppl < tok.vocab_size as f32,
            "trained PPL {trained_ppl:.2} should be below uniform {}",
            tok.vocab_size
        );
    }

    /// Mean PPL over all sliding windows of length `seq`, predicting t+1.
    /// Matches the trainer's loss definition for a fair comparison.
    fn window_ppl(lm: &ToyLm, ids: &[u32], seq: usize, vocab: usize) -> f32 {
        let mut total = 0.0;
        let mut count = 0usize;
        for start in 0..ids.len().saturating_sub(seq + 1) {
            let tokens = &ids[start..start + seq];
            let targets = &ids[start + 1..start + 1 + seq];
            let (_, logits) = forward_cached(lm, tokens);
            total += mean_ce(&logits, targets, vocab);
            count += 1;
        }
        if count == 0 {
            f32::INFINITY
        } else {
            (total / count as f32).exp()
        }
    }

    fn mean_ce(logits: &[f32], targets: &[u32], vocab: usize) -> f32 {
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
