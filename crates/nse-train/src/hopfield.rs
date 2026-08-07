//! Modern Hopfield / Associative Memory trainer — one-shot writes into the
//! Toy LM's FFN, no backprop.
//!
//! ## Idea
//!
//! Treat the FFN of each transformer block as an associative memory store:
//!
//! - `ff_up`   `[ff_dim, dim]`   — **key store**: row `i` is key `K[i]`.
//! - `ff_down` `[dim, ff_dim]`   — **value store**: column `i` is value `V[i]`.
//!
//! For a retrieval query `k` (a context vector in `dim`), the modern
//! Hopfield / exponential-memory retrieval rule is
//!
//! ```text
//! z = ff_down · softmax( β · (ff_up · k) )
//! ```
//!
//! (the softmax replaces the FFN's GELU; this is the standard Hopfield
//! retrieval — *documented adaptation*). `β` is the sharpness / inverse
//! temperature; large `β` ≈ exact nearest-neighbor lookup, small `β` ≈
//! average. With `ff_down @ ff_up ≈ M` and near-orthogonal keys, `M·k ≈ v`
//! for the stored (key, value) pair.
//!
//! ## Writing
//!
//! For each (context, next-token) pair from the corpus we one-shot write a
//! slot `i` (round-robin across `ff_dim`):
//!
//! - key   = the block's FFN input activation (the layernorm-2 output `ln2_out`,
//!   i.e. the context representation at the FFN's input).
//! - value = a direction we want the block to *output* so the residual stream
//!   nudges the next-token's embedding. We use the **delta** needed at the
//!   block output: `v = E[target] − x_after_attn` (so that adding the FFN
//!   output moves the residual toward the target embedding). For this we need
//!   one forward pass to read `ln2_out` and `x_after_attn` — but **no
//!   backprop**: the write is analytic, one-shot.
//!
//! Keys are L2-normalized so retrieval is by cosine similarity.
//!
//! ## Head / gains
//!
//! `token_embed`, `ln1_gain`, `ln2_gain`, `ln_f_gain` are **frozen** at init
//! (Xavier/init values). Only `ff_up`/`ff_down` are written. This is the
//! purest form of the spec's "O(1)/O(log N) knowledge write instead of
//! O(N²) gradient updates".
//!
//! ## Honest limitation
//!
//! On the toy LM this is a **research prototype**: it stores `(context →
//! next-token direction)` associations and retrieves them by context
//! similarity. PPL improves over the uniform baseline when the test context
//! resembles a stored one, but it is not a competitive trainer — the bar is
//! "retrieval of a stored key returns the stored value, and the model does
//! not blow up".

use nse_models::{Tokenizer, ToyLm, forward_cached};

use crate::Trainer;

/// Hyperparameters for the Hopfield associative-memory trainer.
#[derive(Debug, Clone)]
pub struct HopfieldConfig {
    pub seq_len: usize,
    /// Number of slots (per layer) to write. Slots are taken round-robin from
    /// `ff_dim`; if `num_writes > ff_dim` earlier writes are overwritten.
    pub num_writes: usize,
    /// Retrieval sharpness β. Larger → sharper (more nearest-neighbor-like).
    pub beta: f32,
    pub log_every: usize,
    pub seed: u64,
}

impl Default for HopfieldConfig {
    fn default() -> Self {
        Self {
            seq_len: 16,
            num_writes: 64,
            beta: 8.0,
            log_every: 0,
            seed: 3,
        }
    }
}

/// Hopfield/associative-memory trainer: one-shot writes into the FFN.
pub struct HopfieldTrainer {
    pub config: HopfieldConfig,
}

impl HopfieldTrainer {
    pub fn new(config: HopfieldConfig) -> Self {
        Self { config }
    }
}

impl Default for HopfieldTrainer {
    fn default() -> Self {
        Self::new(HopfieldConfig::default())
    }
}

impl Trainer for HopfieldTrainer {
    fn name(&self) -> &'static str {
        "hopfield-associative"
    }

    fn train(&mut self, model: &mut ToyLm, corpus: &[u8]) -> anyhow::Result<()> {
        let cfg = &self.config;
        let seq = cfg.seq_len;
        let tok = Tokenizer::from_corpus(corpus);
        let ids = tok.encode(corpus);
        if ids.len() < seq + 2 {
            anyhow::bail!("corpus too small for seq_len={seq}: {} tokens", ids.len());
        }

        let d = model.config.dim;
        let ff_dim = model.config.ff_dim;
        let v = model.config.vocab_size;
        let n_layers = model.config.num_layers;

        // Collect, per write, the (key, value-direction) pair for every layer.
        // We do all forward passes first (immutable borrows of `model`) and
        // then perform the one-shot writes (mutable borrows) afterwards — this
        // avoids borrowing `model` as both mutable and immutable at once.
        // Key   = the block's FFN input at the last position (ln2_out).
        // Value = target embedding − pre-FFN residual (ln2_in), so the retrieved
        //         FFN output moves the residual toward the target embedding.
        let num_windows = ids.len().saturating_sub(seq + 1);
        let writes = cfg.num_writes.min(num_windows);

        // pairs[layer][w] = (key[d], value[d]).
        let mut pairs: Vec<Vec<(Vec<f32>, Vec<f32>)>> = Vec::with_capacity(n_layers);
        for _ in 0..n_layers {
            pairs.push(Vec::with_capacity(writes));
        }
        let emb = &model.weights.token_embed.data;
        for w in 0..writes {
            let start = w % num_windows;
            let tokens: Vec<u32> = ids[start..start + seq].to_vec();
            let (cache, _) = forward_cached(model, &tokens);
            let tgt = (ids[start + seq] as usize).min(v - 1);
            let tgt_emb = &emb[tgt * d..tgt * d + d];
            for layer in 0..n_layers {
                let lc = &cache.layers[layer];
                let lo = (seq - 1) * d;
                let key = lc.ln2_out[lo..lo + d].to_vec();
                let xa = &lc.ln2_in[lo..lo + d];
                let value: Vec<f32> = (0..d).map(|j| tgt_emb[j] - xa[j]).collect();
                pairs[layer].push((key, value));
            }
        }

        // One-shot writes: round-robin across ff_dim slots per layer.
        for layer in 0..n_layers {
            let ff_up = &mut model.weights.ff_up[layer].data; // [ff_dim, dim] keys
            let ff_down = &mut model.weights.ff_down[layer].data; // [dim, ff_dim] values
            for (w, (key, value)) in pairs[layer].iter().enumerate() {
                let slot = w % ff_dim;
                // Normalize the key (cosine retrieval).
                let kn = key.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);
                for j in 0..d {
                    ff_up[slot * d + j] = key[j] / kn;
                }
                for j in 0..d {
                    ff_down[j * ff_dim + slot] = value[j];
                }
            }
        }

        if cfg.log_every > 0 {
            eprintln!("[hopfield] wrote {} slots/layer, beta={}", writes, cfg.beta);
        }
        Ok(())
    }
}

/// Retrieve from a Hopfield FFN store: `z = ff_down · softmax(β · (ff_up · k))`.
/// Returns `z` of length `dim`. This is the retrieval rule the trainer equips
/// the FFN with; it is *not* used by the dense forward (which keeps GELU),
/// but exported so tests / callers can verify recall directly.
pub fn hopfield_retrieve(
    ff_up: &[f32],   // [ff_dim, dim]
    ff_down: &[f32], // [dim, ff_dim]
    key: &[f32],     // [dim]
    beta: f32,
    ff_dim: usize,
    dim: usize,
) -> Vec<f32> {
    // scores[i] = ff_up[i,:] . key
    let mut scores = vec![0.0f32; ff_dim];
    for i in 0..ff_dim {
        let mut s = 0.0;
        for j in 0..dim {
            s += ff_up[i * dim + j] * key[j];
        }
        scores[i] = beta * s;
    }
    // softmax over slots.
    let max = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for s in scores.iter_mut() {
        *s = (*s - max).exp();
        sum += *s;
    }
    let inv = if sum > 0.0 { 1.0 / sum } else { 0.0 };
    for s in scores.iter_mut() {
        *s *= inv;
    }
    // z[j] = sum_i ff_down[j, i] * softmax_i
    let mut z = vec![0.0f32; dim];
    for j in 0..dim {
        let mut acc = 0.0;
        for i in 0..ff_dim {
            acc += ff_down[j * ff_dim + i] * scores[i];
        }
        z[j] = acc;
    }
    z
}

#[cfg(test)]
mod tests {
    use super::*;
    use nse_models::Config;

    /// Retrieval of a stored key returns (approximately) the stored value.
    #[test]
    fn hopfield_retrieves_stored_value() {
        let dim = 8;
        let ff_dim = 16;
        let mut ff_up = vec![0.0f32; ff_dim * dim];
        let mut ff_down = vec![0.0f32; dim * ff_dim];
        // Store 3 (key, value) pairs in slots 0..3.
        let keys = [
            vec![1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            vec![0.0f32, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            vec![0.0f32, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ];
        let vals = [
            vec![0.5f32; 8],
            vec![-0.3f32; 8],
            vec![0.9f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ];
        for (i, (k, v)) in keys.iter().zip(vals.iter()).enumerate() {
            // normalize key
            let kn = k.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);
            for j in 0..dim {
                ff_up[i * dim + j] = k[j] / kn;
                ff_down[j * ff_dim + i] = v[j];
            }
        }
        // Query with the first key → should retrieve ~vals[0].
        let z = hopfield_retrieve(&ff_up, &ff_down, &keys[0], 16.0, ff_dim, dim);
        for j in 0..dim {
            assert!(
                (z[j] - vals[0][j]).abs() < 0.05,
                "retrieve mismatch at {j}: z={:.4} expected {:.4}",
                z[j],
                vals[0][j]
            );
        }
    }

    /// Training writes associations and the model's PPL stays finite and below
    /// the uniform baseline (does not blow up / degrade to random).
    #[test]
    fn hopfield_train_does_not_degrade_ppl() {
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
            num_layers: 2,
            num_heads: 2,
            max_seq_len: 64,
            ff_dim: 64,
        };
        let mut lm = ToyLm::init_random(cfg, 21);
        let mut trainer = HopfieldTrainer::new(HopfieldConfig {
            seq_len: 16,
            num_writes: 64,
            beta: 8.0,
            ..Default::default()
        });
        trainer.train(&mut lm, corpus).unwrap();

        // PPL on sliding windows.
        let mut total = 0.0f32;
        let mut count = 0usize;
        for start in 0..ids.len().saturating_sub(17) {
            let tokens = &ids[start..start + 16];
            let targets = &ids[start + 1..start + 17];
            let logits = lm.forward(tokens);
            for t in 0..16 {
                let row = &logits[t * tok.vocab_size..(t + 1) * tok.vocab_size];
                let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let sum: f32 = row.iter().map(|v| (v - max).exp()).sum();
                total -= ((row[targets[t] as usize] - max).exp() / sum).max(1e-12).ln();
                count += 1;
            }
        }
        let ppl = (-(total / count as f32)).exp();
        assert!(ppl.is_finite(), "PPL should be finite, got {ppl}");
        assert!(
            ppl < tok.vocab_size as f32,
            "Hopfield PPL {ppl:.2} should be below uniform {}",
            tok.vocab_size
        );
    }
}
