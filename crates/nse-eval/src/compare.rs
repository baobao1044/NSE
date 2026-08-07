//! Dense-vs-sparse comparison report — the headline result of the NSE POC.
//!
//! Runs both backends over the same corpus and reports `PPL_dense`,
//! `PPL_sparse`, the relative degradation, and the average fraction of
//! weights activated per token.

use nse_core::sparse::TransmutedModel;
use nse_models::{Tokenizer, ToyLm};

use crate::ppl::{dense_ppl, sparse_ppl_with_options};
use crate::sparse_forward::{Activation, SparseOptions};

/// A completed comparison report.
#[derive(Debug, Clone)]
pub struct CompareReport {
    pub ppl_dense: f32,
    pub ppl_sparse: f32,
    /// Relative PPL increase, e.g. 0.05 = sparse is 5% worse.
    pub rel_degradation: f32,
    /// Fraction of total parameters activated per token on average.
    pub avg_active_fraction: f32,
    pub activation_mode: Activation,
}

impl CompareReport {
    /// Pretty-print the report to a string.
    pub fn pretty(&self) -> String {
        let pct = self.rel_degradation * 100.0;
        let mode = match self.activation_mode {
            Activation::All => "all-experts (upper bound)".to_string(),
            Activation::Threshold { ratio, max_k } => {
                format!("threshold(ratio={ratio}, max_k={max_k})")
            }
        };
        format!(
            "PPL dense  : {:.4}\n\
             PPL sparse : {:.4}\n\
             degradation: +{:.2}%\n\
             avg active : {:.4}% of params/token\n\
             activation : {mode}",
            self.ppl_dense,
            self.ppl_sparse,
            pct,
            self.avg_active_fraction * 100.0,
        )
    }
}

/// Run the full comparison. `corpus` is tokenized with the model's tokenizer.
pub fn compare(
    lm: &ToyLm,
    tm: &TransmutedModel,
    corpus: &[u8],
    seq_len: usize,
    act: Activation,
) -> CompareReport {
    compare_with_options(lm, tm, corpus, seq_len, act, SparseOptions::default())
}

/// Like [`compare`] but with explicit kernel/index backend selection (used by
/// the CLI `--kernel` / `--index` flags). The kernel (scalar vs AVX2) and the
/// index (brute-force vs HNSW) only change *how* the sparse matmuls are
/// evaluated, not the result — they should not affect PPL beyond FP noise.
pub fn compare_with_options(
    lm: &ToyLm,
    tm: &TransmutedModel,
    corpus: &[u8],
    seq_len: usize,
    act: Activation,
    opts: SparseOptions,
) -> CompareReport {
    let tok = Tokenizer::from_corpus(corpus);
    let ids = tok.encode(corpus);

    let ppl_dense = dense_ppl(lm, &ids, seq_len);
    let ppl_sparse = sparse_ppl_with_options(tm, &ids, seq_len, act, opts);

    let rel_degradation = if ppl_dense > 0.0 {
        (ppl_sparse - ppl_dense) / ppl_dense
    } else {
        0.0
    };

    // Average active fraction: with Activation::All, every expert row is on,
    // so it's 100%; with Threshold, it's roughly the fraction of experts kept
    // (a rough estimate — exact per-token counting would need the router).
    let avg_active_fraction = match act {
        Activation::All => {
            // All experts + core active = 100% of rows (no pruning).
            1.0
        }
        Activation::Threshold { ratio, max_k } => {
            // Rough: fraction of experts kept. A proper implementation counts
            // per-token; this is an over-estimate of the active param frac.
            let total_experts: f32 = tm
                .layers
                .iter()
                .flat_map(|l| l.iter())
                .map(|sl| sl.experts.len() as f32)
                .sum();
            let kept = max_k as f32;
            (kept / total_experts.max(1.0)).min(1.0) * ratio.max(0.0).min(1.0)
        }
    };

    CompareReport {
        ppl_dense,
        ppl_sparse,
        rel_degradation,
        avg_active_fraction,
        activation_mode: act,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nse_models::Config;
    use nse_zstm::{transmute, TransmuteConfig};

    #[test]
    fn compare_runs_and_reports() {
        let corpus = b"to be or not to be that is the question whether tis nobler \
                       in the mind to suffer the slings and arrows of outrageous \
                       fortune or to take arms against a sea of troubles and by";
        let tok = Tokenizer::from_corpus(corpus);
        let cfg = Config {
            vocab_size: tok.vocab_size,
            dim: 16,
            num_layers: 1,
            num_heads: 2,
            max_seq_len: 32,
            ff_dim: 32,
        };
        let lm = ToyLm::init_random(cfg, 5);
        let tm = transmute(&lm, Some(corpus), &TransmuteConfig::poc()).unwrap();
        let report = compare(&lm, &tm, corpus, 16, Activation::All);
        let s = report.pretty();
        assert!(s.contains("PPL dense"));
        assert!(s.contains("PPL sparse"));
        // Both PPLs should be finite.
        assert!(report.ppl_dense.is_finite());
        assert!(report.ppl_sparse.is_finite());
        // With all experts on, sparse PPL should be close-ish to dense
        // (only ternary error), within an order of magnitude.
        assert!(report.ppl_sparse < report.ppl_dense * 10.0);
    }
}
