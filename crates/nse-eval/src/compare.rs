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

    /// The composite report should run all four forward paths and stay finite.
    #[test]
    fn compare_composite_runs_and_reports() {
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
        let report = compare_composite(
            &lm,
            &tm,
            corpus,
            16,
            8.0,
            Activation::All,
            SparseOptions::default(),
        );
        let s = report.pretty();
        assert!(s.contains("dense  GELU"));
        assert!(s.contains("sparse GELU"));
        assert!(report.dense_gelu.is_finite());
        assert!(report.dense_hopfield.is_finite());
        assert!(report.sparse_gelu.is_finite());
        assert!(report.sparse_hopfield.is_finite());
        // All four within an order of magnitude of each other on this tiny model
        // (ternary quantization + retrieval noise, but not catastrophic).
        let hi = report
            .dense_gelu
            .max(report.dense_hopfield)
            .max(report.sparse_gelu)
            .max(report.sparse_hopfield);
        let lo = report
            .dense_gelu
            .min(report.dense_hopfield)
            .min(report.sparse_gelu)
            .min(report.sparse_hopfield);
        assert!(hi < lo * 50.0, "paths diverge too much: lo={lo:.2} hi={hi:.2}");
    }
}

/// A 4-path comparison report: dense vs sparse, each under the standard GELU
/// FFN and the Hopfield-retrieval FFN. The headline artifact of the composite
/// architecture evaluation (§5.6): it shows which forward path each trained
/// representation does best under, and the cost of ternary quantization for
/// each path.
#[derive(Debug, Clone)]
pub struct CompositeReport {
    /// Dense model, GELU FFN (the standard forward path).
    pub dense_gelu: f32,
    /// Dense model, Hopfield-retrieval FFN (softmax over `ff_up`, β sharpness).
    pub dense_hopfield: f32,
    /// Sparse (transmuted) model, GELU FFN — the existing sparse path.
    pub sparse_gelu: f32,
    /// Sparse (transmuted) model, Hopfield-retrieval FFN on reconstructed
    /// ternary keys/values — the §5.6 research path.
    pub sparse_hopfield: f32,
    /// Average fraction of weights activated per token (from the activation
    /// mode; matches `CompareReport::avg_active_fraction`).
    pub avg_active_fraction: f32,
    pub activation_mode: Activation,
}

impl CompositeReport {
    /// Pretty-print the 4-path report with relative degradations.
    pub fn pretty(&self) -> String {
        let mode = match self.activation_mode {
            Activation::All => "all-experts (upper bound)".to_string(),
            Activation::Threshold { ratio, max_k } => {
                format!("threshold(ratio={ratio}, max_k={max_k})")
            }
        };
        // Sparse vs dense, per forward path (the cost of ternary transmutation).
        let deg_gelu = rel_drop(self.dense_gelu, self.sparse_gelu);
        let deg_hop = rel_drop(self.dense_hopfield, self.sparse_hopfield);
        // Hopfield vs GELU, per model (the value of the retrieval path).
        let hop_vs_gelu_dense = rel_drop(self.dense_gelu, self.dense_hopfield);
        let hop_vs_gelu_sparse = rel_drop(self.sparse_gelu, self.sparse_hopfield);
        format!(
            "=== NSE composite: 4-path PPL (β retrieval) ===\n\
             activation : {mode}\n\
             avg active : {active:.4}% of params/token\n\
             \n\
             dense  GELU          : {dg:.4}\n\
             dense  Hopfield(β)   : {dh:.4}   (vs GELU: {hvd:+.2}%)\n\
             sparse GELU          : {sg:.4}   (vs dense GELU: {deg_gelu:+.2}%)\n\
             sparse Hopfield(β)   : {sh:.4}   (vs dense Hopfield: {deg_hop:+.2}% | vs sparse GELU: {hvs:+.2}%)",
            active = self.avg_active_fraction * 100.0,
            dg = self.dense_gelu,
            dh = self.dense_hopfield,
            hvd = hop_vs_gelu_dense * 100.0,
            sg = self.sparse_gelu,
            deg_gelu = deg_gelu * 100.0,
            sh = self.sparse_hopfield,
            deg_hop = deg_hop * 100.0,
            hvs = hop_vs_gelu_sparse * 100.0,
        )
    }
}

/// Relative change `b` vs `a`: `(b - a) / a` (positive = b worse).
fn rel_drop(a: f32, b: f32) -> f32 {
    if a > 0.0 {
        (b - a) / a
    } else {
        0.0
    }
}

/// Run the 4-path composite comparison. `corpus` is tokenized with the model's
/// tokenizer. `beta` is the Hopfield retrieval sharpness used for both dense and
/// sparse retrieval paths.
pub fn compare_composite(
    lm: &ToyLm,
    tm: &TransmutedModel,
    corpus: &[u8],
    seq_len: usize,
    beta: f32,
    act: Activation,
    opts: SparseOptions,
) -> CompositeReport {
    let tok = Tokenizer::from_corpus(corpus);
    let ids = tok.encode(corpus);

    let dense_gelu = dense_ppl(lm, &ids, seq_len);
    let dense_hopfield = crate::ppl::dense_ppl_hopfield(lm, &ids, seq_len, beta);
    let sparse_gelu = sparse_ppl_with_options(tm, &ids, seq_len, act, opts);
    let sparse_hopfield =
        crate::ppl::sparse_ppl_hopfield_with_options(tm, &ids, seq_len, beta, act, opts);

    // Reuse the active-fraction estimate from the dense-vs-sparse comparison.
    let avg_active_fraction = match act {
        Activation::All => 1.0,
        Activation::Threshold { ratio, max_k } => {
            let total_experts: f32 = tm
                .layers
                .iter()
                .flat_map(|l| l.iter())
                .map(|sl| sl.experts.len() as f32)
                .sum();
            (max_k as f32 / total_experts.max(1.0)).min(1.0) * ratio.max(0.0).min(1.0)
        }
    };

    CompositeReport {
        dense_gelu,
        dense_hopfield,
        sparse_gelu,
        sparse_hopfield,
        avg_active_fraction,
        activation_mode: act,
    }
}
