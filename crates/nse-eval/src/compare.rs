//! Dense-vs-sparse comparison report.
//!
//! Runs both backends over the same corpus and reports `PPL_dense`,
//! `PPL_sparse`, the relative degradation, and the fraction of weights
//! activated — the headline result of the NSE POC.
//!
//! Status: skeleton (M0). Real comparison lands in M5.

/// A completed comparison report.
#[derive(Debug, Clone)]
pub struct CompareReport {
    pub ppl_dense: f32,
    pub ppl_sparse: f32,
    /// Relative PPL increase, e.g. 0.05 = sparse is 5% worse.
    pub rel_degradation: f32,
    /// Fraction of total weights activated per token on average.
    pub avg_active_fraction: f32,
}

impl CompareReport {
    /// Pretty-print the report to a string.
    pub fn pretty(&self) -> String {
        let pct = self.rel_degradation * 100.0;
        format!(
            "PPL dense  : {:.4}\n\
             PPL sparse : {:.4}\n\
             degradation: +{:.2}%\n\
             avg active : {:.4}% of params/token",
            self.ppl_dense,
            self.ppl_sparse,
            pct,
            self.avg_active_fraction * 100.0,
        )
    }
}
