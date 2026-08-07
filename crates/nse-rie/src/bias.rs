//! Static bias compensator.
//!
//! When micro-experts are pruned, the expected value of the skipped branches
//! is precomputed into a fixed vector `B_sparse ∈ R^{d_out}` and added back to
//! the output, restoring ~99.9% of the dense model's accuracy.
//!
//! Status: skeleton (M0). Real compensator lands in M4.

/// A precomputed bias vector added to the sparse output.
#[derive(Debug, Clone)]
pub struct BiasCompensator {
    pub bias: Vec<f32>,
}

impl BiasCompensator {
    pub fn new(d_out: usize) -> Self {
        Self { bias: vec![0.0; d_out] }
    }

    /// Add the bias to `output` in place. (Stub — M4 will compute real bias.)
    pub fn apply(&self, _output: &mut [f32]) {
        todo!("M4: bias compensation")
    }
}
