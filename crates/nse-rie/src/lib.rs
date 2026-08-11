//! # nse-rie
//!
//! Routing & Indexing Engine (online).
//!
//! For each input activation, finds the relevant micro-experts without
//! scanning the whole model:
//!
//! - `index` — Maximum Inner Product Search (MIPS) over expert centroids.
//!   Brute-force exact (O(N)) for the POC; HNSW/LSH scaffolded for scale-out.
//! - `router` — adaptive threshold router: prune centroids below `θ(x)`,
//!   keep the dynamic top-K (or keep all as the upper-bound reference).
//! - `bias` — static bias compensator: add the precomputed `B[i]` to restore
//!   the expected contribution of pruned experts.
//!
//! The sparse-linear primitive that ties RIE + LLER together for one layer is
//! [`sparse_linear`]: given a [`nse_core::SparseLayer`] and an input vector,
//! produce the output vector using the dense core, activated experts, and
//! bias.

#![allow(dead_code)]

pub mod bias;
pub mod hnsw;
pub mod index;
pub mod router;

pub use bias::{apply as apply_bias, apply_layer};
pub use hnsw::HnswIndex;
pub use index::{Hit, MipsIndex};
pub use router::{route_all, route_by_ratio, RouterConfig};

// Re-export the kernel selector so callers can choose Scalar vs AVX2.
pub use nse_ller::KernelKind;

use nse_core::sparse::SparseLayer;
use nse_ller::{
    apply_bias as ller_apply_bias, apply_bias_pruned_only as ller_apply_bias_pruned_only,
    compute_dense_core_dispatch, compute_pq_micro_expert_dispatch,
    compute_ternary_micro_expert_dispatch,
};

/// Which MIPS backend to use for expert routing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IndexKind {
    /// Exact brute-force (O(N), canonical).
    Brute,
    /// Approximate HNSW (O(log N)).
    Hnsw,
}

impl Default for IndexKind {
    fn default() -> Self {
        IndexKind::Brute
    }
}

/// Trait abstracting "return every expert hit, sorted by score descending",
/// so the router can work with either backend.
pub trait MipsQuery {
    fn query_all(&self, x: &[f32]) -> Vec<Hit>;
}

impl<'a> MipsQuery for MipsIndex<'a> {
    fn query_all(&self, x: &[f32]) -> Vec<Hit> {
        MipsIndex::query_all(self, x)
    }
}

impl MipsQuery for HnswIndex {
    fn query_all(&self, x: &[f32]) -> Vec<Hit> {
        // Return every expert (k = N), sorted descending — matches brute force.
        // For the POC's small expert counts, querying with k = N and a large
        // ef_search gives exact recall; at the spec's scale HNSW's value is
        // the sub-linear search, not recall on tiny graphs.
        self.query(x, self.num_experts())
    }
}

/// Build an HNSW index over a [`SparseLayer`]'s experts with POC-friendly
/// defaults (large ef so small expert counts give exact recall).
pub fn build_hnsw_for_layer(sl: &SparseLayer) -> HnswIndex {
    HnswIndex::new(&sl.experts, 8, 32, sl.experts.len().max(16))
}

/// Sparse forward of one linear layer: `y = core(x) + sum_activated experts(x)
/// + bias`. `activated` is the set of expert indices (into `sl.experts`) to
/// compute; the rest are pruned and their average contribution is covered by
/// the static bias. `kind` selects the compute kernel (`Scalar` = canonical,
/// `Auto` = AVX2 if available, else scalar).
///
/// Each expert dispatches by its quantization scheme: `pq: Some` → PQ kernel
/// (decodes against `sl.pq_codebook`); `pq: None` → ternary kernel. A layer is
/// homogeneous (all experts share one scheme, set at ZSTM time), but the
/// dispatch handles mixed layers defensively: a PQ expert needs the codebook,
/// a ternary expert never does.
pub fn sparse_linear_with_kernel(
    sl: &SparseLayer,
    x: &[f32],
    activated: &[usize],
    kind: KernelKind,
) -> Vec<f32> {
    let mut y = vec![0.0f32; sl.out_dim];
    // Dense core (always on).
    compute_dense_core_dispatch(&sl.dense_core, &sl.core_row_ids, x, &mut y, kind);
    // Activated micro-experts — dispatch by quantization scheme.
    let codebook = sl.pq_codebook.as_ref();
    for &eid in activated {
        let expert = &sl.experts[eid];
        match (&expert.pq, codebook) {
            (Some(_), Some(cb)) => {
                compute_pq_micro_expert_dispatch(expert, x, &mut y, cb, kind);
            }
            (Some(_), None) => {
                // Defensive: expert claims PQ but layer has no codebook
                // (shouldn't happen — ZSTM sets both together). Skip rather
                // than panic so a corrupt model degrades gracefully.
            }
            (None, _) => {
                compute_ternary_micro_expert_dispatch(expert, x, &mut y, kind);
            }
        }
    }
    // Bias application — dispatch by mode:
    // - empty `row_to_expert`  → legacy (unconditional add; reproduces M8).
    // - non-empty + no table   → pruned-only mean-input bias (S1 correctness fix).
    // - non-empty + table      → adaptive per-token bias (S4; handled below).
    if sl.row_to_expert.is_empty() {
        // Legacy: unconditional add to every output row.
        ller_apply_bias(&sl.bias, &mut y);
    } else {
        // Build the activated-expert boolean mask (size n_experts).
        let mut activated_set = vec![false; sl.experts.len()];
        for &eid in activated {
            if eid < activated_set.len() {
                activated_set[eid] = true;
            }
        }
        // Adaptive (S4): if a per-token bias table + activation codebook are
        // present, encode `x` and look up per-code bias for pruned rows;
        // otherwise fall back to the fixed mean-input bias for pruned rows.
        let used_adaptive = match (&sl.bias_table, &sl.input_codebook) {
            (Some(table), Some(cb)) => {
                crate::bias::apply_adaptive(
                    table,
                    &mut y,
                    &sl.row_to_expert,
                    &activated_set,
                    x,
                    cb,
                    sl.out_dim,
                );
                true
            }
            _ => false,
        };
        if !used_adaptive {
            ller_apply_bias_pruned_only(&sl.bias, &mut y, &sl.row_to_expert, &activated_set);
        }
    }
    y
}

/// Sparse forward with the default (`Auto`) kernel — backward-compatible with
/// the original POC entry point.
pub fn sparse_linear(sl: &SparseLayer, x: &[f32], activated: &[usize]) -> Vec<f32> {
    sparse_linear_with_kernel(sl, x, activated, KernelKind::Auto)
}

/// Sparse forward with **all** experts activated — the upper-bound reference.
/// With all experts on, the only degradation vs dense is ternary
/// quantization error (the bias is then zero on average for the prunable
/// rows, since nothing is pruned).
pub fn sparse_linear_all(sl: &SparseLayer, x: &[f32]) -> Vec<f32> {
    let activated: Vec<usize> = (0..sl.experts.len()).collect();
    sparse_linear(sl, x, &activated)
}

#[cfg(test)]
mod tests {
    use super::*;
    use nse_core::sparse::MicroExpert;
    use nse_core::tensor::Matrix;

    fn mk_layer(core: &[[f32; 3]], core_ids: &[u32], expert_rows: &[(&[i8], f32, u32)]) -> SparseLayer {
        // expert_rows: (ternary row, scale, original row id)
        let in_dim = 3;
        let dense_core = Matrix {
            rows: core.len(),
            cols: in_dim,
            data: core.iter().flat_map(|r| r.iter().copied()).collect(),
        };
        let expert = MicroExpert {
            row_ids: expert_rows.iter().map(|(_, _, id)| *id).collect(),
            ternary: expert_rows.iter().flat_map(|(t, _, _)| t.iter().copied()).collect(),
            row_scales: expert_rows.iter().map(|(_, s, _)| *s).collect(),
            centroid: vec![1.0; in_dim],
            mean_input: vec![0.0; in_dim],
            pq: None,
        };
        let out_dim = 1 + expert_rows.len();
        let bias = vec![0.0; out_dim];
        SparseLayer {
            out_dim,
            in_dim,
            dense_core,
            core_row_ids: core_ids.to_vec(),
            experts: vec![expert],
            bias,
            mean_input: vec![0.0; in_dim],
            pq_codebook: None,
            // Legacy mode (empty) so existing tests keep the unconditional
            // bias-add behavior they were written against.
            row_to_expert: Vec::new(),
            input_codebook: None,
            bias_table: None,
        }
    }

    #[test]
    fn sparse_linear_with_all_matches_reconstruction() {
        // core owns row 0: [1,2,3]; expert owns row 1: scale 2, ternary [1,-1,0]
        let sl = mk_layer(&[[1.0, 2.0, 3.0]], &[0], &[(vec![1i8, -1, 0].as_slice(), 2.0, 1u32)]);
        let x = vec![0.5, 1.0, 2.0];
        let y = sparse_linear_all(&sl, &x);
        // y[0] = 1*.5+2*1+3*2 = 8.5 ; y[1] = 2*(+0.5 -1.0 +0) = 2*(-0.5) = -1.0
        assert!((y[0] - 8.5).abs() < 1e-5);
        assert!((y[1] - (-1.0)).abs() < 1e-5);
    }

    #[test]
    fn sparse_linear_with_none_uses_bias_only() {
        // bias[i] = 0 for core rows, set a nonzero bias for the expert row.
        let mut sl = mk_layer(&[[1.0, 0.0, 0.0]], &[0], &[(vec![1i8, 0, 0].as_slice(), 1.0, 1u32)]);
        sl.bias[1] = 7.0;
        let x = vec![2.0, 0.0, 0.0];
        let y = sparse_linear(&sl, &x, &[]); // no experts activated
        // y[0] = core = 1*2 = 2 ; y[1] = bias only = 7 (expert pruned)
        assert!((y[0] - 2.0).abs() < 1e-5);
        assert!((y[1] - 7.0).abs() < 1e-5);
    }

    /// Phase 8 / S1 correctness fix: when `row_to_expert` is non-empty the bias
    /// is applied **pruned-only**, not unconditionally. An activated expert
    /// row must NOT receive the bias (no double-count of the mean-input term);
    /// a pruned expert row DOES receive it; a core row never does.
    #[test]
    fn bias_pruned_only_no_double_count() {
        // 2-expert layer: core owns row 0, expert 0 owns row 1, expert 1 owns row 2.
        // Both experts activated → row 1 & row 2 computed; none should get bias.
        let in_dim = 3;
        let dense_core = Matrix {
            rows: 1,
            cols: in_dim,
            data: vec![1.0, 2.0, 3.0],
        };
        let e0 = MicroExpert {
            row_ids: vec![1],
            ternary: vec![1, -1, 0],
            row_scales: vec![2.0],
            centroid: vec![1.0; in_dim],
            mean_input: vec![0.0; in_dim],
            pq: None,
        };
        let e1 = MicroExpert {
            row_ids: vec![2],
            ternary: vec![0, 1, -1],
            row_scales: vec![1.5],
            centroid: vec![1.0; in_dim],
            mean_input: vec![0.0; in_dim],
            pq: None,
        };
        // Non-zero bias on the expert rows (simulating a real-corpus mean).
        let mut bias = vec![0.0f32; 3];
        bias[1] = 100.0; // would double-count if added to activated e0 row
        bias[2] = 200.0; // would double-count if added to activated e1 row
        // row_to_expert: row 0 → -1 (core), row 1 → 0 (expert 0), row 2 → 1 (expert 1).
        let row_to_expert = vec![-1i32, 0, 1];
        let sl = SparseLayer {
            out_dim: 3,
            in_dim,
            dense_core,
            core_row_ids: vec![0],
            experts: vec![e0, e1],
            bias,
            mean_input: vec![0.0; in_dim],
            pq_codebook: None,
            row_to_expert,
            input_codebook: None,
            bias_table: None,
        };
        let x = vec![0.5, 1.0, 2.0];

        // --- Case A: all experts activated → NO bias added (pruned-only) ---
        let y_all = sparse_linear(&sl, &x, &[0, 1]);
        // y[0] = core = 1*.5+2*1+3*2 = 8.5
        // y[1] = e0 = 2*(+.5 -1.0 +0) = -1.0  (NO +100 bias — no double-count)
        // y[2] = e1 = 1.5*(0 +1.0 -2.0) = -1.5  (NO +200 bias)
        assert!((y_all[0] - 8.5).abs() < 1e-5, "core row: {}", y_all[0]);
        assert!((y_all[1] - (-1.0)).abs() < 1e-5, "e0 row should have NO bias: {}", y_all[1]);
        assert!((y_all[2] - (-1.5)).abs() < 1e-5, "e1 row should have NO bias: {}", y_all[2]);

        // --- Case B: only expert 0 activated → row 2 (expert 1) pruned → gets bias ---
        let y_half = sparse_linear(&sl, &x, &[0]);
        // y[1] = e0 computed = -1.0 (no bias)
        // y[2] = bias only = 200.0 (expert 1 pruned)
        assert!((y_half[1] - (-1.0)).abs() < 1e-5, "e0 still no bias when activated: {}", y_half[1]);
        assert!((y_half[2] - 200.0).abs() < 1e-5, "pruned e1 row should get bias: {}", y_half[2]);
        // core row unchanged
        assert!((y_half[0] - 8.5).abs() < 1e-5);

        // --- Case C: no experts activated → both expert rows pruned → both get bias ---
        let y_none = sparse_linear(&sl, &x, &[]);
        assert!((y_none[1] - 100.0).abs() < 1e-5, "all-pruned e0 should get bias: {}", y_none[1]);
        assert!((y_none[2] - 200.0).abs() < 1e-5, "all-pruned e1 should get bias: {}", y_none[2]);
        assert!((y_none[0] - 8.5).abs() < 1e-5, "core never gets bias: {}", y_none[0]);
    }

    /// Phase 8 / S4 adaptive bias: when `input_codebook` + `bias_table` are
    /// present, the bias for a pruned row depends on the token's `x` — two
    /// different inputs that encode against the activation codebook to
    /// different codes must produce different bias values for the same
    /// pruned row. This is the core "per-token, not mean" property.
    #[test]
    fn bias_adaptive_depends_on_x() {
        use nse_core::sparse::PqCodebook;
        let in_dim = 3;
        // Two well-separated activation centroids (M=1, nbits=1 → 2 codes).
        let input_codebook = PqCodebook {
            num_sub_vectors: 1,
            nbits: 1,
            sub_dim: in_dim,
            codebook: vec![10.0, 0.0, 0.0, -10.0, 0.0, 0.0],
        };
        // Dense core owns row 0 (computed exactly, no bias). Expert 0 owns
        // row 1 (prunable). No experts activated → row 1 gets adaptive bias.
        let dense_core = Matrix {
            rows: 1,
            cols: in_dim,
            data: vec![1.0, 2.0, 3.0],
        };
        let expert = MicroExpert {
            row_ids: vec![1],
            ternary: vec![1, -1, 0],
            row_scales: vec![2.0],
            centroid: vec![1.0; in_dim],
            mean_input: vec![0.0; in_dim],
            pq: None,
        };
        let out_dim = 2;
        // bias_table[c * out_dim + i]: code 0 → bias 5.0 for row 1;
        // code 1 → bias 50.0 for row 1. Core row 0 left 0 in both codes.
        let bias_table = vec![
            0.0, 5.0,  // code 0: row 0 = 0 (core), row 1 = 5.0
            0.0, 50.0, // code 1: row 0 = 0 (core), row 1 = 50.0
        ];
        let row_to_expert = vec![-1i32, 0];
        let sl = SparseLayer {
            out_dim,
            in_dim,
            dense_core,
            core_row_ids: vec![0],
            experts: vec![expert],
            bias: vec![0.0, 0.0], // legacy mean bias (unused in adaptive mode)
            mean_input: vec![0.0; in_dim],
            pq_codebook: None,
            row_to_expert,
            input_codebook: Some(input_codebook),
            bias_table: Some(bias_table),
        };
        // x_a = [10, 0, 0] encodes to code 0 (nearest centroid) → bias 5.0.
        // x_b = [-10, 0, 0] encodes to code 1 → bias 50.0.
        // No experts activated → row 1 (pruned) gets the adaptive bias.
        let y_a = sparse_linear(&sl, &[10.0, 0.0, 0.0], &[]);
        let y_b = sparse_linear(&sl, &[-10.0, 0.0, 0.0], &[]);
        // Core row 0: 1*10+2*0+3*0 = 10 (x_a), 1*-10 = -10 (x_b).
        assert!((y_a[0] - 10.0).abs() < 1e-5, "core x_a: {}", y_a[0]);
        assert!((y_b[0] - (-10.0)).abs() < 1e-5, "core x_b: {}", y_b[0]);
        // Pruned row 1: adaptive bias depends on x.
        assert!((y_a[1] - 5.0).abs() < 1e-5, "x_a → code 0 → bias 5.0: {}", y_a[1]);
        assert!((y_b[1] - 50.0).abs() < 1e-5, "x_b → code 1 → bias 50.0: {}", y_b[1]);
        // The two pruned-row values MUST differ — the per-token property.
        assert!((y_a[1] - y_b[1]).abs() > 1e-5, "adaptive bias must depend on x");
    }
}
