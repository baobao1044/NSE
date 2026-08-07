//! HNSW (Hierarchical Navigable Small World) index (performance scaffold).
//!
//! The POC uses brute-force exact MIPS ([`crate::index::MipsIndex`], O(N)).
//! HNSW achieves O(log N) approximate MIPS by building a multi-layer
//! navigable graph: queries descend from the top layer (sparse) to the
//! bottom (dense), greedily moving toward the query at each level, then
//! exploring neighbors at the base layer with a bounded beam search.
//!
//! For the POC's small expert counts (K ~ 10–100), brute force is faster and
//! exact. HNSW matters at the spec's scale (millions of experts).
//!
//! Status: scaffold (M6). Implementation deferred to the scale-out phase.
//! Must return the same top-K as brute force (up to approximation tolerance).

#![allow(dead_code)]

use crate::index::Hit;
use nse_core::sparse::MicroExpert;

/// HNSW graph index over micro-expert centroids.
///
/// Build parameters (per the HNSW paper):
/// - `M`: max connections per node (layer ≥ 1).
/// - `M0`: max connections at layer 0 (= 2*M).
/// - `ef_construction`: beam width during build.
/// - `ef_search`: beam width during query.
pub struct HnswIndex {
    pub experts: Vec<MicroExpert>,
    pub m: usize,
    pub ef_construction: usize,
    pub ef_search: usize,
    // Graph layers would live here: Vec<Vec<Vec<u32>>> (layer -> node -> neighbors).
}

impl HnswIndex {
    /// Build the HNSW graph from a set of micro-experts.
    ///
    /// (Stub — implemented in the scale-out phase.)
    pub fn build(experts: Vec<MicroExpert>, m: usize, ef_construction: usize) -> Self {
        let _ = (experts, m, ef_construction);
        todo!("scale-out phase: HNSW graph build")
    }

    /// Query the top-K experts by inner product.
    ///
    /// (Stub — implemented in the scale-out phase.)
    pub fn query(&self, _x: &[f32], _k: usize) -> Vec<Hit> {
        todo!("scale-out phase: HNSW query")
    }
}
