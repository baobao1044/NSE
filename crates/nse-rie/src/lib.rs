//! # nse-rie
//!
//! Routing & Indexing Engine (online).
//!
//! For each input token, finds the relevant micro-experts in sub-linear time
//! without scanning the whole model:
//!
//! - `index` — Maximum Inner Product Search (MIPS) tree (HNSW / LSH). The POC
//!   uses brute-force exact MIPS (correct, O(N)) — HNSW is scaffolded.
//! - `router` — adaptive threshold router: prune centroids below a dynamic
//!   threshold `θ(x)`, keep only the dynamic top-K.
//! - `bias` — static bias compensator: a precomputed vector `B_sparse` added
//!   back to restore the dense model's expectation after pruning.
//!
//! Status: skeleton (M0). Brute-force MIPS + threshold + bias land in M4.

#![allow(dead_code)]

pub mod index;
pub mod router;
pub mod bias;
