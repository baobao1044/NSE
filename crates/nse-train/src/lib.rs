//! # nse-train
//!
//! Training backends for NSE.
//!
//! The POC implements a vanilla [`SgdTrainer`] (real backprop) that trains the
//! Toy LM to a reasonable perplexity, plus three research training
//! architectures that explore training without huge GPU clusters:
//!
//! - [`ForwardForwardTrainer`] — Hinton's Forward-Forward: local per-block
//!   goodness (positive vs negative passes), no global backprop, with a
//!   light Hebbian association rule for the tied head. Research prototype.
//! - [`HopfieldTrainer`] — modern Hopfield / associative memory: one-shot
//!   writes into the FFN (keys/values), retrieval by cosine similarity.
//!   No backprop. Research prototype.
//! - [`LshSparseTrainer`] — LSH-indexed sparse weight updates (dense backprop
//!   + per-row LSH gradient masking, ~`sparse_fraction` of rows updated/step).
//!   Closest to SGD; shares the LSH index idea with the inference router.
//!
//! Shared momentum + gradient-clipping helpers live in [`sgd_apply`] and are
//! reused by SGD and LSH-sparse.
//!
//! Status: SGD (M2) + AVX2/HNSW/LSH/FF/Hopfield (N1-N5) implemented and tested.

#![allow(dead_code)]

pub mod trainer;
pub mod sgd;
pub mod sgd_apply;
pub mod forward_forward;
pub mod hopfield;
pub mod lsh;
pub mod lsh_sparse;

pub use sgd::{SgdConfig, SgdTrainer};
pub use forward_forward::{ForwardForwardConfig, ForwardForwardTrainer, Homeostasis};
pub use hopfield::{HopfieldConfig, HopfieldTrainer, hopfield_retrieve};
pub use lsh_sparse::{LshSparseConfig, LshSparseTrainer};
pub use trainer::Trainer;
