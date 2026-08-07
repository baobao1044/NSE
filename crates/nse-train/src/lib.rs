//! # nse-train
//!
//! Training backends for NSE.
//!
//! The POC implements a vanilla [`SgdTrainer`] (real backprop) that trains the
//! Toy LM to a reasonable perplexity. Three research training architectures are
//! scaffolded as traits + skeletons for later implementation:
//!
//! - `ForwardForwardTrainer` — Hinton's Forward-Forward / predictive coding
//!   (local goodness, no backprop, ~zero VRAM overhead).
//! - `HopfieldTrainer` — modern Hopfield / associative memory (energy-based,
//!   one-shot/few-shot via vector projection).
//! - `LshSparseTrainer` — LSH-indexed sparse weight updates (~0.01% weights
//!   per step), sharing the LSH index with the inference router.
//!
//! Status: skeleton (M0). `SgdTrainer` lands in M2.

#![allow(dead_code)]

pub mod trainer;
pub mod sgd;
pub mod forward_forward;
pub mod hopfield;
pub mod lsh_sparse;

pub use sgd::{SgdConfig, SgdTrainer};
pub use trainer::Trainer;
