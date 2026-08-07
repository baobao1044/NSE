//! # nse-eval
//!
//! Perplexity evaluation and dense-vs-sparse comparison — the headline metric
//! of the NSE POC.
//!
//! - `sparse_forward`: the sparse forward pass over a [`TransmutedModel`],
//!   mirroring the dense forward but with matmuls replaced by sparse linears.
//! - `ppl`: perplexity for both the dense [`nse_models::ToyLm`] and the
//!   sparse [`nse_core::TransmutedModel`], over the same sliding windows.
//! - `compare`: runs both and produces a [`CompareReport`] with `PPL_dense`,
//!   `PPL_sparse`, the relative degradation, and the active-fraction.

#![allow(dead_code)]

pub mod compare;
pub mod ppl;
pub mod sparse_forward;

pub use compare::{compare, compare_with_options, CompareReport};
pub use ppl::{dense_ppl, sparse_ppl, sparse_ppl_with_options, perplexity_from_logprobs, logprobs};
pub use sparse_forward::{sparse_forward, sparse_forward_with_options, Activation, SparseOptions};
