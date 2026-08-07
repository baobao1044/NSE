//! # nse-eval
//!
//! Perplexity evaluation and dense-vs-sparse comparison.
//!
//! Computes language-model perplexity (PPL) for both the dense source model and
//! the transmuted sparse NSE model over the same corpus, then reports the
//! relative PPL degradation — the headline metric of the NSE POC.
//!
//! Status: skeleton (M0). PPL runners + comparison report land in M5.

#![allow(dead_code)]

pub mod ppl;
pub mod compare;
