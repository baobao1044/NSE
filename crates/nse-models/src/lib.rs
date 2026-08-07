//! # nse-models
//!
//! Model definitions and weight loading for NSE.
//!
//! The POC ships a [`toy_lm`] (a small transformer-style language model) plus a
//! char-level tokenizer, used as the dense source for transmutation and PPL
//! evaluation. A `safetensors` loader and flexible [`Config`] are provided so
//! externally trained models can be plugged in later.
//!
//! Status: skeleton (M0). Toy LM forward + tokenizer land in M1.

#![allow(dead_code)]

pub mod config;
pub mod tokenizer;
pub mod toy_lm;
pub mod loader;
pub mod autograd;

pub use config::Config;
pub use tokenizer::Tokenizer;
pub use toy_lm::{ToyLm, ToyLmWeights};
pub use autograd::{
    ForwardCache, ToyLmGrads, block_backward_local, forward_cached, backward,
};
