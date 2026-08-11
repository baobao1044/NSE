//! # nse-zstm
//!
//! Zero-Shot Transmutation Module (offline).
//!
//! Converts a dense model's weights into the NSE sparse representation without
//! retraining, in three stages matching the NSE spec:
//!
//! 1. `outlier` — extract high-amplitude outlier rows into a fixed dense
//!    core (kept on the "L1 cache" path at inference).
//! 2. `cluster` — spherical k-means partitioning of the remaining rows into
//!    micro-experts (cache-sized), grouped by direction in input space.
//! 3. `quantize` — sub-1-bit ternary compression `{−1, 0, 1}` plus a per-row
//!    scale (BitNet-style).
//!
//! The `transmuter` driver assembles the three stages into a
//! [`nse_core::TransmutedModel`] and precomputes the static bias.

#![allow(dead_code)]

pub mod cluster;
pub mod outlier;
pub mod pq;
pub mod quantize;
pub mod transmuter;

pub use transmuter::{
    QuantSchemeConfig, TransmuteConfig, load_transmuted, save_transmuted,
    transmute, transmute_matrix,
};
