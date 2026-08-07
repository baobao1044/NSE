//! # nse-zstm
//!
//! Zero-Shot Transmutation Module (offline).
//!
//! Converts a dense model's weights into the NSE sparse representation without
//! retraining, in three stages matching the NSE spec:
//!
//! 1. `outlier` — extract high-amplitude outlier channels into a fixed dense
//!    core (kept on L1 at inference time).
//! 2. `cluster` — spherical k-means / SVD partitioning of the remaining
//!    columns into micro-experts (cache-sized).
//! 3. `quantize` — sub-1-bit compression: ternary `{−1, 0, 1}` (4 weights /
//!    byte) and Product Quantization (PQ) into a shared codebook.
//!
//! Status: skeleton (M0). Outlier + k-means + ternary land in M3.

#![allow(dead_code)]

pub mod outlier;
pub mod cluster;
pub mod quantize;
pub mod transmuter;
