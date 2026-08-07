//! # nse-ller
//!
//! Low-Level Execution Runtime (online, CPU).
//!
//! Executes the sparse computation with maximal use of CPU hardware:
//!
//! - `tiling` — L3 cache tiling engine: load data in blocks sized to L3 cache.
//! - `kernel` — SIMD compute kernels. A **scalar reference kernel** (ternary
//!   math) is the mathematical ground truth used for PPL. AVX2
//!   `_mm256_*` kernels (with scalar fallback) and a PQ `shuffle_epi8`
//!   lookup kernel are scaffolded for performance work later.
//!
//! Status: skeleton (M0). Scalar reference ternary kernel lands in M4.

#![allow(dead_code)]

pub mod tiling;
pub mod kernel;
