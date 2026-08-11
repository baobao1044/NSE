//! # nse-ller
//!
//! Low-Level Execution Runtime (online, CPU).
//!
//! Executes the sparse computation:
//!
//! - `kernel` — compute kernels. A **scalar reference kernel** (ternary
//!   add/sub/skip, dense core mat-vec, bias) is the mathematical ground truth
//!   used for PPL. AVX2 kernels are scaffolded for later performance work and
//!   must match the scalar results bit-for-bit.
//! - `tiling` — L3 cache tiling (no-op in the POC; experts are already
//!   cache-sized by ZSTM).
//! - `avx2` — real AVX2 kernels with runtime auto-detection (`KernelKind`).

#![allow(dead_code)]

pub mod avx2;
pub mod kernel;
pub mod tiling;

pub use avx2::{
    compute_dense_core_dispatch, compute_pq_micro_expert_dispatch,
    compute_ternary_micro_expert_dispatch, KernelKind,
};
pub use kernel::{
    apply_bias, apply_bias_pruned_only, compute_dense_core,
    compute_pq_micro_expert_scalar, compute_ternary_micro_expert_scalar,
};
