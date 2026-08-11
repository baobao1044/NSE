//! # nse-core
//!
//! Core types, traits, error types, and the `.nse` binary format for the
//! Neuro-Sparse Engine (NSE).
//!
//! Everything else in the workspace builds on top of this crate. The key
//! pieces defined here are:
//!
//! - [`NSEFileHeader`] / [`MicroExpertMeta`]: the on-disk layout of an `.nse`
//!   model file, matching the NSE technical specification (magic `"NSE1"`).
//! - [`format`]: read/write helpers for `.nse` files, `mmap`-friendly.
//! - [`tensor`]: a small dense matrix view used across crates.
//!
//! The file format layout is: header -> dense core -> codebook ->
//! micro-expert data -> MIPS tree. See [`format`] for details.

//! Unsafe usage is limited to `mmap` reads in [`format`] (via `memmap2`); all
//! public APIs stay safe.

pub mod error;
pub mod format;
pub mod sparse;
pub mod tensor;

pub use error::{NseError, NseResult};
pub use format::{MicroExpertMeta, NSEFileHeader, NSE_MAGIC, NSE_VERSION};
pub use sparse::{ConfigStub, MicroExpert, PqCodebook, PqExpertData, SparseLayer, TransmutedModel};
