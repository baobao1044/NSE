//! Safetensors loader for dense model weights.
//!
//! Status: skeleton (M0). Real load/save land in M1.

use std::path::Path;

use anyhow::Result;

/// Save Toy LM weights to a `.safetensors` file. (Stub — implemented in M1.)
pub fn save_weights(_path: impl AsRef<Path>) -> Result<()> {
    todo!("M1: save_weights")
}

/// Load Toy LM weights from a `.safetensors` file. (Stub — implemented in M1.)
pub fn load_weights(_path: impl AsRef<Path>) -> Result<()> {
    todo!("M1: load_weights")
}
