//! Error types for the NSE workspace.

use thiserror::Error;

/// All errors produced by NSE crates funnel into [`NseError`].
pub type NseResult<T> = Result<T, NseError>;

/// Top-level error enum. Crates may wrap their own errors via the
/// `Other` / `Io` / custom variants, but the common categories live here.
#[derive(Debug, Error)]
pub enum NseError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    #[error("invalid NSE file: {0}")]
    InvalidFile(String),

    #[error("invalid magic: expected {expected:?}, got {got:?}")]
    BadMagic { expected: [u8; 4], got: [u8; 4] },

    #[error("unsupported NSE version: {0}")]
    UnsupportedVersion(u32),

    #[error("shape mismatch: expected {expected:?}, got {got:?}")]
    ShapeMismatch {
        expected: Vec<usize>,
        got: Vec<usize>,
    },

    #[error("index out of bounds: {index} (len {len})")]
    IndexOutOfBounds { index: usize, len: usize },

    #[error("tensor error: {0}")]
    Tensor(String),

    #[error("model error: {0}")]
    Model(String),

    #[error("transmutation error: {0}")]
    Transmutation(String),

    #[error("config error: {0}")]
    Config(String),

    #[error("{0}")]
    Other(String),
}

impl From<serde_json::Error> for NseError {
    fn from(e: serde_json::Error) -> Self {
        NseError::Other(format!("serde_json: {e}"))
    }
}
