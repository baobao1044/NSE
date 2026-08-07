//! Extreme codebook quantization (ZSTM stage 3).
//!
//! Compresses per-micro-expert weights to sub-1-bit:
//!
//! - **Ternary** `{−1, 0, 1}` (BitNet-style): 4 weights packed per byte.
//! - **Product Quantization (PQ)**: weights become indices into a shared
//!   codebook (< 1 MB, kept on L3 cache).
//!
//! Status: skeleton (M0). Ternary encode/decode lands in M3.

/// Pack layout for ternary weights: 4 weights per byte, 2 bits each.
/// Encoding: `00 -> 0`, `01 -> +1`, `10 -> -1`, `11 -> reserved`.
pub const TRINARY_PER_BYTE: usize = 4;

/// Encode a slice of `{-1,0,1}` values into packed bytes (4 per byte).
/// (Stub — implemented in M3.)
pub fn encode_ternary(_values: &[i8]) -> anyhow::Result<Vec<u8>> {
    todo!("M3: ternary pack")
}

/// Decode packed ternary bytes back into `{-1,0,1}` values.
/// (Stub — implemented in M3.)
pub fn decode_ternary(_packed: &[u8], _count: usize) -> anyhow::Result<Vec<i8>> {
    todo!("M3: ternary unpack")
}
