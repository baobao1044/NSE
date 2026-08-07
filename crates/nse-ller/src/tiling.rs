//! L3 cache tiling engine.
//!
//! Loads data in blocks sized to fit the L3 cache (16–64 MB) so cache misses
//! approach 0% over a token's compute cycle. The POC scalar kernel processes
//! whole micro-experts in one pass (they're already cache-sized by ZSTM
//! design), so explicit tiling is a no-op here. The real tiled engine lands in
//! the performance phase (post-M6) and does not change numerical results.

#![allow(dead_code)]

#[derive(Debug, Clone)]
pub struct TilingConfig {
    /// Target tile size in bytes (default 32 MB).
    pub tile_bytes: usize,
}

impl Default for TilingConfig {
    fn default() -> Self {
        Self { tile_bytes: 32 * 1024 * 1024 }
    }
}
