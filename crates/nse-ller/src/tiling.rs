//! L3 cache tiling engine.
//!
//! Loads data in blocks sized to fit the L3 cache (16–64 MB) so cache misses
//! approach 0% over a token's compute cycle.
//!
//! Status: skeleton (M0). Real tiling lands post-M6 (performance work; does
//! not change numerical results).

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
