//! Extreme codebook quantization (ZSTM stage 3).
//!
//! Compresses each weight row to ternary `{−1, 0, 1}` plus a per-row scale,
//! so the reconstructed weight is `~w[j] = scale * ternary[j]`. This is the
//! BitNet-style scheme: a row's scale is its mean absolute magnitude, and a
//! weight is rounded to `sign(w)` when `|w| > 0.5 * scale`, else `0`.
//!
//! The packed 4-weights-per-byte layout (2 bits each) is provided for the
//! on-disk `.nse` format; the in-memory [`SparseLayer`] keeps ternary codes as
//! `i8` for simplicity and correctness during the POC.

/// Pack 4 ternary values into one byte: `00 -> 0`, `01 -> +1`, `10 -> -1`.
/// Values outside `{-1,0,1}` are clamped.
pub fn encode_ternary(values: &[i8]) -> Vec<u8> {
    let n = values.len();
    let mut out = Vec::with_capacity((n + 3) / 4);
    for chunk in values.chunks(4) {
        let mut byte = 0u8;
        for (shift, &v) in chunk.iter().enumerate() {
            let code: u8 = match v {
                1 => 0b01,
                -1 => 0b10,
                _ => 0b00,
            };
            byte |= code << (shift * 2);
        }
        out.push(byte);
    }
    out
}

/// Unpack packed ternary bytes back into `{-1,0,1}` values. `count` is the
/// number of values to decode (may exceed the byte boundary).
pub fn decode_ternary(packed: &[u8], count: usize) -> Vec<i8> {
    let mut out = Vec::with_capacity(count);
    for i in 0..count {
        let byte = packed[i / 4];
        let code = (byte >> ((i % 4) * 2)) & 0b11;
        let v = match code {
            0b01 => 1,
            0b10 => -1,
            _ => 0,
        };
        out.push(v);
    }
    out
}

/// Ternary encoding of one weight row. Returns `(ternary codes, scale)`.
///
/// `scale = mean(|w|)`, and `ternary[j] = sign(w[j])` if `|w[j]| > 0.5*scale`
/// else `0`. Reconstruction `~w[j] = scale * ternary[j]`.
pub fn quantize_row(row: &[f32]) -> (Vec<i8>, f32) {
    let scale = row.iter().map(|v| v.abs()).sum::<f32>() / row.len().max(1) as f32;
    let thresh = 0.5 * scale;
    let ternary: Vec<i8> = row
        .iter()
        .map(|&v| {
            if v.abs() > thresh {
                if v >= 0.0 { 1 } else { -1 }
            } else {
                0
            }
        })
        .collect();
    (ternary, scale)
}

/// Ternary-quantize a whole matrix row by row. Returns `(ternary [rows*in],
/// scales [rows])`.
pub fn quantize_matrix(m: &nse_core::tensor::Matrix) -> (Vec<i8>, Vec<f32>) {
    let in_dim = m.cols;
    let mut ternary = Vec::with_capacity(m.data.len());
    let mut scales = Vec::with_capacity(m.rows);
    for r in 0..m.rows {
        let row = &m.data[r * in_dim..(r + 1) * in_dim];
        let (codes, scale) = quantize_row(row);
        ternary.extend_from_slice(&codes);
        scales.push(scale);
    }
    (ternary, scales)
}

/// Reconstruct a row from ternary codes + scale (for verification).
pub fn reconstruct_row(ternary: &[i8], scale: f32) -> Vec<f32> {
    ternary.iter().map(|&t| scale * t as f32).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use nse_core::tensor::Matrix;

    #[test]
    fn pack_unpack_roundtrip() {
        let v = vec![1i8, 0, -1, 1, -1, 0, 0, 1, -1];
        let packed = encode_ternary(&v);
        let back = decode_ternary(&packed, v.len());
        assert_eq!(back, v);
    }

    #[test]
    fn quantize_preserves_sign_and_mean_mag() {
        let row = vec![0.6, -0.7, 0.01, 0.55, -0.5, 0.2];
        let (codes, scale) = quantize_row(&row);
        let recon = reconstruct_row(&codes, scale);
        // Sign matches for the large-magnitude entries.
        assert!(codes[0] == 1 && codes[1] == -1);
        // Small entry should be zero.
        assert_eq!(codes[2], 0);
        // Mean abs is roughly preserved.
        let recon_mean = recon.iter().map(|v| v.abs()).sum::<f32>() / recon.len() as f32;
        let orig_mean = row.iter().map(|v| v.abs()).sum::<f32>() / row.len() as f32;
        assert!((recon_mean - orig_mean).abs() < 0.15);
    }

    #[test]
    fn matrix_quantize_shapes() {
        let m = Matrix::zeros(3, 4);
        let (codes, scales) = quantize_matrix(&m);
        assert_eq!(codes.len(), 12);
        assert_eq!(scales.len(), 3);
    }
}
