//! Product Quantization (PQ) codebook — real implementation (Phase 7 / M8).
//!
//! PQ compresses each weight row into a short code of sub-vector indices,
//! each pointing into a shared codebook of < 1 MB (per spec, kept on L3
//! cache). At inference, the kernel gathers codebook entries and dots with
//! the input — no per-element multiply, the codebook *is* the quantized
//! weight. Compared to ternary `{-1,0,1}` + per-row scale (3 levels, 1
//! scalar/row), PQ with `nbits=8` gives 256 levels per sub-vector and a
//! learned codebook — far more expressive, at the cost of a codebook
//! gather instead of an add/sub/skip.
//!
//! ## Scheme
//!
//! A row `w` of length `in_dim` splits into `M = num_sub_vectors`
//! contiguous sub-vectors of length `sub_dim = in_dim / M`. Each sub-vector
//! is independently quantized against its own sub-codebook of `2^nbits`
//! centroids (learned by k-means on the training rows' sub-vectors). The
//! encoded row is `M` bytes (one centroid index per sub-vector); decoding
//! concatenates the `M` centroid vectors. A per-row scalar `scale`
//! (BitNet-style, `mean(|w|)`) bounds the magnitude so decode is
//! `scale * concat(centroids)`.
//!
//! The codebook is shared across all experts of a layer (`SparseLayer::
//! pq_codebook`), matching the spec's "shared codebook < 1 MB on L3".

use nse_core::sparse::PqCodebook;

/// Train a PQ codebook from a set of weight vectors (the residual rows of
/// a transmuted matrix) using per-sub-vector k-means.
///
/// - `weights`: the rows to quantize, each of length `in_dim`.
/// - `num_sub_vectors`: `M`; `in_dim` must be divisible by `M`.
/// - `nbits`: bits per code (8 → 256 centroids per sub-codebook).
/// - `iters`: k-means iterations per sub-vector.
/// - `seed`: deterministic centroid init.
///
/// Returns a [`PqCodebook`] with `codebook` laid out as
/// `[subvec m][centroid c][dim j]` (see [`PqCodebook::codebook`]).
pub fn train_pq(
    weights: &[Vec<f32>],
    num_sub_vectors: usize,
    nbits: usize,
    iters: usize,
    seed: u64,
) -> PqCodebook {
    let in_dim = weights.first().map(|r| r.len()).unwrap_or(0);
    let m = num_sub_vectors.max(1);
    debug_assert!(
        in_dim % m == 0,
        "in_dim {in_dim} must be divisible by num_sub_vectors {m}"
    );
    let sub_dim = if m > 0 { in_dim / m } else { 0 };
    let n_entries = (1usize << nbits).max(1);

    // Cap the number of centroids to the number of distinct sub-vectors
    // available (k-means with k > samples degenerates to empty clusters).
    let n = weights.len();
    let k = n_entries.min(n.max(1));

    let mut codebook = vec![0.0f32; m * n_entries * sub_dim];

    for sm in 0..m {
        // Collect sub-vector `sm` from every row.
        let mut subvectors: Vec<Vec<f32>> = Vec::with_capacity(n);
        for row in weights {
            let s = &row[sm * sub_dim..(sm + 1) * sub_dim];
            subvectors.push(s.to_vec());
        }
        let centroids = kmeans_l2(&subvectors, k, sub_dim, iters, seed.wrapping_add(sm as u64));
        // Write into the global codebook slot [sm][c][j].
        for (c, cent) in centroids.iter().enumerate() {
            let off = sm * n_entries * sub_dim + c * sub_dim;
            codebook[off..off + sub_dim].copy_from_slice(cent);
        }
        // Zero-fill unused centroids (k < n_entries) so decode is well-defined.
        for c in k..n_entries {
            let off = sm * n_entries * sub_dim + c * sub_dim;
            for j in 0..sub_dim {
                codebook[off + j] = 0.0;
            }
        }
    }

    PqCodebook {
        num_sub_vectors: m,
        nbits,
        sub_dim,
        codebook,
    }
}

/// Encode a weight row into PQ codes: one byte per sub-vector (nbits <= 8),
/// the index of the nearest centroid in that sub-vector's sub-codebook.
///
/// `row.len()` must equal `codebook.num_sub_vectors * codebook.sub_dim`.
pub fn encode_pq(row: &[f32], codebook: &PqCodebook) -> Vec<u8> {
    let m = codebook.num_sub_vectors;
    let sub_dim = codebook.sub_dim;
    let n_entries = codebook.num_entries();
    let mut codes = vec![0u8; m];
    for sm in 0..m {
        let sub = &row[sm * sub_dim..(sm + 1) * sub_dim];
        let base = sm * n_entries * sub_dim;
        let mut best = 0usize;
        let mut best_d = f32::INFINITY;
        for c in 0..n_entries {
            let cent = &codebook.codebook[base + c * sub_dim..base + (c + 1) * sub_dim];
            let d = squared_dist(sub, cent);
            if d < best_d {
                best_d = d;
                best = c;
            }
        }
        codes[sm] = best as u8;
    }
    codes
}

/// Decode PQ codes back to an approximate weight vector: concatenate the
/// `M` indexed centroid sub-vectors. (No per-row scale here — the caller
/// applies `row_scales[i]` if the expert stores one.)
pub fn decode_pq(codes: &[u8], codebook: &PqCodebook) -> Vec<f32> {
    let m = codebook.num_sub_vectors;
    let sub_dim = codebook.sub_dim;
    let n_entries = codebook.num_entries();
    let mut out = vec![0.0f32; m * sub_dim];
    for sm in 0..m {
        let c = (codes[sm] as usize).min(n_entries - 1);
        let base = sm * n_entries * sub_dim + c * sub_dim;
        out[sm * sub_dim..(sm + 1) * sub_dim]
            .copy_from_slice(&codebook.codebook[base..base + sub_dim]);
    }
    out
}

/// Reconstruct a single row from its codes + per-row scale
/// (`scale * decode(codes)`). Convenience for tests / `reconstruct_dense`.
pub fn reconstruct_pq_row(codes: &[u8], codebook: &PqCodebook, scale: f32) -> Vec<f32> {
    decode_pq(codes, codebook).iter().map(|v| scale * v).collect()
}

/// Squared L2 distance between two equal-length slices.
fn squared_dist(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum()
}

/// Plain (Euclidean, non-spherical) k-means. Weight values are centered
/// near 0 (Gaussian-ish), so L2 — not cosine — is the right metric (unlike
/// the spherical k-means used for routing centroids in `cluster.rs`).
fn kmeans_l2(
    points: &[Vec<f32>],
    k: usize,
    dim: usize,
    iters: usize,
    seed: u64,
) -> Vec<Vec<f32>> {
    let n = points.len();
    if n == 0 {
        return vec![vec![0.0; dim]; k];
    }
    let k = k.min(n).max(1);
    // Init: pick K distinct points by seeded round-robin (matches the
    // codebase's no-external-rand pattern; `cluster.rs` uses the same Lcg).
    let mut rng = Lcg::new(seed);
    let mut order: Vec<usize> = (0..n).collect();
    for i in (1..n).rev() {
        let j = (rng.next_u32() as usize) % (i + 1);
        order.swap(i, j);
    }
    let mut centroids: Vec<Vec<f32>> = order.iter().take(k).map(|&i| points[i].clone()).collect();
    let mut assignment = vec![0usize; n];

    for _ in 0..iters {
        let mut changed = false;
        // Assign.
        for (i, p) in points.iter().enumerate() {
            let best = nearest_l2(p, &centroids);
            if assignment[i] != best {
                assignment[i] = best;
                changed = true;
            }
        }
        // Update centroids = mean of members.
        let mut new_c = vec![vec![0.0f32; dim]; k];
        let mut counts = vec![0usize; k];
        for (i, p) in points.iter().enumerate() {
            let c = assignment[i];
            for j in 0..dim {
                new_c[c][j] += p[j];
            }
            counts[c] += 1;
        }
        for c in 0..k {
            if counts[c] == 0 {
                // Keep old centroid to avoid empty clusters.
                new_c[c] = centroids[c].clone();
            } else {
                for j in 0..dim {
                    new_c[c][j] /= counts[c] as f32;
                }
            }
        }
        centroids = new_c;
        if !changed {
            break;
        }
    }
    centroids
}

/// Index of the nearest centroid by squared L2 distance.
fn nearest_l2(p: &[f32], centroids: &[Vec<f32>]) -> usize {
    let mut best = 0usize;
    let mut best_d = f32::INFINITY;
    for (i, c) in centroids.iter().enumerate() {
        let d = squared_dist(p, c);
        if d < best_d {
            best_d = d;
            best = i;
        }
    }
    best
}

/// Minimal xorshift RNG (matches the codebase's `cluster.rs` Lcg pattern;
/// kept local to avoid pulling a dependency).
struct Lcg {
    state: u64,
}

impl Lcg {
    fn new(seed: u64) -> Self {
        Self { state: if seed == 0 { 0x9e3779b97f4a7c15 } else { seed } }
    }
    fn next_u32(&mut self) -> u32 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        (x & 0xffff_ffff) as u32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn near(a: &[f32], b: &[f32], tol: f32) -> bool {
        a.len() == b.len()
            && a.iter().zip(b.iter()).all(|(x, y)| (x - y).abs() < tol)
    }

    /// Two well-separated sub-vector clusters per sub-vector → k-means finds
    /// them, and encode/decode reconstructs to the nearest centroid.
    #[test]
    fn train_pq_clusters_subvectors() {
        // in_dim=4, M=2 sub-vectors of dim 2. Rows come in two clusters:
        // cluster A near [0,0 | 0,0], cluster B near [10,10 | -10,-10].
        let rows = vec![
            vec![0.1, 0.0, 0.0, -0.1],
            vec![-0.1, 0.0, 0.0, 0.1],
            vec![10.0, 10.0, -10.0, -10.0],
            vec![10.1, 9.9, -9.9, -10.1],
        ];
        let cb = train_pq(&rows, 2, 1, 20, 7); // nbits=1 → 2 centroids
        assert_eq!(cb.num_sub_vectors, 2);
        assert_eq!(cb.sub_dim, 2);
        assert_eq!(cb.num_entries(), 2);
        // Encode a row in cluster A → both sub-vector codes pick the A centroid.
        let codes_a = encode_pq(&rows[0], &cb);
        let codes_b = encode_pq(&rows[2], &cb);
        assert_ne!(codes_a[0], codes_b[0]);
        assert_ne!(codes_a[1], codes_b[1]);
        // Decode of cluster A's codes lands near [0,0,0,0].
        let recon_a = decode_pq(&codes_a, &cb);
        assert!(near(&recon_a, &[0.0, 0.0, 0.0, 0.0], 0.2));
    }

    /// Encode → decode roundtrip for a single row gives the nearest-centroid
    /// reconstruction (MSE within the cluster radius).
    #[test]
    fn encode_decode_roundtrip() {
        let rows: Vec<Vec<f32>> = (0..40)
            .map(|i| {
                vec![
                    (i as f32) * 0.1,
                    (i as f32) * 0.1 + 0.05,
                    -(i as f32) * 0.05,
                    (i as f32) * 0.07,
                ]
            })
            .collect();
        let cb = train_pq(&rows, 2, 4, 10, 3); // 16 centroids per sub-vector
        for r in &rows {
            let codes = encode_pq(r, &cb);
            let recon = decode_pq(&codes, &cb);
            // Per-sub-vector reconstruction is within the k-means quantization
            // radius; with 16 centroids over 40 smooth points this is tight.
            for sm in 0..2 {
                let d = squared_dist(&r[sm * 2..(sm + 1) * 2], &recon[sm * 2..(sm + 1) * 2]);
                assert!(d < 0.5, "subvec {sm} dist {d:.4} too large");
            }
        }
    }

    /// Headline: PQ (8-bit, 4 sub-vectors) has lower reconstruction MSE than
    /// ternary on random Gaussian weights. This is the accuracy argument
    /// for PQ over ternary — the Phase 7 motivation.
    #[test]
    fn pq_lower_error_than_ternary() {
        use crate::quantize::quantize_row;
        // 256 rows of dim 32, Gaussian-ish (sum of uniforms → ~Gaussian).
        let mut rng = Lcg::new(11);
        let mut rows: Vec<Vec<f32>> = Vec::with_capacity(256);
        for _ in 0..256 {
            let row: Vec<f32> = (0..32)
                .map(|_| {
                    let mut s = 0.0;
                    for _ in 0..4 {
                        s += (rng.next_u32() as f32 / u32::MAX as f32) * 2.0 - 1.0;
                    }
                    s
                })
                .collect();
            rows.push(row);
        }
        // PQ: M=4 sub-vectors of dim 8, 8-bit codebook (256 centroids each).
        let cb = train_pq(&rows, 4, 8, 20, 5);
        let mut mse_pq = 0.0f32;
        let mut mse_tern = 0.0f32;
        for r in &rows {
            // PQ reconstruction (no per-row scale here; codebook learned the
            // magnitudes directly).
            let codes = encode_pq(r, &cb);
            let recon_pq = decode_pq(&codes, &cb);
            for (a, b) in r.iter().zip(recon_pq.iter()) {
                mse_pq += (a - b) * (a - b);
            }
            // Ternary reconstruction (scale * ternary).
            let (tern, scale) = quantize_row(r);
            for (a, &c) in r.iter().zip(tern.iter()) {
                let recon = scale * c as f32;
                mse_tern += (a - recon) * (a - recon);
            }
        }
        mse_pq /= (256 * 32) as f32;
        mse_tern /= (256 * 32) as f32;
        eprintln!("[pq_lower_error_than_ternary] PQ MSE={mse_pq:.6} ternary MSE={mse_tern:.6}");
        assert!(
            mse_pq < mse_tern,
            "PQ MSE {mse_pq:.6} should be lower than ternary MSE {mse_tern:.6}"
        );
    }
}
