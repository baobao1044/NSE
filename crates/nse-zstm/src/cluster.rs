//! Micro-expert clustering (ZSTM stage 2).
//!
//! Partitions the residual output rows of a weight matrix into `K`
//! micro-experts via spherical k-means: rows are L2-normalized and assigned
//! to the nearest centroid by cosine similarity; centroids are recomputed as
//! the normalized mean of their members. The centroid (in input space) is
//! what the RIE router matches the input activation against.
//!
//! Clustering only groups rows; it does not change weight values, so the
//! transform up to this stage is exact (a permutation of output rows).

use nse_core::tensor::Matrix;

#[derive(Debug, Clone)]
pub struct ClusterConfig {
    /// Number of micro-experts (`K`). If 0, derived as `sqrt(num_rows)`.
    pub num_experts: usize,
    /// K-means iterations.
    pub iters: usize,
    /// Random seed for centroid initialization.
    pub seed: u64,
}

impl Default for ClusterConfig {
    fn default() -> Self {
        Self { num_experts: 0, iters: 10, seed: 42 }
    }
}

#[derive(Debug, Clone)]
pub struct ClusterResult {
    /// `assignment[r] = expert id` for each residual row (in input order).
    pub assignment: Vec<u32>,
    /// Centroid per expert, shape `[K, in]` (normalized, unit length).
    pub centroids: Matrix,
    /// `members[k] = list of residual-row indices` in expert k.
    pub members: Vec<Vec<usize>>,
}

/// Cluster the residual rows of `weights [n, in]` into `K` micro-experts.
pub fn cluster(weights: &Matrix, cfg: &ClusterConfig) -> anyhow::Result<ClusterResult> {
    let n = weights.rows;
    let in_dim = weights.cols;
    let k = if cfg.num_experts > 0 {
        cfg.num_experts
    } else {
        (n as f32).sqrt().round().max(1.0) as usize
    };
    let k = k.clamp(1, n.max(1));

    if n == 0 {
        return Ok(ClusterResult {
            assignment: vec![],
            centroids: Matrix::zeros(k, in_dim),
            members: vec![Vec::new(); k],
        });
    }

    // Normalize rows to unit length (spherical). Keep zero-norm rows as-is.
    let mut unit = vec![0.0f32; n * in_dim];
    let mut norms = vec![0.0f32; n];
    for r in 0..n {
        let row = &weights.data[r * in_dim..(r + 1) * in_dim];
        let nr = row.iter().map(|v| v * v).sum::<f32>().sqrt();
        norms[r] = nr;
        let inv = if nr > 1e-12 { 1.0 / nr } else { 0.0 };
        for j in 0..in_dim {
            unit[r * in_dim + j] = row[j] * inv;
        }
    }

    // Init centroids by picking K distinct rows (round-robin by seed).
    let mut rng = Lcg::new(cfg.seed);
    let mut first: Vec<usize> = (0..n).collect();
    // Shuffle via Fisher-Yates with the LCG.
    for i in (1..n).rev() {
        let j = (rng.next_u32() as usize) % (i + 1);
        first.swap(i, j);
    }
    let init_ids: Vec<usize> = first.iter().take(k).copied().collect();

    let mut centroids = vec![0.0f32; k * in_dim];
    for (ki, &rid) in init_ids.iter().enumerate() {
        centroids[ki * in_dim..(ki + 1) * in_dim]
            .copy_from_slice(&unit[rid * in_dim..(rid + 1) * in_dim]);
    }
    // Renormalize centroids (handle duplicates).
    normalize_rows(&mut centroids, k, in_dim);

    let mut assignment = vec![0u32; n];
    for _ in 0..cfg.iters {
        // Assign each row to nearest centroid (cosine = dot of unit vectors).
        let mut changed = false;
        for r in 0..n {
            let best = nearest_centroid(&unit[r * in_dim..(r + 1) * in_dim], &centroids, k, in_dim);
            if assignment[r] as usize != best {
                assignment[r] = best as u32;
                changed = true;
            }
        }
        // Recompute centroids = normalized mean of members.
        let mut new_c = vec![0.0f32; k * in_dim];
        let mut counts = vec![0usize; k];
        for r in 0..n {
            let c = assignment[r] as usize;
            for j in 0..in_dim {
                new_c[c * in_dim + j] += unit[r * in_dim + j];
            }
            counts[c] += 1;
        }
        for ki in 0..k {
            if counts[ki] == 0 {
                // Keep old centroid to avoid empty clusters.
                new_c[ki * in_dim..(ki + 1) * in_dim]
                    .copy_from_slice(&centroids[ki * in_dim..(ki + 1) * in_dim]);
            }
        }
        normalize_rows(&mut new_c, k, in_dim);
        centroids = new_c;
        if !changed {
            break;
        }
    }

    let mut members: Vec<Vec<usize>> = vec![Vec::new(); k];
    for (r, &c) in assignment.iter().enumerate() {
        members[c as usize].push(r);
    }

    Ok(ClusterResult {
        assignment,
        centroids: Matrix { rows: k, cols: in_dim, data: centroids },
        members,
    })
}

fn nearest_centroid(row: &[f32], centroids: &[f32], k: usize, in_dim: usize) -> usize {
    let mut best = 0usize;
    let mut best_sim = f32::NEG_INFINITY;
    for ki in 0..k {
        let c = &centroids[ki * in_dim..(ki + 1) * in_dim];
        let sim: f32 = row.iter().zip(c.iter()).map(|(a, b)| a * b).sum();
        if sim > best_sim {
            best_sim = sim;
            best = ki;
        }
    }
    best
}

fn normalize_rows(data: &mut [f32], rows: usize, cols: usize) {
    for r in 0..rows {
        let row = &data[r * cols..(r + 1) * cols];
        let nr = row.iter().map(|v| v * v).sum::<f32>().sqrt();
        let inv = if nr > 1e-12 { 1.0 / nr } else { 0.0 };
        for j in 0..cols {
            data[r * cols + j] *= inv;
        }
    }
}

/// Minimal xorshift RNG (kept local; no external dep).
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

    #[test]
    fn clusters_two_groups() {
        // Two well-separated row directions.
        let mut w = Matrix::zeros(4, 2);
        w.data[0..2].copy_from_slice(&[1.0, 0.0]);
        w.data[2..4].copy_from_slice(&[0.9, 0.1]);
        w.data[4..6].copy_from_slice(&[-1.0, 0.0]);
        w.data[6..8].copy_from_slice(&[-0.9, -0.1]);
        let r = cluster(&w, &ClusterConfig { num_experts: 2, iters: 20, seed: 1 }).unwrap();
        // Rows 0,1 should share a cluster; 2,3 the other.
        assert_eq!(r.assignment[0], r.assignment[1]);
        assert_eq!(r.assignment[2], r.assignment[3]);
        assert_ne!(r.assignment[0], r.assignment[2]);
        assert_eq!(r.centroids.rows, 2);
    }

    #[test]
    fn empty_input() {
        let w = Matrix::zeros(0, 3);
        let r = cluster(&w, &ClusterConfig { num_experts: 2, iters: 5, seed: 1 }).unwrap();
        assert!(r.assignment.is_empty());
    }
}
