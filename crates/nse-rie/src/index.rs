//! Maximum Inner Product Search (MIPS) index.
//!
//! The POC uses exact brute-force MIPS (correct, O(N)). For each query, it
//! computes `score_k = x . centroid_k` for every micro-expert centroid and
//! returns them sorted descending. An HNSW/LSH index with O(log N) lookup is
//! scaffolded for the scale-out phase — it must return the same top-K as
//! brute force (up to ties).

use nse_core::sparse::MicroExpert;

/// A query hit: expert id and its inner-product score.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Hit {
    pub expert_id: usize,
    pub score: f32,
}

/// Brute-force exact MIPS index over the centroids of a layer's micro-experts.
pub struct MipsIndex<'a> {
    pub experts: &'a [MicroExpert],
}

impl<'a> MipsIndex<'a> {
    pub fn new(experts: &'a [MicroExpert]) -> Self {
        Self { experts }
    }

    /// Query all experts, returning every hit sorted by score descending.
    pub fn query_all(&self, x: &[f32]) -> Vec<Hit> {
        let mut hits: Vec<Hit> = self
            .experts
            .iter()
            .enumerate()
            .map(|(i, e)| Hit {
                expert_id: i,
                score: dot(x, &e.centroid),
            })
            .collect();
        hits.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        hits
    }

    /// Query the top-K experts by inner product.
    pub fn query_topk(&self, x: &[f32], k: usize) -> Vec<Hit> {
        let mut all = self.query_all(x);
        all.truncate(k);
        all
    }
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mk_expert(centroid: &[f32]) -> MicroExpert {
        MicroExpert {
            row_ids: vec![0],
            ternary: vec![0],
            row_scales: vec![0.0],
            centroid: centroid.to_vec(),
            mean_input: vec![0.0; centroid.len()],
            pq: None,
        }
    }

    #[test]
    fn topk_sorted_desc() {
        let experts = vec![
            mk_expert(&[1.0, 0.0]), // dot with [1,1] = 1
            mk_expert(&[0.0, 1.0]), // dot = 1
            mk_expert(&[1.0, 1.0]), // dot = 2  -> highest
        ];
        let idx = MipsIndex::new(&experts);
        let top = idx.query_topk(&[1.0, 1.0], 2);
        assert_eq!(top[0].expert_id, 2);
        assert!((top[0].score - 2.0).abs() < 1e-6);
        assert_eq!(top.len(), 2);
    }

    #[test]
    fn empty_query() {
        let experts: Vec<MicroExpert> = vec![];
        let idx = MipsIndex::new(&experts);
        assert!(idx.query_all(&[1.0]).is_empty());
    }
}
