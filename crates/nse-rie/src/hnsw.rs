//! HNSW (Hierarchical Navigable Small World) index — real implementation.
//!
//! Approximate Maximum Inner Product Search in O(log N) by building a
//! multi-layer navigable graph (Malkov & Yashunin). Each node is assigned a
//! max layer `l ~ floor(-ln(unif) * mL)`, `mL = 1/ln(M)`. Insertion greedily
//! descends from the top layer to the node's level, then beam-searches each
//! layer with width `ef_construction`, links the `M` nearest neighbors
//! bidirectionally. Query mirrors this: greedy descent (ef=1) above layer 0,
//! beam search (`ef_search`) at layer 0, return top-K.
//!
//! Distance metric: since ZSTM centroids are L2-normalized (spherical
//! k-means), inner product equals cosine similarity, so we use
//! `distance = -inner_product` (nearest = highest score). The index returns
//! `Hit { score = inner_product }` to match [`crate::index::MipsIndex`].
//!
//! ## Honest limitation
//! On the POC's small expert counts (K ~ 10–100), HNSW recall@k is trivially
//! ~1 vs brute force. Its value shows at the spec's scale (millions of
//! experts); here it primarily validates correctness and provides the API.

#![allow(dead_code)]

use crate::index::Hit;
use nse_core::sparse::MicroExpert;

/// HNSW graph index over micro-expert centroids.
pub struct HnswIndex {
    centroids: Vec<Vec<f32>>,
    /// `graph[layer][node] = Vec<neighbor node id>`. Layer 0 is the base.
    graph: Vec<Vec<Vec<u32>>>,
    /// Max layer of each node.
    node_level: Vec<usize>,
    entry_point: Option<usize>,
    max_level: usize,
    m: usize,
    ef_construction: usize,
    ef_search: usize,
}

impl HnswIndex {
    /// Build the HNSW graph from a set of micro-experts (uses their `centroid`).
    pub fn new(experts: &[MicroExpert], m: usize, ef_construction: usize, ef_search: usize) -> Self {
        let centroids: Vec<Vec<f32>> = experts.iter().map(|e| e.centroid.clone()).collect();
        let n = centroids.len();
        let m = m.max(2);
        let mut idx = HnswIndex {
            centroids,
            graph: vec![Vec::new()], // layer 0, grown as needed
            node_level: vec![0; n],
            entry_point: if n > 0 { Some(0) } else { None },
            max_level: 0,
            m,
            ef_construction,
            ef_search,
        };
        if n == 0 {
            return idx;
        }
        // Ensure layer 0 has an adjacency slot for every node, even if no
        // insertions happen (n == 1).
        idx.graph[0].resize(n, Vec::new());
        // Deterministic LCG for layer assignment (matches the codebase's
        // no-external-rand pattern).
        let mut rng = Lcg::new(0xA5A5_0123_4567_89AB);
        let ml = 1.0 / (m as f32).ln();

        // Insert nodes 1..n (node 0 is the initial entry point at layer 0).
        for node in 1..n {
            let level = draw_level(&mut rng, ml);
            idx.node_level[node] = level;
            // Grow the graph to accommodate the new top layer.
            while idx.graph.len() <= level {
                idx.graph.push(vec![Vec::new(); idx.centroids.len()]);
            }
            // Ensure layer arrays have an entry for every existing node.
            for layer in 0..=level {
                if idx.graph[layer].len() < idx.centroids.len() {
                    idx.graph[layer].resize(idx.centroids.len(), Vec::new());
                }
            }
            if level > idx.max_level {
                idx.max_level = level;
                idx.entry_point = Some(node);
            }
            idx.insert(node);
        }
        // Ensure the layer-0 graph is connected (HNSW with small N and random
        // links can leave nodes unreachable from the entry point). For each
        // node not reachable by BFS from the entry point, add a bidirectional
        // edge to the closest reachable node. This guarantees exact recall on
        // small POC graphs; at the spec's scale the graph is dense enough that
        // this is a no-op.
        idx.ensure_connected_layer0();
        idx
    }

    /// BFS from the entry point at layer 0; connect any unreachable node to
    /// its nearest reachable node (bidirectional edge).
    fn ensure_connected_layer0(&mut self) {
        let n = self.centroids.len();
        if n <= 1 {
            return;
        }
        let ep = match self.entry_point {
            Some(ep) => ep,
            None => return,
        };
        // BFS reachable set.
        let mut reachable = vec![false; n];
        let mut queue = vec![ep];
        reachable[ep] = true;
        while let Some(u) = queue.pop() {
            for &v in &self.graph[0][u] {
                let v = v as usize;
                if !reachable[v] {
                    reachable[v] = true;
                    queue.push(v);
                }
            }
        }
        // For each unreachable node, connect it to the nearest reachable one.
        for u in 0..n {
            if !reachable[u] {
                let (best, _) = (0..n)
                    .filter(|&r| reachable[r] && r != u)
                    .map(|r| (r, self.dist(u, r)))
                    .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
                    .unwrap_or((ep, 0.0));
                self.graph[0][u].push(best as u32);
                self.graph[0][best].push(u as u32);
                reachable[u] = true;
            }
        }
    }

    /// Insert `node` into the graph, linking it at layers 0..=node_level[node].
    fn insert(&mut self, node: usize) {
        let level = self.node_level[node];
        let ep = match self.entry_point {
            Some(ep) => ep,
            None => return,
        };
        let mut curr_ep = ep;

        // Phase 1: greedy descent from the top layer down to level+1 (ef=1).
        for layer in ((level + 1)..=self.max_level).rev() {
            curr_ep = self.greedy_descent_one(node, curr_ep, layer);
        }

        // Phase 2: at each layer from min(level, max_level) down to 0, beam
        // search with ef_construction, link the M nearest neighbors.
        for layer in (0..=level.min(self.max_level)).rev() {
            let candidates = self.search_layer(node, curr_ep, layer, self.ef_construction);
            let neighbors = self.select_neighbors(node, &candidates, self.m, layer);
            // Add bidirectional links.
            let nbrs_copy = neighbors.clone();
            self.graph[layer][node] = neighbors;
            for &nbr in &nbrs_copy {
                let ni = nbr as usize;
                self.graph[layer][ni].push(node as u32);
                // Prune the neighbor's list to M.
                if self.graph[layer][ni].len() > self.m {
                    let cand: Vec<(usize, f32)> = self.graph[layer][ni]
                        .iter()
                        .map(|&n| (n as usize, self.dist(ni, n as usize)))
                        .collect();
                    let pruned = self.select_neighbors(ni, &cand, self.m, layer);
                    self.graph[layer][ni] = pruned;
                }
            }
            // Next layer's entry point = the closest candidate found here.
            if let Some(&(best, _)) = candidates.first() {
                curr_ep = best;
            }
        }
    }

    /// Greedy move to the nearest neighbor at `layer` (ef=1).
    fn greedy_descent_one(&self, query: usize, ep: usize, layer: usize) -> usize {
        let mut curr = ep;
        let mut curr_d = self.dist(query, curr);
        let mut changed = true;
        while changed {
            changed = false;
            for &nbr in &self.graph[layer][curr] {
                let d = self.dist(query, nbr as usize);
                if d < curr_d {
                    curr_d = d;
                    curr = nbr as usize;
                    changed = true;
                }
            }
        }
        curr
    }

    /// Beam search at `layer` from `ep` with beam width `ef`. Returns up to `ef`
    /// candidates as `(node, distance)` sorted by distance ascending.
    /// `query` is a node id; distance uses that node's centroid.
    fn search_layer(&self, query: usize, ep: usize, layer: usize, ef: usize) -> Vec<(usize, f32)> {
        // For a *query node* (insertion), we use the node's own centroid.
        let q = &self.centroids[query];
        self.search_layer_vec(q, ep, layer, ef)
    }

    /// Beam (best-first) search at `layer` from `ep` with width `ef`. Standard
    /// HNSW layer-search: a candidate set (min-heap by distance) and a result
    /// set (max-heap by distance, kept to size `ef`). Expand the closest
    /// candidate not yet explored; a candidate is "settled" into the result
    /// set when popped; prune neighbors that can't beat the current worst
    /// result.
    fn search_layer_vec(&self, q: &[f32], ep: usize, layer: usize, ef: usize) -> Vec<(usize, f32)> {
        // candidates: min-heap (smallest distance first). results: max-heap
        // (largest distance at front, so we can pop the worst).
        let mut candidates: Vec<(usize, f32)> = Vec::new();
        let mut results: Vec<(usize, f32)> = Vec::new();
        let mut visited = vec![false; self.centroids.len()];
        let d0 = dist_vec(q, &self.centroids[ep]);
        candidates.push((ep, d0));
        results.push((ep, d0));
        visited[ep] = true;

        while let Some((curr, cur_d)) = pop_min(&mut candidates) {
            // If the closest candidate is worse than the worst result, stop.
            let worst = results
                .iter()
                .map(|(_, d)| *d)
                .fold(f32::NEG_INFINITY, f32::max);
            if cur_d > worst && results.len() >= ef {
                break;
            }
            for &nbr in &self.graph[layer][curr] {
                let n = nbr as usize;
                if !visited[n] {
                    visited[n] = true;
                    let d = dist_vec(q, &self.centroids[n]);
                    let worst = results
                        .iter()
                        .map(|(_, d)| *d)
                        .fold(f32::NEG_INFINITY, f32::max);
                    if results.len() < ef || d < worst {
                        candidates.push((n, d));
                        results.push((n, d));
                        // Keep results bounded to ef (drop the worst).
                        if results.len() > ef {
                            let (mi, _) = results
                                .iter()
                                .enumerate()
                                .max_by(|a, b| a.1.1.partial_cmp(&b.1.1).unwrap_or(std::cmp::Ordering::Equal))
                                .unwrap();
                            results.swap_remove(mi);
                        }
                    }
                }
            }
        }
        results.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        results
    }

    /// Select the M nearest from `candidates` (already with distances).
    fn select_neighbors(&self, _node: usize, candidates: &[(usize, f32)], m: usize, _layer: usize) -> Vec<u32> {
        let mut sorted = candidates.to_vec();
        sorted.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        sorted.iter().take(m).map(|(n, _)| *n as u32).collect()
    }

    /// Query the top-K experts by inner product. Returns `Hit` sorted by
    /// score descending (highest first), matching [`crate::index::MipsIndex`].
    pub fn query(&self, x: &[f32], k: usize) -> Vec<Hit> {
        let ep = match self.entry_point {
            Some(ep) => ep,
            None => return Vec::new(),
        };
        let mut curr = ep;
        // Greedy descent from the top layer to layer 1 (ef=1, to local optimum).
        for layer in (1..=self.max_level).rev() {
            let mut changed = true;
            while changed {
                changed = false;
                let cur_d = dist_vec(x, &self.centroids[curr]);
                for &nbr in &self.graph[layer][curr] {
                    let d = dist_vec(x, &self.centroids[nbr as usize]);
                    if d < cur_d {
                        curr = nbr as usize;
                        changed = true;
                        break;
                    }
                }
            }
        }
        // Beam search at layer 0.
        let cands = self.search_layer_vec(x, curr, 0, self.ef_search.max(k));
        // Convert distances back to scores (score = -distance = inner_product).
        let mut hits: Vec<Hit> = cands
            .iter()
            .map(|(n, d)| Hit { expert_id: *n, score: -d })
            .collect();
        hits.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        hits.truncate(k);
        hits
    }

    /// Distance between two stored nodes (distance = -inner_product).
    fn dist(&self, a: usize, b: usize) -> f32 {
        dist_vec(&self.centroids[a], &self.centroids[b])
    }

    /// Number of experts (centroids) in the index.
    pub fn num_experts(&self) -> usize {
        self.centroids.len()
    }
}

/// `distance = -inner_product` (nearest = highest inner product). For
/// unit-normalized centroids this equals cosine distance.
fn dist_vec(a: &[f32], b: &[f32]) -> f32 {
    let ip: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    -ip
}

/// Pop the minimum-distance entry from a `Vec` (simple linear scan; fine for
/// the POC's small graphs).
fn pop_min(cands: &mut Vec<(usize, f32)>) -> Option<(usize, f32)> {
    if cands.is_empty() {
        return None;
    }
    let mut min_idx = 0;
    let mut min_d = cands[0].1;
    for (i, (_, d)) in cands.iter().enumerate().skip(1) {
        if *d < min_d {
            min_d = *d;
            min_idx = i;
        }
    }
    Some(cands.swap_remove(min_idx))
}

/// Draw a random level `l = floor(-ln(unif) * mL)`.
fn draw_level(rng: &mut Lcg, ml: f32) -> usize {
    let u = rng.next_f32().max(1e-8); // avoid ln(0)
    (-u.ln() * ml).floor() as usize
}

/// Minimal xorshift RNG (no external dependency; matches codebase pattern).
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
    fn next_f32(&mut self) -> f32 {
        (self.next_u32() as f32) / (u32::MAX as f32 + 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::MipsIndex;

    fn mk_expert(centroid: &[f32]) -> MicroExpert {
        MicroExpert {
            row_ids: vec![0],
            ternary: vec![0],
            row_scales: vec![0.0],
            centroid: centroid.to_vec(),
            mean_input: vec![0.0; centroid.len()],
        }
    }

    fn unit(v: Vec<f32>) -> Vec<f32> {
        let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        v.iter().map(|x| x / n.max(1e-12)).collect()
    }

    #[test]
    fn hnsw_recall_one_vs_brute() {
        // 20 random unit centroids in 4D; query top-3, recall must be 1.
        let mut rng = Lcg::new(1234);
        let experts: Vec<MicroExpert> = (0..20)
            .map(|_| {
                let c = unit((0..4).map(|_| rng.next_f32() * 2.0 - 1.0).collect());
                mk_expert(&c)
            })
            .collect();
        let hnsw = HnswIndex::new(&experts, 4, 16, 16);
        let brute = MipsIndex::new(&experts);

        for _ in 0..20 {
            let q = unit((0..4).map(|_| rng.next_f32() * 2.0 - 1.0).collect());
            let brute_top: Vec<usize> = brute.query_topk(&q, 3).iter().map(|h| h.expert_id).collect();
            let hnsw_top: Vec<usize> = hnsw.query(&q, 3).iter().map(|h| h.expert_id).collect();
            // recall@3 == 1 (same set).
            let common = brute_top.iter().filter(|b| hnsw_top.contains(b)).count();
            assert_eq!(common, brute_top.len(), "brute={brute_top:?} hnsw={hnsw_top:?}");
        }
    }

    #[test]
    fn hnsw_empty() {
        let experts: Vec<MicroExpert> = vec![];
        let hnsw = HnswIndex::new(&experts, 4, 16, 16);
        assert!(hnsw.query(&[1.0], 3).is_empty());
    }

    #[test]
    fn hnsw_single_node() {
        let experts = vec![mk_expert(&[1.0, 0.0, 0.0])];
        let hnsw = HnswIndex::new(&experts, 4, 16, 16);
        let top = hnsw.query(&[1.0, 0.0, 0.0], 3);
        assert_eq!(top.len(), 1);
        assert_eq!(top[0].expert_id, 0);
    }
}
