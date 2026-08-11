//! HNSW vs brute-force MIPS query-latency benchmark.
//!
//! Builds an index over many unit-norm synthetic centroids and measures
//! wall-clock query latency at three scales, reporting throughput
//! (queries/s), recall@10 vs brute-force, and the HNSW speedup. Run with:
//!
//! ```bash
//! cargo run --release --example bench_hnsw -p nse-rie
//! ```

use std::time::Instant;

use nse_rie::{HnswIndex, MipsIndex, Hit};

fn main() {
    // Use rand to build unit-norm centroids (no need for high quality — just
    // a realistic spread for timing).
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};
    let mut rng = StdRng::seed_from_u64(7);

    let dim = 64usize;
    for &n in &[1_000usize, 5_000, 20_000] {
        // Unit-norm centroids.
        let centroids: Vec<Vec<f32>> = (0..n)
            .map(|_| {
                let v: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>() * 2.0 - 1.0).collect();
                let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);
                v.iter().map(|x| x / norm).collect()
            })
            .collect();

        // Wrap centroids in MicroExpert shells (HNSW/MIPS take &[MicroExpert]).
        // The index only reads `centroid`, so fill the rest minimally.
        let experts: Vec<nse_core::sparse::MicroExpert> = centroids
            .iter()
            .map(|c| nse_core::sparse::MicroExpert {
                row_ids: vec![0],
                ternary: vec![],
                row_scales: vec![1.0],
                centroid: c.clone(),
                mean_input: vec![0.0; dim],
                pq: None,
            })
            .collect();

        // 100 random query vectors (unit-norm).
        let queries: Vec<Vec<f32>> = (0..100)
            .map(|_| {
                let v: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>() * 2.0 - 1.0).collect();
                let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);
                v.iter().map(|x| x / norm).collect()
            })
            .collect();

        println!("\n=== n={n} experts, dim={dim}, 100 queries ===");

        // Brute force.
        let brute = MipsIndex::new(&experts);
        let t0 = Instant::now();
        let brute_results: Vec<Vec<Hit>> = queries.iter().map(|q| brute.query_all(q)).collect();
        let brute_us = t0.elapsed().as_secs_f64() * 1e6 / queries.len() as f64;
        let brute_qps = queries.len() as f64 / t0.elapsed().as_secs_f64();

        // HNSW.
        let hnsw = HnswIndex::new(&experts, 8, 64, 64.max(n.min(256)));
        // warm-up (first query builds entry-point path)
        let _ = hnsw.query(&queries[0], 10);
        let t0 = Instant::now();
        let k = 10usize;
        let hnsw_results: Vec<Vec<Hit>> = queries.iter().map(|q| hnsw.query(q, k)).collect();
        let hnsw_us = t0.elapsed().as_secs_f64() * 1e6 / queries.len() as f64;
        let hnsw_qps = queries.len() as f64 / t0.elapsed().as_secs_f64();
        let speedup = brute_us / hnsw_us;

        // Recall@10: fraction of HNSW's top-10 that are in brute's top-10.
        let mut recall_sum = 0.0;
        for (h, b) in hnsw_results.iter().zip(brute_results.iter()) {
            let bset: std::collections::HashSet<usize> =
                b.iter().take(k).map(|h| h.expert_id).collect();
            let hits = h.iter().filter(|hh| bset.contains(&hh.expert_id)).count();
            recall_sum += hits as f64 / k as f64;
        }
        let recall = recall_sum / queries.len() as f64;

        println!(
            "brute: {brute_us:.1} us/query  ({brute_qps:.0} q/s)\n\
             HNSW:  {hnsw_us:.1} us/query  ({hnsw_qps:.0} q/s)  speedup {speedup:.2}x  recall@{k} {recall:.3}"
        );
    }
}
