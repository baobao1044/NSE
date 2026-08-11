//! PQ micro-expert kernel benchmark: PQ scalar vs PQ AVX2 throughput, plus
//! PQ vs ternary on both speed and reconstruction MSE.
//!
//! The headline accuracy metric is the reconstruction MSE: PQ's 256-level
//! per-sub-vector codebook should reconstruct weights far more faithfully
//! than ternary's 3 levels (`{-1,0,1}`) — this is the Phase 7 / M8
//! sparse-quality recovery argument. The speed comparison is honest: PQ
//! uses a codebook gather + dot, which is more expensive than ternary's
//! add/sub/skip, so PQ is expected to be *slower* per call. Its value is in
//! accuracy (lower degradation → fewer experts needed → net win), not raw
//! kernel speed.
//!
//! Run with:
//! ```bash
//! cargo run --release --example bench_pq -p nse-ller
//! ```

use std::time::Instant;

use nse_core::sparse::{MicroExpert, PqCodebook, PqExpertData};
use nse_ller::{
    compute_pq_micro_expert_dispatch, compute_ternary_micro_expert_dispatch, KernelKind,
};
use nse_zstm::pq::{decode_pq, encode_pq, train_pq};

fn main() {
    let avx2 = std::is_x86_feature_detected!("avx2");
    println!("AVX2 available: {avx2}");
    if !avx2 {
        eprintln!("CPU has no AVX2 — benchmark will only measure scalar for PQ.");
    }

    // ---- Setup: synthetic Gaussian-ish weights, in_dim=64, 512 rows ----
    // dim=64 matches the plan's PQ evaluation config (M=4, sub_dim=16).
    let in_dim = 64usize;
    let n_rows = 512usize;
    let m = 4usize; // sub-vectors
    let nbits = 8usize; // 256 centroids per sub-codebook
    let iters_kmeans = 20usize;
    let seed = 11u64;

    let mut rng = Lcg::new(seed);
    let weights: Vec<Vec<f32>> = (0..n_rows)
        .map(|_| {
            (0..in_dim)
                .map(|_| {
                    // Sum-of-uniforms ~ Gaussian, centered at 0.
                    let mut s = 0.0f32;
                    for _ in 0..4 {
                        s += (rng.next_u32() as f32 / u32::MAX as f32) * 2.0 - 1.0;
                    }
                    s
                })
                .collect()
        })
        .collect();
    let x: Vec<f32> = (0..in_dim).map(|i| (i as f32) * 0.01 - 0.3).collect();

    // ---- Train PQ codebook on the normalized weights ----
    // Per-row scale = mean(|w|); codebook trained on w/scale (shape only).
    let scales: Vec<f32> = weights
        .iter()
        .map(|r| {
            let s = r.iter().map(|v| v.abs()).sum::<f32>() / in_dim as f32;
            s.max(1e-8)
        })
        .collect();
    let normalized: Vec<Vec<f32>> = weights
        .iter()
        .zip(scales.iter())
        .map(|(r, s)| r.iter().map(|&v| v / s).collect())
        .collect();
    let codebook = train_pq(&normalized, m, nbits, iters_kmeans, seed);
    println!(
        "PQ codebook: M={m}, nbits={nbits} ({} centroids), sub_dim={}",
        codebook.num_entries(),
        codebook.sub_dim
    );

    // Encode every row → PQ expert.
    let mut codes: Vec<u8> = Vec::with_capacity(n_rows * m);
    for r in &normalized {
        codes.extend_from_slice(&encode_pq(r, &codebook));
    }
    let pq_expert = MicroExpert {
        row_ids: (0..n_rows as u32).collect(),
        ternary: vec![],
        row_scales: vec![],
        centroid: vec![0.0; in_dim],
        mean_input: vec![0.0; in_dim],
        pq: Some(PqExpertData {
            codes,
            row_scales: scales.clone(),
            num_sub_vectors: m,
        }),
    };

    // ---- Ternary expert on the same weights (for the speed + MSE compare) ----
    let (ternary, tern_scales) = ternary_quantize_matrix(&weights, in_dim);
    let tern_expert = MicroExpert {
        row_ids: (0..n_rows as u32).collect(),
        ternary,
        row_scales: tern_scales,
        centroid: vec![0.0; in_dim],
        mean_input: vec![0.0; in_dim],
        pq: None,
    };

    let iters = 2000usize;
    // FLOPs per call: n_rows * in_dim (multiply-add = 2). Same for PQ and
    // ternary (both are one dot per row); PQ additionally does M codebook
    // gathers per row but those aren't counted as FLOPs.
    let flops_per_call = (n_rows * in_dim * 2) as f64;

    // ---- PQ kernel throughput (scalar vs AVX2) ----
    println!("\n=== PQ micro-expert (rows={n_rows}, in_dim={in_dim}, M={m}) ===");
    let mut y_pq = vec![0.0f32; n_rows];
    let (pq_scalar_ns, pq_avx2_ns) = bench_pq(&pq_expert, &codebook, &x, &mut y_pq, iters);
    report("pq", pq_scalar_ns, pq_avx2_ns, flops_per_call, iters);

    // ---- Ternary kernel throughput (scalar vs AVX2) on the same weights ----
    let mut y_tern = vec![0.0f32; n_rows];
    let (tern_scalar_ns, tern_avx2_ns) = bench_ternary(&tern_expert, &x, &mut y_tern, iters);
    println!("\n=== Ternary micro-expert (rows={n_rows}, in_dim={in_dim}) ===");
    report("ternary", tern_scalar_ns, tern_avx2_ns, flops_per_call, iters);

    // ---- Headline: PQ vs ternary reconstruction MSE ----
    let mut mse_pq = 0.0f64;
    let mut mse_tern = 0.0f64;
    for (r, &s) in weights.iter().zip(scales.iter()) {
        // PQ: scale * decode(encode(normalized)).
        let row_codes = encode_pq(&(r.iter().map(|&v| v / s).collect::<Vec<_>>()), &codebook);
        let recon_pq = decode_pq(&row_codes, &codebook);
        for (a, b) in r.iter().zip(recon_pq.iter()) {
            mse_pq += ((a - s * b) as f64).powi(2);
        }
        // Ternary: scale * ternary.
        let (tern, ts) = ternary_quantize_row(r);
        for (a, &c) in r.iter().zip(tern.iter()) {
            mse_tern += ((a - ts * c as f32) as f64).powi(2);
        }
    }
    mse_pq /= (n_rows * in_dim) as f64;
    mse_tern /= (n_rows * in_dim) as f64;

    println!("\n=== Reconstruction MSE (lower is better) ===");
    println!("PQ      MSE: {mse_pq:.6}");
    println!("Ternary MSE: {mse_tern:.6}");
    let ratio = if mse_pq > 0.0 { mse_tern / mse_pq } else { f64::INFINITY };
    println!("Ternary/PQ MSE ratio: {ratio:.2}x (PQ is {ratio:.2}x more accurate)");

    // ---- Headline: PQ vs ternary scalar kernel speed ----
    println!("\n=== Scalar kernel speed (PQ vs ternary) ===");
    let slowdown = pq_scalar_ns / tern_scalar_ns.max(1e-9);
    println!(
        "PQ: {pq_scalar_ns:.1} ns/call  |  Ternary: {tern_scalar_ns:.1} ns/call  |  PQ/ternary: {slowdown:.2}x"
    );
    println!(
        "(PQ gather+dot is expected to be slower than ternary add/sub/skip; PQ's value is accuracy.)"
    );

    // ---- Correctness cross-check: PQ kernel output matches decode·x ----
    let mut y_ref = vec![0.0f32; n_rows];
    for i in 0..n_rows {
        let row_codes = &pq_expert.pq.as_ref().unwrap().codes[i * m..(i + 1) * m];
        let recon = decode_pq(row_codes, &codebook);
        let s = scales[i];
        for j in 0..in_dim {
            y_ref[i] += s * recon[j] * x[j];
        }
    }
    let mut y_kern = vec![0.0f32; n_rows];
    compute_pq_micro_expert_dispatch(&pq_expert, &x, &mut y_kern, &codebook, KernelKind::Scalar);
    let max_err = y_ref
        .iter()
        .zip(y_kern.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    println!("\n=== Correctness (scalar kernel vs decode·x) ===");
    println!("max abs error: {max_err:.2e} (should be ~0)");
}

fn bench_pq(
    expert: &MicroExpert,
    codebook: &PqCodebook,
    x: &[f32],
    y: &mut [f32],
    iters: usize,
) -> (f64, f64) {
    let t0 = Instant::now();
    for _ in 0..iters {
        compute_pq_micro_expert_dispatch(expert, x, y, codebook, KernelKind::Scalar);
    }
    let scalar_ns = t0.elapsed().as_nanos() as f64 / iters as f64;

    if !std::is_x86_feature_detected!("avx2") {
        return (scalar_ns, f64::NAN);
    }
    let t0 = Instant::now();
    for _ in 0..iters {
        compute_pq_micro_expert_dispatch(expert, x, y, codebook, KernelKind::Auto);
    }
    let avx2_ns = t0.elapsed().as_nanos() as f64 / iters as f64;
    (scalar_ns, avx2_ns)
}

fn bench_ternary(
    expert: &MicroExpert,
    x: &[f32],
    y: &mut [f32],
    iters: usize,
) -> (f64, f64) {
    let t0 = Instant::now();
    for _ in 0..iters {
        compute_ternary_micro_expert_dispatch(expert, x, y, KernelKind::Scalar);
    }
    let scalar_ns = t0.elapsed().as_nanos() as f64 / iters as f64;

    if !std::is_x86_feature_detected!("avx2") {
        return (scalar_ns, f64::NAN);
    }
    let t0 = Instant::now();
    for _ in 0..iters {
        compute_ternary_micro_expert_dispatch(expert, x, y, KernelKind::Auto);
    }
    let avx2_ns = t0.elapsed().as_nanos() as f64 / iters as f64;
    (scalar_ns, avx2_ns)
}

fn report(name: &str, scalar_ns: f64, avx2_ns: f64, flops: f64, iters: usize) {
    let scalar_gflops = flops / scalar_ns;
    if avx2_ns.is_nan() {
        println!(
            "{name}: scalar {scalar_ns:.1} ns/call  {scalar_gflops:.2} GFLOP/s  (no AVX2)"
        );
        return;
    }
    let avx2_gflops = flops / avx2_ns;
    let speedup = scalar_ns / avx2_ns;
    println!(
        "{name}: scalar {scalar_ns:.1} ns ({scalar_gflops:.2} GFLOP/s)  |  AVX2 {avx2_ns:.1} ns ({avx2_gflops:.2} GFLOP/s)  |  speedup {speedup:.2}x"
    );
    let _ = iters;
}

/// Local copy of the ternary quantizer (mirrors `nse_zstm::quantize`) so this
/// example doesn't need `nse-zstm`'s quantizer re-exported; keeps the bench
/// self-contained.
fn ternary_quantize_row(row: &[f32]) -> (Vec<i8>, f32) {
    let scale = row.iter().map(|v| v.abs()).sum::<f32>() / row.len().max(1) as f32;
    let thresh = 0.5 * scale;
    let ternary: Vec<i8> = row
        .iter()
        .map(|&v| if v.abs() > thresh { if v >= 0.0 { 1 } else { -1 } } else { 0 })
        .collect();
    (ternary, scale)
}

fn ternary_quantize_matrix(weights: &[Vec<f32>], in_dim: usize) -> (Vec<i8>, Vec<f32>) {
    let mut ternary = Vec::with_capacity(weights.len() * in_dim);
    let mut scales = Vec::with_capacity(weights.len());
    for r in weights {
        let (codes, scale) = ternary_quantize_row(r);
        ternary.extend_from_slice(&codes);
        scales.push(scale);
    }
    (ternary, scales)
}

/// Minimal xorshift RNG (matches `nse_zstm::pq::Lcg`; kept local).
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
