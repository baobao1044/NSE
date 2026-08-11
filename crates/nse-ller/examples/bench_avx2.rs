//! AVX2 vs scalar throughput benchmark for the ternary micro-expert and
//! dense-core kernels.
//!
//! Measures wall-clock time over many invocations on a large synthetic
//! expert/core, and reports throughput (rows/s, effective GFLOP/s) and the
//! AVX2 speedup. Run with:
//!
//! ```bash
//! cargo run --release --example bench_avx2 -p nse-ller
//! ```

use std::time::Instant;

use nse_core::sparse::MicroExpert;
use nse_core::tensor::Matrix;
use nse_ller::{
    compute_dense_core_dispatch, compute_ternary_micro_expert_dispatch, KernelKind,
};

fn main() {
    let avx2 = std::is_x86_feature_detected!("avx2");
    println!("AVX2 available: {avx2}");
    if !avx2 {
        eprintln!("CPU has no AVX2 — benchmark will only measure scalar.");
    }

    // ---- Ternary micro-expert benchmark ----
    // One expert owning many rows, large in_dim.
    let in_dim = 256usize;
    let n_rows = 1024usize;
    let expert = MicroExpert {
        row_ids: (0..n_rows as u32).collect(),
        ternary: (0..(n_rows * in_dim))
            .map(|i| match i % 3 { 0 => 1, 1 => -1, _ => 0 })
            .collect(),
        row_scales: (0..n_rows).map(|i| 0.01 + (i as f32) * 0.001).collect(),
        centroid: vec![0.0; in_dim],
        mean_input: vec![0.0; in_dim],
        pq: None,
    };
    let x = vec![0.5f32; in_dim];
    let mut y = vec![0.0f32; n_rows];

    // FLOPs per invocation: n_rows * in_dim (multiply-add counted as 2).
    let flops_per_call = (n_rows * in_dim * 2) as f64;
    let iters = 2000usize;

    println!("\n=== Ternary micro-expert (rows={n_rows}, in_dim={in_dim}) ===");
    let (scalar_ns, avx2_ns) = bench_ternary(&expert, &x, &mut y, iters);
    report("ternary", scalar_ns, avx2_ns, flops_per_call, iters);

    // ---- Dense-core benchmark ----
    let core_rows = 512usize;
    let core_in = 256usize;
    let mut core = Matrix::zeros(core_rows, core_in);
    for (i, v) in core.data.iter_mut().enumerate() {
        *v = ((i as f32) * 0.001 - 0.5).sin();
    }
    let row_ids: Vec<u32> = (0..core_rows as u32).collect();
    let x2 = vec![0.3f32; core_in];
    let mut y2 = vec![0.0f32; core_rows];
    let flops_core = (core_rows * core_in * 2) as f64;

    println!("\n=== Dense core (rows={core_rows}, in_dim={core_in}) ===");
    let (scalar_ns, avx2_ns) = bench_dense(&core, &row_ids, &x2, &mut y2, iters);
    report("dense_core", scalar_ns, avx2_ns, flops_core, iters);
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

fn bench_dense(
    core: &Matrix,
    row_ids: &[u32],
    x: &[f32],
    y: &mut [f32],
    iters: usize,
) -> (f64, f64) {
    let t0 = Instant::now();
    for _ in 0..iters {
        compute_dense_core_dispatch(core, row_ids, x, y, KernelKind::Scalar);
    }
    let scalar_ns = t0.elapsed().as_nanos() as f64 / iters as f64;

    if !std::is_x86_feature_detected!("avx2") {
        return (scalar_ns, f64::NAN);
    }
    let t0 = Instant::now();
    for _ in 0..iters {
        compute_dense_core_dispatch(core, row_ids, x, y, KernelKind::Auto);
    }
    let avx2_ns = t0.elapsed().as_nanos() as f64 / iters as f64;
    (scalar_ns, avx2_ns)
}

fn report(name: &str, scalar_ns: f64, avx2_ns: f64, flops: f64, iters: usize) {
    let scalar_gflops = flops / scalar_ns; // ns already 1e-9, so flops/ns = GFLOP/s
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
