# NSE Cargo Examples

NSE ship 3 cargo example để benchmark kernel + index. Tất cả chạy nhanh (bench_avx2 và bench_pq < 10s, bench_hnsw < 30s). Chạy `--release` (build debug quá chậm cho benchmark).

```bash
cargo run --release --example bench_avx2 -p nse-ller   # ternary + dense-core AVX2 vs scalar
cargo run --release --example bench_pq    -p nse-ller   # PQ scalar vs AVX2, PQ vs ternary speed + MSE
cargo run --release --example bench_hnsw  -p nse-rie    # HNSW vs brute-force query latency/recall
```

Mỗi example dưới đây: đo gì, cách chạy, và output thật (capture trên máy build `--release`, CPU x86_64 có AVX2). Số tuyệt đối phụ thuộc máy; pattern định tính ổn định.

---

## 1. `bench_avx2` (nse-ller)

**Đo gì:** throughput của 2 kernel LLER — (a) ternary micro-expert (add/sub/skip) và (b) dense-core mat-vec — dưới scalar reference vs AVX2 (FMA). Báo ns/call, GFLOP/s, và speedup. Ternary: 1 expert own 1024 row, in_dim=256. Dense core: 512 row, in_dim=256.

**Cách chạy:**

```bash
cargo run --release --example bench_avx2 -p nse-ller
```

**Output thật (capture):**

```text
AVX2 available: true

=== Ternary micro-expert (rows=1024, in_dim=256) ===
ternary: scalar 1043236.7 ns (0.50 GFLOP/s)  |  AVX2 961960.4 ns (0.55 GFLOP/s)  |  speedup 1.08x

=== Dense core (rows=512, in_dim=256) ===
dense_core: scalar 591356.2 ns (0.44 GFLOP/s)  |  AVX2 125559.5 ns (2.09 GFLOP/s)  |  speedup 4.71x
```

**Đọc kết quả:**
- **Ternary kernel speedup thấp (1.08x)** — bất ngờ vì ternary add/sub/skip có branch per-element, khó vectorize triệt để; AVX2 dùng mask add/sub nhưng horizontal reduction + tail giữ cost. Đây là động lực cho PQ (xem `bench_pq`).
- **Dense-core speedup cao (4.71x)** — FMA 8-lane accumulate thuần, không branch, vô hướng lý tưởng cho AVX2. Đây là lý do dense core (outlier row) rẻ trên CPU.
- Nếu CPU không có AVX2: dòng `AVX2 available: false`, chỉ đo scalar.

> **Lưu ý:** AVX2 kernel **không bit-identical** với scalar (SIMD reduction + FMA thay FP rounding), agree trong `~1e-5` relative — dưới noise floor PPL POC. `KernelKind::Scalar` = canonical ground truth dùng cho PPL.

---

## 2. `bench_pq` (nse-ller, mới — Phase 7 / M8)

**Đo gì:** (a) PQ micro-expert kernel scalar vs AVX2 throughput; (b) PQ vs ternary trên cả speed lẫn reconstruction MSE; (c) correctness cross-check (PQ kernel output vs `decode·x`). Headline accuracy là **reconstruction MSE**: PQ 256-level/sub-vector reconstruct weight tốt hơn ternary 3-level — argument phục hồi chất lượng thưa (Phase 7). Speed honest: PQ gather+dot đắt hơn ternary add/sub/skip, nên per-call chậm hơn mong đợi; giá trị PQ ở accuracy (→ ít expert cần active → net win end-to-end), không phải raw kernel speed.

Setup: synthetic Gaussian-ish weight, in_dim=64, 512 row, M=4 sub-vector, nbits=8 (256 centroid), 20 k-means iter.

**Cách chạy:**

```bash
cargo run --release --example bench_pq -p nse-ller
```

**Output thật (capture):**

```text
AVX2 available: true
PQ codebook: M=4, nbits=8 (256 centroids), sub_dim=16

=== PQ micro-expert (rows=512, in_dim=64, M=4) ===
pq: scalar 132294.7 ns (0.50 GFLOP/s)  |  AVX2 81349.8 ns (0.81 GFLOP/s)  |  speedup 1.63x

=== Ternary micro-expert (rows=512, in_dim=64) ===
ternary: scalar 877855.6 ns (0.07 GFLOP/s)  |  AVX2 115766.6 ns (0.57 GFLOP/s)  |  speedup 7.58x

=== Reconstruction MSE (lower is better) ===
PQ      MSE: 0.293689
Ternary MSE: 0.322201
Ternary/PQ MSE ratio: 1.10x (PQ is 1.10x more accurate)

=== Scalar kernel speed (PQ vs ternary) ===
PQ: 132294.7 ns/call  |  Ternary: 877855.6 ns/call  |  PQ/ternary: 0.15x
(PQ gather+dot is expected to be slower than ternary add/sub/skip; PQ's value is accuracy.)

=== Correctness (scalar kernel vs decode·x) ===
max abs error: 1.91e-6 (should be ~0)
```

**Đọc kết quả:**
- **Reconstruction MSE:** PQ 0.293689 vs ternary 0.322201 → **PQ 1.10x chính xác hơn** (256 level/sub-vector vs 3). Đây là headline của Phase 7 / M8 và là cơ sở trực tiếp cho việc PQ giảm sparse PPL degradation từ +32.8% (ternary) xuống +18.4% (PQ) trên dim=64 (paper §5.7.2). Số này khớp paper §5.7.3 (PQ 0.294 vs ternary 0.322).
- **PQ AVX2 speedup 1.63x** — FMA chặt codebook lookup vào 8-lane accumulate; paper §5.7.3 báo 1.92x (cùng config, máy khác). Khớp về bậc lớn.
- **PQ scalar nhanh hơn ternary scalar 6.6x (0.15x ratio)** — trái dự đoán "gather expensive", vì ternary scalar có branch per-element (chậm), còn PQ scalar có tight dot loop. AVX2 thu hẹp gap (ternary 7.58x speedup nhờ add/sub vectorizable).
- **PQ AVX2 (81µs) vs ternary AVX2 (116µs): PQ nhanh hơn ~30%** trong config dim=64/sub_dim=16 — FMA chặt lookup. Trên sub_dim lớn (≥32), gather cost có thể đảo ngược; tài liệu hóa trung thực (paper §5.7.5).
- **Correctness:** max abs error 1.91e-6 (PQ kernel vs decode·x reference) — gần 0, xác nhận kernel đúng.
- Nếu CPU không có AVX2: chỉ đo scalar cho PQ.

> So sánh với paper §5.7.3 (bảng chuẩn): PQ scalar 117673 ns / AVX2 61325 ns / 1.92x; ternary scalar 797618 ns / AVX2 87442 ns / 9.12x. Số bạn chạy khác tuyệt đối (máy, load, turbo) nhưng pattern ổn định: PQ scalar >> ternary scalar, AVX2 thu hẹp gap, PQ AVX2 cạnh tranh được với ternary AVX2. MSE 0.294 vs 0.322 (1.10x) khớp chính xác.

---

## 3. `bench_hnsw` (nse-rie)

**Đo gì:** HNSW (Hierarchical Navigable Small World) vs brute-force MIPS — query latency (µs/query, q/s), speedup, và recall@10. Chạy 3 scale: n=1000, 5000, 20000 expert (dim=64, unit-norm centroid), 100 query random.

**Cách chạy:**

```bash
cargo run --release --example bench_hnsw -p nse-rie
```

**Output thật (capture):**

```text
=== n=1000 experts, dim=64, 100 queries ===
brute: 666.9 us/query  (1499 q/s)
HNSW:  1893.9 us/query  (528 q/s)  speedup 0.35x  recall@10 0.993

=== n=5000 experts, dim=64, 100 queries ===
brute: 2513.2 us/query  (398 q/s)
HNSW:  2966.0 us/query  (337 q/s)  speedup 0.85x  recall@10 0.868

=== n=20000 experts, dim=64, 100 queries ===
brute: 10365.1 us/query  (96 q/s)
HNSW:  3349.9 us/query  (299 q/s)  speedup 3.09x  recall@10 0.547
```

**Đọc kết quả:**
- **Cross-over scale:** ở n=1000 HNSW *chậm hơn* brute (0.35x) — graph build + traversal overhead vượt lợi ích khi expert ít; brute-force exact MIPS O(N) đủ rẻ. Đến n=20000, HNSW thắng 3.09x (brute 10.4ms vs HNSW 3.3ms/query) — đúng kỳ vọng O(log N).
- **Recall@10 trade-off:** recall giảm khi n tăng (0.993 → 0.547) với `ef_search` mặc định (64..256). Tăng `ef_search` (param thứ 4 của `HnswIndex::new`) đổi recall vs speed. POC dùng `build_hnsw_for_layer` (M=8, ef=32, ef_search=max(16, n)) — POC-friendly cho expert count nhỏ (~10-100) nơi recall ~1.
- **POC context:** ở expert count nhỏ (K ~ 10-100 như toy LM), HNSW recall@k trivially ~1 vs brute; giá trị HNSW chỉ hiện ở spec's scale (triệu expert). Benchmark này validate correctness + provide API, không phải production tuning.
- So paper §x (HNSW table): n=20000, brute 4.925ms → 2.144ms, speedup 2.30x, recall 0.547. Số bạn chạy (3.09x) cùng bậc; recall 0.547 khớp chính xác. Sai khác tuyệt đối do máy + warm-up.

> **Lưu ý warm-up:** dòng warm-up (`hnsw.query(&queries[0], 10)`) chạy trước timer để build entry-point path; không tính vào kết quả.

---

## Tổng kết benchmark

| Example | Headline | Pattern |
|---------|----------|---------|
| `bench_avx2` | dense-core AVX2 4.71x scalar; ternary 1.08x | FMA vectorize tốt cho dense, kém cho ternary branch |
| `bench_pq` | PQ 1.10x chính xác hơn ternary (MSE); PQ AVX2 1.63x scalar; PQ scalar 6.6x nhanh hơn ternary scalar | Giá trị PQ ở accuracy (→ ít expert → net win), không raw speed |
| `bench_hnsw` | HNSW thắng brute ở n=20000 (3.09x); thua ở n=1000 (0.35x); recall giảm 0.993 → 0.547 | Cross-over scale; POC dùng brute mặc định, HNSW cho scale-out |

Tất cả 3 example chạy thành công trong `--release`. Không có example fail. Để reproduce chính xác số paper, xem command trong `paper/PAPER.md` §5.7.4.
