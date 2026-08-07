# Plan: Scaffold toàn hệ thống Neuro-Sparse Engine (NSE) bằng Rust

## 0. Bối cảnh & định hướng

Spec gốc (inference) + mở rộng training thay thế từ bạn → hệ thống thống nhất gồm:

- **Inference (NSE gốc, Zero-Shot):** ZSTM (chuyển đổi offline) + RIE (định tuyến O(log N)) + LLER (kernel SIMD + cache tiling) + format `.nse`.
- **Training thay thế (đột phá, không phụ thuộc cluster GPU khổng lồ):**
  1. **Forward-Forward / Predictive Coding** (Hinton) — local goodness, không backprop, zero VRAM overhead.
  2. **Hopfield / Associative Memory** — energy-based, one-shot/few-shot learning bằng vector projection.
  3. **LSH Sparse Weight Training** — chỉ cập nhật ~0.01% trọng số mỗi step (dùng chung index LSH với RIE inference).
- **POC:** Toy LM đơn giản (Rust), **kết quả quan trọng nhất = đo sụt giảm PPL** dense (gốc) vs sparse (NSE).

Ngôn ngữ: **Rust**. Thư mục `C:\Users\Admin\Downloads\NSE` hiện trống → greenfield.

## 1. Cấu trúc workspace (Cargo workspace, 8 crate)

```
NSE/
├── Cargo.toml                 # [workspace]
├── README.md
├── docs/
│   ├── 01-nse-inference-spec.md   # spec bạn paste (ZSTM/RIE/LLER/.nse/pseudocode)
│   └── 02-training-vision.md      # 3 thuật toán training thay thế (FF/Hopfield/LSH-sparse)
├── data/
│   └── corpus.txt             # corpus nhỏ (vd subset tinyshakespeare) cho train Toy LM
├── crates/
│   ├── nse-core/     # types, traits, errors, format .nse (mmap), Tensor, ModelSource
│   ├── nse-models/   # Toy LM + tokenizer + safetensors loader + config
│   ├── nse-train/    # Trainer trait + SgdTrainer(baseline REAL) + FF/Hopfield/LshSparse(STUB)
│   ├── nse-zstm/     # offline: outlier + clustering(k-means) + quantization(ternary/PQ)
│   ├── nse-rie/      # MIPS index(HNSW/LSH) + threshold router + static bias compensator
│   ├── nse-ller/     # cache tiling + SIMD kernel(ternary/PQ) — scalar ref + AVX2
│   ├── nse-eval/     # PPL dense vs sparse + báo cáo so sánh
│   └── nse-cli/      # bin: train / transmute / infer / eval
├── tests/            # integration tests (round-trip .nse, correctness, PPL sanity)
└── benches/          # benchmark skeleton (dense vs sparse latency)
```

Phụ thuộc tối giản, ưu tiên pure-Rust để build sạch trên Windows: `ndarray`, `ndarray-rand`/`rand`, `memmap2` (mmap `.nse`), `safetensors` (load), `rayon` (song song), `clap` (CLI), `anyhow`/`thiserror`. HNSW/k-means tự viết bản đơn giản cho POC (không phụ thuộc crate nặng).

## 2. Độ "thật" của từng module (scaffold nhưng PPL path chạy được)

| Crate | Phần REAL (chạy được) | Phần SCAFFOLD (trait + stub + TODO) |
|---|---|---|
| **nse-core** | `NSEFileHeader`, `MicroExpertMeta` đúng spec; read/write `.nse` + mmap; `Tensor`/`Matrix`; `ModelSource` trait; error types; magic `"NSE1"` | — |
| **nse-models** | Toy LM (mini transformer-LM 2 layer, d~128, vocab~1k); tokenizer char-level (bundled); safetensors loader linh hoạt; `Config` | kiến trúc Llama thật (sau) |
| **nse-train** | `SgdTrainer` baseline (train Toy LM ra PPL hợp lý) | `ForwardForwardTrainer`, `HopfieldTrainer`, `LshSparseTrainer` — trait + skeleton + TODO + doc |
| **nse-zstm** | Outlier extraction (top-k theo biên độ); spherical k-means (bản đơn giản); ternary encode (4 weight/byte) | SVD decomposition; PQ codebook |
| **nse-rie** | Brute-force MIPS (đúng cho POC); adaptive threshold router; static bias compensator | HNSW; LSH scale lớn |
| **nse-ller** | **Scalar reference kernel** (ternary math đúng → dùng cho PPL) | AVX2 `_mm256_*` kernel (target_feature, có scalar fallback); L3 cache tiling; PQ shuffle kernel |
| **nse-eval** | Tính PPL; runner dense; runner sparse; báo cáo so sánh (PPL + % sụt giảm) | — |
| **nse-cli** | `train`, `transmute`, `eval dense`, `eval sparse`, `eval compare` | subcommand FF/Hopfield/LSH (stub) |

**Lý do:** Để đo "sụt giảm PPL" đúng, chỉ cần kernel **đúng về mặt toán học** — AVX2/HNSW/L3-tiling là tối ưu hiệu năng, không đổi kết quả số học → để scaffold. Toàn bộ optimization & 3 thuật toán training nghiên cứu để skeleton + trait, có sẵn chỗ cắm.

## 3. Pipeline POC end-to-end (deliverable chính = báo cáo PPL)

```
1. cargo run --release -- train          → train Toy LM (SgdTrainer) trên data/corpus.txt → toy_lm.safetensors
2. cargo run --release -- eval dense    → PPL_dense  (baseline)
3. cargo run --release -- transmute     → ZSTM: outlier + k-means + ternary → model.nse
4. cargo run --release -- eval sparse   → RIE + LLER(scalar) + bias → PPL_sparse
5. cargo run --release -- eval compare  → in bảng: PPL_dense | PPL_sparse | % sụt giảm | num params active
```

Mỗi bước là một lệnh CLI độc lập, có artifact trung gian (`.safetensors`, `.nse`) → dễ debug từng giai đoạn.

## 4. Format `.nse` (theo đúng spec, struct trong nse-core)

```rust
struct NSEFileHeader {
    magic: [u8; 4],            // "NSE1"
    total_params: u64,
    num_layers: u32,
    dense_core_size: u32,      // bytes — Outlier Core
    codebook_size: u32,        // bytes — Codebook (PQ)
    index_tree_offset: u64,   // con trỏ MIPS Tree
}
struct MicroExpertMeta {
    expert_id: u32,
    num_channels: u32,
    data_offset: u64,          // vị trí data nén sub-1-bit
    // centroid_vector: float[dim]  // biến length, nằm kế tiếp
}
```
Layout tối ưu mmap: header → dense core → codebook → micro-experts data → MIPS tree. Load bằng `memmap2`, zero-copy truy cập.

## 5. Tests & benches

- **tests:** round-trip `.nse` (write→read→so); ternary encode/decode; transmutation correctness (sparse output ≈ dense cho input cố định, trong tolerance); bias compensation bù đúng; PPL sanity (PPL_sparse không tệ hơn ngưỡng cho phép).
- **benches:** skeleton đo latency dense vs sparse inference (criterion, sau).

## 6. Milestones

- **M0:** Verify toolchain (`cargo`/`rustc`/`git`); init git + Cargo workspace + 8 crate skeletons (compiles, `cargo build`/`cargo test` pass rỗng).
- **M1:** `nse-core` (.nse round-trip + mmap) + `nse-models` (Toy LM forward + tokenizer). Toy LM forward chạy được.
- **M2:** `nse-train` SgdTrainer → train Toy LM ra PPL hợp lý trên corpus nhỏ.
- **M3:** `nse-zstm` (outlier + k-means + ternary) → xuất `.nse` từ Toy LM đã train.
- **M4:** `nse-rie` (brute-force MIPS + threshold + bias) + `nse-ller` scalar kernel → sparse inference đúng.
- **M5:** `nse-eval` + `nse-cli` → **báo cáo so sánh PPL dense vs sparse** (deliverable chính).
- **M6:** Scaffold phần còn lại: AVX2 kernel, HNSW, L3 cache tiling, PQ, và 3 thuật toán training (FF/Hopfield/LSH-sparse) — trait + stub + doc + TODO.

## 7. Ngoài scope (sau)

- Tối ưu AVX2 thực tế + benchmark hiệu năng.
- HNSW/LSH ở quy mô 2.7T; L3 cache tiling engine thật.
- PQ codebook đầy đủ.
- Implement thật 3 thuật toán training thay thế (FF/Hopfield/LSH-sparse) — scaffold lần này.
- Tích hợp Llama-3-8B thật.

## 8. Lệnh build/run dự kiến

```bash
cargo build --release
cargo test
cargo run --release -- train
cargo run --release -- eval dense
cargo run --release -- transmute --in toy_lm.safetensors --out model.nse
cargo run --release -- eval sparse --nse model.nse
cargo run --release -- eval compare --nse model.nse --weights toy_lm.safetensors
```

## Lưu ý
- Bước **M0** đầu tiên tôi sẽ kiểm tra xem máy có Rust toolchain chưa; nếu chưa có, sẽ báo bạn và hướng dẫn cài trước khi scaffold.
- Toàn bộ math dùng **scalar reference đúng** làm chân lý để đo PPL; AVX2/HNSW chỉ là tối ưu sau, không ảnh hưởng kết quả PPL → POC trung thực.
