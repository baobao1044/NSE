# Neuro-Sparse Engine (NSE)

Framework AI chạy LLM trên CPU/Edge không cần GPU — Rust implementation.

## Mục tiêu

Chạy mô hình ngôn ngữ (POC: Toy LM, sau đó Llama-3-8B) trên CPU/Edge bằng:

- **Zero-Shot Transmutation (ZSTM):** biến weight dense → đồ thị thưa động, không retrain.
- **Routing & Indexing (RIE):** định tuyến O(log N) chỉ kích hoạt ~0.01% trọng số/token.
- **Low-Level Execution (LLER):** kernel SIMD + L3 cache tiling.
- **Training thay thế (scaffold):** Forward-Forward, Hopfield/Associative Memory, LSH Sparse Training.

Kết quả quan trọng nhất của POC: **đo sụt giảm PPL** dense vs sparse.

## Workspace

| Crate | Vai trò |
|---|---|
| `nse-core` | Types, errors, format `.nse` (mmap) |
| `nse-models` | Toy LM + tokenizer + safetensors loader |
| `nse-train` | Trainer trait + SGD (real) + FF/Hopfield/LSH (scaffold) |
| `nse-zstm` | Outlier + k-means + ternary/PQ quantization |
| `nse-rie` | MIPS index + threshold router + bias compensator |
| `nse-ller` | Cache tiling + SIMD kernel (scalar ref + AVX2) |
| `nse-eval` | PPL dense vs sparse + báo cáo |
| `nse-cli` | CLI: train / eval / transmute |

## Build & test

```bash
cargo build --workspace
cargo test --workspace
```

## Pipeline POC (sau khi hoàn thiện)

```bash
cargo run --release -- train          # -> toy_lm.safetensors
cargo run --release -- eval dense     # -> PPL_dense
cargo run --release -- transmute      # -> model.nse
cargo run --release -- eval sparse    # -> PPL_sparse
cargo run --release -- eval compare   # -> báo cáo so sánh
```

## Trạng thái

- [x] **M0**: Workspace scaffold, build + test pass.
- [ ] **M1**: nse-core (.nse) + nse-models (Toy LM forward).
- [ ] **M2**: nse-train SgdTrainer.
- [ ] **M3**: nse-zstm (outlier + k-means + ternary → .nse).
- [ ] **M4**: nse-rie + nse-ller scalar kernel.
- [ ] **M5**: nse-eval + nse-cli (báo cáo PPL).
- [ ] **M6**: Scaffold AVX2/HNSW/PQ + 3 thuật toán training.

Xem `docs/` cho spec đầy đủ.
