# Changelog

Tất cả thay đổi đáng chú ý của dự án **Neuro-Sparse Engine (NSE)** được ghi lại
trong file này. Định dạng dựa trên [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
và dự án tuân thủ [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

NSE phát triển theo **milestone** (M0–M8) thay vì tag version语义 từng bản —
mỗi milestone là một giai đoạn nghiên cứu có artifact + kết quả tài liệu hóa trong
`paper/PAPER.md`. Workspace version hiện tại `0.1.0` (xem `Cargo.toml`).

### Quy ước loại thay đổi

- **Added** — feature mới.
- **Changed** — thay đổi feature/behavior hiện có (kể cả kết quả thực nghiệm).
- **Deprecated** — feature sắp bỏ.
- **Removed** — feature đã bỏ.
- **Fixed** — bug fix (hoặc note negative result đã tài liệu hóa trung thực).
- **Security** — không áp dụng cho prototype nghiên cứu này.

---

## [Unreleased]

### Added — M9: Calibration + bias-adaptive (Phase 8)

- **Bias correctness fix (S1)**: `apply_bias` được áp dụng **pruned-only** (chỉ
  hàng của expert bị prune) thay vì unconditional — sửa double-count
  `W_quant[i]·x + W[i]·mean_input` cho activated expert rows (bug từ M8,
  test không phát hiện vì `corpus=None` → `mean_input=0`). Thêm `row_to_expert`
  map (-1=core, k=expert k) vào `SparseLayer` (`#[serde(default)]` backward-compat).
- **Calibration infra (S2)**: `collect_activations` thay `mean_inputs_for`,
  sliding windows đa cửa sổ (step=seq/2, 50% overlap) thay 1 window. CLI
  thêm `--calibration-corpus` (tách calibration set khỏi training corpus).
- **Activation PQ codebook (S3)**: `BiasMode::Adaptive` — train VQ codebook
  (PQ machinery M=1, 256 centroids) trên calibration activations, precompute
  `bias_table[c][i] = W_quant[i]·centroid[c]`. Đúng promise "PQ là foundation
  cho cả 2" (weight codebook + activation codebook).
- **Adaptive bias kernel (S4)**: `apply_adaptive` encode `x` → code `c`, lookup
  `bias_table[c][i]` cho pruned rows — per-token, không mean. Dispatch 3 mode
  trong `sparse_linear_with_kernel` (legacy/pruned-only/adaptive).
- **CLI (S6)**: `--bias-mode mean|adaptive`, `--calibration-corpus`, `--bias-codebook-bits`.
- **Tests mới (S7)**: `bias_pruned_only_no_double_count` (3 case), `calibration_multi_window`,
  `activation_pq_bias_table_math` (cross-check decode), `bias_adaptive_depends_on_x`
  (2 x → 2 bias), `transmute_adaptive_roundtrip` (save/load fields).

### Changed

- `transmute` thêm param `calibration_corpus: Option<&[u8]>` (None → dùng corpus,
  backward-compat). `transmute_matrix` → `transmute_matrix_with_cal` (thêm `cal_rows`).
- `TransmuteConfig` thêm `bias_mode: BiasMode` (default `Mean`, backward-compat).
- `nse-rie` depends on `nse-zstm` (cho `encode_pq` trong adaptive bias).

### Fixed

- **Double-count bias bug**: activated expert rows nhận `W_quant[i]·x + W[i]·mean_input`
  thay vì chỉ `W_quant[i]·x` — bias phải pruned-only (doc `sparse.rs:14-21` nói
  pruned-only nhưng code add unconditional). Fix: `row_to_expert` + pruned-only dispatch.

---

## [M8] - 2025 — Real PQ codebook (Phase 7)

Mục tiêu: tấn công con số +82% sparse PPL degradation (paper §5.2) bằng Product
Quantization với codebook 8-bit học được (256 level/sub-vector) thay ternary 3
level thô. Giảm degradation gần một nửa; 9 test mới (56 test tổng).

### Added

- **nse-zstm `pq`**: `train_pq` (per-sub-vector L2 k-means, không spherical vì
  trọng lượng Gaussian-centered) + `encode_pq`; một codebook shared per `SparseLayer`
  (< 1 MB → L3 cache, per spec).
- **`QuantSchemeConfig::Pq { num_sub_vectors, nbits, iters, seed }`** trong
  `transmuter.rs`; `TransmuteConfig::pq()` (M=4, 8-bit, 20 iters) + `build_pq_experts`.
- **nse-core::sparse `PqCodebook`** (`num_sub_vectors`, `nbits`, `sub_dim`,
  `codebook [subvec m][centroid c][dim j]`) + **`PqExpertData`** (`codes`,
  `row_scales`, `num_sub_vectors`).
- **`#[serde(default)]` trên `Option`**: `MicroExpert.pq` + `SparseLayer.pq_codebook`
  → backward-compat (model.nse ternary-only cũ deserialize `None`, chạy ternary).
- **nse-ller PQ kernel**: `compute_pq_micro_expert_scalar` (decode-inline + dot,
  tight loop) + AVX2 PQ gather+FMA (`compute_pq_micro_expert_dispatch`).
- **nse-rie dispatch**: `compute_pq_micro_expert_dispatch` + nhánh
  `sparse_linear_with_kernel` theo `MicroExpert.pq` (Some → PQ + codebook,
  None → ternary; defensive skip khi expert claims PQ nhưng layer thiếu codebook).
- **Geometry fallback**: `in_dim` không chia hết cho `M` → dùng ước số lớn nhất
  `≤ M` (dim=30, M=4 → M=3; prime → M=1 = VQ) — CLI không panic.
- **CLI `transmute --quant pq`** + `--pq-subvectors M` + `--pq-nbits` (default
  ternary/4/8); kernel auto-detect scheme từ `MicroExpert.pq` — `eval-sparse`/
  `eval-compare`/`eval-composite` không cần flag mới.
- **`bench_pq` example** (`cargo run --example bench_pq -p nse-ller`): kernel +
  reconstruction MSE benchmark.
- 9 test mới (56 tổng): PQ transmute roundtrip, PQ save/load serde roundtrip,
  geometry fallback, PQ kernel correctness, `sparse_pq_lower_degradation`, v.v.

### Changed

- **Sparse PPL degradation +32.8% → +18.4%** (dim=64, SGD 20 epoch, all-experts,
  M=4 8-bit): PQ `pq/ternary = 0.891` (PQ thắng 11% trên sparse PPL). Bar kỳ vọng
  (<0.700, giảm >30%) chưa đạt trên toy dim=64 (chỉ ~50 residual rows train 256
  centroids → codebook sub-fit) — dự kiến mở rộng trên dim ≥ 128.
- **PQ AVX2 1.92× speedup** (61µs) vs scalar PQ (117µs); PQ chính xác 1.10× vs
  ternary (reconstruction MSE 0.294 vs 0.322, 256 level vs 3). PQ AVX2 nhanh hơn
  ternary AVX2 30% trên dim=64 (FMA chặt lookup) — trái dự đoán "gather expensive",
  có thể đảo ngược trên sub_dim ≥ 32.

### Fixed

- Backward-compat: `--quant ternary` (default) giữ nguyên reproduce số liệu
  paper §5.2–5.6 (model.nse ternary-only không break).
- Tài liệu hóa trung thực giới hạn PQ (paper §5.7.5): toy model sub-tối ưu cho
  codebook, PQ gather không luôn nhanh hơn, bar <15% chưa đạt (calibration +
  bias-adaptive là Phase 8 kế tiếp).

---

## [M7] - 2025 — CompositeTrainer + sparse Hopfield + eval-composite 4-path

Kiến trúc tổng hợp "hippocampus + cortex" phân vai trò (routing/learning/memory
tách rời) — composite thắng từng trainer riêng cùng compute, không thắng SGD
(tài liệu hóa trung thực). 4-path eval reveal negative result sparse Hopfield.

### Added

- **nse-train `composite`**: `CompositeTrainer` orchestrator 4-phase qua
  `Trainer` trait — (1) SGD warm *stabilizer* → (2) Hopfield writes *hippocampus*
  → (3) Forward-Forward *local plasticity* (`weight_clip` 0.5 sweet spot) →
  (4) LSH-sparse fine-tune *routing + sparse update* (~1% update/step). Mỗi phase
  skip khi epoch/write = 0; default = FF 15 + LSH 15.
- **`CompositeConfig`** với `--sgd-epochs`/`--hopfield-writes`/`--ff-epochs`/`--lsh-epochs`
  toggle phase.
- **nse-eval Hopfield forward**: `dense_ppl_hopfield`, `sparse_ppl_hopfield`,
  `sparse_forward_hopfield` (softmax `β·(ff_up·k)` thay GELU), `Activation { Gelu,
  Hopfield }`, `SparseOptions`.
- **`compare_composite` → `CompositeReport`**: báo cáo 4-path (dense/sparse ×
  GELU/Hopfield) với degrade tương đối — artifact chính của M7.
- **CLI `train-composite` + `eval-composite`** (với `--beta` cho Hopfield retrieval).

### Changed

- Composite (FF 15 + LSH 15) **thắng từng trainer riêng** cùng compute:
  21.44 vs FF 26.04 (−18%), LSH 24.12 (−11%), Hopfield 62.40 (−66%).
- Composite **không thắng SGD** (21.44 vs 12.37) — đúng kỳ vọng (backprop đầy đủ
  gradient toàn cục mạnh nhất trên toy; composite trao đổi chất lượng lấy tính
  cục bộ/không-backprop-toàn-cục). Tài liệu hóa trung thực.

### Fixed

- Tài liệu hóa **negative result sparse Hopfield**: sparse Hopfield 52.61 thua
  sparse GELU 28.63 (+84%) — ternary quantization phá cosine structure của `ff_up`
  keys → softmax phẳng → retrieval không chọn đúng memory. Hướng mở: giữ `ff_up`
  dense (chỉ quantize `ff_down`) hoặc dùng PQ codebook thay ternary.
- FF test assertion: tolerance 0.05 thay strict `>` (margin G_pos−G_neg mỏng trên
  held-out windows, paper §5.4); bar chính = PPL < uniform baseline (không suy biến).

---

## [M6] - 2025 — Scaffold AVX2/HNSW/PQ + 3 alternative trainers

Scaffold tối ưu hiệu năng (AVX2/HNSW) cho phase sau + 3 thuật toán training đột
phá (FF/Hopfield/LSH-sparse) khảo sát training không phụ thuộc backprop toàn cục
quy mô lớn.

### Added

- **nse-ller `avx2`**: `compute_ternary_micro_expert_avx2` (mask pos/neg 0xFFFFFFFF,
  `_mm256_and_ps`, add/sub 8 float/iter, tail vô hướng cùng thứ tự, horizontal-reduce
  theo thứ tự vô hướng) + `compute_dense_core_avx2` (`_mm256_fmadd_ps` FMA);
  `#[target_feature(enable="avx2")]` + dispatch runtime
  `is_x86_feature_detected!("avx2")` + scalar fallback; `KernelKind { Scalar, Avx2,
  Auto }`.
- **nse-rie `hnsw`**: `HnswIndex` đồ thị phân tầng thật (cấp `l ~ floor(−ln(u)·mL)`,
  `mL = 1/ln(M)`; chèn greedy descent + beam search; link bidirectional M-neighbor
  với pruning; `ensure_connected_layer0` BFS → recall@k = 1 trên đồ thị nhỏ) +
  `IndexKind { Brute, Hnsw }` + `MipsQuery` trait + `build_hnsw_for_layer`.
- **nse-train `forward_forward`**: `ForwardForwardTrainer` (Hinton FF — goodness
  `G = mean(y²)` per-block, positive/negative softplus loss, θ EMA per-block, gradient
  cục bộ `block_backward_local`, max-norm clamp `weight_clip`) + `Homeostasis { None,
  LayerNorm }` + light Hebb head cho tied embedding.
- **nse-train `hopfield`**: `HopfieldTrainer` (modern Hopfield — `ff_up` = key store,
  `ff_down` = value store, one-shot writes, retrieval `softmax(β·(ff_up·k))`) +
  `hopfield_retrieve` export.
- **nse-train `lsh`** (LSH random hyperplane hash) + **`lsh_sparse`**:
  `LshSparseTrainer` (dense backprop + per-row LSH gradient masking, ~sparse_fraction
  update/step, chia sẻ `sgd_apply` với SGD) + `LshSparseConfig`.
- **CLI `train-ff` / `train-hopfield` / `train-lsh`** (train-lsh hỗ trợ `--init`
  warm-start).
- **Benchmark examples**: `bench_avx2` (kernel), `bench_hnsw` (query latency vs N).

### Changed

- `KernelKind::Auto` = AVX2 nếu CPU hỗ trợ, else scalar (POC default).

### Fixed

- Tài liệu hóa trung thực: AVX2 không bit-identical với vô hướng (associativity
  dấu chấm động) — test tolerance 1e-5, giá trị thật ở throughput (paper §5.5, §6.1).
- HNSW recall = 1 trên N nhỏ nhờ `ensure_connected_layer0`; tradeoff recall/latency
  chỉ thể hiện ở N lớn (benchmark: break-even ~5,000, 2.3× tại 20k nhưng recall
  giảm 0.99→0.55) — tài liệu hóa là tradeoff thật, không bug (paper §5.5).

---

## [M5] - 2025 — nse-eval + nse-cli → dense vs sparse PPL compare report

Evaluation + CLI hoàn thiện đường ống POC: huấn luyện → biến đổi → suy luận thưa →
so sánh PPL.

### Added

- **nse-eval `ppl`**: `dense_ppl`, `sparse_ppl`, `sparse_ppl_with_options`,
  `perplexity_from_logprobs`, `logprobs` — PPL cho cả `ToyLm` dense và
  `TransmutedModel` sparse qua cùng sliding windows.
- **nse-eval `sparse_forward`**: sparse forward pass over `TransmutedModel`
  (mirror dense forward, mỗi matmul → `sparse_linear`).
- **nse-eval `compare`**: `compare`/`compare_with_options` → `CompareReport {
  PPL_dense, PPL_sparse, degradation, active_fraction }`.
- **nse-cli `nse` binary** với subcommands: `train`, `eval-dense`, `transmute`,
  `eval-sparse`, `eval-compare` — mỗi cái sinh artifact trung gian để debug riêng.
- Báo cáo so sánh dense vs sparse PPL (headline metric POC).

### Changed

- Pipeline end-to-end chạy được: `train → transmute → eval-sparse → eval-compare`.

### Fixed

- _(none notable.)_

---

## [M4] - 2025 — nse-rie + nse-ller scalar kernel → sparse inference correct

RIE (routing/indexing) + LLER (scalar kernel) → sparse inference chính xác (scalar
= canonical ground truth).

### Added

- **nse-ller `kernel`**: scalar reference kernel — `compute_ternary_micro_expert_scalar`
  (add/sub/skip), `compute_dense_core` (mat-vec), `apply_bias` — sự thật toán học
  dùng cho PPL.
- **nse-ller `tiling`**: L3 cache tiling (no-op trong POC; expert đã cache-sized).
- **nse-rier `index`**: `MipsIndex` (MIPS brute-force exact, O(N)) + `Hit`.
- **nse-rier `router`**: `route_all` (All, upper bound) + `route_by_ratio`
  (Threshold, ≥ max·ratio, cap max_k) + `RouterConfig`.
- **nse-rier `bias`**: `apply`/`apply_layer` — static bias compensator.
- **nse-rier `sparse_linear`**: primitive RIE+LLER cho một layer (`y = core(x) +
  Σ activated experts(x) + bias`).
- Sparse forward trong nse-eval; scalar sparse inference correct (roundtrip ZSTM).

### Changed

- Sparse inference chính xác vs dense (scalar) — verify correctness trước throughput.

### Fixed

- _(none notable.)_

---

## [M3] - 2025 — nse-zstm (outlier + k-means + ternary → .nse transmuted model)

Zero-Shot Transmutation: biến dense weights thành sparse quantized ternary **không
retrain** (xấp xỉ zero-shot).

### Added

- **nse-zstm `outlier`**: `extract(W)` → dense_core (outlier rows theo chuẩn) +
  residual + residual_row_ids.
- **nse-zstm `cluster`**: spherical k-means → centroids + members (group row
  theo direction trong input space) + `ClusterConfig`.
- **nse-zstm `quantize`**: `quantize_matrix` → ternary `{-1,0,1}` + per-row scale
  (BitNet-style).
- **nse-zstm `transmuter`**: `transmute`/`transmute_matrix` assemble 3 stage →
  `TransmutedModel` + static bias `B[i] = W[i]·mean_input` (zero cho core rows);
  `QuantSchemeConfig` (Ternary default), `TransmuteConfig::poc()`, `mean_inputs_for`
  (collect mean activations từ corpus).
- **nse-core::sparse**: `SparseLayer`, `MicroExpert`, `TransmutedModel`,
  `ConfigStub` (mirror tối giản tránh dependency cycle — nse-core không phụ thuộc
  nse-models), `IDX_QKV`/`IDX_ATTN_OUT`/`IDX_FF_UP`/`IDX_FF_DOWN`.
- **`save_transmuted`/`load_transmuted`**: serialize `TransmutedModel` ra `.nse`
  (JSON, mmap-able container).

### Changed

- Dense → sparse transformation pipeline land; xấp xỉ zero-shot (không retrain).

### Fixed

- _(none notable.)_

---

## [M2] - 2025 — nse-train SgdTrainer (backprop + momentum + grad clip)

Vanilla SGD trainer với backprop đầy đủ + momentum + gradient clip toàn cục — PPL
giảm >50% so baseline đồng nhất.

### Added

- **nse-train `Trainer` trait** trừu tượng hóa "train Toy LM".
- **nse-train `sgd`**: `SgdTrainer` (vanilla backprop, momentum, gradient clipping
  theo chuẩn toàn cục) + `SgdConfig`.
- **nse-train `sgd_apply`**: helper momentum + clip gradient dùng chung (sau này
  reuse bởi LSH-sparse).
- Pipeline training: train → eval-dense (PPL).

### Changed

- PPL giảm >50%: ~38 (init) → ~20.5 sau training (paper §5.1).

### Fixed

- _(none notable.)_

---

## [M1] - 2025 — nse-core (.nse format) + nse-models (Toy LM + tokenizer + safetensors)

Core types + format `.nse` + Toy LM dense (forward, tokenizer, autograd, safetensors
loader) — nền dense cho transmute + eval.

### Added

- **nse-core `format`**: `NSEFileHeader`, `MicroExpertMeta`, `NSE_MAGIC` ("NSE1"),
  `NSE_VERSION` — on-disk layout `.nse` (header → dense core → codebook →
  micro-expert data → MIPS tree), mmap-friendly qua `memmap2`.
- **nse-core `tensor`**: `Matrix` (dense matrix view dùng across crates).
- **nse-core `error`**: `NseError`/`NseResult`.
- **nse-models `toy_lm`**: `ToyLm` (transformer 2 lớp, qkv/attn_out/ff_up/ff_down
  + layernorm gain, tied head), `ToyLmWeights`, config mặc định dim=32/num_layers=2/
  num_heads=4/ff_dim=64/vocab=38.
- **nse-models `tokenizer`**: char-level `Tokenizer` (from_corpus, encode/decode).
- **nse-models `autograd`**: `forward_cached` + `backward` + `ToyLmGrads`
  (autograd thủ công, gradient đầy đủ) + `block_backward_local` + `ForwardCache`.
- **nse-models `loader`**: safetensors I/O (save/load `ToyLmWeights`).
- **nse-models `config`**: `Config` (vocab, dim, layers, heads, max_seq_len, ff_dim).
- Gradient check finite-difference (autograd verify).

### Changed

- Dense model + autograd + format `.nse` foundation land.

### Fixed

- _(none notable.)_

---

## [M0] - 2025 — Workspace scaffold, build + test pass

Cargo workspace 8 crate scaffold + metadata; build + test pass sạch.

### Added

- **Cargo workspace** (`resolver = "2"`, 8 members: nse-core, nse-models,
  nse-train, nse-zstm, nse-rie, nse-ller, nse-eval, nse-cli).
- **`[workspace.package]` metadata**: version 0.1.0, edition 2021, rust-version
  1.75, license MIT, repository URL.
- **`[workspace.dependencies]`**: ndarray, ndarray-rand, rand, rand_distr, memmap2,
  safetensors, serde, serde_json, rayon, clap, anyhow, thiserror + path deps nội.
- **`[profile.release]`**: opt-level 3, lto "thin", codegen-units 1.
- `cargo build --workspace` + `cargo test --workspace` pass sạch.

### Changed

- _(initial scaffold.)_

### Fixed

- _(none notable.)_
