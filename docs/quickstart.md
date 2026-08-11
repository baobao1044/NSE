# Quickstart — Neuro-Sparse Engine (NSE)

NSE (Neuro-Sparse Engine) là một Rust framework chạy LLM trên CPU/Edge mà không cần GPU cluster. Quickstart này dẫn bạn từ `git clone` đến lần inference thưa (sparse) đầu tiên, qua hai track: **Fast path** (dim=32, ternary) và **PQ path** (dim=64, so sánh ternary vs Product Quantization).

Toàn bộ pipeline chạy trên `data/corpus.txt` (38-token Shakespeare excerpt đi kèm repo). Mỗi subcommand sinh ra một artifact trung gian để debug riêng từng stage.

> Toàn bộ lệnh dưới đây dùng dạng `cargo run --release --quiet -p nse-cli -- <subcommand>` vì binary `nse` có thể chưa nằm trên `PATH`. Dùng `--release` vì build debug quá chậm cho POC pipeline (lựa `--quiet` để bớt noise compile). Nếu muốn cài binary trực tiếp: `cargo install --path crates/nse-cli` (sau đó `nse <subcommand>`), hoặc alias:
>
> ```bash
> alias nse="cargo run --release --quiet -p nse-cli --"
> ```

## Yêu cầu

- Rust 1.75+ (`rustup` ổn định).
- CPU x86_64; AVX2 được auto-detect ở runtime (không có thì fallback scalar, vẫn chạy).
- Repo đã clone và `cd` vào thư mục gốc:

```bash
git clone <repo-url> NSE-master
cd NSE-master
```

Kiểm tra corpus đi kèm:

```bash
ls data/corpus.txt   # data/corpus.txt
```

Build toàn bộ workspace (8 crate) ở release:

```bash
cargo build --workspace --release
```

```text
   Compiling nse-core v0.1.0 (...)
   Compiling nse-models v0.1.0 (...)
   Compiling nse-ller v0.1.0 (...)
   Compiling nse-rie v0.1.0 (...)
   Compiling nse-zstm v0.1.0 (...)
   Compiling nse-eval v0.1.0 (...)
   Compiling nse-train v0.1.0 (...)
   Compiling nse-cli v0.1.0 (...)
    Finished `release` profile [optimized] target(s) in ~1m 12s
```

Pipeline tổng quan (mỗi mũi tên là một artifact trung gian):

```text
nse train      -> toy_lm.safetensors   (SgdTrainer, dense baseline)
nse transmute  -> model.nse            (ZSTM: outlier + k-means + quantize)
nse eval-compare -> report            (PPL_dense | PPL_sparse | % degradation)
```

---

## Track 1 — Fast path (dim=32, ternary)

Track mặc định: train Toy LM ở `dim=32`, transmute sang sparse ternary `{-1,0,1}`, rồi so sánh PPL dense vs sparse. Đây là đường ngắn nhất đến một con số headline.

### Bước 1: train dense model

```bash
cargo run --release --quiet -p nse-cli -- train
```

Mặc định: `--corpus data/corpus.txt`, `--out toy_lm.safetensors`, `--dim 32`, `--layers 2`, `--heads 4`, `--seq-len 32`, `--ff-dim 64`, `--epochs 80`, `--lr 0.05`.

```text
Training Toy LM: 39 vocab, 32 dim, 2 layers
[epoch 0/80] mean_loss=2.1678 ppl=8.74 lr=0.05000
[epoch 19/80] mean_loss=1.9797 ppl=7.24 lr=0.05000
[epoch 39/80] mean_loss=1.8886 ppl=6.61 lr=0.05000
[epoch 59/80] mean_loss=1.8189 ppl=6.17 lr=0.05000
[epoch 79/80] mean_loss=1.8832 ppl=6.57 lr=0.05000
Saved trained model to toy_lm.safetensors
```

### Bước 2: transmute sang sparse NSE format

```bash
cargo run --release --quiet -p nse-cli -- transmute
```

Mặc định: `--model toy_lm.safetensors`, `--out model.nse`, `--outlier-fraction 0.1`, `--quant ternary`, `--pq-subvectors 4`, `--pq-nbits 8`.

```text
Quantization scheme: ternary ({-1,0,1} + per-row scale)
Transmuting dense model -> sparse NSE format
Saved transmuted model to model.nse
```

### Bước 3: so sánh dense vs sparse PPL

```bash
cargo run --release --quiet -p nse-cli -- eval-compare
```

Mặc định: `--model toy_lm.safetensors`, `--nse model.nse`, `--seq-len 16`, `--kernel auto`, `--index brute`.

```text
=== NSE POC: Dense vs Sparse PPL ===
PPL dense  : 13.8220
PPL sparse : 48.5718
degradation: +251.41%
avg active : 100.0000% of params/token
activation : all-experts (upper bound)
```

`degradation` dương = sparse tệ hơn dense (đúng kỳ vọng: zero-shot transmutation có cost). `activation: all-experts` nghĩa là mọi micro-expert đều active (upper bound) — khi đó phần degradation chỉ còn là **quantization error** (ternary `{-1,0,1}` chỉ 3 level), còn bias thì zero-on-average vì không có expert nào bị prune.

### What just happened

Bạn vừa chạy end-to-end pipeline NSE: SgdTrainer (momentum + global-norm gradient clip) train Toy LM (char-level tokenizer, 2-layer transformer với GELU FFN, tied head) xuống PPL ~6.6 (vượt uniform baseline 39). ZSTM transmute dense weights đó — không retrain — qua ba stage: outlier extraction (10% row biên độ cao giữ dense FP32), spherical k-means clustering các residual row thành micro-expert, rồi ternary quantize `{-1,0,1}` + per-row scale (BitNet-style); kèm static bias `B[i] = W[i]·mean_input` bù lại đóng góp trung bình của expert bị prune. Eval-compare chạy forward thưa (RIE route + LLER compute) trên cùng sliding window, so PPL. Vì toàn bộ expert active, sparse PPL = dense PPL + quantization error + bias residual — con số +251% trên toy dim=32 phản ánh ternary quá thô cho trọng lượng Gaussian-centered (đó chính là động lực cho PQ ở Track 2).

---

## Track 2 — PQ path (dim=64, ternary vs Product Quantization)

Track này tái hiện thí nghiệm §5.7 của paper: train dim=64, transmute hai cách (ternary baseline vs Product Quantization), rồi so degradation. Mục tiêu: chứng minh PQ codebook 8-bit (256 level/sub-vector, học được) phục hồi chất lượng thưa tốt hơn ternary (3 level).

### Bước 1: train dim=64 model

```bash
cargo run --release --quiet -p nse-cli -- train --dim 64 --ff-dim 128 --epochs 20 --out lm64.safetensors
```

```text
Training Toy LM: 39 vocab, 64 dim, 2 layers
[epoch 0/20] mean_loss=2.0698 ppl=7.92 lr=0.05000
[epoch 19/20] mean_loss=1.9377 ppl=6.94 lr=0.05000
Saved trained model to lm64.safetensors
```

### Bước 2: transmute hai cách

Ternary baseline:

```bash
cargo run --release --quiet -p nse-cli -- transmute --model lm64.safetensors --out lm64_tern.nse --quant ternary
```

```text
Quantization scheme: ternary ({-1,0,1} + per-row scale)
Transmuting dense model -> sparse NSE format
Saved transmuted model to lm64_tern.nse
```

PQ (M=4 sub-vectors, 8-bit codebook):

```bash
cargo run --release --quiet -p nse-cli -- transmute --model lm64.safetensors --out lm64_pq.nse --quant pq --pq-subvectors 4
```

```text
Quantization scheme: PQ (M=4 sub-vectors, 8-bit codebook)
Transmuting dense model -> sparse NSE format
Saved transmuted model to lm64_pq.nse
```

`--pq-subvectors 4` chia mỗi row thành 4 sub-vector (sub_dim = in_dim/4 = 16 cho dim=64). `--pq-nbits 8` (mặc định) → 256 centroid/sub-codebook. Codebook chia sẻ/layer, train offline trên residual row chuẩn hóa, < 1 MB → L3 cache (spec). Nếu `in_dim` không chia hết cho `M`, NSE fallback về ước số lớn nhất `≤ M` (không panic).

### Bước 3: so sánh dense vs sparse cho cả hai

Ternary:

```bash
cargo run --release --quiet -p nse-cli -- eval-compare --model lm64.safetensors --nse lm64_tern.nse
```

```text
=== NSE POC: Dense vs Sparse PPL ===
PPL dense  : 16.8208
PPL sparse : 27.1859
degradation: +61.62%
avg active : 100.0000% of params/token
activation : all-experts (upper bound)
```

PQ:

```bash
cargo run --release --quiet -p nse-cli -- eval-compare --model lm64.safetensors --nse lm64_pq.nse
```

```text
=== NSE POC: Dense vs Sparse PPL ===
PPL dense  : 16.8208
PPL sparse : 23.0333
degradation: +36.93%
avg active : 100.0000% of params/token
activation : all-experts (upper bound)
```

### Kết quả reference (paper §5.7) và giải thích

Bảng chuẩn §5.7 (paper, `dim=64`, SGD 20 epoch, all-experts, seed kiểm soát):

| Path | PPL | Degradation vs dense |
|------|-----|---------------------|
| Dense (SGD) | 13.531 | — |
| Sparse ternary (all) | 17.973 | **+32.8%** |
| Sparse PQ (all, M=4) | 16.017 | **+18.4%** |

PQ **giảm degradation gần một nửa** (+32.8% → +18.4%); `pq/ternary = 0.891` (PQ thắng ~11% trên sparse PPL). Đây là kết quả thật trên cùng mô hình SGD dim=64, cùng corpus, cùng activation (all-experts) — chỉ khác quantization scheme.

> **Lưu ý về số tuyệt đối:** số bạn vừa chạy qua CLI (+61.62% ternary, +36.93% PQ) khác số paper §5.7 (+32.8% / +18.4%) vì CLI hardcode `seed=1337` cho `ToyLm::init_random` và `SgdTrainer` (xem `crates/nse-cli/src/cli.rs`), trong khi bảng §5.7 dùng `seed=7` (thí nghiệm kiểm soát trong `crates/nse-eval/src/compare.rs::sparse_pq_lower_degradation`). Mô hình toy dim=64 nhạy với seed (chỉ ~50 residual row train codebook). **Pattern định tính vẫn reproduces**: PQ giảm degradation đáng kể so với ternary (ở đây 36.93% vs 61.62%, tức PQ thắng ~40% trên sparse PPL). Để reproduce chính xác số §5.7, chạy test đó trực tiếp: `cargo test --release sparse_pq_lower_degradation -p nse-eval -- --nocapture`.

Tại sao PQ thắng? Ternary `{-1,0,1}` chỉ 3 level — quá thô cho trọng lượng Gaussian-centered. PQ codebook 8-bit có 256 level/sub-vector, học được trên phân phối thật thay vì cố định sign, nên reconstruction MSE thấp hơn ~1.10x (xem `bench_pq`). Per-row scale `s = mean(|w|)` (BitNet-style) giữ magnitude chính xác, codebook chỉ học *shape* trên row chuẩn hóa `w/s`.

Trung thực về giới hạn (paper §5.7.5): trên toy dim=64, chỉ ~50 residual row train 256 centroid → nhiều cluster trống, codebook kém fit; giá trị PQ ở đây là **lower bound**, dự kiến mở rộng trên dim ≥ 128. Bar kỳ vọng (<15% degradation) chưa đạt (PQ +18.4%); calibration + bias-adaptive (Phase 8) là foundation tiếp theo.

### What just happened

Bạn đã tái hiện thí nghiệm phục hồi chất lượng thưa (Phase 7 / M8): cùng một dense model dim=64, transmute bằng hai quantization scheme khác nhau. Ternary baseline (+32.8% / +61.62%) phản ánh chi phí của 3-level quantization; PQ codebook 8-bit học được (+18.4% / +36.93%) phục hồi gần nửa degradation nhờ 256 level/sub-vector fit phân phối trọng lượng thật. Kernel tự auto-detect scheme qua `MicroExpert.pq: Some|None` — `eval-compare`/`eval-sparse`/`eval-composite` không cần flag mới để chọn PQ vs ternary. Mọi thứ vẫn all-experts (upper bound); giá trị PQ ở accuracy (→ ít expert cần active → net win end-to-end), không phải raw kernel speed (xem `docs/examples.md` cho benchmark PQ vs ternary).

---

## Bước tiếp theo

- **`docs/api.md`** — public API surface của 8 crate (struct, function signature, usage snippet).
- **`docs/examples.md`** — 3 cargo example: `bench_avx2`, `bench_pq`, `bench_hnsw` với output thật.
- **`paper/PAPER.md` §5.7** — phân tích đầy đủ PQ codebook, kernel benchmark, limitation.
- Thử các trainer thay thế: `train-ff` (Forward-Forward), `train-hopfield` (associative memory), `train-lsh` (LSH-sparse), `train-composite` (hippocampus + cortex). Xem `nse <subcommand> --help` cho flag đầy đủ.
- Thử routing threshold: `eval-sparse --mode threshold --threshold-ratio 0.5 --max-k 16` (prune expert dưới θ, bù bằng bias).
- Thử backend: `--kernel scalar|avx2|auto`, `--index brute|hnsw`.
