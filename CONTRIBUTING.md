# Contributing to NSE (Neuro-Sparse Engine)

Cảm ơn bạn quan tâm đến NSE — một prototype nghiên cứu bằng Rust khảo sát khả năng
chạy LLM trên CPU/Edge không cần GPU cluster. Tài liệu này mô tả cách thiết lập môi
trường, phong cách code kỳ vọng, cách mở rộng từng phần, và triết lý "kết quả
trung thực" mà repo tuân theo.

Repository: <https://github.com/local/nse>

---

## 1. Dev setup

### Rust toolchain

NSE yêu cầu Rust **1.75** (xem `rust-version` trong `Cargo.toml`) với edition
**2021**. Cài đặt qua [rustup](https://rustup.rs/) nếu chưa có:

```bash
rustup default 1.75.0   # hoặc mới hơn tương thích
rustc --version
```

### Build & test

```bash
cargo build --workspace                    # build toàn bộ 8 crate
cargo test --workspace                     # chạy toàn bộ test suite (56 test)
cargo run --example bench_pq -p nse-ller    # benchmark PQ kernel + reconstruction MSE
```

**Lưu ý về thời gian test**: `cargo test --workspace` ở debug build chạy khoảng
**~40 phút** vì có nhiều test training (SGD/FF/Hopfield/LSH-sparse/Composite) và
`sparse_pq_lower_degradation` (~18 phút mỗi cái ở debug). Cho iteration nhanh:

```bash
cargo test --release --workspace           # release build nhanh hơn nhiều cho training test
cargo test -p nse-zstm                      # chạy test theo crate để lặp nhanh
cargo test -p nse-core                      # core không có test chậm
```

Nếu chỉ muốn verify correctness của một path cụ thể, chạy test theo tên:

```bash
cargo test -p nse-ller compute_ternary      # chỉ test ternary kernel
cargo test -p nse-zstm transmute_pq         # chỉ test PQ transmute roundtrip
```

### Chạy pipeline end-to-end

```bash
cargo run --release -- train --epochs 10 --out lm.safetensors
cargo run --release -- eval-dense --model lm.safetensors
cargo run --release -- transmute --model lm.safetensors --out lm.nse
cargo run --release -- eval-compare --model lm.safetensors --nse lm.nse
cargo run --release -- train-composite --out lm_comp.safetensors
cargo run --release -- eval-composite --model lm_comp.safetensors --nse lm_comp.nse
```

Xem `paper/PAPER.md` phụ lục "Reproduce" cho đầy đủ lệnh.

---

## 2. Code style

NSE là prototype nghiên cứu, không production, nhưng vẫn giữ một số quy ước:

- **Match surrounding code**: phong cách đặt tên, mật độ comment, và cấu trúc
  module phải khớp với code xung quanh. Mỗi crate có doc comment `//!` ở đầu
  `lib.rs` mô tả vai trò — đọc nó trước khi sửa.
- **Naming**: `snake_case` cho item, `PascalCase` cho type/struct/enum, `SCREAMING_SNAKE` cho constant (theo Rust convention). Tên hàm mô tả hành động
  (`compute_ternary_micro_expert_scalar`, `route_by_ratio`).
- **Comment density**: codebase comment dày, kiểu "research log" — giải thích
  *tại sao* chọn cách này, đặc biệt cho quyết định lượng tử hóa/routing. Khi
  thêm code, comment tương tự; khi sửa code có comment, cập nhật comment theo.
- **`#![allow(dead_code)]`**: một số crate (nse-models, nse-train, nse-zstm,
  nse-rie, nse-ller, nse-eval) đặt `#![allow(dead_code)]` ở đầu `lib.rs` vì đây
  là POC — nhiều item giữ lại cho phase tối ưu hiệu năng sau (scaffold) chưa được
  caller dùng. **Đừng xóa attribute này** và đừng báo "unused" qua clippy như
  lỗi; nó là intentional. Nếu thêm item mới thực sự không dùng và không phải
  scaffold, vẫn giữ attribute crate-level (không thêm `#[allow(dead_code)]` per
  item trừ khi có lý do).
- **Scalar kernel = canonical ground truth**: trong `nse-ller`, kernel vô hướng
  (`kernel.rs`) là **sự thật toán học** dùng cho PPL. AVX2 (`avx2.rs`) là tối ưu
  hiệu năng và **phải match scalar trong tolerance 1e-5** (không bit-identical do
  tính kết hợp của dấu chấm động — xem `paper/PAPER.md` §6.1). Khi sửa kernel,
  luôn giữ scalar chính xác trước, rồi verify AVX2 match.

---

## 3. How to extend

### (a) Thêm một trainer mới

Implement trait `Trainer` trong `nse-train/src/trainer.rs`, đặt module mới trong
`nse-train/src/<tên>.rs` và re-export từ `lib.rs`. Trainer nhận `ToyLm` (hoặc
warm-start từ `init`), corpus, config riêng; chia sẻ helper `sgd_apply`
(momentum + clip gradient theo chuẩn toàn cục) nếu cần. Thêm test: (1) gradient
hoặc update thực sự thay đổi weight đúng cách, (2) PPL < baseline đồng nhất (không
suy biến). Cuối cùng, thêm subcommand CLI trong `nse-cli/src/cli.rs` nếu trainer
có flag riêng. Xem `sgd.rs` (backprop đầy đủ) và `lsh_sparse.rs` (gần SGD nhất,
che gradient theo LSH) làm mẫu.

### (b) Thêm một kernel mới

Thêm hàm vô hướng trong `nse-ller/src/kernel.rs` (canonical), rồi bản AVX2 trong
`avx2.rs` với `#[target_feature(enable="avx2")]` + dispatch runtime
(`is_x86_feature_detected!("avx2")`) + fallback vô hướng. AVX2 **phải match
scalar trong tolerance 1e-5** — viết test so sánh trên matrix ngẫu nhiên nhiều
size. Cuối cùng, wire dispatch qua `KernelKind` và các hàm
`*_dispatch` trong `avx2.rs`. Pattern chuẩn: xem `compute_ternary_micro_expert_*`
(scalar: add/sub/skip theo mask pos/neg; AVX2: mask vector 0xFFFFFFFF/giữ 0,
`_mm256_and_ps`, horizontal-reduce theo thứ tự vô hướng) và
`compute_dense_core_*` (AVX2 FMA `_mm256_fmadd_ps`).

### (c) Thêm một quant scheme mới

Mở rộng enum `QuantSchemeConfig` trong `nse-zstm/src/transmuter.rs` (thêm variant
với config riêng), thêm hàm `build_<scheme>_experts` (song song với
`build_ternary_experts` / `build_pq_experts`) gọi trong nhánh `match cfg.quant`
của `transmute_matrix`. Nếu scheme cần codebook/data per-expert, thêm struct
data trong `nse-core/src/sparse.rs` với `#[serde(default)]` trên `Option` field
của `MicroExpert`/`SparseLayer` để backward-compat với `model.nse` cũ (xem
`PqExpertData`/`PqCodebook` làm mẫu). Wire kernel: thêm `compute_<scheme>_micro_expert_scalar` trong `nse-ller/src/kernel.rs`, bản AVX2 trong `avx2.rs`, và
nhánh dispatch trong `sparse_linear_with_kernel` (`nse-rie/src/lib.rs`) theo
`MicroExpert.pq`-tương-đương. Cuối cùng, thêm flag `--quant <scheme>` + các
flag config trong CLI `transmute`. Test: roundtrip serde + covered_rows =
out_dim + PPL so với dense.

---

## 4. Triết lý "kết quả trung thực"

NSE là prototype nghiên cứu, và **giá trị cốt lõi là tính trung thực**, không phải
hype. Một số nguyên tắc:

- **Kết quả âm (negative results) cũng có giá trị** — tài liệu hóa chúng, không
  giấu. Xem `paper/PAPER.md` §5.4 (FF homeostasis LayerNorm fail — phân tích tại
  sao chuẩn hóa mất "hướng" pos/neg), §5.4.3 (Hopfield trên dense GELU không cải
  thiện PPL — mismatch kiến trúc), §5.6.2 (sparse Hopfield trên ternary keys phá
  cosine structure), và §5.7.5 (giới hạn PQ: toy model sub-tối ưu cho codebook,
  PQ gather không luôn nhanh hơn ternary).
- **Đừng fudge số liệu**. Nếu một experiment cho PPL tệ hơn baseline, báo cáo
  đúng vậy và giải thích tại sao — đó là *kết quả*, không phải lỗi. Nếu bar kỳ
  vọng không đạt (ví dụ PQ <15% degradation, đạt +18.4%), nói rõ "chưa đạt" và
  dự kiến phase tiếp theo.
- **POC values correctness + honesty over hype**: kernel vô hướng là ground
  truth, AVX2/HNSW/PQ verify correctness trước, throughput sau. Số liệu PPL tuyệt
  đối trên toy LM (dim=32/64) không đại diện cho mô hình lớn — giá trị ở tính
  *tương đối*.
- Khi viết commit message, PR description, hoặc doc, mô tả **cả thành công lẫn
  giới hạn**. Ví dụ mẫu: "Composite thắng từng trainer riêng (21.44 vs FF 26.04)
  nhưng không thắng SGD (12.37) — đúng kỳ vọng, tài liệu hóa."

---

## 5. PR process

1. **Branch off `master`**: tạo branch mô tả, ví dụ
   `feat/pq-calibration` hoặc `fix/avx2-ternary-tail`.
2. **Small focused PR**: một PR giải quyết một vấn đề (một trainer, một kernel,
   một fix). Tránh PR trộn nhiều thay đổi không liên quan.
3. **`cargo test --workspace` phải pass** trước khi PR. Nếu test chậm, dùng
   `cargo test --release --workspace` hoặc chạy theo crate rồi verify full suite
   trước merge.
4. **PR description**: mô tả **cái gì đổi + tại sao**. Nếu là experiment, báo
   cả kết quả (PPL, speedup, recall) và giới hạn. Nếu là negative result, nói rõ
   đó là negative và giải thích. Link issue nếu có.
5. **Không tự ý bump version/scope**: NSE ở `0.1.0`; milestone M0–M8 là đơn vị
   release (xem `CHANGELOG.md`). Thảo luận trước khi tạo milestone mới.
6. **Không commit artifact**: `*.nse`, `*.safetensors`, `target/` đã trong
   `.gitignore`. Chỉ commit source + doc + test.

---

## 6. Testing notes

- **Test chậm (~18 phút mỗi cái ở debug)**: `sparse_pq_lower_degradation` (nse-eval)
  và các training test (SGD, FF, Hopfield, LSH-sparse, Composite). Dùng
  `cargo test --release` hoặc chạy theo crate (`cargo test -p nse-zstm`,
  `cargo test -p nse-core`) cho iteration nhanh.
- **FF test margin mỏng**: trên một số platform (Rust 1.97/Linux) separation
  `G_pos − G_neg` trên held-out windows có thể bị noise flip (margin mỏng, xem
  PAPER §5.4). Assertion dùng tolerance 0.05 thay vì strict `>`; bar chính là
  `PPL < uniform baseline` (không suy biến), đúng tinh thần paper.
- **AVX2 test**: so sánh scalar vs AVX2 trên matrix ngẫu nhiên, tolerance 1e-5.
  AVX2 không bit-identical (associativity dấu chấm động) — đây là expected, không
  phải bug.
- **HNSW test**: `recall@k = 1` vs brute-force trên đồ thị nhỏ nhờ
  `ensure_connected_layer0` (BFS + kết nối nút cô lập). Trên N lớn tradeoff
  recall/latency mới thể hiện — benchmark `bench_hnsw` cho số liệu thật.
- **Khi thêm test**: đặt test trong module `#[cfg(test)] mod tests` cùng file
  (pattern codebase), hoặc `tests/` integration test nếu cần cross-crate. Test
  training phải verify PPL < baseline đồng nhất (không suy biến) — đây là bar
  tối thiểu cho mọi trainer.

---

## 7. Cấu trúc repo

```
Cargo.toml              workspace (8 crate)
crates/
  nse-core/    types, errors, .nse format (mmap), sparse structs
  nse-models/  Toy LM, tokenizer, autograd, safetensors loader
  nse-train/   Trainer trait + SGD/FF/Hopfield/LSH-sparse/Composite
  nse-zstm/    Zero-Shot Transmutation: outlier + k-means + ternary/PQ
  nse-rie/     Routing & Indexing: MIPS brute/HNSW + router + bias
  nse-ller/    Low-Level Execution: scalar + AVX2 kernels
  nse-eval/    PPL dense/sparse + Hopfield forward + compare reports
  nse-cli/     `nse` binary: train / transmute / eval subcommands
data/           corpus (data/corpus.txt)
docs/           inference spec + training vision
paper/          PAPER.md (báo cáo nghiên cứu)
```

Xem `ARCHITECTURE.md` cho deep-dive kiến trúc, và `paper/PAPER.md` cho kết quả
thực nghiệm + phân tích failure mode.
