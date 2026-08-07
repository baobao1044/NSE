# Plan: Implement AVX2 kernels, HNSW index, and 3 breakthrough training algorithms

Bar đã chốt: **nguyên mẫu chạy được** — mỗi thuật toán chạy end-to-end, objective riêng cải thiện / mô hình không suy biến (PPL < uniform baseline), có test đúng; tích hợp CLI + runtime. Trung thực: FF/Hopfield trên toy LM là prototype nghiên cứu, khó bằng SGD.

## N1 — AVX2 kernels (nse-ller)

**Thực:** `compute_ternary_micro_expert_avx2` + `compute_dense_core_avx2` bằng `core::arch::x86_64::*`:
- Ternary: decode 2-bit codes → 2 mask vectors (`mask_pos`/`mask_neg`, 0xFFFFFFFF để giữ / 0x00000000 để bỏ, tạo bằng `_mm256_cmpgt_ps`/blend). `pos_vals = _mm256_and_ps(x, mask_pos)`, `accum = _mm256_add_ps(accum, pos_vals)`, `accum = _mm256_sub_ps(accum, neg_vals)`. 8 floats/iter, tail scalar theo cùng thứ tự. Cuối horizontal-reduce theo thứ tự vô hướng.
- Dense core: dot product mỗi row bằng `_mm256_fmadd_ps` (FMA), tail scalar.
- Cả hai `#[target_feature(enable = "avx2")]` + `#[cfg(target_arch = "x86_64")]`.

**Wire:** thêm `KernelKind { Scalar, Auto }` vào nse-rie; `sparse_linear` nhận `KernelKind`, gọi `compute_ternary_micro_expert_dispatch` (đã có runtime `is_x86_feature_detected!("avx2")`) khi `Auto`. `sparse_linear` cũ giữ = `Auto`.

**Test:** random expert+x, so AVX2 vs scalar trong tolerance 1e-5 (không bit-identical do FP non-associativity — tài liệu hóa). dense_core AVX2 vs scalar.

## N2 — HNSW index thật (nse-rie/hnsw.rs)

**Thực:** HNSW đồ thị phân tầng:
- Build: layer mỗi node `l ~ floor(-ln(unif)*mL)`, mL=1/ln(M). Chèn: greedy search từ entry point tầng cao xuống, mỗi tầng chọn `M` neighbor gần nhất (distance = `-inner_product`, centroid đã unit-norm từ ZSTM → cosine). Lưu adjacency `Vec<Vec<Vec<u32>>>` (layer→node→neighbors).
- Query: greedy descent tầng >0 (ef=1) → beam search tầng 0 (ef=ef_search) → trả top-K theo score giảm.
- `HnswIndex::new(experts, m, ef_construction, ef_search)`, `query(x, k) -> Vec<Hit>`.

**Wire:** thêm `IndexKind { Brute, Hnsw }`; trait `MipsQuery { fn query_all(&self, x) -> Vec<Hit> }` với 2 impl (MipsIndex, HnswIndex). `sparse_forward` nhận `IndexKind` trong struct `SparseOptions { kernel, index }`.

**Test:** recall@k == 1 vs brute-force trên N nhỏ (centroid unit ngẫu nhiên). Top-K khớp thứ tự.

## N3 — LSH Sparse Training (nse-train/lsh_sparse.rs)

**Gần SGD nhất (dùng backprop):** `forward_cached` + `backward` lấy `ToyLmGrads` đầy đủ, rồi **mask** grad theo LSH.
- LSH: random hyperplanes (dùng `rand`), hash mỗi weight-row của mỗi matmul → bucket. Mỗi step, hash *input activation* của matmul (lấy từ `LayerCache`: `ln1_out` cho qkv, `attn_out` cho attn_out, `ln2_out` cho ff_up, `ff_up_act` cho ff_down) → chọn các row cùng bucket = relevant; zero grad các row còn lại. `sparse_fraction` điều khiển num hashes/buckets.
- Reuse helper `apply` (momentum+clip) của SGD — factor ra `sgd_apply.rs` chung, cả SGD + LSH dùng.
- Thêm module `lsh.rs` (local, dùng `rand`) — không thêm cross-crate dep.

**Test:** mask thực sự thưa (đếm row selected ≈ sparse_fraction); PPL < uniform baseline sau train.

## N4 — Forward-Forward (nse-train/forward_forward.rs)

**Không backprop toàn cục.** Faithful Hinton FF cho Toy LM:
- Mỗi block (layer) có goodness `G = (1/d)·Σ a²` (a = residual-stream output block, đã LN).
- Positive pass: window token thật. Negative pass: window token bị permute (giữ shape).
- Local loss: `softplus(θ − G_pos) + softplus(G_neg − θ)`; cập nhật **chỉ weight của block đó** bằng local gradient `dG/dW` — viết `block_local_forward`/`block_local_backward` tái dùng helper của autograd (layernorm/matmul/attention/gelu) nhưng **một block**.
- Tied head `token_embed`/`ln_f`: thêm goodness head riêng (logits positive=next-token đúng, negative=sai) để embed không suy biến; hoặc để Xavier init + light Hebbian. Tài liệu hóa adaptation.
- Per-layer train tuần tự (layer-wise, song song khả thi).

**Test:** G_pos tăng, G_neg giảm sau train trên tiny corpus; mô hình ra PPL < uniform.

## N5 — Hopfield / Associative Memory (nse-train/hopfield.rs)

**One-shot writes, không backprop.** Dùng FFN làm associative memory:
- `ff_up` [ff_dim, dim] = **key store** (mỗi row 1 key), `ff_down` [dim, ff_dim] = **value store**. Tích `ff_down @ ff_up ≈ M` có `M·k ≈ v` với key trực giao.
- Với mỗi (context key `k`, target value `v`) từ corpus: gán slot `i` (round-robin ff_dim), viết `ff_up[i,:] = k` (normalize), `ff_down[:,i] = v`. Key = activation context (ln2_out), value = embedding token mục tiêu (để retrieval → dự đoán next-token qua tied head).
- `beta` = sharpness; retrieval `z = ff_down @ softmax(β·(ff_up @ k))` (softmax thay GELU cho retrieval chuẩn Hopfield — tài liệu hóa).
- Gains/token_embed: freeze hoặc set analytic.

**Test:** với key đã lưu, retrieval z khớp value (trong tolerance); viết nhiều cặp, recall đúng; mô hình không suy biến.

## N6 — CLI + runtime integration (nse-cli)

- Subcommand mới: `train-ff`, `train-hopfield`, `train-lsh` (mirror `train`, gọi trainer mới, save safetensors).
- `eval-sparse`/`eval-compare`: thêm cờ `--kernel scalar|avx2|auto` và `--index brute|hnsw`, thread qua `SparseOptions` vào `sparse_forward`.
- Chạy end-to-end: train-lsh → eval-compare với --kernel avx2 --index hnsw.

## Test tổng
- 32 test cũ vẫn pass; thêm: AVX2 vs scalar, HNSW recall, FF goodness, Hopfield retrieval, LSH sparsity+PPL.
- Build sạch `cargo build --workspace`; `cargo test --workspace` pass.

## Milestones
- **N1** AVX2 kernels + wire + tests
- **N2** HNSW + IndexKind + tests + wire sparse_forward
- **N3** LSH-sparse trainer (reuses backprop) + factor apply helper + tests
- **N4** Forward-Forward (per-block local goodness) + tests
- **N5** Hopfield (associative writes) + tests
- **N6** CLI (train-ff/hopfield/lsh, --kernel, --index) + end-to-end

## Trung thực về giới hạn
- AVX2 không bit-identical (FP non-associativity) → test trong tolerance, tài liệu hóa.
- FF/Hopfield trên toy LM = prototype nghiên cứu, PPL khó bằng SGD → bar = chạy được + objective cải thiện + không suy biến.
- HNSW recall = 1 trên N nhỏ (giá trị thật ở quy mô lớn) → test nhỏ + tài liệu hóa.