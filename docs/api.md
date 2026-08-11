# NSE Public API

Tài liệu này liệt kê **public API surface** của mỗi crate trong workspace NSE — tức các `pub use`, `pub fn`, `pub struct`, `pub enum` mà code ngoài sẽ gọi. Mỗi mục có mô tả 1 dòng và snippet nhỏ khi cần. Tất cả signature lấy trực tiếp từ source (đừng đoán — xem `crates/*/src/lib.rs` và module re-export).

Workspace có 8 crate:

```text
nse-core    — core types, .nse format, tensor
nse-models  — ToyLm, Tokenizer, Config, autograd, loader
nse-zstm    — Zero-Shot Transmutation (offline): dense -> sparse
nse-rie     — Routing & Indexing Engine (online): route + bias
nse-ller    — Low-Level Execution Runtime (online): scalar + AVX2 kernels
nse-eval    — perplexity + dense-vs-sparse comparison
nse-train   — Trainer trait + 5 trainer (SGD, FF, Hopfield, LSH, Composite)
nse-cli     — binary `nse`, 10 subcommand
```

> Quy ước: `&[f32]` là một slice row-major; `Matrix` (nse-core::tensor) là `{rows, cols, data: Vec<f32>}` row-major. Tất cả PPL tính `exp(mean cross-entropy)`, position `t` predict token `t+1`.

---

## nse-core

Core types chia sẻ giữa ZSTM (sản xuất), RIE+LLER (tiêu thụ), eval (so sánh). Source: `crates/nse-core/src/lib.rs` re-export từ `sparse`, `format`, `tensor`, `error`.

### Public items

- `pub struct ConfigStub` — bản sao tối thiểu của `nse_models::Config` (tránh dependency cycle). Fields: `vocab_size`, `dim`, `num_layers`, `num_heads`, `max_seq_len`, `ff_dim` (toàn `usize`). Convert to/from `Config` qua `From` impl.
- `pub struct PqCodebook` — trained PQ codebook/lớp: `M` sub-codebook, mỗi cái `2^nbits` centroid. Fields:
  - `num_sub_vectors: usize` — số sub-vector `M`.
  - `nbits: usize` — bit/code (8 → 256 entry).
  - `sub_dim: usize` — `= in_dim / M`.
  - `codebook: Vec<f32>` — layout `[subvec m][centroid c][dim j]`.
  - `pub fn num_entries(&self) -> usize` — `2^nbits`.
- `pub struct PqExpertData` — data PQ/expert: codes + per-row scale. Fields: `codes: Vec<u8>` (`[rows*M]`), `row_scales: Vec<f32>`, `num_sub_vectors: usize`. Decode = `scale * concat(reconstructed sub-vectors)`.
- `pub struct MicroExpert` — một micro-expert (group of output rows + ternary codes + centroid route). Fields:
  - `row_ids: Vec<u32>` — output-row index gốc.
  - `ternary: Vec<i8>` — codes `{-1,0,1}`, length `rows*in_dim`, row-major. Unused khi `pq: Some`.
  - `row_scales: Vec<f32>` — per-row scale `s` (BitNet). Unused khi `pq: Some`.
  - `centroid: Vec<f32>` — vector input-space (`in_dim`), router score = `x·centroid`.
  - `mean_input: Vec<f32>` — cache cho bias bookkeeping.
  - `pq: Option<PqExpertData>` (`#[serde(default)]`) — `None` → ternary path; `Some` → PQ path. Backward-compat với file `.nse` cũ.
- `pub struct SparseLayer` — một linear layer `W [out, in]` đã transmute. Fields:
  - `out_dim`, `in_dim: usize`.
  - `dense_core: Matrix` — outlier row FP32 (`[n_core, in]`), always active.
  - `core_row_ids: Vec<u32>`.
  - `experts: Vec<MicroExpert>`.
  - `bias: Vec<f32>` — static bias `B[out]`: `B[i] = W[i]·mean_input` cho prunable row, 0 cho core row.
  - `mean_input: Vec<f32>` — mean activation (`[in]`) qua transmutation corpus.
  - `pq_codebook: Option<PqCodebook>` (`#[serde(default)]`) — shared codebook khi expert nào đó có `pq: Some`.
  - `pub fn num_experts(&self) -> usize`
  - `pub fn covered_rows(&self) -> usize` — total output row (core + expert), nên bằng `out_dim`.
  - `pub fn active_fraction(&self, avg_experts_on: f32) -> f32` — xấp xỉ fraction param active/token.
- `pub struct TransmutedModel` — toàn bộ model đã transmute. Fields: `config: ConfigStub`, `token_embed: Matrix` (dense, kept as-is), `layers: Vec<[SparseLayer; 4]>` (mỗi layer 4 matmul: qkv, attn_out, ff_up, ff_down), `ln1_gain: Vec<Vec<f32>>`, `ln2_gain: Vec<Vec<f32>>`, `ln_f_gain: Vec<f32>`.
- `pub const IDX_QKV: usize = 0` / `IDX_ATTN_OUT = 1` / `IDX_FF_UP = 2` / `IDX_FF_DOWN = 3` — index vào per-layer `[SparseLayer; 4]`.
- `pub struct Matrix` (`nse_core::tensor::Matrix`) — dense matrix view `{rows, cols, data: Vec<f32>}`. Method chính: `Matrix::zeros(rows, cols)`, `Matrix::from_vec(rows, cols, data)`, `.row(i)`, `.transposed()`, `.view()`.
- `pub use error::{NseError, NseResult}` — error/result type (thiserror).
- `pub use format::{MicroExpertMeta, NSEFileHeader, NSE_MAGIC, NSE_VERSION}` — on-disk layout `.nse` (magic `"NSE1"`).

```rust
use nse_core::sparse::{SparseLayer, IDX_FF_UP};

let sl: &SparseLayer = &tm.layers[0][IDX_FF_UP];
println!("ff_up: out={} in={} experts={}", sl.out_dim, sl.in_dim, sl.num_experts());
assert_eq!(sl.covered_rows(), sl.out_dim);
```

---

## nse-models

Model definition + weight loading. POC ship Toy LM (transformer nhỏ, char-level) + tokenizer. Source: `crates/nse-models/src/lib.rs`.

### Public items

- `pub struct Config` (Serialize/Deserialize, PartialEq) — hyperparameter. Fields: `vocab_size`, `dim`, `num_layers`, `num_heads`, `max_seq_len`, `ff_dim` (usize). `Config::toy_default(vocab)` → dim=128, 2 layer, 4 head, max_seq=256, ff=512. `Default` → toy_default(256).
- `pub struct ToyLm` — toy transformer. Fields: `pub config: Config`, `pub weights: ToyLmWeights`.
  - `ToyLm::new(config)` — zero-init.
  - `ToyLm::init_random(config, seed: u64)` — Xavier-ish random init (dùng cho training).
  - `ToyLm::num_params(&self) -> u64`.
  - `ToyLm::forward(&self, tokens: &[u32]) -> Vec<f32>` — logits `[seq, vocab]` row-major; position `t` predict `t+1`. GELU FFN (standard path).
  - `ToyLm::forward_hopfield(&self, tokens: &[u32], beta: f32) -> Vec<f32>` — FFN thay bằng retrieval `ff_down·softmax(β·(ff_up·h2))` (modern Hopfield). Đường Hopfield trainer thiết kế cho.
- `pub struct ToyLmWeights` — weights layout: `token_embed: Matrix`, `ln1_gain: Vec<Vec<f32>>`, `qkv: Vec<Matrix>`, `attn_out: Vec<Matrix>`, `ln2_gain: Vec<Vec<f32>>`, `ff_up: Vec<Matrix>`, `ff_down: Vec<Matrix>`, `ln_f_gain: Vec<f32>`.
- `pub struct Tokenizer` — char-level. Fields: `id_by_byte: [u32; 256]`, `byte_by_id: Vec<u8>`, `vocab_size: usize`. Method:
  - `Tokenizer::byte_level()` — 256-byte vocab.
  - `Tokenizer::from_corpus(corpus: &[u8])` — vocab = byte xuất hiện trong corpus (sắp xếp).
  - `encode(&self, text: &[u8]) -> Vec<u32>`.
  - `decode(&self, ids: &[u32]) -> Vec<u8>`.
- `pub fn forward_cached(lm: &ToyLm, tokens: &[u32]) -> (ForwardCache, Vec<f32>)` (nse_models::autograd) — forward + cache đầy đủ để backward; trả `(cache, logits)`.
- `pub fn backward(lm, cache, targets) -> (f32, ToyLmGrads)` — cross-entropy loss + gradient full backprop.
- `pub fn block_backward_local(...)` — local gradient cho 1 block (dùng Forward-Forward; không flow gradient sang block khác/head).
- `pub struct ForwardCache`, `pub struct LayerCache`, `pub struct ToyLmGrads` — cache/grad buffer (xem `crates/nse-models/src/autograd.rs` cho field đầy đủ; `ToyLmGrads::zeros(&cfg)` init zero).
- `pub mod loader`:
  - `loader::save_toy_lm(path, lm: &ToyLm) -> Result<()>` — serialize `.safetensors` (mỗi matrix 1 tensor f32 + `__config__` JSON tensor).
  - `loader::load_toy_lm(path) -> Result<ToyLm>` — load lại; resilient theo name.

```rust
use nse_models::{Config, ToyLm, Tokenizer, loader};

let corpus = std::fs::read("data/corpus.txt")?;
let tok = Tokenizer::from_corpus(&corpus);
let cfg = Config { vocab_size: tok.vocab_size, dim: 32, num_layers: 2,
                   num_heads: 4, max_seq_len: 32, ff_dim: 64 };
let lm = ToyLm::init_random(cfg, 1337);
loader::save_toy_lm("toy.safetensors", &lm)?;
let lm2: ToyLm = loader::load_toy_lm("toy.safetensors")?;
```

---

## nse-zstm

Zero-Shot Transmutation Module (offline). Convert dense weight → sparse NSE representation **không retrain**, 3 stage: outlier extraction → spherical k-means cluster → quantize. Source: `crates/nse-zstm/src/lib.rs` re-export từ `transmuter` (+ module `outlier`, `cluster`, `pq`, `quantize`).

### Public items

- `pub enum QuantSchemeConfig` — scheme quantize expert weight. Đặt trên `TransmuteConfig`, áp dụng uniform cho mọi matrix.
  - `Ternary` — `{-1,0,1}` + per-row scale (BitNet). Default, backward-compat với file cũ.
  - `Pq { num_sub_vectors: usize, nbits: usize, iters: usize, seed: u64 }` — Product Quantization: row chia `num_sub_vectors` sub-vector, mỗi cái quantize chống shared 8-bit codebook (256 centroid) train/lớp trên residual row chuẩn hóa. `num_sub_vectors=4, nbits=8, iters=20, seed=7` (mặc định CLI).
- `pub struct TransmuteConfig` — full config. Fields: `outlier: OutlierConfig` (`{ fraction: f32 }`), `cluster: ClusterConfig` (`{ num_experts, iters, seed }`), `quant: QuantSchemeConfig`.
  - `TransmuteConfig::poc()` — default tốt cho POC (ternary, outlier fraction 0.1).
  - `TransmuteConfig::pq()` — PQ variant (M=4, 8-bit, 20 iter).
- `pub fn transmute(lm: &ToyLm, corpus: Option<&[u8]>, cfg: &TransmuteConfig) -> anyhow::Result<TransmutedModel>` — transmute toàn bộ Toy LM. `corpus` (nếu có) được tokenize + chạy forward dense để collect mean input activation cho mỗi weight (dùng precompute bias + seed centroid).
- `pub fn transmute_matrix(w: &Matrix, mean_input: &[f32], cfg: &TransmuteConfig) -> anyhow::Result<SparseLayer>` — transmute 1 matrix `W [out, in]` → `SparseLayer`. Branch theo `cfg.quant` (ternary → build_ternary_experts; PQ → build_pq_experts + shared codebook). Bias `B[i]=W[i]·mean_input`, zero cho core row. Scheme-agnostic (dùng W gốc, không dùng quantized form).
- `pub fn save_transmuted(model: &TransmutedModel, path: impl AsRef<Path>) -> anyhow::Result<()>` — serialize JSON (POC container; `#[serde(default)]` cho PQ field → backward-compat).
- `pub fn load_transmuted(path: impl AsRef<Path>) -> anyhow::Result<TransmutedModel>` — load lại từ JSON.

```rust
use nse_zstm::{transmute, save_transmuted, TransmuteConfig, QuantSchemeConfig};

let corpus = std::fs::read("data/corpus.txt")?;
let lm = nse_models::loader::load_toy_lm("toy_lm.safetensors")?;

// Ternary (default)
let tm_tern = transmute(&lm, Some(&corpus), &TransmuteConfig::poc())?;
save_transmuted(&tm_tern, "model.nse")?;

// PQ (Phase 7 / M8)
let tm_pq = transmute(&lm, Some(&corpus), &TransmuteConfig::pq())?;
save_transmuted(&tm_pq, "model_pq.nse")?;
```

Module thấp hơn (dùng khi cần custom): `outlier::extract`, `cluster::cluster`, `quantize::quantize_matrix`, `pq::{train_pq, encode_pq, decode_pq}`.

---

## nse-rie

Routing & Indexing Engine (online). Với mỗi input activation, tìm micro-expert relevant mà không scan toàn model: MIPS index + adaptive threshold router + static bias compensator. Source: `crates/nse-rie/src/lib.rs`.

### Public items

- `pub fn sparse_linear(sl: &SparseLayer, x: &[f32], activated: &[usize]) -> Vec<f32>` — sparse forward 1 layer, kernel `Auto` (backward-compat entry point). `y = core(x) + Σ_activated experts(x) + bias`.
- `pub fn sparse_linear_with_kernel(sl, x, activated, kind: KernelKind) -> Vec<f32>` — explicit kernel. Dispatch per-expert theo scheme: `pq: Some` → PQ kernel (dùng `sl.pq_codebook`); `pq: None` → ternary kernel.
- `pub fn sparse_linear_all(sl, x) -> Vec<f32>` — tất cả expert active (upper bound). Khi đó chỉ còn quantization error, bias zero-on-average.
- `pub enum IndexKind { Brute, Hnsw }` (`Default = Brute`) — MIPS backend. `Brute` = exact O(N); `Hnsw` = approximate O(log N).
- `pub struct MipsIndex<'a>` (`nse_rie::index`) — brute-force exact MIPS. `MipsIndex::new(experts: &[MicroExpert])`, `query_all(x) -> Vec<Hit>` (sort descending), `query_topk(x, k)`.
- `pub struct HnswIndex` (`nse_rie::hnsw`) — HNSW graph. `HnswIndex::new(experts, m, ef_construction, ef_search)`, `query(x, k) -> Vec<Hit>`, `num_experts()`.
- `pub fn build_hnsw_for_layer(sl: &SparseLayer) -> HnswIndex` — build HNSW với POC-friendly default (M=8, ef=32, ef_search=max(16, n_experts)).
- `pub struct Hit` (`nse_rie::index`) — `{ expert_id: usize, score: f32 }`.
- `pub trait MipsQuery { fn query_all(&self, x: &[f32]) -> Vec<Hit>; }` — abstract "trả mọi hit sort descending" để router work với cả backend. Impl cho `MipsIndex` và `HnswIndex`.
- `pub struct RouterConfig` — `{ threshold_ratio: f32, max_k: usize }` (`Default`: 0.5, 64).
- `pub fn route_all(hits: &[Hit]) -> Vec<Hit>` — active mọi expert (upper bound).
- `pub fn route_by_ratio(hits, cfg: &RouterConfig) -> Vec<Hit>` — `θ = max·ratio`, keep `score >= θ`, cap `max_k`. Assume `hits` sort descending.
- `pub use nse_ller::KernelKind` — re-export kernel selector.
- `pub use bias::{apply as apply_bias, apply_layer}` — `apply_bias(bias, output)` cộng bias; `apply_layer(sl, output)` cho cả layer.

```rust
use nse_rie::{MipsIndex, route_all, route_by_ratio, RouterConfig, sparse_linear_with_kernel, KernelKind};

let idx = MipsIndex::new(&sl.experts);
let hits = idx.query_all(x);                  // sort desc
let activated: Vec<usize> = route_by_ratio(&hits, &RouterConfig { threshold_ratio: 0.5, max_k: 16 })
    .iter().map(|h| h.expert_id).collect();
let y = sparse_linear_with_kernel(&sl, x, &activated, KernelKind::Auto);
```

---

## nse-ller

Low-Level Execution Runtime (online, CPU). Execute sparse compute: scalar reference (canonical ground truth) + AVX2 kernels (runtime auto-detect). Source: `crates/nse-ller/src/lib.rs`.

### Public items

- `pub enum KernelKind { Scalar, Avx2, Auto }` (`Default = Auto`) — chọn kernel. `Scalar` = canonical (match dense math exact). `Avx2` = force (panics nếu no AVX2... thực tế dispatch re-check + fallback scalar). `Auto` = AVX2 nếu CPU có, else scalar. Method `use_avx2(self) -> bool`.
- `pub fn compute_dense_core_dispatch(core: &Matrix, row_ids: &[u32], x: &[f32], y: &mut [f32], kind: KernelKind)` — dense-core mat-vec: `y[row_ids[i]] += W[i]·x`. Dispatch AVX2 (FMA) / scalar.
- `pub fn compute_ternary_micro_expert_dispatch(expert: &MicroExpert, x, y: &mut [f32], kind)` — ternary accumulate: `y[row] += scale·Σ_j ternary[j]·x[j]`. AVX2 dùng mask add/sub/skip.
- `pub fn compute_pq_micro_expert_dispatch(expert, x, y: &mut [f32], codebook: &PqCodebook, kind)` — PQ: decode M code → centroid, dot với x sub-vector, FMA accumulate. `expert.pq` phải `Some`.
- `pub fn apply_bias(bias: &[f32], y: &mut [f32])` — cộng static bias vào output.
- Scalar reference (canonical, dùng cho PPL): `compute_dense_core`, `compute_ternary_micro_expert_scalar`, `compute_pq_micro_expert_scalar` (trong `nse_ller::kernel`).

> **Lưu ý numerical:** AVX2 kernel **không bit-identical** với scalar (SIMD reduction + FMA thay FP rounding), nhưng agree trong `~1e-5` relative — dưới noise floor PPL POC. `KernelKind::Scalar` = ground truth.

```rust
use nse_ller::{compute_ternary_micro_expert_dispatch, KernelKind};

let mut y = vec![0.0f32; sl.out_dim];
for &eid in activated {
    compute_ternary_micro_expert_dispatch(&sl.experts[eid], x, &mut y, KernelKind::Auto);
}
nse_ller::apply_bias(&sl.bias, &mut y);
```

---

## nse-eval

Perplexity + dense-vs-sparse comparison — headline metric NSE POC. Source: `crates/nse-eval/src/lib.rs`.

### Public items

- `pub enum Activation` (`Default = All`) — cách active expert trong sparse forward.
  - `All` — mọi expert (upper bound; chỉ còn quantization error).
  - `Threshold { ratio: f32, max_k: usize }` — adaptive threshold routing.
- `pub struct SparseOptions` (`Default: Scalar + Brute`) — runtime option. Fields: `kernel: nse_rie::KernelKind`, `index: nse_rie::IndexKind`.
- `pub fn sparse_forward(tm: &TransmutedModel, tokens: &[u32], act: Activation) -> Vec<f32>` — sparse forward, default option (scalar + brute). Mirror dense forward, 4 matmul/layer thay bằng `sparse_linear`.
- `pub fn sparse_forward_with_options(tm, tokens, act, opts: SparseOptions) -> Vec<f32>` — explicit kernel + index.
- `pub fn sparse_forward_hopfield(tm, tokens, beta: f32, act) -> Vec<f32>` / `sparse_forward_hopfield_with_options(...)` — FFN thay bằng Hopfield retrieval trên reconstructed ternary key/value store (§5.6 research path).
- `pub fn dense_ppl(lm: &ToyLm, ids: &[u32], seq_len: usize) -> f32` — PPL dense qua sliding window.
- `pub fn dense_ppl_hopfield(lm, ids, seq_len, beta) -> f32` — PPL dense dưới Hopfield retrieval FFN.
- `pub fn sparse_ppl(tm, ids, seq_len, act) -> f32` / `sparse_ppl_with_options(tm, ids, seq_len, act, opts)` — PPL sparse.
- `pub fn sparse_ppl_hopfield(...)` / `sparse_ppl_hopfield_with_options(...)` — PPL sparse + Hopfield FFN.
- `pub fn logprobs(logits, targets, vocab) -> Vec<f32>` / `perplexity_from_logprobs(lp) -> f32` — primitive PPL.
- `pub struct CompareReport` — `{ ppl_dense, ppl_sparse, rel_degradation, avg_active_fraction, activation_mode }`. `pretty() -> String` in report.
- `pub fn compare(lm, tm, corpus, seq_len, act) -> CompareReport` — dense vs sparse, default option.
- `pub fn compare_with_options(lm, tm, corpus, seq_len, act, opts) -> CompareReport` — explicit kernel + index (dùng CLI `--kernel`/`--index`).
- `pub struct CompositeReport` — 4-path: `{ dense_gelu, dense_hopfield, sparse_gelu, sparse_hopfield, avg_active_fraction, activation_mode }`. `pretty()` in 4-path report với degrade tương đối. Headline artifact §5.6.
- `pub fn compare_composite(lm, tm, corpus, seq_len, beta, act, opts) -> CompositeReport` — chạy cả 4 forward path (dense/sparse × GELU/Hopfield).

```rust
use nse_eval::{compare_with_options, Activation, SparseOptions, nse_rie::{KernelKind, IndexKind}};

let opts = SparseOptions { kernel: KernelKind::Auto, index: IndexKind::Brute };
let report = compare_with_options(&lm, &tm, &corpus, 16, Activation::All, opts);
println!("{}", report.pretty());
// PPL dense  : 13.8220
// PPL sparse : 48.5718
// degradation: +251.41%
```

---

## nse-train

Training backend. POC có vanilla `SgdTrainer` (real backprop) + 3 research trainer (Forward-Forward, Hopfield, LSH-sparse) + composite "hippocampus + cortex". Source: `crates/nse-train/src/lib.rs`.

### Public items

- `pub trait Trainer` — `fn name(&self) -> &'static str` + `fn train(&mut self, model: &mut ToyLm, corpus: &[u8]) -> anyhow::Result<()>`. Implement bởi mọi trainer.

#### SgdTrainer (dense baseline, real backprop)

- `pub struct SgdConfig` — `{ learning_rate, seq_len, epochs, lr_decay, log_every, seed }`. Default: lr=0.3, seq=32, epochs=200, decay=0.995, seed=1337.
- `pub struct SgdTrainer { config: SgdConfig, momentum: f32, max_grad_norm: f32, .. }` — SGD momentum + global-norm clip. `SgdTrainer::new(config)` (momentum=0.9, max_grad_norm=1.0).

#### ForwardForwardTrainer (Hinton FF, local goodness, no global backprop)

- `pub enum Homeostasis { None, LayerNorm }` (`Default = None`) — goodness normalization. `None` = raw `G=mean(y²)` (cần `weight_clip`); `LayerNorm` = standardize G (experimentally fails, kept cho repro).
- `pub struct ForwardForwardConfig` — `{ learning_rate, seq_len, epochs, lr_decay, momentum, max_grad_norm, hebbian_embed_lr, theta_ema, weight_clip, homeostasis, log_every, seed }`. `weight_clip` sweet spot 0.5 (§5.4).
- `pub struct ForwardForwardTrainer` — `new(config)`. Per-block local goodness, light Hebbian head.

#### HopfieldTrainer (modern Hopfield, one-shot writes, no backprop)

- `pub struct HopfieldConfig` — `{ seq_len, num_writes, beta, value_scale, log_every, seed }`. Default: seq=16, writes=64, beta=8.0, value_scale=0.1.
- `pub struct HopfieldTrainer` — `new(config)`. Write `(context → next-token direction)` vào FFN store round-robin; key L2-normalized; retrieval bằng cosine.
- `pub fn hopfield_retrieve(...)` — helper retrieval (xem `crates/nse-train/src/hopfield.rs:193`).

#### LshSparseTrainer (dense backprop + per-row LSH gradient mask)

- `pub struct LshSparseConfig` — `{ learning_rate, seq_len, epochs, lr_decay, sparse_fraction, momentum, max_grad_norm, log_every, seed }`. `sparse_fraction` (e.g. 0.01 = 1% row update/step) → `num_bits = round(log2(1/frac))`. Default: lr=0.05, epochs=40, frac=0.01.
- `pub struct LshSparseTrainer` — `new(config)`. Dense backprop + mask gradient theo LSH bucket.

#### CompositeTrainer (hippocampus + cortex, 4 phase)

- `pub struct CompositeConfig` — `{ sgd_warm: SgdConfig, hopfield: HopfieldConfig, ff: ForwardForwardConfig, lsh: LshSparseConfig, eval_seq_len, eval_beta, log_every }`. Mỗi phase skip khi count=0. Default (§5.4.2): FF warm-start + LSH fine-tune là synthesis; SGD warm + Hopfield off mặc định.
- `pub struct CompositeTrainer` — `new(config)`. Chạy 4 phase tuần tự: SGD warm (stabilizer) → Hopfield writes (hippocampus) → FF (plasticity) → LSH (routing + sparse update). Log PPL giữa phase.

- `pub mod sgd_apply` — helper `apply_step` (momentum update + clip) chia sẻ bởi SGD + LSH.

```rust
use nse_train::{SgdTrainer, SgdConfig, Trainer};

let mut trainer = SgdTrainer::new(SgdConfig {
    learning_rate: 0.05, seq_len: 16, epochs: 30,
    lr_decay: 1.0, log_every: 0, seed: 7,
});
trainer.max_grad_norm = 1.0;
trainer.train(&mut lm, &corpus)?;
```

---

## nse-cli (binary `nse`)

Binary CLI, không phải library. 10 subcommand, mỗi cái sinh artifact trung gian. Source: `crates/nse-cli/src/cli.rs`. Run dạng `cargo run --release --quiet -p nse-cli -- <subcommand>` (binary `nse` có thể chưa trên PATH). Tất cả flag có `--long`, default trong ngoặc.

### `nse train` — SGD baseline → `toy_lm.safetensors`

| Flag | Default | Mô tả |
|------|---------|------|
| `--corpus` | `data/corpus.txt` | corpus text |
| `--out` | `toy_lm.safetensors` | output model |
| `--dim` | 32 | hidden dim |
| `--layers` | 2 | transformer layer |
| `--heads` | 4 | attention head |
| `--seq-len` | 32 | context length |
| `--ff-dim` | 64 | FFN intermediate |
| `--epochs` | 80 | SGD epoch |
| `--lr` | 0.05 | learning rate |

### `nse train-ff` — Forward-Forward → `toy_lm_ff.safetensors`

| Flag | Default | Mô tả |
|------|---------|------|
| `--corpus`, `--out`, `--dim`, `--layers`, `--heads`, `--seq-len` (16), `--ff-dim`, `--epochs` (60), `--lr` (0.02) | (như train) | config FF |
| `--hebb-lr` | 0.01 | Hebbian head lr |
| `--weight-clip` | 1.0 | per-weight max-norm clamp (FF stabilize; 0 disable) |
| `--homeostasis` | `none` | `none` (raw G, cần clip) hoặc `layernorm` (standardize G — fails, kept repro) |

### `nse train-hopfield` — associative writes → `toy_lm_hop.safetensors`

| Flag | Default | Mô tả |
|------|---------|------|
| `--corpus`, `--out`, `--dim`, `--layers`, `--heads`, `--seq-len` (16), `--ff-dim` | (như train) | config |
| `--num-writes` | 64 | slot/layer write round-robin |
| `--beta` | 8.0 | retrieval sharpness |
| `--value-scale` | 0.1 | scale value written (sau unit-normalize) |

### `nse train-lsh` — LSH-sparse → `toy_lm_lsh.safetensors`

| Flag | Default | Mô tả |
|------|---------|------|
| `--corpus`, `--out`, `--dim`, `--layers`, `--heads`, `--seq-len` (16), `--ff-dim`, `--epochs` (40), `--lr` (0.05) | (như train) | config |
| `--init` | (none) | optional warm-start: load model thay vì random init |
| `--sparse-fraction` | 0.01 | fraction row update/step |

### `nse train-composite` — hippocampus + cortex → `toy_lm_comp.safetensors`

| Flag | Default | Mô tả |
|------|---------|------|
| `--corpus`, `--out`, `--dim`, `--layers`, `--heads`, `--seq-len` (16), `--ff-dim` | (như train) | config |
| `--sgd-epochs` | 0 | Phase 1: SGD warm (0 skip) |
| `--hopfield-writes` | 0 | Phase 2: Hopfield writes/layer (0 skip) |
| `--ff-epochs` | 15 | Phase 3: FF epoch (0 skip) |
| `--lsh-epochs` | 15 | Phase 4: LSH fine-tune (0 skip) |
| `--ff-clip` | 0.5 | FF max-norm clamp (§5.4 sweet spot) |
| `--lsh-frac` | 0.01 | LSH sparse-fraction |
| `--eval-beta` | 8.0 | β cho between-phase PPL probe |

### `nse eval-dense` — baseline PPL

| Flag | Default | Mô tả |
|------|---------|------|
| `--corpus` | `data/corpus.txt` | corpus |
| `--model` | `toy_lm.safetensors` | trained model |
| `--seq-len` | 16 | sliding window |
| `--forward` | `gelu` | `gelu` (standard) hoặc `hopfield` (softmax retrieval) |
| `--beta` | 8.0 | retrieval sharpness (chỉ `--forward hopfield`) |

### `nse transmute` — dense → sparse `.nse`

| Flag | Default | Mô tả |
|------|---------|------|
| `--corpus` | `data/corpus.txt` | corpus (collect mean input) |
| `--model` | `toy_lm.safetensors` | dense model |
| `--out` | `model.nse` | output sparse |
| `--outlier-fraction` | 0.1 | fraction outlier row → dense core |
| `--quant` | `ternary` | `ternary` (default) hoặc `pq` (Phase 7/M8) |
| `--pq-subvectors` | 4 | số PQ sub-vector `M` (chỉ `--quant pq`; `in_dim` chia hết cho `M`, else fallback ước số lớn nhất ≤ M) |
| `--pq-nbits` | 8 | bit/code (chỉ `--quant pq`; 8 → 256 centroid) |

### `nse eval-sparse` — sparse PPL

| Flag | Default | Mô tả |
|------|---------|------|
| `--corpus` | `data/corpus.txt` | corpus |
| `--nse` | `model.nse` | transmuted model |
| `--seq-len` | 16 | sliding window |
| `--mode` | `all` | `all` (all expert, upper bound) hoặc `threshold` |
| `--threshold-ratio` | 0.5 | `θ = max·ratio` (chỉ `--mode threshold`) |
| `--max-k` | 16 | cap số expert (chỉ `--mode threshold`) |
| `--kernel` | `auto` | `scalar` / `avx2` / `auto` (LLER backend) |
| `--index` | `brute` | `brute` (exact MIPS) / `hnsw` (approximate) |

### `nse eval-compare` — dense vs sparse report (headline)

| Flag | Default | Mô tả |
|------|---------|------|
| `--corpus` | `data/corpus.txt` | corpus |
| `--model` | `toy_lm.safetensors` | dense model |
| `--nse` | `model.nse` | transmuted model |
| `--seq-len` | 16 | sliding window |
| `--kernel` | `auto` | `scalar` / `avx2` / `auto` |
| `--index` | `brute` | `brute` / `hnsw` |

### `nse eval-composite` — 4-path report (§5.6 artifact)

| Flag | Default | Mô tả |
|------|---------|------|
| `--corpus` | `data/corpus.txt` | corpus |
| `--model` | `toy_lm_comp.safetensors` | dense model |
| `--nse` | `model.nse` | transmuted model |
| `--seq-len` | 16 | sliding window |
| `--beta` | 8.0 | Hopfield retrieval sharpness (dense + sparse retrieval path) |
| `--kernel` | `auto` | `scalar` / `avx2` / `auto` |
| `--index` | `brute` | `brute` / `hnsw` |

> `--kernel`/`--index` chỉ đổi *cách* tính sparse matmul, không đổi kết quả (chỉ FP noise). `--quant` chọn scheme khi transmute; `eval-sparse`/`eval-compare`/`eval-composite` auto-detect scheme từ `MicroExpert.pq` → không cần flag mới cho PQ.
