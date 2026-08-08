# Neuro-Sparse Engine (NSE): Chạy LLM trên CPU/Edge không cần GPU cluster

**Một prototype nghiên cứu bằng Rust**

> Mã nguồn: <https://github.com/baobao1044/NSE>

---

## Tóm tắt

Báo cáo này mô tả **Neuro-Sparse Engine (NSE)** — một framework thử nghiệm bằng
Rust nhằm khảo sát khả năng chạy mô hình ngôn ngữ (LM) trên CPU/edge mà không cần
GPU cluster lớn. NSE gồm ba trục: (1) **ZSTM** — "biến đổi không-zero-shot"
(zero-shot transmutation) một LM dày đặc thành dạng thưa (sparse) lượng tử hóa
ternary; (2) **RIE** — định tuyến & chỉ mục (routing & indexing) bằng HNSW để
chỉ kích hoạt một tập con các chuyên gia (experts) mỗi token; (3) **LLER** —
kernel AVX2/SIMD cấp thấp để đánh giá các nhân thưa với chi phí nhỏ. Bên cạnh
đường ống suy luận (inference), chúng tôi triển khai **ba thuật toán huấn luyện
đột phá** để khảo sát việc huấn luyện không phụ thuộc backprop toàn cục quy mô
lớn: Forward-Forward (Hinton), bộ nhớ liên tưởng Hopfield, và huấn luyện thưa
LSH.

Toàn bộ hệ thống được phát triển như một prototype chạy end-to-end: huấn luyện →
lượng tử hóa → suy luận thưa → so sánh PPL. Chúng tôi báo cáo số liệu thực trên
một toy LM (transformer 2 lớp, dim=32, vocab=38) và trung thành về giới hạn: FF
và Hopfield trên toy LM là prototype nghiên cứu, khó bằng SGD; AVX2 không
bit-identical với vô hướng do tính chất kết hợp của dấu chấm động.

---

## 1. Giới thiệu

Việc huấn luyện và chạy các LLM quy mô lớn đòi hỏi cụm GPU đáng kể. Câu hỏi đặt ra:
có thể chạy — và một mức độ nào đó, huấn luyện — mô hình trên phần cứng thưa/CPU/edge?
NSE tiếp cận vấn đề này ở cấp độ prototype, thử nghiệm các kỹ thuật sau:

- **Tách biệt suy luận khỏi huấn luyện dày đặc**: sau khi một LM dày đặc được
  huấn luyện (bằng phương pháp thông thường), ZSTM "biến đổi" nó sang dạng
  thưa lượng tử hóa ternary {-1, 0, +1} với scale theo hàng (kiểu BitNet),
  không cần huấn luyện lại. Suy luận sau đó chỉ kích hoạt một tập con các hàng
  chuyên gia mỗi token.
- **Định tuyến hiệu quả**: để chọn tập con chuyên gia nhanh, RIE xây dựng chỉ
  mục HNSW (Hierarchical Navigable Small World) trên các centroid chuyên gia
  và truy vấn MIPS (maximum inner-product search) thay vì quét brute-force.
- **Kernel cấp thấp**: LLER triển khai kernel AVX2 thực cho dot product dày đặc
  và đánh giá chuyên gia ternary, với dispatch runtime và fallback vô hướng.
- **Thuật toán huấn luyện thay thế**: để khảo sát huấn luyện không backprop toàn
  cục, chúng tôi triển khai Forward-Forward (goodness cục bộ theo khối), bộ nhớ
  liên tưởng Hopfield (ghi một lần, không backprop), và huấn luyện thưa LSH
  (backprop dày đặc + che gradient theo hàng theo LSH).

Tiêu chuẩn chất lượng đã thống nhất: **prototype chạy được** — chạy end-to-end,
mỗi thuật toán có mục tiêu riêng cải thiện hoặc ít nhất không suy biến (PPL <
baseline đồng nhất), test đúng, có CLI. Báo cáo này trung thành về việc FF và
Hopfield trên toy LM khó bằng SGD.

---

## 2. Kiến trúc

NSE là một Cargo workspace 8 crate:

```
nse-core     — tensor, định dạng .nse (mmap), mô hình thưa TransmutedModel
nse-models   — Toy LM (transformer), tokenizer, autograd thủ công, safetensors I/O
nse-zstm     — Zero-Shot Transmutation: outlier + k-means + lượng tử hóa ternary
nse-rie      — Routing & Indexing: router, MIPS brute-force, HNSW
nse-ller     — Low-Level Execution: kernel vô hướng + AVX2
nse-eval     — PPL dày đặc/thưa, báo cáo so sánh
nse-train    — SGD, Forward-Forward, Hopfield, LSH-sparse
nse-cli      — CLI `nse` với các subcommand
```

### 2.1 Toy LM

Mô hình thử nghiệm là một transformer nhỏ với token embedding (tied head), các
khối gồm: qkv [3·dim, dim], attn_out [dim, dim], ff_up [ff_dim, dim],
ff_down [dim, ff_dim], layernorm gain ln1/ln2/ln_f. Cấu hình mặc định:
dim=32, num_layers=2, num_heads=4, ff_dim=64, vocab=38 (char-level tokenizer).
Autograd thủ công (`forward_cached` + `backward`) cung cấp gradient đầy đủ với
kiểm tra gradient finite-difference.

### 2.2 ZSTM — Biến đổi không-zero-shot

Từ một Toy LM dày đặc, ZSTM trích xuất các hàng ngoại lệ (outlier) theo chuẩn,
gom cụm phần còn lại bằng k-means cầu (spherical), và lượng tử hóa ternary mỗi
centroid với scale theo hàng. Kết quả là `TransmutedModel` với mỗi khối chứa 4
lớp thưa (qkv, attn_out, ff_up, ff_down), mỗi lớp gồm "core" dày đặc (hàng
ngoại lệ) + các chuyên gia ternary (centroid). Lưu dưới định dạng `.nse` (JSON,
mmap-able). Việc biến đổi **không huấn luyện lại** — đây là xấp xỉ zero-shot.

### 2.3 RIE — Định tuyến & chỉ mục

Mỗi token, router tính điểm cho mỗi chuyên gia (MIPS giữa activation đầu vào
và centroid chuyên gia). Hai chế độ kích hoạt: `All` (mọi chuyên gia, upper
bound) và `Threshold` (giữ chuyên gia có điểm ≥ max·ratio, cap max_k). Hai
backend chỉ mục: brute-force (MIPS chính xác) và **HNSW** (xấp xỉ).

#### HNSW thực

`HnswIndex` triển khai đồ thị phân tầng thật: cấp mỗi nút `l ~ floor(−ln(u)·mL)`,
mL = 1/ln(M); chèn bằng greedy descent (ef=1) tầng cao + beam search
(ef_construction) mỗi tầng, liên kết bidirectional M-neighbor với pruning. Truy
vấn: greedy descent xuống tầng 1, beam search tầng 0 (ef_search), trả top-K theo
điểm giảm. `ensure_connected_layer0` (BFS từ entry point, kết nối các nút không
đạt được với nút gần nhất) đảm bảo recall@k = 1 trên đồ thị nhỏ.

### 2.4 LLER — Kernel AVX2

Hai kernel thực bằng `core::arch::x86_64::*`:

- **`compute_ternary_micro_expert_avx2`**: giải mã ternary thành 2 mask vector
  (mask_pos/mask_neg, 0xFFFFFFFF giữ / 0 bỏ), `pos = _mm256_and_ps(x, mask_pos)`,
  `accum = add(accum, pos) − neg`, 8 float/iter, tail vô hướng cùng thứ tự,
  horizontal-reduce theo thứ tự vô hướng.
- **`compute_dense_core_avx2`**: dot product mỗi hàng bằng `_mm256_fmadd_ps` (FMA).

Cả hai `#[target_feature(enable="avx2")]` + dispatch runtime
`is_x86_feature_detected!("avx2")` với fallback vô hướng. `KernelKind { Scalar,
Avx2, Auto }`.

---

## 3. Thuật toán huấn luyện đột phá

### 3.1 LSH Sparse Training (N3)

**Gần SGD nhất (dùng backprop):** `forward_cached` + `backward` lấy `ToyLmGrads`
đầy đủ, rồi **che** gradient theo hàng theo LSH. LSH (random hyperplane): băm
mỗi vector activation đầu vào của matmul thành bucket; một hàng trọng số `r`
giữ gradient chỉ khi `r mod num_buckets` là bucket bị activation trúng. Với
`num_bits = round(log2(1/sparse_fraction))`, khoảng `sparse_fraction` hàng được
cập nhật mỗi bước. Helper `apply_step` (momentum + clip gradient theo chuẩn toàn
cục) dùng chung với SGD.

Test: mask thực sự thưa (≤4/32 hàng active); PPL < baseline đồng nhất.

### 3.2 Forward-Forward (N4)

**Không backprop toàn cục.** Faithful Hinton FF cho toy LM transformer:

- **Goodness** mỗi khối `G = (1/N)·Σ y²`, `y = ln2_in + ff_down_out` (output
  residual stream, **không** layer-norm trước khi bình phương — LN sẽ pin
  G≡1 làm objective suy biến).
- **Positive**: cửa sổ token thật. **Negative**: cùng cửa sổ nhưng token bị
  permute (giữ shape, phá thứ tự).
- **Loss**: `softplus(θ − G_pos) + softplus(G_neg − θ)`. θ per-block, khởi tạo =
  G_pos ban đầu, cập nhật EMA để theo dõi năng lượng khối tăng (θ cố định làm
  dương bão hòa và ngừng học).
- **Gradient cục bộ**: chỉ trọng số của khối đó được cập nhật, qua
  `block_backward_local` (không gradient lan ra khối trước hoặc head — input
  khối coi như đóng băng cho cập nhật cục bộ).
- **Tied head**: head/embedding không nằm trong objective FF; thêm quy tắc
  Hebb nhẹ (nudge embedding token kế tiếp về biểu diễn context) để head không
  suy biến. `ln_f_gain` giữ init.

**Ổn định**: objective FF không có box constraint; năng lượng khối tăng vô
hạn → residual stream phình → NaN. Max-norm clamp (`weight_clip`, mặc định 1.0)
sau mỗi bước chặn divergence — kỹ thuật chuẩn để ổn định FF.

Test: `block_backward_local` cho grad khác 0 cho khối đó, 0 cho khối khác/head;
G_pos > G_neg (trung bình trên nhiều cửa sổ) + PPL < baseline đồng nhất.

### 3.3 Hopfield / Associative Memory (N5)

**Ghi một lần, không backprop.** FFN làm khoá liên tưởng:

- `ff_up` [ff_dim, dim] = **key store** (hàng i = key i), `ff_down` [dim, ff_dim]
  = **value store** (cột i = value i).
- Quy tắc retrieval: `z = ff_down · softmax(β·(ff_up·k))` (softmax thay GELU cho
  retrieval Hopfield chuẩn — adaptation đã tài liệu hóa).
- Ghi: cho mỗi (context, next-token) từ corpus, slot i (round-robin ff_dim):
  key = activation context (ln2_out, L2-normalized), value = hướng target
  (unit-norm × `value_scale`).
- Head/gains: đóng băng tại init.

`hopfield_retrieve` export cho test/caller verify recall trực tiếp.

Test: retrieval key đã lưu trả về ~value (trong tolerance); ghi nhiều cặp, recall
đúng; mô hình không suy biến (PPL hữu hạn < baseline đồng nhất trên corpus test).

---

## 4. CLI

CLI `nse` với pipeline:

```
nse train          — huấn luyện SGD baseline → safetensors
nse train-ff       — huấn luyện Forward-Forward
nse train-hopfield — ghi bộ nhớ liên tưởng Hopfield
nse train-lsh      — huấn luyện LSH-sparse
nse transmute      — ZSTM → .nse
nse eval-dense     — PPL dày đặc
nse eval-sparse    — PPL thưa (--kernel scalar|avx2|auto, --index brute|hnsw)
nse eval-compare   — báo cáo so sánh dày đặc/thưa
```

Chạy end-to-end: `train-lsh → transmute → eval-sparse(--kernel avx2 --index hnsw)
→ eval-compare`.

---

## 5. Kết quả thực nghiệm

**Môi trường**: Windows x64, Rust 1.96, debug build. Corpus: "To be, or not to
be..." (Shakespeare Hamlet soliloquy, 14 dòng, 607 byte, vocab 38 ký tự). Toy LM
dim=32, 2 lớp, 4 heads, ff_dim=64.

### 5.1 Bảng PPL — các trainer

| Trainer          | PPL (dense eval) | Baseline đồng nhất | So baseline   |
|------------------|-----------------:|-------------------:|--------------:|
| (không train)    | ~38.0 (init)     | 38                 | ~1.0×         |
| SGD (10 epochs)  | **20.50**        | 38                 | **0.54×** ✓   |
| LSH-sparse (15)  | **25.21**        | 38                 | **0.66×** ✓   |
| Forward-Forward (30, clipped) | **30.24** | 38          | **0.80×** ✓   |
| Hopfield (64 writes) | 53.18       | 38                 | 1.40× ✗       |

**Diễn giải**: SGD (backprop đầy đủ) tốt nhất. LSH-sparse (backprop + che theo
hàng) gần SGD — chỉ che gradient mà vẫn học. FF (goodness cục bộ, không backprop
toàn cục) đánh bại baseline nhưng kém SGD — đúng kỳ vọng prototype. Hopfield
trên dense-forward (GELU) **không** cải thiện PPL — đúng giới hạn đã tài liệu hóa:
retrieval Hopfield cần forward-path softmax, không tương thích với GELU dày đặc;
test retrieval riêng cho thấy recall đúng.

### 5.2 Bảng PPL — suy luận thưa (từ mô hình SGD)

| Backend                    | PPL (sparse, all-experts) |
|----------------------------|-------------------------:|
| Dense (baseline)           | 20.50                    |
| Sparse, scalar, brute      | 37.29                    |
| Sparse, AVX2, brute        | 37.29                    |
| Sparse, AVX2, HNSW         | 37.29                    |

**Diễn giải**: PPL thưa = 37.29 (degradation +82% so dense) phản ánh **chi phí
lượng tử hóa ternary** — đây là upper bound (all-experts, chỉ lỗi lượng tử hóa,
không pruning). Quan trọng: **scalar/AVX2/HNSW cho cùng PPL** (chính xác) — đúng
như mong đợi: kernel và index chỉ thay đổi *cách* đánh giá, không kết quả (chỉ
FP noise). AVX2 match vô hướng, HNSW match brute (recall=1 trên đồ thị nhỏ nhờ
`ensure_connected_layer0`).

### 5.3 Test suite

`cargo test --workspace` pass sạch (0 failure). Bao gồm:
- Gradient check finite-difference (autograd)
- AVX2 vs vô hướng trong tolerance 1e-5 (không bit-identical)
- HNSW recall@k = 1 vs brute-force
- LSH mask thưa + PPL < baseline
- FF goodness phân tách (G_pos > G_neg) + PPL < baseline
- Hopfield retrieval khớp value + PPL hữu hạn
- ZSTM roundtrip, RIE routing, định dạng .nse

### 5.4 Phân tích failure mode: FF vs Hopfield

Số liệu 5.1 cho thấy FF (PPL 27.35) đánh bại baseline đồng nhất (38) nhưng kém
SGD (20.50), còn Hopfield (53.18) thậm chí kém baseline. Câu hỏi quan trọng:
đó là **thiếu năng lực** của thuật toán, hay một **lỗi có thể sửa**? Để phân
biệt, chúng ta sweep tham số ổn định của từng trainer và xem PPL có cải thiện
thêm hay chỉ "đứng yên ở mức ổn định".

**FF — sweep `weight_clip`** (max-norm clamp, 30 epochs):

| clip | G_pos | G_neg | margin (G_pos−G_neg) | G_neg/G_pos | PPL |
|----:|------:|------:|--------------------:|------------:|----:|
| 0.2 | 3.34 | 3.31 | 0.035 | 0.990 | 32.19 |
| **0.5** | 1.74 | 1.71 | **0.027** | **0.985** | **27.35** |
| 0.7 | 1.89 | 1.89 | 0.001 | 1.000 | 28.73 |
| 1.0 | 1.67 | 1.67 | 0.002 | 0.999 | 30.50 |
| 2.0 | 23.20 | 23.20 | 0.0001 | 1.0000 | 32.20 |
| 3.0 | 3186.7 | 2958.5 | 228.1 | 0.928 | 20548.6 |

Đường cong PPL theo `weight_clip` có **dạng U rõ ràng**, cực tiểu tại 0.5. Đây
là dấu hiệu của một hệ có **tín hiệu học thật**: quá chặt (0.2) → underfit, quá
lỏng (≥1.0) → exploit, vừa đủ (0.5) → học được. Quan sát chìa khóa: khi clip lỏng,
G_pos và G_neg **tăng cùng nhau theo cấp số nhân** (1.67 → 23 → 3186) nhưng
margin (G_pos−G_neg) **suy giảm về ~0** (0.0001 tại clip=2.0, ratio 1.0000). Tức
là mạng đang **"hét to" cả positive lẫn negative** thay vì *tách* chúng — nó
hack objective `softplus(θ−G_pos)+softplus(G_neg−θ)` bằng cách đẩy cả hai G ra xa
θ, không phải bằng cách làm G_pos > G_neg. Clip chặn exploit này, lộ năng lực
học thật (PPL 27.35). Khoảng cách tới SGD (20.50) ≈ 33% — chưa thắng, nhưng
**hoàn toàn không phải "FF vô dụng"**.

Điều này gợi ý FF nguyên bản thiếu cơ chế sinh học: neuron được "tự quyết định
goodness" nhưng không có giới hạn mức năng lượng — clip đóng vai trò **ức chế
sinh học / homeostasis**. FF có thể hợp với sparse training: điểm yếu backprop
trong sparse setting là weight inactive → gradient không tới, trong khi FF tự
đánh giá goodness cục bộ không cần gradient toàn cục — nhưng cần thêm
normalization/bounded activation để tránh tự kích hoạt.

#### 5.4.1 LayerNorm homeostasis — một negative result có ý nghĩa

Câu hỏi tự nhiên: clip cố định là "phanh tay"; não có **homeostasis tự thích ứng**
(firing-rate / synaptic scaling). Có thể thay clip bằng chuẩn hóa goodness?
Chúng ta thử variant `Homeostasis::LayerNorm`: `Ĝ = (G − run_mean)/(run_std + ε)`,
θ = run_mean (EMA Welford) — chuẩn hóa G trước softplus để objective thưởng
*separation* (G_pos trên chuẩn, G_neg dưới) thay vì magnitude.

Kết quả: **LayerNorm fail** — G_pos ≈ G_neg ở mọi epoch (ratio 1.0000), mạng
không học phân tách. Lý do rõ ràng: khi cả G_pos và G_neg cùng được chuẩn hóa
chống cùng running stats, hai gradient của softplus `(0−Ĝ_pos)` và `(Ĝ_neg−0)`
trở nên **đối xứng và triệt tiêu** — mạng không nhận được signal *hướng*
(pos vs neg). Đây là rủi ro của chuẩn hóa mất thông tin hướng, đúng cảnh báo.
Insight: **homeostasis phải giữ hướng (pos vs neg), không chỉ rescale
magnitude**. Clip giữ hướng (chỉ chặn magnitude) nên work; LayerNorm mất hướng
nên fail. Hướng đúng là percentile θ (θ = percentile cao của G_pos history,
không chuẩn hóa G) — giữ hướng + ngăn phình, nhưng cần history buffer, để
Phase tiếp theo. Variant `LayerNorm` giữ lại trong code (`--homeostasis
layernorm`) để reproduce negative result này.

#### 5.4.2 LSH + FF hybrid (warm-start) — tổng hợp thắng từng phần

Phân tích 5.4 chỉ ra FF (goodness cục bộ, cần ức chế) và LSH-sparse (backprop +
che gradient theo locality) giải đúng vấn đề của nhau: FF cung cấp *plasticity*
(học cục bộ, không cần gradient toàn cục), LSH cung cấp *locality* (biết "đi
đâu"). Thí nghiệm warm-start: huấn luyện FF (15 epochs, clip 0.5 sweet spot) →
fine-tune LSH-sparse (15 epochs), so với LSH thuần (random init) cùng tổng
compute 30 epochs:

| Cấu hình | Tổng epochs | PPL |
|---|---:|---:|
| LSH thuần (random init) | 30 | 19.49 |
| **Hybrid (FF warm 15 + LSH 15)** | 30 | **18.26** (−1.23, −6%) |

Hybrid thắng LSH thuần ~6% khi cùng tổng compute: FF warm-start đặt model ở
basin tốt hơn random init (goodness cục bộ cho plasticity), LSH-sparse
fine-tune (che gradient theo locality) tận dụng điểm khởi đầu đó. Objective
exploit của FF bị tránh bằng cách "hand-off" sang LSH sớm; "biết đi đâu, chưa
biết học thế nào" của LSH được FF warm-start cung cấp "how". Đây là bằng chứng
cho ý tưởng kiến trúc tổng hợp: các vai trò tách rời (memory/routing/learning)
có thể cộng tác.

#### 5.4.3 Hopfield retrieval forward — mismatch hypothesis xác nhận

5.4 kết luận Hopfield cần forward-path softmax (thay GELU). Thí nghiệm: huấn
luyện Hopfield (CLI corpus, regime non-trivial, PPL 53 > uniform 38) rồi eval
cùng model dưới hai forward path:

| Forward path | PPL |
|---|---:|
| GELU (standard dense) | 62.40 |
| Hopfield retrieval (β=8) | **50.86** (−11.54, −18%) |
| Hopfield retrieval (β=4) | 50.97 |
| Hopfield retrieval (β=16) | 50.83 |

Hopfield retrieval thắng GELU **11.5 PPL (−18%)** — **mismatch hypothesis xác
nhận**: writes của Hopfield được thiết kế cho `ff_down · softmax(β·(ff_up·k))`,
GELU (`gelu(h2·ff_up)·ff_down`) không match cơ chế retrieval. β không nhạy
(4/8/16 → ~50.8). Lưu ý: PPL vẫn > uniform (38) — forward-path đúng cải thiện
nhưng chưa đủ để vượt baseline, vì Hopfield writes là memory lookup (recall
đúng) chứ không phải representation learner (generalize yếu). Điều này củng cố
kết luận 5.4: Hopfield làm *hippocampus* (memory retrieval) tốt hơn *cortex*
(representation learning) — cần kết hợp với learning pathway (FF/LSH) để có
giá trị đầy đủ.

**Hopfield — sweep `value_scale`** (num_writes=64):

| value_scale | PPL |
|---:|---:|
| 0.05 | 63.62 |
| 0.1 | 62.40 |
| 0.3 | 59.70 |
| 0.5 | 58.90 |

PPL giảm nhẹ theo value_scale (63.62 → 58.90) nhưng **đơn điệu, không có sweet
spot**, và **mọi giá trị đều > baseline đồng nhất (38)**. Đây là pattern rất
khác FF: không có cực tiểu → không có "lộ năng lực" khi nới tham số. Vấn đề
không phải "value quá lớn" mà là **retrieval mechanism sai loại** — Hopfield
hiện đại cần `query → similarity → normalized retrieval (softmax) → value
mixture`, nhưng dense eval dùng `raw activation → GELU → value injection`, thiếu
cạnh tranh giữa các ký ức và xác suất retrieval. value_scale chỉ nudge biên độ
residual, không sửa được sự không tương thích căn bản này. **Hopfield cần
forward-path riêng** (thay GELU bằng softmax retrieval trong forward) mới có khả
năng cải thiện PPL — đây là giới hạn kiến trúc, không phải tuning.

**Phân biệt hai loại failure**: phân tích này tách rõ hai failure mode rất khác:
(i) FF = **vấn đề objective/optimization** → đã cứu được một phần bằng ức chế
sinh học (clip); (ii) Hopfield = **mismatch kiến trúc** → cần thay đổi nguyên
lý, không phải tham số. Cái nhìn này có giá trị khoa học hơn vài điểm PPL: nó
chỉ ra rằng "local learning không đủ mạnh" chưa phải kết luận — đúng hơn là
"local learning cần cơ chế chống tự kích hoạt".

**Gợi ý kiến trúc tổng hợp**: thay vì ép Hopfield thành trainer, NSE có thể đi
theo phân vai trò giống thần kinh học (routing/learning/memory tách rời như
basal ganglia / cortex / hippocampus): `token → Hopfield retrieval (normalized,
chọn ký ức) → LSH route (tìm nhanh) → FF learning (học pathway, có clip) →
sparse update`. SGD giữ vai trò pretraining stabilizer. Đây là hướng mở, chưa
triển khai.

---

## 6. Giới hạn & thảo luận trung thành

1. **AVX2 không bit-identical**: do tính kết hợp (associativity) của dấu chấm
   động, kết quả AVX2 khác vô hướng ở mức FP noise. Test trong tolerance 1e-5,
   tài liệu hóa trong code. Giá trị thật của AVX2 ở quy mô lớn (throughput), test
   nhỏ chỉ verify correctness.

2. **HNSW recall = 1 trên N nhỏ**: nhờ `ensure_connected_layer0` (BFS + kết nối
   nút cô lập). Giá trị thật (tradeoff recall/latency) chỉ thể hiện ở quy mô lớn;
   test nhỏ verify correctness của graph + search, không throughput.

3. **FF/Hopfield là prototype nghiên cứu, PPL khó bằng SGD**:
   - FF: goodness cục bộ + Hebb head đánh bại baseline đồng nhất nhưng kém SGD.
     Goodness `G = mean(y²)` (raw energy) cần max-norm clamp để ổn định; separation
     G_pos/G_neg mỏng trên corpus nhỏ. Đây là khám phá ý tưởng, không trainer sản
     xuất.
   - Hopfield: retrieval (softmax) không tương thích dense-forward (GELU) —
     PPL không cải thiện trên eval-dense. Test retrieval riêng cho recall đúng.
     Để tận dụng đầy đủ cần forward-path Hopfield riêng (thay GELU bằng softmax
     retrieval trong eval) — hướng mở.

4. **Toy LM nhỏ**: dim=32, vocab=38, 2 lớp. Số liệu PPL tuyệt đối không đại diện
     cho mô hình quy mô lớn; giá trị ở **tính tương đối** (trainer nào cải thiện,
     kernel nào chính xác, pipeline chạy end-to-end).

5. **LSH-sparse**: che gradient giảm FLOPs update nhưng trên toy model nhỏ saving
     tuyệt đối khiêm tốn; chất lượng phụ thuộc activation clusterable.

---

## 7. Kết luận

NSE triển khai end-to-end một pipeline prototype: huấn luyện → biến đổi thưa →
suy luận thưa (AVX2 + HNSW) → so sánh PPL, với ba thuật toán huấn luyện thay
thế. Kết quả chính:
- Pipeline chạy được, PPL đo được, test đúng.
- AVX2 + HNSW cho kết quả chính xác như vô hướng/brute (correctness verified).
- LSH-sparse (backprop + che) gần SGD; FF đánh bại baseline đồng nhất; Hopfield
  recall đúng nhưng không cải thiện dense PPL (giới hạn đã tài liệu hóa).
- Trung thành: FF/Hopfield là prototype nghiên cứu, không production.

Hướng mở: forward-path Hopfield thực (softmax thay GELU trong eval), HNSW ở quy
mô lớn, AVX2 throughput benchmark, LSH-sparse trên model lớn hơn.

---

## Phụ lục: Reproduce

```bash
# Train SGD baseline + so sánh dày đặc/thưa
nse train --epochs 10 --out lm.safetensors
nse transmute --model lm.safetensors --out lm.nse
nse eval-compare --model lm.safetensors --nse lm.nse --kernel avx2 --index hnsw

# Trainers thay thế
nse train-lsh --epochs 15 --out lm_lsh.safetensors
nse train-ff --epochs 30 --out lm_ff.safetensors
nse train-hopfield --num-writes 64 --out lm_hop.safetensors

# Test toàn workspace
cargo test --workspace
```

Mã nguồn: <https://github.com/baobao1044/NSE>
