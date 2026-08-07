# Training Vision: 3 kiến trúc thay thế backprop

Mục tiêu: train mô hình lớn KHÔNG cần cụm GPU khổng lồ. Ba hướng nghiên cứu được scaffold trong `nse-train` (implement sau M6).

## 1. Forward-Forward / Predictive Coding (Hinton)
Thay vì forward+backward để tính gradient, chạy **2 forward passes**:
- Positive data (thật) + Negative data (nhiễu/giả).
- Mỗi layer tự cập nhật weight theo độ "goodness" cục bộ, không cần chờ layer sau.

**Lợi ích:**
- Local updates: mỗi layer độc lập.
- Zero VRAM overhead: không lưu activations của hàng chục layer.
- Parallel layer training: không nghẽn pipeline.

**Scaffold:** `nse-train/src/forward_forward.rs` → `ForwardForwardTrainer`.

## 2. Hopfield / Associative Memory (Energy-Based)
Coi training = bài toán tối ưu năng lượng. Mỗi dữ liệu mới = "thung lũng năng lượng" (energy minimum).
- "Train" dữ liệu mới = vector projection ghi đè điểm năng lượng vào associative memory matrix.
- Chi phí nạp tri thức: từ O(N²)/O(N³) xuống O(1)/O(log N).

**Lợi ích:** one-shot / few-shot learning, không tính lại gradient toàn ma trận.

**Scaffold:** `nse-train/src/hopfield.rs` → `HopfieldTrainer`.

## 3. LSH Sparse Weight Training
Giữ kiến trúc transformer nhưng chỉ cập nhật ~0.01% trọng số mỗi step.
- LSH (Locality-Sensitive Hashing) chỉ ra đúng 0.01% trọng số liên quan tới câu train.
- Chỉ 0.01% này tính gradient + update; 99.99% còn lại đóng băng.
- FLOPs/step giảm 1000–10000×.

**Lợi ích:** train mô hình cực lớn trên vài CPU/GPU phổ thông.

**Scaffold:** `nse-train/src/lsh_sparse.rs` → `LshSparseTrainer` (dùng chung LSH index với RIE inference).

## So sánh

| Phương pháp | Cốt lõi | Bộ nhớ | Tốc độ |
|---|---|---|---|
| Backprop truyền thống | Forward + Backward (AdamW) | Cực lớn (lưu toàn activations) | Tốn hàng nghìn GPU |
| Forward-Forward | Local Goodness | Siêu thấp (1 layer) | Nhanh 10–50×, không nghẽn bộ nhớ |
| Hopfield Associative | Energy Minimization | Trung bình (state matrix) | One-shot/few-shot |
| LSH Sparse Training | Dynamic routing + sub-network | Thấp (0.01% weights) | Tiết kiệm 99.9% FLOPs |

## Định hướng POC
- Baseline REAL: `SgdTrainer` (backprop) → Toy LM ra PPL hợp lý (M2).
- 3 thuật toán trên: trait + skeleton + doc + TODO (M6 scaffold, implement sau).
