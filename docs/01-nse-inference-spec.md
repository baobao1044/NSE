# NSE Inference Spec (tóm tắt từ technical specification)

## Tổng quan
Neuro-Sparse Engine (NSE) chạy LLM (100B–2.7T) trên CPU/Edge (x86 AVX2/AVX-512) không cần GPU cluster.

Nguyên lý cốt lõi:
1. **Post-Training Architecture Transmutation**: Dense matrix → dynamic sparse graph, không retrain.
2. **Sub-1-Bit / Micro-Expert Segmentation**: chia weight thành hàng triệu micro-experts, nén codebook (VQ-PQ / Ternary).
3. **Input-Dependent Dynamic Activation**: chỉ kích hoạt trọng số liên quan tới từng token (0.001%–0.1%).
4. **Cache-Centric Execution**: toàn bộ data cho 1 bước tính toán gói trong L1/L2/L3 cache, loại bỏ RAM bandwidth bottleneck.

## Module

### ZSTM (Offline Transmutation)
- **A. Outlier Preserving Dense Core**: tách kênh activation outlier, FP16/INT8 cố định, luôn kích hoạt, nằm L1 cache.
- **B. Micro-Expert Clustering**: mỗi cột weight = vector trong không gian `din`; Spherical K-Means/SVD rã cột thành N micro-experts (64KB–2MB vừa cache). Chỉ hoán vị chỉ số (permutation), giữ nguyên giá trị số học.
- **C. Extreme Codebook Quantization**: Ternary `{−1,0,1}` (4 giá trị/byte) hoặc PQ (indices trỏ shared codebook < 1MB, L3 cache).

### RIE (Routing & Indexing)
- **A. MIPS Index Tree**: HNSW/LSH cho centroid, tra cứu O(log N) thay vì O(N).
- **B. Adaptive Threshold Router**: `Score_k = Sim(X, C_k)`, nếu `Score_k < θ(X)` gán 0 + prune; giữ dynamic top-K.
- **C. Static Bias Compensator**: kỳ vọng nhánh bị bỏ qua → vector `B_sparse ∈ R^dout`, cộng vào output để khôi phục độ chính xác.

### LLER (Low-Level Execution)
- **A. L3 Cache Tiling Engine**: nạp theo block vừa L3 (16MB–64MB), cache miss tiệm cận 0%.
- **B. SIMD Bitwise Compute Kernel**: thay FPU mul bằng AVX2/SSE4.1:
  - Ternary: `+1` → `_mm256_add_ps`, `−1` → `_mm256_sub_ps`, `0` → skip.
  - PQ: `_mm256_shuffle_epi8` tra cứu codebook trên L1 cache.

## Format `.nse` (mmap-friendly)
Layout: header → dense core → codebook → micro-expert data → MIPS tree.

```rust
struct NSEFileHeader {
    magic: [u8; 4],            // "NSE1"
    total_params: u64,
    num_layers: u32,
    dense_core_size: u32,
    codebook_size: u32,
    index_tree_offset: u64,
}
struct MicroExpertMeta {
    expert_id: u32,
    num_channels: u32,
    data_offset: u64,
    centroid_vector: [float],  // biến length
}
```

## Pipeline
**Offline**: Dense Model → Outlier Extraction → Micro-Expert Clustering → VQ → Build MIPS Tree + Bias Vector → `.nse` (30–50GB cho 2.7T).

**Online**: Token Input → [Dense Core (L1) || MIPS Lookup (L3)] → Dynamic Top-K Micro-Experts → Bitwise AVX2 Kernel → + Static Bias → Next Token.

## Roadmap
- Phase 1 (POC, 2 tháng): Llama-3-8B, đánh giá PPL sụt giảm.
- Phase 2 (Core Kernel, 3 tháng): C++20/Rust, AVX2/AVX-512, L3 tiling, MIPS HNSW/LSH.
