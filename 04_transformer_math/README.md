# Section 4: Transformer Math

**From**: [How to Scale Your Model - Transformers](https://jax-ml.github.io/scaling-book/transformers/)

**My Q&A Document**: [Google Doc](https://docs.google.com/document/d/1ByEHwNMeRgl9OI_fTwRn6jY4IeHdX6Q2kFKmTgZ4oQg/edit?usp=sharing)

**Visual Reference**: [Transformer Operation Details](transformers_operation_details.pdf)

## 🎯 Key Concepts

This section covers the **accounting** of modern Transformers:
- **Parameter counting**: MLP dominates with `3DF` params per layer
- **FLOPs estimation**: Training requires `6PT` FLOPs (P = params, T = tokens)
- **Attention complexity**: Becomes dominant when `T > 8D`
- **KV caching**: Inference memory scales as `2SLKH` bytes

**The Golden Rule**: For reasonable context lengths, `FLOPs ≈ 6 × Parameters × Tokens`.

---

## 📝 Problems & Solutions

### Problem 1: Parameter Counting
**Question**: Model with `D=4096`, `F=4D`, `V=32K`, `L=64`. How many params? What fraction is attention? KV cache size per token (int8)?

**My Answer**:
- **KVQ params**: `3DNH` (Wₖ, Wᵥ, Wᵩ matrices)
- **Attention out**: `NHD` (Wₒ)
- **MLP params**: `3DF` (gating einsum: Wᵢₙ₁, Wᵢₙ₂, Wₒᵤₜ)
- **Vocab**: `VD`
- **Total**: `VD + L(4D² + 3DF) = 16B params`

**Attention fraction**: `4D²/(4D² + 3DF) = 1/4`

**KV cache per token**: `2LKH = 512 KB/token`

**Key Insight**: MLP dominates parameter count (75%) when `F` is large.

---

### Problem 2: Sharded FLOPs
**Question**: FLOPs for `A[Bₓ,Dᵧ] · W[Dᵧ,F]` on mesh `{X:4, Y:8, Z:4}`?

**My Answer**:
- **Total FLOPs**: `2BDFZ` (Z is replicated)
- **Per TPU**: `2BDF/(XY)`

**Key Insight**: Unused mesh axes (Z) cause full replication → `Z×` FLOPs overhead.

---

### Problem 3: Batched Contraction FLOPs
**Question**: FLOPs for `A[I,J,K,L] · B[I,J,M,N,O] → C[K,L,M,N,O]`?

**My Answer**: `2IJKLMNO`

**Key Insight**: Contract over `I,J`, batch over nothing, output `K,L,M,N,O` → multiply all dims × 2.

---

### Problem 4: Attention Arithmetic Intensity
**Question**: AI of self-attention (no projections) as function of `T,S`. When is it FLOPs-bound?

**My Answer**:
- **Bytes**: `3BTNH` (K, Q, V)
- **FLOPs**: `2BNHT² + 2BT²NH = 4BT²NH`
- **AI**: `4BT²NH / 3BTNH = (4T²) / 3T = 4T/3`
- **FLOPs-bound when**: `AI > 240 → T > ~180`

**Key Insight**: Attention AI scales linearly with sequence length. Becomes compute-bound around `T ≈ 240` on TPUs.

---

### Problem 5: Attention vs Projection FLOPs
**Question**: At what sequence length do attention FLOPs equal projection FLOPs?

**My Answer**:
- **Attention FLOPs**: `4BT²NH`
- **Projection FLOPs**: `4BTDNH` (Q, K, V, O)
- **Equal when**: `T² = TD → T = D`

Wait, let me recalculate:
- **Attention**: `2BNHTT` (Q·Kᵀ + Attention·V)
- **Projections**: `4BTDNH` (since `D = NH`)
- **Equal when**: `2BT²NH = 4BTDNH → T = 2D`

**Key Insight**: Attention dominates beyond 2× the model dimension.

---

### Problem 6: Gradient Checkpointing Overhead
**Question**: If we save only 7 main matmul outputs (Q,K,V,O + 3 MLP), how many extra FLOPs to rematerialize?

**My Answer**:
- **Saved**: `4BTDNH + 3BTDF`
- **Extra FLOPs**: `2(2BT²NH + BTF + BTFD)`

**Key Insight**: We need to recompute attention operations and activation functions → roughly doubling backward pass FLOPs.

---

### Problem 7: DeepSeek v3 Hardware Utilization
**Question**: Trained for 2.79M H800-hours on 14.8T tokens with 37B activated params (FP8). What's the utilization?

**My Answer**:
- **Required FLOPs**: `6 × 37e9 × 14.8e12 = 3.28e24`
- **H800 FP8 FLOPs/s**: `1.513e15` (no structured sparsity)
- **Compute done**: `1.513e15 × 2.79e6 × 3600 = 1.52e25`
- **Utilization**: `3.28e24 / 1.52e25 = 21.5%`

**Key Insight**: ~20% MFU is reasonable for MoE models with communication overhead.

---

### Problem 8: MoE Batch Size
**Question**: Batch size for compute-bound MoE (int8) on v5e? For DeepSeek (`E=256`, `k=8`)?

**My Answer**:
- **MoE FLOPs per token**: `6kBFD`
- **MoE loads**: `3EDF` (all experts)
- **Per-token activation compute**: `6kFD`
- **Per-token loads**: `3EDF/B`
- **Compute-bound when**: `B > (E/2k) × 240`

**DeepSeek**: `B > (256/16) × 240 = 3840 tokens`

**Key Insight**: MoE requires **much larger batches** to saturate compute due to loading all `E` experts.

---

## 🧠 Personal Reflection

This section crystallized the **economics** of LLM training:
1. **The 6PT rule** is shockingly accurate — training really is just `6 × params × tokens`.
2. **Attention isn't the bottleneck** until `T > 8D` — for most training, MLPs dominate.
3. **MoE's hidden cost**: You need `E/k` times the batch size to be compute-bound (3840 tokens for DeepSeek!).



---

**Next**: Section 5 - Training at Scale
