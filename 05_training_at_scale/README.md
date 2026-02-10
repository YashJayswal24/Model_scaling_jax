# Section 5: Training at Scale - Parallelism Strategies

**From**: [How to Scale Your Model - Training](https://jax-ml.github.io/scaling-book/training/)

**My Q&A Document**: [Google Doc](https://docs.google.com/document/d/150mR_Q3ju3ujFz752uEocnvLjjQUg0IaNOMmmPwqLRU/edit?usp=sharing)

**Visual Reference**: [Training in TPU Part 5](training_in_tpu_part_5.pdf)

## 🎯 Key Concepts

This section covers the **4 main parallelism strategies** for LLM training:
1. **Data Parallelism (DP)**: Batch-sharded activations, replicated parameters
2. **Fully-Sharded Data Parallelism (FSDP/ZeRO-3)**: Batch-sharded everything, AllGather weights just-in-time
3. **Tensor Parallelism (TP/Megatron)**: Activations sharded along D, weights along F
4. **Pipeline Parallelism**: Weights sharded along layers, microbatching

**The Golden Rule**: Strong scaling requires hiding communication with compute. Each strategy has a **critical batch size** below which it becomes communication-bound.

**The 6PT Rule**: Training FLOPs ≈ `6 × Parameters × Tokens` (this is your rule of thumb!)

---

## 📊 Reference Model: LLaMA-2 13B

| Hyperparameter | Value |
|---|---|
| **L** (layers) | 40 |
| **D** (d_model) | 5,120 |
| **F** (d_ff) | 13,824 |
| **N** (num heads) | 40 |
| **K** (KV heads) | 40 |
| **H** (head dim) | 128 |
| **V** (vocab) | 32,000 |

---

## 📝 Problems & Solutions

### Problem 1: Parameter Counting
**Question**: How many parameters does LLaMA-2 13B have? (LLaMA uses gating einsum with 3 MLP matrices)

**My Answer**:

**MLP Block** (per layer, gating einsum):
- Wᵢₙ₁`[D,F]` = 5,120 × 13,824 = **71M**
- Wᵢₙ₂`[D,F]` = 5,120 × 13,824 = **71M**
- Wₒᵤₜ`[F,D]` = 13,824 × 5,120 = **71M**
- **Subtotal**: 213M × 40 layers = **8.5B**

**Attention Block** (per layer):
- Wᵩ, Wₖ, Wᵥ: `[D, N×H]` each = 5,120 × 5,120 = **26M × 3**
- Wₒ: `[N×H, D]` = **26M**
- **Subtotal**: 104M × 40 layers = **4.2B**

**Vocabulary**:
- Embedding + Output: `[V,D]` × 2 = 32,000 × 5,120 × 2 = **0.33B**

**Grand Total**: `8.5B + 4.2B + 0.33B ≈ 13B` ✓

**Key Insight**: MLP dominates with ~65% of parameters (gating einsum adds 50% more MLP params vs. standard Transformer).

---

### Problem 2: Memory Breakdown
**Question**: Training with BS=16M tokens using Adam. How much memory for params, optimizer state, and activations? (bf16 params, fp32 optimizer, checkpoint 3 matmuls/layer)

**My Answer**:

**Parameters** (bf16):
- 13B × 2 bytes = **26 GB**

**Optimizer State** (Adam, fp32):
- Momentum: 13B × 4 bytes = 52 GB
- Velocity: 13B × 4 bytes = 52 GB
- **Total**: **104 GB**

**Activations** (bf16, checkpointing):
- Per layer: 3 matmul outputs = `2(BF + BF + BD)` bytes
- = `2 × 40 × (16M×13.8K + 16M×13.8K + 16M×5.1K)`
- = `2 × 40 × 16M × (13.8K + 13.8K + 5.1K)`
- = **42 TB**

**Key Insight**: Activations are **1,615× larger** than parameters! Without gradient checkpointing, this would be ~280 TB. Checkpointing is survival, not optimization.

---

### Problem 3: Sharding Strategy Selection
**Question**: Train with BS=3M tokens, 32K context on TPU v5p 16×16×16 (4096 chips). bf16 weights, fp32 optimizer. Which strategies work?

---

#### Part 1: Pure Data Parallelism?
**My Answer**: ❌ **No** (two reasons)

**Memory Check**:
- Per-chip requirement: 130 GB (params + optimizer)
- Available per chip: 96 GB
- **Verdict**: Doesn't fit!

**Communication Check**:
- `B/N = 3M/4096 = 732`
- Requirement: `B/N > C/W_ici = 2550`
- **Verdict**: Communication-bound (732 < 2550)

**Why not**: Model doesn't fit in memory AND we're comms-bound.

---

#### Part 2: Pure FSDP?
**My Answer**: ⚠️ **Barely, but communication-bound**

**Memory Check**:
- Total activations (from Problem 2, scaled): `42 TB × (3M/16M) = 7.85 TB`
- Per chip: `7.85 TB / 4096 = 2 GB` ✓ **Fits!**

**Communication Check** (3-axis FSDP):
- `B/(M_X × M_Y × M_Z) = 3M / (3 axes) = 1M per axis`
- Per chip: `3M/4096 = 732`
- Requirement: `B/N > 2550/M_X = 2550/3 = 850`
- **Verdict**: `732 < 850` → **Communication-bound**

**Key Insight**: FSDP saves memory but we're still **communication-bound** because batch size is too small for this many chips.

---

#### Part 3: Mixed FSDP + Tensor Parallelism?
**My Answer**: ✅ **Yes!**

**Compute-Bound Check**:
- Requirement: `B/N > α²/(2M_X M_Y F)`
- = `2550² / (2 × 2 × 1 × 13,824)` (assuming M_X=2, M_Y=1)
- = `6,502,500 / 55,296 = 235`
- Actual: `3M/4096 = 732`
- **Verdict**: `732 > 235` ✓ **Compute-bound!**

**Optimal Sharding**:
- Formula: `X_opt = √(BN × M_X/(M_Y × F))`
- = `√(3M × 4096 × 2/(1 × 13,824))`
- = `√(24,576M / 13,824)`
- = `√1,778,439` = **1,333**

**Chosen Sharding**:
- **X = 1024** (FSDP, closest power of 2)
- **Y = 4** (Tensor Parallelism)
- `1024 × 4 = 4096` ✓

**Memory per Device**: ~2 GB (same as FSDP)

**Training Time** (40% MFU):
- Total FLOPs: `6 × 13B × 3M = 2.34e17`
- Available FLOPs: `4096 chips × 4.6e14 FLOPS/s × 0.4 MFU = 7.54e17 FLOPS/s`
- **Time**: `2.34e17 / 7.54e17 = 0.31 sec ≈ 300 ms/step`

**Key Insight**: Mixed FSDP+TP lets us **drop batch size by ~3.6×** compared to pure FSDP (235 vs 850 tokens/chip)!

---

## 🧠 Personal Reflection & Nuances

### What I Got Right
1. **The 6PT rule**: Training is fundamentally `6 × params × tokens` FLOPs
2. **Memory dominance**: Activations are 1,615× larger than parameters — checkpointing is mandatory
3. **Critical batch sizes**: Each strategy has a wall below which you're paying for idle chips
4. **Optimal sharding**: X_opt = 1333, choosing X=1024, Y=4 is close

### Calculation Nuances & Corrections
1. **Vocab parameters**: I correctly used `2 × V × D` because LLaMA has separate embedding and output matrices
2. **FSDP communication**: The `B/(M_X M_Y M_Z)` formula accounts for 3-axis parallelism
3. **Mixed FSDP+TP**: The critical batch size formula `α²/(2M_X M_Y F)` gives us 235, which is **~8× smaller** than pure FSDP's 850!

### What About 32K+ Context?
**Your question is crucial!** Here's what changes with longer sequences:

1. **Attention becomes dominant**:
   - MLP: `O(BDF)` — independent of T
   - Attention: `O(BT²NH)` — quadratic in T!
   - Crossover: Attention dominates when `T > 8D` (~40K for LLaMA-2)

2. **Required strategies**:
   - **Flash Attention** (mandatory beyond 8K) — reduces memory from `O(T²)` to `O(T)`
   - **Ring Attention** — splits attention across chips, each handles `T/N` tokens
   - **Context Parallelism** — treat sequence dimension like batch dimension

3. **Memory explosion**:
   - Activation memory: `O(BTD)` + `O(BT²NH)`
   - At T=32K: Attention activations ~4× MLP activations
   - At T=128K: Attention activations ~16× MLP activations

4. **Why the book ignores it**:
   - For **training** at T<32K, MLP still dominates (~75% of FLOPs)
   - For **inference**, KV cache is the bottleneck, not attention FLOPs
   - Ring attention is relatively new (2023) and not yet standard

**The Cool Realization**:
JAX's `Mesh` abstraction is **topology-agnostic**! You specify logical axes `{X:16, Y:16, Z:16}` and JAX maps them to physical TPU coordinates. This means the same sharding code works on 4×4×4, 8×8×8, or 16×16×16 slices — JAX handles the routing! 🤯

---

## 📐 Quick Reference: Parallelism Strategies

| Strategy | When to use | Critical batch size (v5p) | Formula |
|---|---|---|---|
| **Data Parallel** | Model fits on 1 chip | `B/N > 2550` | `T_comp > 4DF/W_ici` |
| **FSDP** | Model doesn't fit | `B/N > 2550/M_X` | Same as DP |
| **Tensor Parallel** | N/A (batch-independent) | `Y < M_Y F/2550` | `8-16 way` for most models |
| **FSDP + TP** | Smallest batches | `B/N > α²/(2M_X M_Y F)` | `X_opt = √(BN M_X/(M_Y F))` |

**Key Formulas**:
- **Optimal FSDP**: `X_opt = √(BN M_X/(M_Y F))`
- **Critical batch (mixed)**: `B/N > 2550²/(2 × 2 × 1 × F) = 235` (for LLaMA-2)
- **Training time**: `6PT / (C × N × MFU)`

---

**Next**: Section 6 - Inference at Scale (KV caching, prefill vs. generation)
