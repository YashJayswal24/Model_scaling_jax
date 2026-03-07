# Section 7: All About Transformer Inference

**From**: [How to Scale Your Model - Inference](https://jax-ml.github.io/scaling-book/inference/)

**My Q&A Document**: [Google Doc](https://docs.google.com/document/d/1xqNwFtALOCplvjS7mF7yf0pwJXZ9EFJYwiDskFVUdwo/edit?usp=sharing)

**Visual Reference**: [Inference TPU Part 7](inference_tpu_part_7.pdf)

---

## 🎯 Key Concepts

Inference is **two fundamentally different problems** disguised as one:
- **Prefill**: Processing the prompt — compute-bound, like training
- **Generation**: Sampling token-by-token — **always memory-bandwidth-bound**

The central data structure: **the KV cache** = `2 × L × K × H × T` bytes per sequence.

### Critical Numbers (TPU v5e)
| Hardware | Value |
|---|---|
| HBM capacity | 16 GB/chip |
| HBM bandwidth | 8.1e11 bytes/s/chip |
| FLOPs (bf16) | 1.97e14/chip |
| FLOPs (int8) | 3.94e14/chip |
| ICI bandwidth | 9.0e10 bytes/s (bidi) |
| **B_crit** (bf16 params) | **240 tokens** |
| **B_crit** (int8 params, bf16 FLOPs) | **120 tokens** |
| **β = W_hbm/W_ici** | **~ 9** |

---

## 📊 Model for Problems

| Hyperparam | Value |
|---|---|
| **L** | 64 |
| **D** | 4,096 |
| **F** | 16,384 |
| **N** (query heads) | 32 |
| **K** (KV heads) | 8 |
| **H** (head dim) | 256 |
| **V** | 32,128 |

---

## 🧪 Pop Quiz: Latency at Small vs Large Batch

**Setup**: 30B dense model, TPU v5e 4×4 (16 chips), int8 params + bf16 FLOPs, 8192 context, 100 kB/token KV cache.

### Batch = 4 tokens

**My answer**: `30e9/8.1e11/16 = 2.3ms`

**Correction**: I forgot the KV cache term in the small batch formula! The correct formula is:

```
T_step = (B × kv_size + params_bytes) / (N_chips × W_hbm)
= (4 × 8192 × 100e3 + 30e9) / (16 × 8.1e11)
= (3.28e9 + 30e9) / 1.296e13
= 33.28e9 / 1.296e13
= 2.57 ms ≈ 2.5 ms
```

The KV cache is ~11% of the params here, so forgetting it underestimates by ~0.3ms.

### Batch = 256 tokens  

**My answer**: KV = 16ms, MLP = 4.8ms, Total ≈ 21ms ✓

```
KV time:  256 × 8192 × 100e3 / (16 × 8.1e11) = 16.2ms  ✓
MLP time: 2 × 30e9 × 256 / (16 × 1.97e14) = 4.87ms     ✓
Total ≈ 21ms ✓
```

**Key Insight**: At B=256, we're compute-bound on MLP but still **bandwidth-bound on KV attention** (always!). The two terms don't compete — you just sum them.

---

## 📝 Problems & Solutions

### Problem 1: Parameter Count & KV Cache Size

**Question**: Parameters (shared input/output projections)? KV cache per token in int8?

**My Answer**:
- MLP: `L × 3DF = 64 × 3 × 4096 × 16384 = 12.88B` ✓
- Attention: `L × 2DH(N+K) = 64 × 2 × 4096 × 256 × 40 = 5.37B` ✓
- Vocab: `D × V = 4096 × 32128 = 0.132B`

**Correction on vocab**: I wrote `2×V×D = 0.26B` but the problem says **shared** input/output matrices → only **1 matrix** = `D×V = 0.132B`. This is a classic trap!

**Total**: `12.88 + 5.37 + 0.132 = 18.4B` ✓ (Book: 18.4B)

**KV cache per token (int8)**:
```
2 × L × K × H = 2 × 64 × 8 × 256 = 262,144 bytes = 262 kB/token ✓
```

---

### Problem 2: Max Batch Size on 4×4 TPU v5e

**Question**: Max batch for 128k context, int8 everything? Drop K=8→1?

**My Answer** (K=8):
```
B × KV_per_seq + params ≤ total HBM
B × (262e3 × 128e3) + 18.4e9 ≤ 16 chips × 16e9
B × 33.5e9 ≤ 256e9 - 18.4e9 = 237.6e9
B ≤ 7 ✓
```

**K=1**: `8× smaller KV → 8×7=56` ✓

**Key Insight**: KV cache is the **dominant memory consumer** at long contexts. With 128k sequences, a single KV cache (33.5 GB) is nearly 2× the entire model! GQA (K=8→1) is transformative for long-context inference.

---

### Problem 3: Lower Bound on Per-Step Latency

**Question**: Time to load all params into MXU? (fully sharded 4×4, int8)

```
T_step ≥ params / (N_chips × W_hbm)
= 18.4e9 / (16 × 8.1e11)
= 1.42 ms
```

**My answer**: 1.427 ms ✓

**Why this is a lower bound**: This assumes 100% HBM bandwidth utilization with perfect overlap. Reality: KV caches also need loading, communication adds overhead. Real latency ≈ 1.5–2× this floor.

---

### Problem 4: Sharding Strategy (4×4 v5e, int8)

**ICI topology**: 4×4 = 2D mesh, W_ici = 9.0e10 bytes/s bidi, β = W_hbm/W_ici = 8.1e11/9.0e10 ≈ **9**

#### Prefill Sharding

**Megatron TP limit**:
```
Y < M_Y × F / α_ICI = 1 × 16384 / (1.97e14/9e10) = 16384/2189 ≈ 7.5
```
→ **Up to 8-way TP** is feasible on the 4×4.

**My answer**: "cores ≤ 14" — slightly off. The correct limit is ~7.5 (≈8 way). With 16 chips total and 8-way TP, we have 2 chips per sequence axis for sequence parallelism.

**Final prefill scheme**: `8-way TP + 2-way sequence parallelism = 16 chips`

#### Generation Sharding (Key Rules!)

**Critical constraints** — during generation, you CANNOT use:
1. ❌ FSDP (moving weights over ICI is slower than HBM bandwidth by β=9×)
2. ❌ Data parallelism (replicates params, doesn't help)
3. ❌ Sequence sharding (only 1 token at a time!)

**Only option**: Model parallelism (Megatron-style).

**ICI vs HBM bound for generation**:
```
Better to shard more if: T_ICI < T_HBM
i.e., B × D / (Y × W_ici) < D / W_hbm
→ B < Y × W_hbm/W_ici = Y × β = Y × 9
```

At 16-way model parallel: compute-bound threshold = `16 × 9 = 144 tokens`

For small batches (B < 144): **pure HBM bandwidth-bound** → fully shard model across all 16 chips.

**Per-step latency** (small batch, all 16 chips model-parallel):
```
T_step = params / (N_chips × W_hbm) = 18.4e9 / (16 × 8.1e11) = 1.42 ms ✓
```

**KV sharding**: With K=8 heads and 16 chips, shard K across Y=8 chips, then shard batch (B) across remaining Z=2 chips.

---

### Problem 5: MoE Inference (E=16, k=2)

**Question**: Total/activated params, FLOPs-bound batch, KV size, FLOPs per T tokens?

**My Answers**:

**Total params**: `MLP × E + attention + vocab = 12.88 × 16 + (5.37 + 0.13) = 205.5 + 5.5 = 211B ≈ 212B` ✓

**Activated params**: `MLP × k + attention + vocab = 12.88 × 2 + 5.5 = 25.76 + 5.5 = 31.3B` ✓

**Critical batch size for MoE** (the key insight!):
```
B_crit_MoE = B_crit_dense × (E/k) = 240 × (16/2) = 1920 tokens
```
Why? We load `E` times more weights (MoE params) but only do `k` times more FLOPs → need `E/k` times bigger batch to be compute-bound.

**KV cache**: Same as dense (MoE only changes FFW, not attention) ✓

**FLOPs per T tokens**: `2 × activated_params × T = 2 × 31.3e9 × T = 62.6e9 × T` ✓

**Key Insight**: MoE has 12× more total params but only ~2× more activated params and 8× worse B_crit. This is the inference cost of MoE — hard to be compute-bound without massive batches.

---

### Problem 6: Expert Sharding on 8×16 v5e

**Setup**: 128 chips total, Y=8 (F dim), Z=16 (expert dim). Int8.

**HBM loading time per chip**:
```
FFW params per chip = Total_MoE_FFW / (Y × Z) = (16 × 12.88B) / 128 = 1.61B bytes
T_HBM = 1.61e9 / 8.1e11 = 1.99 ms ≈ 2ms ✓
```

**Free HBM per chip**: `16GB - 1.61GB ≈ 14.4 GB` ✓

**Smallest slice to fit 212B int8 model**:
```
Minimum chips = 212GB / 16GB = 13.25 → 16 chips (4×4) ✓
```

But also need Z ≥ E = 16 for expert sharding, so **4×4 = 16 chips is both the memory minimum AND the expert sharding minimum**!

---

### Problem 7: 2D Weight-Stationary Sharding

**Algorithm** (sharding `Win[DX, FYZ]`, `Wout[FYZ, DX]`):
```
1. In[B, DX]      = AllGatherYZ(In[B, DXYZ])
2. Tmp[B, FYZ]    = In[B, DX] *D Win[DX, FYZ]   (local matmul, partial sum)
3. Tmp[B, FYZ]    = AllReduceX(Tmp[B, FYZ])
4. Out[B, DX]     = Tmp[B, FYZ] *F Wout[FYZ, DX] (local matmul)
5. Out[B, DXYZ]   = ReduceScatterYZ(Out[B, DX])
```

**T_math** = `4BDF/(N × C)` (all FLOPs sharded N ways) ✓

**T_comms**:
- Step 1 (AllGather YZ, size BD/X × YZ axes): `BD/(X × M_YZ × W_ici)`
- Step 3 (AllReduce X, size BF/YZ, 1 axis): `2BF/(YZ × W_ici)` — AllReduce = 2×
- Step 5 (ReduceScatter YZ, size BD/X): `BD/(X × M_YZ × W_ici)`

Total: `T_comms = 2BD/(X × M_YZ × W_ici) + 2BF/(YZ × W_ici)`

**Minimizing over X** (with YZ = N/X and M_YZ = M_Y + M_Z ≈ 2 axes):
Set `∂T_comms/∂X = 0`:
```
X_opt = √(N × D / (2F))
```
With F ≈ 4D: `X_opt = √(N/8)` ✓

**Optimal total comms**:
```
T_2D = √(8/N) × BD/W_ici
```

**vs traditional 3D model parallel**: `T_3D = 4BD/(3 × W_ici)`

**2D wins when**:
```
√(8/N) < 4/3 → 8/N < 16/9 → N > 72 ≈ 81 chips
```

**My answer**: N > 72, book gets N > 81. The discrepancy comes from the `M_YZ` factor and exactly how many axes you use. The key insight is the same: **2D sharding is better when N ≳ 80 chips**.

**Why this matters**: At N > 80, even though we're communication-bound, adding more chips **still reduces latency** because T_comms ∝ 1/√N. That's remarkable — you can keep scaling and keep improving!

---

## 🧠 Personal Insights

### What I Got Right
- Pop Quiz batch=256 (16ms + 4.8ms = 21ms) ✓
- All 7 problem directions correct
- MoE B_crit = 1920 = 240 × E/k logic ✓
- Expert sharding loading time ✓
- 2D sharding X_opt = √(N/8) derivation ✓

### Corrections & Nuances
1. **Pop quiz (batch=4)**: Forgot KV cache in small-batch formula → off by 0.3ms (2.3 vs 2.5ms)
2. **Q1 vocab**: "shared matrices" = 1 matrix (`V×D`), not 2. Off by 0.13B (doesn't change conclusions)
3. **Q4 Megatron limit**: Said "cores≤14", correct is `F/α ≈ 7.5` → 8-way for prefill
4. **Q7 threshold**: Got N>72, book gets N>81. Small rounding difference in axis-count assumptions

### The Big Picture
Inference = 2 modes:

| | **Prefill** | **Generation** |
|---|---|---|
| Bound | Compute (like training) | **Memory BW (always)** |
| Sharding | TP + Sequence | **Model parallel only** |
| KV cache | Negligible (Flash Attn) | Dominant |
| Batch effect | Minor | Critical |

The surprising fact: generation's arithmetic intensity is **constant** — you can never improve it by adding FLOPs. The only path is reducing weight/KV bytes (GQA, quantization, pruning).

---

**Next**: Part 8 - Serving LLaMA 3 on TPUs (Applied Inference)
