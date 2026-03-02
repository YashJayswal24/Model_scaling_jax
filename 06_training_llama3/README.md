# Section 6: Training LLaMA 3 on TPUs

**From**: [How to Scale Your Model - Applied Training](https://jax-ml.github.io/scaling-book/applied-training/)

**My Q&A Document**: [Google Doc](https://docs.google.com/document/d/11DZQ8raX94P4AZ2b3f3GpPLzK5fI4-nwV1d9g0TAEvE/edit?usp=sharing)

**Visual Reference**: [Training LLaMA 3 Part 6](training_llama3_part_6.pdf)

---

## 🎯 Goal

Apply the parallelism equations from Section 5 to a **real model**: LLaMA 3-70B on TPU v5p.  
The key skill here is working with actual numbers and making practical decisions.

---

## 📊 LLaMA 3-70B Architecture

| Hyperparam | Value |
|---|---|
| **L** (n_layers) | 80 |
| **D** (d_model) | 8,192 |
| **F** (d_ff) | 28,672 |
| **N** (n_heads) | 64 |
| **K** (n_kv_heads, GQA) | 8 |
| **H** (d_qkv) | 128 |
| **V** (vocab size) | 128,256 |

---

## 📝 Parameter Counting

### My Answer

| Component | Formula | Count |
|---|---|---|
| **Vocabulary** | `2 × D × V` | `2 × 8,192 × 128,256 = 2.1B` |
| **Attention (per layer)** | `2DH(N+K)` = `2(DNH + DKH)` | `2×8,192×128×(64+8) = 0.151B/layer` → **12.1B total** |
| **MLP (per layer)** | `3 × D × F` | `3×8,192×28,672 = 0.704B/layer` → **56.3B total** |
| | | **Total: ~70.5B** ✓ |

**Nuances in my formula vs. official**:
- The book writes attention as `n_layers × [2 × D × N × H + 2 × D × K × H]`
  - Factor of 2 for Q/output projection AND factor of 2 for K/V
  - My `2DH(N+K)` is algebraically identical → ✓ correct!
- MLP: `3 × DF` because LLaMA uses **SwiGLU** = 2 up-projections (gate + value) + 1 down-projection
  - The book uses `D × F × 3.5` because `d_ff = 2/3 × 4D`, but since F is already given as 28,672 (the actual ffn dim, not intermediate), we use `3 × D × F` directly

**Key Takeaway**: FFW parameters = **56B of 70B** (80%!). Attention is non-trivial (17%) but FLOPs-wise tends to be dwarfed by MLP.

---

## ⚡ FLOPs Counting

### FLOPs per token
**My answer**: `6 × 70B = 420 GFLOPs/token` ✓

### Total training FLOPs (15T tokens)
**My answer**: `6 × 15e12 × 70e9 = 6.3e24 FLOPs` ✓ 

### Training time on a full TPU v5p pod (8,960 chips)
**My answer**:
```
T = 6.3e24 / (8960 × 4.59e14 × 0.4) = 3.83e6 seconds ≈ 44 days
```
✓ **Correct!** (The book gets 3.8e6 sec — same result)

---

## 🧠 Memory Calculation: Minimum Chips Needed

**Question**: Train BS=4M tokens, bf16 params, fp32 optimizer, 4 activation checkpoints/layer.

**My answer**:
```
Parameters (bf16):   70B × 2 bytes = 140 GB
Optimizer (fp32 Adam):
  momentum = 70B × 4 = 280 GB
  velocity = 70B × 4 = 280 GB
  total = 560 GB
Optimizer total:     560 GB
```
Wait — I wrote `2×2×140=560 GB`. Let me unpack:
- bf16 params = 140 GB
- fp32 momentum = 280 GB (70B × 4)  
- fp32 velocity = 280 GB (70B × 4)
- **params + optimizer = 140 + 560 = 700 GB** (not 560 as I wrote)

**Activation checkpointing** (4 checkpoints/layer):
- Checkpoint after each of 4 positions in the FFW block
- Per layer: `2(BF + BF + BF + BD)` bytes (3 FF activations + 1 D-dim)
  - Wait, with 4 checkpoints: shapes are `BF, BF, BF, BD` → `2(3BF + BD)` bytes
- Total: `2 × 80 × (3 × 4M × 28,672 + 4M × 8,192) × 2 bytes`
- = `2 × 80 × (3 × 4e6 × 28,672 + 4e6 × 8,192) × 2`
- ≈ **~110 TB** for activations

> **My answer wrote `8×BD×L = 21 TB`** — this used the wrong shapes! It used BD instead of BF (the FFW intermediate). Since F >> D (28,672 >> 8,192), activations are ~3.5× larger.

**Correct total**: ~110 TB + 0.7 TB (params + opt) ≈ **111 TB**

**Minimum chips**: `111 TB / 96 GB ≈ 1,156 chips` minimum

> **My answer**: 230 chips — this was too low because I only counted the BD (D-dim) activations and missed the dominant BF (F-dim) activations. The book's hidden answer should be around 1,100+ chips.

### Per-chip memory with 8,960 chips
**My answer**: 2.42 GB — this would be correct for the smaller activation estimate. With the corrected total:
```
111 TB / 8960 ≈ 12.4 GB/chip
```
Still well under 96 GB/chip limit → sharding works!

---

## 🔀 Sharding Strategy for LLaMA 3-70B

**Setup**: 4M token batch, 8,960 chips on a TPUv5p pod.  
Sequence length = 4,096 → 1,024 sequences.

### Can we use FSDP alone (no sequence parallelism)?

**My answer**: We'd only be able to use up to 1,024 chips (one per sequence). The remaining ~7,936 chips would be idle. → **No, major waste**.

✓ **Correct reasoning!** The book confirms: without sequence parallelism, max parallelism = batch size = 1,024 sequences, so only 1,024 chips can be utilized.

### FSDP with sequence parallelism?

**My answer (corrected)**:
- Per-chip batch size = `4M / 8960 = 446 tokens`
- Need `B/N > α/M_X = 2550/3 = 850` to be compute-bound
- `446 < 850` → **Communication-bound** ❌

> Note: I wrote `B > 8960 × 2550/3 = 7.6M`, which means we'd need a batch >7.6M to be compute-bound. Since we only have 4M tokens, pure FSDP doesn't work.

✓ **Correct conclusion**: Can't use pure FSDP — communication-bound.

### Mixed FSDP + Tensor Parallelism?

**Compute-bound check**:
- Need `B/N > α²/(2F)` = `2550²/(2×28,672) = 113`
- Actual `B/N = 4M/8960 = 446`
- `446 > 113` ✓ → **Compute-bound is achievable!**

**Optimal X**:
```
X_opt = √(B × M_X × N / (F × M_Y))
     = √(4.19e6 × 2 × 8960 / (28,672 × 1))
     = √(75.0e9 / 28,672)
     = √2,617,992 ≈ 1,618
```

> **My answer**: 1,619 ✓ (nearly identical, small rounding differences)

**Nearest power-of-2**: X = 2,048 (FSDP), Y = 4 (TP) → `2048 × 4 = 8192 ≈ 8960` ✓

> ⚠️ **Nuance**: I also gave X=512 as an option, but that's X_opt/3, which is farther from optimal. The book goes with 2048-way FSDP and 4-way TP.

**Final scheme**: `1024-way batch + 2-way sequence + 4-way tensor parallelism`

---

## 🏆 Worked Problems

### Q1: Scaling to 4 Pods (35,840 chips), same batch size

**My answer**:
1. **Cross-pod**: Pure data parallelism → 1M tokens/pod
2. **DCN bound check**: `B/pod > C/W_dcn = 71,360 → 1M >> 71,360` ✓ Compute-bound over DCN
3. **Per-pod**: Mixed FSDP+TP (same as single-pod analysis)
4. **Per-pod compute-bound**: `B/N_per_pod = 1M/8960 = 112 > 113`... **barely!** On the edge.

**Training time** (at 40% MFU):
```
T = 6.3e24 / (4 × 8960 × 4.59e14 × 0.4)
  = 6.3e24 / (6.57e18)
  = 9.59e5 sec ≈ 11 days
```

> **My answer**: 11 days ✓ — I used `4.69e14` instead of `4.59e14` for one calculation, but the final answer is right.

**Key insight**: Scaling from 1 pod to 4 pods gives **4× speedup** (44 days → 11 days). This is perfect linear scaling!

---

### Q2: LLaMA 3-405B

#### (a) Architecture and FLOPs

| Hyperparam | Value |
|---|---|
| **L** | 126 |
| **D** | 16,384 |
| **F** | 53,248 |
| **N** | 128 |
| **K** | 8 |
| **H** | 128 |
| **V** | 128,256 |

> Note on H: the book uses H=128 for dqkv even for 405B. My guess of `H=D/N=128` is actually correct: `16,384/128=128` ✓

**Parameter count**:
| Component | Formula | Count |
|---|---|---|
| Vocab | `2 × D × V` | `2 × 16,384 × 128,256 = 4.2B` |
| Attention | `2DH(N+K) × L` | `2×16,384×128×(128+8)×126 = 71.4B` |
| MLP | `3 × D × F × L` | `3 × 16,384 × 53,248 × 126 = 330B` |
| **Total** | | **≈ 406B** ✓ |

> **My answer**: 
> - Vocab: 4.2B ✓
> - Attention: `2DH(N+K) = 0.55B/layer × 126 = 69.76B` — slightly off (should be 71.4B, small rounding)
> - MLP: `3DF = 2.617B/layer × 126 = 329.7B` ✓
> - Total: 403.7B ≈ 404B ✓ (close enough)

**FLOPs per token**: `6 × 405B = 2.43 TFLOPs/token` ✓

**Total for 15T tokens**: `6 × 405e9 × 15e12 = 3.645e25 FLOPs` ✓

#### (b) Sharding on 8 pods (71,680 chips)

**My analysis**:
1. **Cross-pod DCN**: Pure data parallelism → `B/pod > 71,360`. With 4M tokens / 8 pods = 500K tokens/pod → `500K >> 71,360` ✓ Compute-bound over DCN

2. **Pure FSDP** (per pod): Need `B > N × 2550 / M_X = 8960 × 850 = 7.6M`. We have 500K/pod → way too small → ❌

3. **Pure TP**: Need `Y < M_Y × F/2550 = 1 × 53,248/2550 ≈ 20`. So 16-way TP is feasible. ✓
   > My check of `F > 7.6M` was wrong — that's the FSDP condition, not TP. Tensor parallelism depends on the number of TP chips Y, not batch size!

4. **Mixed FSDP+TP**: Need `B/N > α²/(2F) = 2550²/(2×53,248) = 61`
   - Per pod: `500K/8960 = 56 < 61` → **Barely communication-bound!** 😬

**Final answer**: For 405B on 8 pods with only 4M total tokens (500K/pod):
- This is a tight fit — we're at the edge of the mixed FSDP+TP compute-bound regime
- Use **16-way TP** (feasible since `Y=16 < 20`)
- FSDP for remaining axes
- Might need to accept some comms overhead

**Training time** (at 40% MFU):
```
T = 6 × 15e12 × 405e9 / (8 × 8960 × 4.59e14 × 0.4)
  = 3.645e25 / (1.32e19)
  = 2.76e6 sec ≈ 32 days
```

---

## 🧠 Personal Insights

### What I Got Right
- Parameter counting framework: `Vocab + Attention + MLP × L`
- The 6PT rule for FLOPs
- Training time calculations (44 days single pod, 11 days 4 pods)
- Cross-pod DCN analysis
- Optimal X_opt ≈ 1,618 → 2,048-way FSDP

### Calculation Nuances & Corrections
1. **Activation memory was off**: I used `BD` shapes instead of `BF`. Since `F >> D` in LLaMA 3 (28K >> 8K), activations are ~3.5× larger than I estimated. Minimum chips is ~1,156, not 230.
2. **Tensor Parallelism check**: For TP, the condition is `Y < M_Y × F/α` (depends on Y, not batch size). I confused the FSDP condition with the TP condition for 405B.
3. **405B mixed FSDP+TP**: Per-pod batch of 500K tokens gives `B/N = 56 < 61`, meaning we're technically comms-bound per pod. This matters for real deployments.

### The Big Picture
The practice of putting real numbers into these equations is the core skill. The same 3-step decision process applies always:
1. **Does it fit in memory?** (params + optimizer + activations)
2. **Are we compute-saturated?** (FLOPs > communication time)
3. **What's the optimal sharding?** (X_opt formula)

---

**Next**: Section 7 - Inference at Scale
