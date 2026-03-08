# Section 8: Serving LLaMA 3-70B on TPUs

**From**: [How to Scale Your Model - Applied Inference](https://jax-ml.github.io/scaling-book/applied-inference/)

**My Q&A Document**: [Google Doc](https://docs.google.com/document/d/1ZNcs3f8CnHKmdMIh9siDrp8uO57CnV0sJjiHKgQ7w78/edit?usp=sharing)

**Visual Reference**: [Serving LLaMA 3 Part 8](serving_llama3_part8.pdf)

---

## 🎯 Goal

Apply inference theory from Section 7 to **serving LLaMA 3-70B on real hardware** (TPU v5e). The key engineering question: how do you pick the right topology, precision, and batch size?

---

## 📊 LLaMA 3-70B Architecture

| Hyperparam | Value |
|---|---|
| **L** | 80 |
| **D** | 8,192 |
| **F** | 28,672 |
| **N** (query heads) | 64 |
| **K** (KV heads) | 8 |
| **H** | 128 |
| **V** | 128,256 |

### Why TPU v5e?

| TPU | bf16 FLOPs/s | $/hr | FLOPs/$ |
|---|---|---|---|
| H100 | 9.9e14 | $10.8 | 3.3e17 |
| v5p | 4.59e14 | $4.2 | 3.9e17 |
| **v5e** | **1.97e14** | **$1.2** | **5.8e17** |

v5e gives you the **most FLOPs per dollar** — that's the inference chip.

---

## 📝 Questions & Solutions

### KV Cache Size per Token (int8)

```
KV_bytes/token = 2 × K × H × L = 2 × 8 × 128 × 80 = 163,840 bytes ≈ 160 kB/token
```

> **My answer**: 164 kB ✓ (minor rounding — correct formula)

**Scale-up insight**: At 32k tokens (common long context), one sequence = `160e3 × 32,768 = 5.3 GB`. At BS=240, that's **1.3 TB** of KV cache! KV is the real memory hog at scale.

---

### Memory at BS=32, Seq=8192, int8

| Component | Bytes |
|---|---|
| Params (int8) | `70 GB` |
| KV cache | `160e3 × 8192 × 32 = 41.9 GB` |
| **Total** | **111.9 GB ≈ 112 GB** |

**Minimum topology**: `112 GB / 16 GB/chip = 7 chips` → round to **4×2 (8 chips)** or **4×4 (16 chips)** for headroom.

> **My answer**: 113 GB, 4×4 ✓ — correct, though 4×2 technically fits.

---

### Decode Latency at BS=32 (int8 params, bf16 FLOPs)

**B_crit** = 120 for int8 params + bf16 FLOPs. We have B=32 < 120 → **memory-bandwidth bound**.

**On 4×2 (8 chips)**:
```
T_step = (params + KV) / (N_chips × W_hbm)
= (70e9 + 41.9e9) / (8 × 8.1e11)
= 111.9e9 / 6.48e12
= 17.3 ms ≈ 17 ms
```

Throughput: `32 tokens / 0.017s / 8 chips = 235 tok/sec/chip` ✓

**On 4×4 (16 chips)**: Same memory on twice as many chips → **latency halves to 8.5ms**, throughput/chip stays the same.

> **My answer**: 17ms for 4×2, 8.5ms for 4×4, 235 tok/sec/chip ✓

**ICI check** (can model parallel stay compute-bound?):
- ICI bound when: `Y > M_Y × F / α_ICI = 2 × 28,672 / (1.97e14/9e10) = 2 × 28,672 / 2,189 ≈ 26`
- With 8 chips: 8 < 26 → ✓ **Not ICI-bound on 4×2**

---

### Critical Batch Sizes on TPU v5e

| Precision | Formula | B_crit |
|---|---|---|
| **bf16 params + bf16 FLOPs** | `C/W_hbm = 1.97e14/8.1e11` | **240** |
| **int8 params + bf16 FLOPs** | `C/(2×W_hbm)` | **120** |
| **int8 params + int8 FLOPs** | `C_int8/(2×W_hbm) = 3.94e14/3.24e12` | **240** (back to 240 because FLOPs/s also doubled) |

> **My answers**: 240, 120, 240 ✓

**Why int8+int8 stays at 240**: Quantizing activations to int8 doubles compute throughput (3.94e14 vs 1.97e14), which exactly cancels the 2× reduction in bytes-per-param. Net result: same critical batch size.

---

### Minimum Topology (ignoring KV cache, by precision)

| Precision | Param bytes | Min chips (÷16GB) | Topology |
|---|---|---|---|
| bf16 | 140 GB | ceil(8.75) = 9 | **4×4 (16 chips)** |
| int8 | 70 GB | ceil(4.38) = 5 | **4×2 (8 chips)** |
| int4 | 35 GB | ceil(2.19) = 3 | **2×2 (4 chips)** |

> **My answers**: 4×4, 4×2, 2×2 ✓

But running at these minimum topologies gives **terrible FLOPs utilization** (tiny batch size). Prefer larger topologies to push batch size up to B_crit.

---

### Max Batch Size & Latency at Each Topology (8k context)

**KV per sequence (int8)**: `160e3 × 8192 = 1.31 GB/seq`
**KV per sequence (bf16)**: `2.62 GB/seq`
**KV per sequence (int4)**: `0.655 GB/seq`

| Setup | Total HBM | Params | Available for KV | Max Batch |
|---|---|---|---|---|
| bf16, 4×4 (16 chips) | 256 GB | 140 GB | 116 GB | `116/2.62 ≈ 44` seqs |
| int8, 4×2 (8 chips) | 128 GB | 70 GB | 58 GB | `58/1.31 ≈ 44` seqs |
| int4, 2×2 (4 chips) | 64 GB | 35 GB | 29 GB | `29/0.655 ≈ 44` seqs |

Interestingly, all three topologies max out at ~44 sequences! The "minimum" topology naturally fills to the same batch size.

**Step latency** ≈ `HBM_per_chip / W_hbm = 16 GB / 8.1e11 = 19.8ms ≈ 20ms`

(When HBM is fully utilized, latency is just the time to read all of it — same across precisions at these topologies!)

> **My answer**: `16e9/8.1e11 = 20ms` ✓

---

### QPS per Chip (512 decode tokens median)

```
QPS/chip = batch_size / (step_latency × decode_steps × N_chips)
= 44 / (0.020 × 512 × N_chips)
```

| Setup | N chips | QPS/chip |
|---|---|---|
| bf16, 4×4 | 16 | `44/(0.020 × 512 × 16) = 0.27 QPS/chip` |
| int8, 4×2 | 8 | `44/(0.020 × 512 × 8) = 0.54 QPS/chip` |
| int4, 2×2 | 4 | `44/(0.020 × 512 × 4) = 1.07 QPS/chip` |

> ⚠️ **Correction on my QPS calculation**: I divided by 16 chips for the int8 case but that setup is on 8 chips (4×2). The correct answer is `0.54 QPS/chip` for int8, not 0.26. I mixed up chip counts between setups.

**Key insight**: int4 gives **4× better QPS/chip** than bf16, but real accuracy degradation must be checked. int8 params + bf16 FLOPs is the sweet spot (half the chips needed, minimal quality loss, 2× QPS/chip).

---

### Doubling Topology (bf16: 4×4 → 4×8, 32 chips)

```
Available HBM = 32 × 16GB = 512 GB
Params (bf16) = 140 GB
Available for KV = 372 GB
Max batch = 372 / 2.62 = 142 ≈ 143 seqs
```

Throughput multiplier: `143/44 × (16/32) = 3.25× / 2 = 1.6× per chip` ✓

> **My answer**: 143 batch, 3.3× total throughput, 1.6× per chip ✓

---

### Sharding on 4×8 (bf16, B=143)

**Can we use model parallel on all 32 chips without ICI bottleneck?**

**High-batch regime** (FLOPs-ICI bound):
```
ICI-bound when Y > 2F / α_ICI = 2 × 28,672 / 2,189 = 26
```
We have 32 chips → 32 > 26 → **ICI-bound at large batch** ❌

**Low-batch regime** (HBM-ICI bound):
```
ICI-bound when Y > F / (B × β) = 28,672 / (143 × 9) = 22.3
```
32 > 22.3 → **still ICI-bound** ❌

**Conclusion**: Cannot serve bf16 on 4×8 with pure model parallel — must use 4×4 (16 chips). For int8/int4 with smaller effective batch, the threshold is higher and 4×8 may work.

> **My answer**: ICI-bound, prefer 4×4 ✓. The approach is correct (checked both high- and low-batch conditions).

---

### Prefill Latency (40% MFU, 16 chips, 8192 tokens)

```
T_prefill = 2 × params × T / (N_chips × C × MFU)
= 2 × 70e9 × 8192 / (16 × 1.97e14 × 0.4)
= 1.147e15 / 1.261e15
= 0.91 seconds!
```

> **My answer**: 0.91s ✓

**Prefill is slow!** A single long prompt takes nearly a full second. This motivates **disaggregated serving** (separate prefill and generate servers).

---

### Tokens Evicted per Step (B=32, P=8192, G=4096)

```
Sequences finishing/step = B / G = 32 / 4096 ≈ 0.0078 seq/step
Tokens evicted/step = B × (P + G) / G = 32 × 12288 / 4096 = 96 tokens/step
```

> **My answer**: 96 tokens/step ✓ (correctly used the `B × (P+G)/G` formula)

---

### Disaggregated Serving: Prefill:Generate Ratio

**Setup**: P=8192 prefill, G=512 decode, bf16.

Key times:
- Prefill: `0.91s/sequence` (single-sequence prefill)
- Decode step: `(70×2 + 160e3×(8192+512)×32) / (16 × 8.1e11) × 2(bf16) = 17ms/step`

Balancing the pipeline:
```
Rate_prefill × P_servers = Rate_generate × G_servers
P / 0.91 = 32G / (0.017 × 512)
P / 0.91 = 32G / 8.704
P × 8.704 = 32G × 0.91
P/G = 29.2 / 8.704 ≈ 3.35 → P = 3G
```

> **My answer**: P = 3G ✓ (same result using slightly different path)

> ⚠️ **Calculation nuance**: In my work, I computed the decode time without dividing by N_chips in one step, getting 285ms. The correct decode step time is ~17ms. But both derivations happen to arrive at P=3G because I used consistent approximations throughout. Be careful to always divide by N_chips when computing step time from total bytes.

**Business interpretation**: You need **3× as many prefill servers as generate servers** in production. This huge imbalance is why disaggregated serving (separate prefill fleets) is a real infrastructure choice companies make.

---

## 🧠 Personal Insights

### What I Got Right
- KV cache per token formula: `2KHL = 160kB` ✓
- Total memory at BS=32: 112 GB ✓
- Decode latency (17ms/8.5ms) and throughput (235 tok/sec/chip) ✓
- Critical batch sizes (240, 120, 240) ✓
- Minimum topologies (4×4, 4×2, 2×2) ✓
- Max batch and 20ms latency floor ✓ 
- Doubling topology → 143 batch, 1.6× per-chip throughput ✓
- 4×8 is ICI-bound for bf16 at large batch ✓
- Prefill time: 0.91s ✓
- Disaggregated ratio: P=3G ✓

### Corrections & Nuances
1. **QPS calculation**: I divided all setups by 16 chips, but int8 uses 4×2 = 8 chips and int4 uses 2×2 = 4 chips. Correct answers: int8 → 0.54 QPS/chip; int4 → 1.07 QPS/chip (not 0.26 and 1.0)
2. **Disaggregated decode time**: I computed `2×(70+45.6)e9/8.1e11 = 285ms` (forgot to divide by N_chips=16). Correct answer is `223.8e9/1.296e13 = 17.3ms`. The final ratio P=3G still came out right due to consistent scaling.
3. **Prefill ICI-bound check nuance**: For 4×2, MLP ICI-bound at `Y > 2F/2200 = 26`. We have 8 < 26, so ICI-safe. This enables BF16 FLOPs at full throughput.

### The Serving Decision Framework

```
1. What precision?        → B_crit tells you (120 for int8+bf16, 240 for rest)
2. Smallest topology?     → ceil(param_bytes / 16GB)
3. Max batch?             → (N_chips × 16GB - param_bytes) / KV_per_seq
4. Latency bound?         → 16GB / W_hbm = 20ms/chip (when HBM full)
5. ICI-safe sharding?     → Y < min(2F/2200, F/(β×B))
6. Disaggregated ratio?   → P/G = decode_throughput × prefill_time / seq_throughput
```

---

**Next**: Section 9 - XLA and TPU Profiling
