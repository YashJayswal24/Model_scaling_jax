# Section 12: How to Think About GPUs

**From**: [How to Scale Your Model - GPUs](https://jax-ml.github.io/scaling-book/gpus/)

**My Q&A Document**: [Google Doc](https://docs.google.com/document/d/1vO6f_jiVCr2Ql-Bm6pw8V_n6C-idHCFt9Jm4Gfv3uHM/edit?usp=sharing)

**Visual Reference**: [GPU Chapter PDF](gpu_part_12.pdf)

---

## 🎯 Goal

Translate everything we know from TPU roofline analysis to GPUs. Key question: how do GPUs structurally differ from TPUs, and how do the same communication bottlenecks (ICI, HBM bound) map onto NVLink, NVSwitch, and InfiniBand?

---

## 📊 GPU Hardware Reference

### Compute & Memory
| GPU | HBM BW | FLOPs (bf16) | FLOPs (fp8) | FLOPs (fp4) |
|---|---|---|---|---|
| V100 | 9.0e11 | — | — | — |
| A100 | 2.0e12 | 3.1e14 | 6.2e14 | — |
| H100 | 3.4e12 | 9.9e14 | 2.0e15 | — |
| H200 | 4.8e12 | 9.9e14 | 2.0e15 | — |
| B200 | 8.0e12 | 2.3e15 | 4.5e15 | 9.0e15 |

### Networking (NVLink/NVSwitch)
| GPU Gen | NVLink BW (full-duplex GB/s) | Node-to-node BW | Node size |
|---|---|---|---|
| Ampere | 25 | 300 GB/s | 8 GPUs |
| Hopper | 25 | 450 GB/s | 8 GPUs |
| Blackwell | 50 | 900 GB/s | 8 or 72 GPUs |

---

## 📝 Quiz 1: GPU Hardware

### Q1: CUDA Core Counts vs TPU ALUs

```
H100 = 132 SMs × 4 subpartitions × 32 CUDA cores = 16,896
B200 = 148 SMs × 4 subpartitions × 32 CUDA cores = 18,944
TPU v5p = 2 TensorCores × 4 ALUs × 8 × 128 VPU lanes = 8,192
```

> **My Answer**: H100=16,896, B200=18,944, TPU=8,192 ✓

**Insight**: GPUs have ~2× more vector ALUs than a TPU, but the TPU's VPU runs at similar clock speeds. This means the difference in *practical* vector throughput is much smaller than raw CUDA core counts suggest.

---

### Q2: H100 Vector FLOPs

```
Normal: 16,896 ALUs × 1.59 GHz = 26.9 TFLOPs/s
Boost:  16,896 ALUs × 1.98 GHz = 33.5 TFLOPs/s
MatMul FLOPs (Tensor Cores): 990 TFLOPs/s
```

> **My Answer**: ~27 TFLOPs normal, ~33 boost ✓ (slight rounding difference)

**Key ratio**: Tensor Cores give **~30× more FLOPs/s** than CUDA vector cores. This is why GPU AI workloads overwhelmingly rely on large matmuls — everything else is starved for compute.

---

### Q3: Peak Matmul Intensity (FLOPs/s ÷ HBM BW)

| Chip | bf16 | fp8 |
|---|---|---|
| H100 | `990e12 / 3.4e12 = **291**` | ~582 |
| B200 | `2250e12 / 8e12 = **281**` | ~562 |

> **My Answer**: H100=291, B200=287.5 ✓ (essentially equal — this was by design)

**Why is this constant across generations?** FLOPs and bandwidth scale together. This means the **critical batch size** needed to be compute-bound (~280–295 tokens) hasn't changed from H100 → B200. Only cross-node bottlenecks get worse.

---

### Q4: Matmul Runtime on a B200

**fp16[64, 4096] × fp16[4096, 8192]** — batch 64 < B_crit(281) → **HBM bound**:
```
Bytes = 2×BD + 2×DF + 2×BF = 2×64×4096 + 2×4096×8192 + 2×64×8192 = 69 MB
T = 69e6 / 8e12 = 8.6 µs
```

**fp16[512, 4096] × fp16[4096, 8192]** — batch 512 > B_crit(281) → **compute bound**:
```
T = 2×512×4096×8192 / 2.3e15 = 15 µs
```

> **My Answer**: 8.6 µs and 15 µs ✓ — spot on!

---

### Q5: L1/SMEM vs TPU VMEM

```
H100 SMEM:     132 SMs × 256 kB = 33 MB
H100 Register: 132 SMs × 256 kB = 33 MB
Total H100:    ~66 MB
TPU VMEM:      ~120 MB (2×)
```

> **My Answer**: 33 MB each ✓

**Key nuance from the book**: TPU register memory is only 256 kB **total** (not per core), but VMEM latency on TPU is low enough that spills to VMEM are cheap. On GPUs, SMEM spills to L2 → HBM are expensive.

---

### Q6: B200 Clock Speed from Vector FLOPs

Each CUDA core does **2 FLOPs/cycle** (FMA = multiply + add).
```
80 TFLOPs/s = 18,944 cores × 2 FLOPs/cycle × freq
freq = 80e12 / (18,944 × 2) = 80e12 / 37,888 = 2.1 GHz
```

> **My Answer**: 2.1 GHz ✓ — B200 is liquid-cooled for this exact reason.

---

### Q7: Vector Add Runtime (Two fp32[N] Vectors)

**Arithmetic intensity**: `N FLOPs / (3 × 4N bytes) = 1/12` — **extremely memory-bound**.

GPU peak intensity: `33.5 / 3.35 = 10` — so we're bound by HBM.

```
T_comm = 3 × 4N / 3.4e12 = 12N / 3.4e12
T_math = N / 33.5e12
```

Always `T_comm ≫ T_math`.

> **My Answer**: Correctly identified comm-bound. The 3.53 ms figure for N=1024 is an overestimate (latency-dominated, not bandwidth-limited at that size). Book records ~1.5 µs (latency floor), improving to ~4.1ms at N=2^30 (close to roofline).

> ⚠️ **Correction**: For small N (e.g. 1024), you can't fully use 3,000+ CUDA cores. The limit is kernel launch latency (~1–5 µs), not bandwidth. For large N (billions of elements), bandwidth becomes the bottleneck.

---

## 📝 Quiz 2: GPU Nodes (NVLink/NVSwitch)

### Q1: Total Bandwidth on 8×H100 node

```
NVSwitch egress: 4 switches × 64 ports × 25 GB/s = 6.4 TB/s
GPU NVLink:      8 GPUs × 18 links × 25 GB/s = 3.6 TB/s

Min = 3.6 TB/s (GPU-to-switch links are the bottleneck)
```

> **My Answer**: 3.6 TB/s ✓

---

### Q2: Bisection Bandwidth

Bisection = splitting 8 GPUs into two groups of 4. Traffic across the partition:
```
Each GPU can send 450 GB/s. Group of 4 GPUs → 4 × 450 = 1.8 TB/s.
Full duplex (both directions) = 3.6 TB/s bisection bandwidth.
```

> **My Answer**: 3.6 TB/s ✓ — because of the full-mesh NVSwitch fabric, bisection = total node BW.

---

### Q3: AllGather Cost (bf16[4096 × 65536])

Array bytes B = 2 × 4096 × 65536 = 536 MB.

In a ring AllGather across N=8 GPUs, time is:
```
T = (N-1)/N × B / W_link = 7/8 × 536e6 / 450e9 = 1.04 ms
```

> **My Answer**: 1.04 ms ✓

**Pop Quiz 1** (B=1024, F=16384, bf16):
```
B_bytes = 2 × 1024 × 16384 = 33.6 MB
T = 7/8 × 33.6e6 / 450e9 = 65 µs ≈ 75 µs
```

> **Pop Quiz 1 Answer**: ~75 µs ✓

**Pop Quiz 2 (AllToAll with sparsity):**
In a MoE top-k scenario, only k/E fraction of tokens are routed to each expert:
```
T_AllToAll ≈ min(k/E, 1) × (N-1) × B × N / (N²× W_link)
```

> **My Answer**: Correct formula ✓. The min(k/N, 1) captures the effective sparse routing ratio.

---

## 📝 Quiz 4: Collectives at Scale

### Q1: SU AllGather Egress/Ingress

For M nodes, N GPUs/node, array B bytes:
- GPU sends `B/MN` → switch receives `N × B/MN = B/M` (ingress at leaf switch)
- Leaf switch egresses `B/M` to spine
- Leaf switch receives `B(M-1)/M` from spine  
- Leaf switch re-egresses `N × (B - B/MN) = NB - B/M` to local GPUs

**Bottleneck**: Egress at leaf switch = `NB`, so `T = NB / W_node = B / 450e9`

For spine: ingress = B, egress = B(M-1), so `T_spine = B(M-1) / (M × W_node)`

> **My Answer**: Correct analysis of ingress/egress ✓ (slight notation difference from book)

---

### Q2: Single-Node SHARP AllReduce Bytes

With in-network reductions (SHARP), the switch accumulates partial sums:
```
Total ingress = B(N-1) + B = BN
Total egress  = B + B(N-1) = BN
```
So the switch handles exactly **B bytes** net (vs `2B` without SHARP).

> **My Answer**: B bytes total ✓

---

### Q3 & Q4: Cross-Node AllReduce

For `bf16[D×X, F/Y]` sharded `X` ways, with SHARP in-network reductions:
```
T_comms = 2DF / (Y × W_link)
```

At spine level (Y=256, bandwidth ~6.4 TB/s over 4 nodes):
```
T = 2DF / 6.4e12
```

> **My Answer**: Same formula, approach is correct ✓

---

### Q5: 2-Way AllGather Cost (2 Nodes, Precise)

Intra-node portion (ring over 8 GPUs within each node):
```
T_intra = 7/8 × B / 450e9 ≈ B / 514e9
```

Cross-node portion (2 node ring):
```
T_cross = 1/2 × B / 400e9 = B / 800e9
```

Bottleneck: **intra-node** (slower at B/514e9). This is why DeepSeek v3 avoids more than 2-node DP — the NVSwitch intra-node bandwidth is actually the tighter constraint.

> **My Answer**: Identified the node-level ring time ✓

---

## 📝 Quiz 5: LLM Rooflines on GPUs

### Q1: B200 DGX SuperPod Rooflines

B200 has: 2250 TFLOPs, 900 GB/s intra-node BW, 400 GB/s scale-out BW.

**Intra-node** (TP):
```
Critical B_crit = 2250e12 / 900e9 = 2500 tokens
```
Same as H100! FLOPs scaled 2.25×, bandwidth scaled 2×, so they nearly cancel.

**Cross-node** (DP/ZeRO):
```
Critical B_crit = 2250e12 / 400e9 = 5625 tokens
```
Much harder to be compute-bound when doing cross-node data parallelism on B200. Scale-out bandwidth didn't keep up with FLOPs.

> **My Answer**: 2.5×/2× scaling ratio giving ~1.25× tighter DP and same TP ✓ — correct intuition. Exact numbers: intra B_crit=2500, cross-node B_crit=5625.

---

### Q2: Sharding LLaMA-3 70B on H100s

#### Part 1: Minimum GPUs to Store Weights + Optimizer
- bf16 weights: 70B × 2 bytes = 140 GB
- fp32 Adam optimizer state: 70B × 8 bytes = 560 GB
- **Total: 700 GB**
- Per H100: 80 GB → minimum **9 GPUs** → **2 nodes (16 GPUs)**

> **My Answer**: ~700 GB, ~9 GPUs → correct ✓

#### Part 2: Training Time on 4096 H100s at 45% MFU
```
Total FLOPs = 6 × 70e9 × 15e12 = 6.3e24 FLOPs
Rate = 0.45 × 4096 × 990e12 = 1.83e18 FLOPs/s
Time = 6.3e24 / 1.83e18 = 3.44e6 s ≈ 40 days
```

> **My Answer**: ~40 days ✓

#### Part 3: Max Model Parallelism, DP, ZeRO-3, Pipelining

**Max TP (within-node)**:
```
Y < F / (W_link / C) = 28,672 / (450e9 / 990e12) = 28,672 / 1995 ≈ 14 → 8-way TP
```

**Pure DP after 8-way TP** (512 GPUs data parallel):
- Memory check: `700 GB / 8 = 87.5 GB/GPU` → **doesn't fit in 80 GB!** ❌
- Per-GPU batch: `4e6 / 4096 = 976 tokens` → below DP critical batch (2475) ❌
- **Pure DP fails on both memory AND compute-bound criteria.**

**ZeRO-3 + 8-way TP** (512-way ZeRO-3):
- Memory: ✓ (shards everything, fits)
- Per-GPU batch: 976 tokens, but ZeRO-3 moves weights which doubles communication → ~half of DP critical batch → **still not compute-bound** ❌

**8-way Pipelining** (each shard spans 8 nodes):
- Effective scale-out BW: `8 × 400 GB/s = 3200 GB/s`
- Critical batch: `990e12 / 3200e9 = 309 tokens → achievable with per-GPU batch of 976` ✓
- **Pipeline parallelism is the winning strategy!**

> **My Answer**: Got the right 8-way TP limit, and correctly identified pipelining's BW advantage. The key insight on why pure DP fails (memory AND compute bound) was partly captured — the ZeRO-3 per-GPU batch size not being enough is the key gotcha.

> ⚠️ **Correction**: `Y < F/W` formula gave you "around 14" → book gets 14.37. Rounding down to **8** (node boundary) is the practical answer. The ZeRO-3 failure is subtle: per-GPU batch = 976, but DP critical = 2475, and ZeRO adds extra comms so really needs ~2× that = 4950. We're nowhere close.

---

## 🧠 Personal Insights

### GPU vs TPU Mental Model
| Aspect | TPU | GPU |
|---|---|---|
| Vector ALUs | 8,192 (v5p) | 16,896 (H100) |
| Local fast memory | 120 MB VMEM | 66 MB (SMEM+Reg) |
| Compute-bound threshold | ~240 tokens | ~295 tokens |
| Intra-node BW | 900 GB/s (ICI) | 450 GB/s (NVLink) |
| Cross-node BW | 1.8e11 (v5p ICI) | 400 GB/s (InfiniBand) |

### Critical Batch Sizes (H100)
| Strategy | Constraint | B_crit |
|---|---|---|
| TP (intra-node) | 450 GB/s link | ~2,200 tokens |
| DP/ZeRO (cross-node) | 400 GB/s InfiniBand | ~2,475 tokens |
| DP (intra-node, 2-node) | 450 GB/s, halved | ~1,237 tokens |

### The Pipeline Trick
Pipeline parallelism is unique: it makes each model chunk span N nodes, which aggregates the bandwidth of N leaf-level switches. This effectively multiplies cross-node bandwidth by the pipeline depth, making it possible to stay compute-bound at much smaller batch sizes.

---

## 📚 References
- [Chapter Link](https://jax-ml.github.io/scaling-book/gpus/)
- [Google Doc Solutions](https://docs.google.com/document/d/1vO6f_jiVCr2Ql-Bm6pw8V_n6C-idHCFt9Jm4Gfv3uHM/edit?usp=sharing)
- NVIDIA H100 Spec Sheet
- NVLink Gen 4 Architecture Whitepaper
