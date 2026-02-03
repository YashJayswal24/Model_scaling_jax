# Section 1: Roofline Analysis

**From**: [How to Scale Your Model - Rooflines](https://jax-ml.github.io/scaling-book/roofline/)

## 🎯 Key Concepts

This section introduces **roofline models** — a framework for understanding whether an operation is:
- **Compute-bound** (limited by FLOPs/s)
- **Memory-bound** (limited by HBM bandwidth)
- **Communication-bound** (limited by inter-chip bandwidth)

The key metric is **Arithmetic Intensity (AI)**: FLOPs per byte loaded.

---

## 📝 Problems & Solutions

### Question 1: int8 Matmul
**Setup**: `X[B,D] · Y[D,F] → Z[B,F]` in int8 (1 byte/param, 3.94e14 OPs/s peak)

**My Answers**:
- **Bytes loaded**: `BD + DF`
- **Bytes written**: `BF`
- **Total OPs**: `2*BDF` (int8 peak)
- **Arithmetic Intensity**: `2*BDF / (BD + DF + BF) ≈ 2*B` (when `B << D, F`)
- **Compute-bound threshold**: `B > (3.94e14) / (2 * 8.1e11) = 243`
- **Time estimates**:
  - `Tmath ~ 2*BDF / 3.94e14`
  - `Tcomms ~ (BD + DF + BF) / 8.1e11`
  - **Upper bound**: `Tmath + Tcomms`
  - **Lower bound**: `max(Tmath, Tcomms)`

**Key Insight**: int8 quantization doubles the OPs/s but halves the bytes per parameter, pushing the compute-bound threshold to ~243 tokens (vs. 240 for bf16).

---

### Question 2: Mixed Precision (int8 weights + bf16 activations)
**Setup**: `bf16[B,D] * int8[D,F] → bf16[B,F]` (1.97e14 bf16 FLOPs/s)

**My Answer**:
- **FLOPs**: `2*BDF` (still bf16 compute)
- **Bytes**: `2*BD + DF + 2*BF ≈ DF` (weights are int8!)
- **AI**: `2*BDF / DF = 2*B`
- **Compute-bound threshold**: `2*B > 240 → B > 120`

**Key Insight**: Quantizing weights to int8 **halves** the batch size needed to be compute-bound (120 vs. 240), since we load fewer bytes for the weight matrix.

---

### Question 3: Roofline Plot
**Setup**: Plot peak FLOPs/s vs. B for `F=D=4096` and `F=D=1024`

**My Answer**:
Created a [Desmos visualization](https://www.desmos.com/calculator/hrgrfz5hkt?embed) showing the roofline transition.

**Key Insight**: The roofline "knee" (where we transition from memory-bound to compute-bound) shifts based on matrix dimensions. Larger matrices reach peak FLOPs/s at smaller batch sizes.

---

### Question 4: Batched Matmul
**Setup**: `int8[B,D] * int8[B,D,F] → int8[B,F]` (different matrix per batch element)

**My Answer**:
- **Bytes**: `BD + BDF + BF`
- **OPs**: `2*BDF`
- **AI**: `2*BDF / (BD + BDF + BF) = 2*DF / (D + DF + F) ≈ 2` (when `DF >> D, F`)

**Key Insight**: The arithmetic intensity is **constant** (~2) regardless of batch size! This operation will almost always be memory-bound.

---

### Question 5: GPU Rooflines (H100)
**Setup**: Calculate batch size for bf16 matmul to be compute-bound on H100 SXM
- **Peak FLOPs/s**: 1.979 PetaFLOPs (with structured sparsity) → `1.979e15 / 2 = 9.895e14` (actual)
- **HBM Bandwidth**: 3.35 TB/s

**My Answer**:
- **Critical AI**: `9.895e14 / (2 * 3.35e12) = ~295`
- **Threshold**: `B > 295`

**Key Insight**: GPUs have slightly higher compute-to-bandwidth ratios than TPUs, requiring larger batch sizes (~295 vs. ~240) to saturate the compute.

---

## 🧠 Personal Reflection

This section made me realize that **quantization isn't just about model size** — it fundamentally changes the roofline characteristics:
1. **int8 weights** let you be compute-bound at **half** the batch size.
2. **Batched operations** (different matrices per sample) have **fixed AI** and are almost always memory-bound.
3. **GPU vs TPU** rooflines differ by ~20%, making batch size tuning hardware-specific.

At Samsung, this means when we quantize models for deployment, we're not just saving memory — we're shifting the entire performance profile.

---

**Next**: Section 2 - TPU Architecture
