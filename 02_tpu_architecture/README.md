# Section 2: How to Think About TPUs

**From**: [How to Scale Your Model - TPUs](https://jax-ml.github.io/scaling-book/tpus/)

**My Q&A Document**: [Google Doc](https://docs.google.com/document/d/1q3feWPBbFnTLwdJJ7PkSWaHh-fRHcii6JmBY9bpol6s/edit?usp=sharing)

## 🎯 Key Concepts

This section demystifies **TPU architecture**: how the MXU (Matrix Multiply Unit) works, how chips are networked via ICI (Inter-Chip Interconnect), and how bandwidth constraints at different levels (HBM, ICI, PCIe, DCN) affect performance.

**Key Components**:
- **MXU**: Matrix multiply unit (systolic array)
- **VPU**: Vector processing unit (element-wise ops)
- **VMEM**: On-chip scratchpad memory (128 MiB)
- **HBM**: High-bandwidth memory (16-96 GB depending on generation)

**Networking Hierarchy** (fastest → slowest):
1. **HBM ↔ VMEM**: ~1-3 TB/s
2. **ICI**: ~90-180 GB/s per link (nearest neighbors)
3. **PCIe**: ~16-32 GB/s (host ↔ chip)
4. **DCN**: ~6-12 GB/s (datacenter network)

---

## 📝 Problems & Solutions

### Problem 1: Bounding LLM Latency
**Question**: How long to load a 200B parameter bf16 model from HBM across 32 TPU v4p?

**My Answer**:
- **Size**: `200e9 * 2 bytes = 4e11 bytes`
- **HBM bandwidth**: `1.2e12 bytes/s per chip × 32 = 3.84e13 bytes/s`
- **Time**: `4e11 / 3.84e13 = 0.01 sec = 10 ms`

**Key Insight**: With sufficient parallelism, even massive models can be loaded in ~10ms — memory bandwidth matters!

---

### Problem 2: TPU Pod Details
**Question**: Calculate total specs for full TPU v5e and v5p pods.

**My Answers**:

**TPU v5e** (16×16):
- **Hosts**: `(16/4) × (16/2) = 8`
- **TensorCores**: `256 chips = 256 cores`
- **Total FLOPs/s**: `256 × 1.97e14 = 5.04e16`
- **Total HBM**: `256 × 16 GB = 4 TB`

**TPU v5p** (16×20×28):
- **Hosts**: `(16/2) × (20/2) × (28/1) = 2240`
- **TensorCores**: `8960 chips = 8960 cores`
- **Total FLOPs/s**: `8960 × 4.59e14 = 4.11e18`
- **Total HBM**: `8960 × 96 GB = 860 TB`

**Key Insight**: v5p pods are **~80x more powerful** than v5e pods and have **200x more memory**!

---

### Problem 3: PCIe Operational Intensity
**Question**: What batch size needed to be FLOPs-bound over PCIe for `bf16[B,D] @ bf16[D,F]` with `F=4D`?

**Setup**: TPU v6e, PCIe BW = 1.5e10 bytes/s

**My Answer**:
- **AI needed**: `9.20e14 / 1.5e10 = 6.13e4`
- **AI formula**: `2*BDF / (2*(BD + DF + BF)) ≈ 2B` (when `B << D`)
- **Threshold**: `B > 6133`

**Key Insight**: PCIe is **~100x slower** than HBM. If weights are on the host, you need **massive batch sizes** (~6K tokens) to saturate compute.

---

### Problem 4: General Matmul Latency
**Question**: Time for `int8[16384,4096] @ int8[B,4096]` on TPU v5e?

**My Answer**:
- **T_comp**: `2*BDF / 3.94e14`
- **T_comm**: `(BD + DF + BF) / 8.1e11`
- **Threshold**: `B > 250` (compute-bound)
- **Formula** (B > 250): `T = B / 2.93e6` seconds

**From VMEM**: If VMEM BW is ~20x HBM, threshold drops to `B > 12`.

**Key Insight**: VMEM's higher bandwidth lets you be compute-bound at **20x smaller batch sizes**.

---

### Problem 5: ICI Bandwidth
**Question**: Send `bf16[8,128,8192]` from TPU{0,0} → TPU{3,3} in a 4×4 slice.

**My Answer**:
1. **First byte arrival**: 6 hops × 1μs/hop = **6 μs**
2. **Total transfer**: 
   - **Bytes**: `2 × 8 × 128 × 8192 = 1.7e7`
   - **Paths**: 2 (can send via two disjoint paths simultaneously!)
   - **Time**: `1.7e7 / (2 × 4.5e10) = 189 μs`

**Key Insight**: The **torus topology** lets you split communication across multiple paths, effectively doubling bandwidth!

---

### Problem 6: Pulling It All Together (Hard)
**Question**: Copy sharded `int8[128K,128K]` from host DRAM to TPU{0,0}, multiply by `bf16[8,128K]`.

**My Answer**:
- **Copy from hosts via DCN**: `16 GB / (16 chips × 6.25e9) = 66 ms`
- **Copy activation from HBM**: `2 × 8 × 128K / 8.1e11 = 2.59 μs`
- **Copy weight matrix via ICI**: `15 GB / 9e10 = 167 ms`
- **Copy weight to MXU**: `16e9 / 8.1e11 = 19 ms`
- **Compute**: `2 × 128K × 128K / 1.97e14 = 1.3 ms`

**Bottleneck**: **ICI transfer** at 167 ms dominates everything.

**Key Insight**: When aggregating data across chips, **ICI bandwidth becomes the bottleneck** — even for "small" 15GB transfers.

---

## 🧠 Personal Reflection

This section made me appreciate just how **hierarchical** TPU performance is:
1. **VMEM** is your friend for small, high-reuse operations
2. **HBM** is where most weights live, limiting memory-bound ops
3. **ICI** is fast but still ~30x slower than HBM
4. **DCN** is the "last resort" — avoid cross-slice communication

At Samsung, when we talk about "model sharding," we're really navigating this hierarchy. The difference between a 10ms and 200ms latency often comes down to **which link the data has to cross**.

---

**Next**: Section 3 - Sharding Strategies
