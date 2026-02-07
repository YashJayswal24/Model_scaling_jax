# Section 3: Sharded Matrices and Communication Primitives

**From**: [How to Scale Your Model - Sharding](https://jax-ml.github.io/scaling-book/sharding/)

**My Q&A Document**: [Google Doc](https://docs.google.com/document/d/16ogD2ipwt2AZetSoC5S4L7_hvWEAY00QrQBm6JExuqw/edit?usp=sharing)

**Visual Reference**: [Sharding Matrix Diagrams](sharding_matrix_in_tpu_mesh.pdf)

## 🎯 Key Concepts

This section introduces **sharding notation** and the **4 core communication primitives** used to multiply matrices distributed across TPU meshes:
- **AllGather**: Replicates sharded data
- **ReduceScatter**: Sums and re-shards
- **AllReduce**: Sums without re-sharding
- **AllToAll**: Reshards along different axes

**The Golden Rule**: The cost of these operations depends only on array size and bandwidth, NOT the number of devices (when bandwidth-bound).

---

## 📝 Pop Quizzes & Problems

### Pop Quiz 1: 2D Sharding Across 1 Axis
**Question**: `fp32[1024,4096]` with sharding `A[IₓY, J]` on mesh `{X:8, Y:2}`. Memory per device? HBM load time on H100?

**My Answer**:
- **Bytes per TPU**: `4 × 1024 × 4096 / 16 = 1 MB`
- **Time**: `1e6 / 3.4e12 ≈ 300 ns`

**Key Insight**: Sharding along flattened axes (`XY`) splits data across `|X| × |Y|` devices.

---

### Pop Quiz 2: Replication Factor
**Question**: `int8[128,2048]` with `A[IₓY, J]` on mesh `{X:2, Y:8, Z:2}`. Memory per device? Total memory?

**My Answer**:
- **Z=2**: Array is **replicated** across both Z slices
- **Total memory**: `2 × 128 × 2048 = 512 KB`

**Key Insight**: Unused mesh axes cause full replication, doubling memory usage.

---

### Pop Quiz 3: AllGather Time
**Question**: `AllGatherY([EY, F])` on TPU v5e `{X:8, Y:4}` with `E=2048, F=8192` (bf16)?

**My Answer**:
- **Unidirectional**: `2 × 2048 × 8192 / (4.5e10 × 1) = 559 μs`
- **For `E=256, F=256`**: `2.1 μs`

[Desmos visualization](https://www.desmos.com/calculator/5ualxkt9ei)

**Key Insight**: Time scales linearly with array size but is independent of number of devices!

---

## 🔧 Worked Problems

### Problem 1: Replicated Sharding
**Question**: Array `A[Iₓ, J, K, ...]` on mesh `{X:4, Y:8, Z:2}`. Ratio of total bytes to single copy?

**My Answer**: `Y × Z = 16`

**Key Insight**: Each unused mesh axis causes full replication → multiplicative memory overhead.

---

### Problem 2: AllGather Latency
**Question**: `AllGatherₓ([Bₓ, Dᵧ])` on TPU v4p 4×4×4 with `B=1024, D=4096` (bf16)?

**My Answers**:
1. **AllGatherₓ**: `2 × 1024 × 4096 / (9e10) = 23 μs`
2. **AllGatherₓY**: `46 μs` (2 axes → 2× cost)
3. **AllReduceᵤ**: `11.6 μs` (ReduceScatter = AllGather / 2)

[Desmos calculator](https://www.desmos.com/calculator/7n9ee6lsm4)

---

### Problem 3: Latency-Bound AllGather
**Question**: `AllGatherₓ([Bₓ])` with tiny `B=128` (bf16). How long on 4×4×4 v4p?

**My Answer**: `~2 μs` (latency-bound due to 1μs/hop overhead)

[Calculation](https://www.desmos.com/calculator/tl1akoexql)

**Key Insight**: For small arrays, latency (1μs/hop) dominates bandwidth.

---

### Problem 4: Matmul Strategies
**Question**: Compare two strategies for `X[B,D] · Y[Dₓ,F] → Z[B,F]`:
1. **Strategy 1**: AllGather Y, then multiply
2. **Strategy 2**: Multiply locally, then AllReduce

**My Analysis**:
- **Strategy 1**: `T_comm = 2DF/W_ici`, `T_comp = 2BDF/W_flop`
  - Compute-bound when `B > W_flop/W_ici`
- **Strategy 2**: `T_comm = 4BF/W_ici`, `T_comp = 2BDF/(X·W_flop)`
  - Compute-bound when `D > 2X·W_flop/W_ici`

**Key Insight**: 
- **Large batch sizes** → Strategy 1
- **Large models** → Strategy 2

---

### Problem 5: Minimum Latency Sharding
**Question**: Best sharding for `A[I,J] · B[J,K] → C[I,K]` (fully replicated output) on 4×4×4 v4p?

**My Answer**:
- **Best**: `A[I,Jₓyz]` × `B[J,K]` or `A[I,J]` × `B[Jₓyz,K]`
  - `T_comm = IK/W_ici` (AllGather output)
  - `T_comp = 2IJK/(XYZ·W_flop)` (parallelized)
- **Also good**: `A[I,Jₓyz]` × `B[Jₓyz,K]`
  - `T_comm = 2IK/W_ici` (AllReduce)
  - `T_comp = 2IJK/(XYZ·W_flop)`

**Key Insight**: Sharding along contracting dimension maximizes compute parallelism.

---

### Problem 6: Common Sharding Patterns
**Question**: Analyze 3 common patterns on TPU v5e 4×4:

**Part 1**: `A[Iₓ,Jᵧ] · B[Jᵧ,K] → C[Iₓ,K]`
- **Comm**: AllGatherᵧ (twice) → `T_comm = 2JK/(Y·W_ici)`
- **Comp**: `T_comp = 2IJK/(XY·W_flop)`

**Part 2**: `A[Iₓ,J] · B[Jₓ,Kᵧ] → C[Iₓ,Kᵧ]` (Training: DP + TP + ZeRO)
- **Comm**: AllGatherₓ on J → `T_comm = JK/(Y·W_ici)`
- **Comp**: `T_comp = 2IJK/(XY·W_flop)`

**Part 3**: `A[Iₓ,J] · B[J,Kᵧ] → C[Iₓ,Kᵧ]` (Inference: Pure TP + DP)
- **Comm**: None!
- **Comp**: `T_comp = 2IJK/(XY·W_flop)`

**Key Insight**: Inference is communication-free when properly sharded!

---

### Problem 7: Transformer Block Sharding
**Question**: Shard `In[B,D] · Wᵢₙ[D,F] · Wₒᵤₜ[F,D]` on 2×2 v5e with 300MB/TPU. `D=8192, F=32768, B=128` (bf16).

**My Answer**:
- **Sizes**: `In/Out = 2 MB`, `Wᵢₙ/Wₒᵤₜ = 536 MB each`
- **Sharding**: `In[B,Dₓᵧ]`, `Wᵢₙ[D,Fₓᵧ]`, `Wₒᵤₜ[Fₓᵧ,D]`
- **Strategy**:
  1. `Wᵢₙ · Wₒᵤₜ → W_processed[D,D]{Uₓᵧ}`
  2. `ReduceScatterₓᵧ,J → W_collected[D,Dₓᵧ]`
  3. `AllGather + multiply → Out[B,Dₓᵧ]`

**Key Insight**: Intermediate fusion (`Wᵢₙ · Wₒᵤₜ`) reduces communication vs. sequential matmuls.

---

### Problem 10: AllToAll is 4× Faster?
**Question**: Why is AllToAll 4× faster than AllGather in bidirectional rings?

**My Analysis**:

**Unidirectional**:
- **AllGather**: `(D-1) × N²/D` scalars → `N² × (D-1)/D` total
- **AllToAll**: `N²(D-1)/(2D)` scalars (arithmetic progression)
- **Ratio**: `2×`

**Bidirectional**:
- **AllGather**: Hops → `D/2`, bytes/hop → `2N²/D` → Total: `N²`
- **AllToAll**: Hops → `D/4`, bytes/hop → `N²/D` → Total: `N²/4`
- **Ratio**: `4×`

**Key Insight**: Bidirectional AllToAll benefits from both **halved hops** and **halved bytes/hop**.

---

## 🧠 Personal Reflection

This section changed how I think about parallelism:
1. **Communication cost is size-agnostic**: Whether you have 4 or 64 devices, AllGather time depends only on bandwidth.
2. **Strategy matters more than scale**: Choosing between AllGather-then-multiply vs. multiply-then-AllReduce can double performance.
3. **Inference is "free"**: With proper sharding (`A[Iₓ,J] · B[J,Kᵧ]`), there's **zero communication overhead**.

At Samsung, when we shard models, we're not just dividing work — we're choosing a **communication strategy** that can make or break latency.

---

**Next**: Section 4 - Transformer Math
