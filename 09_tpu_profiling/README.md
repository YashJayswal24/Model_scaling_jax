# Section 9: XLA and TPU Profiling

**From**: [How to Scale Your Model - Profiling](https://jax-ml.github.io/scaling-book/profiling/)

**My Q&A Document**: [Google Doc](https://docs.google.com/document/d/1MWuPgoy9c87mX9WlnZlqvsioC8pWfEpPzNtXfKUfpys/edit?usp=sharing)

**Visual Reference**: [Profiling JAX Part 9](profiling_jax_part_9.pdf)

**Code Reference**: [Mock Transformer Sharding Notebook](mock_transformer_sharding.ipynb)

---

## 🎯 The Goal

Theoretical roofline calculations (FLOPs / bandwidth) give us a theoretical limit. **Profiling** tells us whether we actually hit that limit in practice.

In JAX, programs convert to **StableHLO** → **HLO (High Level Optimizer)** → **LLO** → **TPU Machine Code**.
By reading HLO through the JAX Profiler Trace Viewer, we can diagnose sharding inefficiencies, memory layout issues, and sub-optimal fusions.

---

## 🧩 How to Read an XLA Op

An example HLO signature from the trace:
```text
bf16[32,32,8192]{2,1,0:T(8,128)(2,1)S(1)}
```
- **bf16**: Data type (bfloat16)
- **[32,32,8192]**: Logical shape of the tensor.
- **{2,1,0}**: Physical dimension ordering in memory (column or row major).
- **T(8,128)(2,1)**: Tiling format. Tensors are broken down into chunks (tiles) for efficient loading into the systolic array. This padding can increase memory footprint (by ~1.6x in bad cases).
- **S(1)**: Memory space. `S(1)` = VMEM (fast compute memory). `S(0)` = HBM (Global memory).

---

## 📝 Problems & Solutions

### Problem 1: Suspicious Colab Profile (Identifying Sharding from Traces)

**Observation from the Trace:**
- A fusion node showing: `bf16[8192] * bf16[4096, 8192] -> bf16[4096]`
- An `AllReduce` node grouped as `replica_groups={{0,16,32,48,64,80,96,112}, ...}`.

**What does this mean?**
The `AllReduce` tells us this is an **8-way Model Parallel** (Megatron-style) matmul.
The per-shard input shape is `[8192]`, but multiplied across 8 devices, the actual tensor shapes being computed globally are `[8, 8192] * [32768, 8192] -> [8, 32768]`.

> **My Answer**: Correctly identified the 2 large matmul fusions `bf16[8192] * bf16[4096, 8192] -> bf16[4096]` per shard. ✓

---

### Problem 2: Profiling a Mock Transformer

We profile a single matrix multiplication to compare the observed real-world execution time against our mathematical rooflines.

#### Part A: Pure Dense Matmul (No Sharding, 1 TPU)

```python
x = jax.random.normal(key, (32, 1024, 8192)).astype(jnp.bfloat16)
W = jax.random.normal(key, (8192, 8192 * 4)).astype(jnp.bfloat16)  # 8192 * 4 = 32768
y = x @ W
```

**Theoretical Math:**
- FLOPs = `2 × B × T × D × F`
- FLOPs = `2 × 32 × 1024 × 8192 × 32768` = **17.59 TFLOPs**
- TPU v5e speed = `197 TFLOPs/s`
- Expected Time: `17.59 / 197 = 89.3 ms`.

**Observed Time in Profiler:**
> **My Answer**: 94 ms 132µs.
**Conclusion**: We are extremely close to the hardware limit (~95% MFU) on a single chip!

#### Part B: Mock Transformer (Data Parallelism, 8 TPUs)

**Setup:**
- `BS=8, seqlen=1024, vocab=32,768, d_model=8,192, d_ff=16,384`

**First FFW Matmul (`jnp.dot(x, w1)`):**
- Total FLOPs: `2 × 8 × 1024 × 8192 × 16384 = 2.199e12` (2.19 TFLOPs).
- Distributed over 8 parameter replicas (Data Parallelism):
  - FLOPs per chip = `2.19 TFLOPs / 8 = 274.8 GFLOPs`.
- Expected Time per chip: `274.8e9 / 1.97e14 = 1.39 ms`.

**Observed Trace:**
> **My Answer**: The graph shows `1.43 ms` (down to the microsecond!). Perfect match! ✓

**Logits Projection (`jnp.dot(x, embed.T)`):**
- FLOPs per chip relative to FFW: `Vocab (32,768) / d_ff (16,384) = 2× more`.
- Expected Time: `1.39 ms × 2 = 2.78 ms`.

**Observed Trace:**
> **My Answer**: 2.91 ms! Perfect match! ✓

#### Part C: What about FSDP & Tensor Parallelism?

If we Naively switch the code to FSDP (`w1 = jax.device_put(w1, P('data', None))`), what happens?

> **My Answer**: FSDP took 6+6ms just for communication. We become entirely **ICI Bound** (Inter-Chip Interconnect). `W_ici` acts as a huge bottleneck compared to compute time. The overhead is strictly not worth it for these matrix dimensions.

By constraining `jax.lax.with_sharding_constraint`, we can optimize this.
Using pure data parallelism for small batch/small model on 8 v5e chips gave the best results, dropping the per-layer compute down drastically compared to naive compiler setups!

---

## 🧠 Personal Insights

- **The Math Works:** Real-world tracing proved that `2∗BTDF / Hardware_FLOPs` dictates the actual runtime down to the millisecond. It stops being theory and becomes a tangible limit you can read in Perfetto visually.
- **Compilers Fall Short:** `jax.jit` and XLA are brilliant, but they don't natively understand hardware topologies. Without explicit `jax.sharding.PartitionSpec` injections, XLA will schedule terrible All-Reduces that multiply node times by 5x (e.g. 1.4ms becoming 6ms+).
- **Traces > Guesses:** Rather than guessing if a model is compute vs memory bound, clicking on a node and seeing whether `T_hbm`, `T_ici`, or `T_math` was bloated instantly solves the mystery.

---

**Next Steps**: Wrap up with Section 10 - JAX Under the Hood.
