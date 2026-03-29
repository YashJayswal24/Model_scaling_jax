# Part 10: All About JAX — Parallelism & MoE Scaling

> **Book Section:** [Part 10 — All About JAX](https://jax-ml.github.io/scaling-book/jax-stuff/)  
> *How To Scale Your Model* — Austin et al., Google DeepMind (2025)

**Run on:** Kaggle TPU v5e-8 | **Notebook:** [`moe-scaling-on-tpu.ipynb`](../moe-scaling-on-tpu.ipynb) | **Kaggle:** [reptor420/moe-scaling-on-tpu](https://www.kaggle.com/code/reptor420/moe-scaling-on-tpu/notebook)

---

## 📚 Key Concepts

- **Three JAX parallelism modes:** Auto (Shardy/XLA), Explicit (JAX type system), Manual (`shard_map`)
- **`shard_map`** gives a per-device local view — you write only the communication you intend
- **AllGather** is expensive for MoE: causes O(N × S × D) memory per device
- **All-to-All** is the correct primitive for MoE dispatch — O(S × D) memory budget
- **Collective Matmuls**: AllReduce and ReduceScatter matmuls can overlap comm with compute
- **Bidirectional butterfly** (recursive halving) achieves log₂(N) communication steps

---

## 🧮 Mathematical Intuition

### MoE Token Routing
For `E` experts, `S` tokens, `D` hidden dim, `N` devices:

- **Naive AllGather:** Each device needs *all* tokens → `AllGather([S/N, D]) → [S, D]` per device. Memory: `N × (S/N × D) = S × D` replicated on every device. **Expensive.**
- **All-to-All dispatch:** Each device sends `C` tokens to each of the `N` devices that own the relevant experts. Max memory in-flight: `[E, C, D]` where `C << S`. **Cheap.**

### ReduceScatter Variants
For `A[BX, FY] @ W[FY, D] → Out[BX, DY]`:
- **Ring (unidirectional):** N-1 steps, each device sends `D/N` columns clockwise
- **Recursive halving (bidirectional):** log₂(N) steps, each step exchanges half the array with XOR partner
  - At step `k`: device `i` exchanges with device `i XOR 2^k`

---

## 💻 Code Structure

```
10_jax_parallelism_moe/
├── README.md                ← You are here
../moe-scaling-on-tpu.ipynb  ← Full Kaggle notebook with all implementations
```

### Problem 1 — Sharded Average & Roll-Diff

```python
# jax.jit solution (reshape trick)
@jax.jit
def average(arr):
    arr = jnp.mean(arr, axis=(1, 3))  # Mean over non-sharded dims
    return jax.lax.with_sharding_constraint(arr, P('x', 'y'))

# shard_map solution (local view — no communication needed)
@jax.shard_map(in_specs=jax.P('x','y'), out_specs=jax.P('x','y'))
def avg_shard_map(arr):
    return jnp.mean(arr).reshape((1, 1))  # Each device sees its own shard
```

**Q: Was there any communication added?**  
**A:** No. Since each shard is independent, both `jit` and `shard_map` compute the mean locally. The `with_sharding_constraint` is just a safety guard to keep the output sharding explicit.

### Problem 2 — MoE: Naive vs Pipelined All-to-All

```python
# NAIVE: scan over experts with masking — causes AllGather
def process_exp(carry, e):
    mask = (B == e)[:, None]          # [S, 1]
    to_add = A @ W[e]                  # [S, F] — ALL tokens × expert e
    return carry + to_add * mask, None # Wasteful: E-1 matmuls are thrown away

# OPTIMIZED: All-to-All pipeline — only route tokens to their experts
recv_A = jax.lax.all_to_all(send_A, axis_name='x', split_axis=0, concat_axis=1)
recv_Out = jnp.einsum('edf,ecd->ecf', W_local, recv_A)  # Dense, zero waste
```

**Q: Why does `jit` cause an AllGather?**  
**A:** With Auto sharding, XLA chose to shard both `W` and `A` along `x` (data-parallel). This means each device only has `E/N` experts locally, but needs *all* activations to run its scan. So XLA inserts an AllGather. The fix is `shard_map` + explicit All-to-All.

**Q: What is expert capacity `C` and how do you tune it?**  
**A:** `C` is the max tokens sent to one expert per communication round. Setting `C` too large wastes padding compute; too small means more `while_loop` iterations and network round-trips. A good default: `C = 2 × (S / (E × N))` — twice the expected load per expert per device.

### Problem 3 — Collective Matmuls

```python
# AllReduce matmul: A[BX,DY] @ W[DY,F] → Out[BX,F]
def compute_matmul_allreduce(A, W):
    return jax.lax.psum(A @ W, axis_name='y')

# ReduceScatter matmul: Tmp[BX,FY] @ W2[FY,D] → Out[BX,DY]
final_out = jax.lax.psum_scatter(partial_out, axis_name='y',
                                  scatter_dimension=1, tiled=True)

# Recursive halving (bidirectional, log₂N steps)
perm = tuple((j, j ^ (1 << step)) for j in range(y))  # XOR partners
recv_data = jax.lax.ppermute(send_data, axis_name='y', perm=perm)
state = keep_data + recv_data  # Array shrinks by 2× each step
```

**Q: Is the manual ring ReduceScatter faster than `psum_scatter`?**  
**A:** No — in this experiment, `psum_scatter` (~1.01ms) beat the manual unidirectional ring (~1.2ms). XLA's built-in primitive has hardware-level optimizations we can't replicate in Python. The recursive halving variant (~1.1µs) gets close to parity.

**Q: Why does the overlapped Transformer MLP not beat `jax.jit`?**  
**A:** XLA already performs compute-communication overlap internally. Our manual overlap doesn't add extra benefit but *does* add transparency — you can verify in Perfetto that no unintended communication occurred. The value is correctness guarantees, not raw speed.

---

## 📊 Performance Results

| Experiment | Latency | Vs Baseline |
|---|---|---|
| Naive `jit` MoE (AllGather) | 73 ms | — |
| **Pipelined All-to-All MoE** | **1.7 ms** | **43× faster** |
| Top-k MoE (k=2) | 1.7 ms | 0% overhead |
| `jit` Auto-Sharded MLP | 1.2 ms | — |
| `psum_scatter` ReduceScatter | 1.01 ms | 16% faster than ring |
| Recursive halving RS | ~1.1 µs | Near optimum |
| Overlapped Transformer MLP | 1.25 ms | Parity with `jit` |

**Hardware:** Kaggle TPU v5e-8 · **Total runtime:** 72 seconds · **Profiler traces:** 9

---

## ⚠️ Known Issues & Fixes

```python
# BUG: Second line silently overwrites first — async_all_to_all never enabled!
os.environ["LIBTPU_INIT_ARGS"] = "--xla_tpu_enable_async_all_to_all=true"
os.environ["LIBTPU_INIT_ARGS"] = "--xla_tpu_enable_async_collective_fusion=true"

# FIX: Combine into one string
os.environ["LIBTPU_INIT_ARGS"] = (
    "--xla_tpu_enable_async_all_to_all=true "
    "--xla_tpu_enable_async_collective_fusion=true"
)

# DEPRECATED: jax.lax.pvary
initial_out = jax.lax.pvary(initial_out, 'x')          # ❌ deprecated
initial_out = jax.lax.pcast(initial_out, to='varying')  # ✅ current API
```

---

## 📖 Citation

```bibtex
@article{scaling-book,
  title     = {How to Scale Your Model},
  author    = {Austin, Jacob and Douglas, Sholto and Frostig, Roy and
               Levskaya, Anselm and Chen, Charlie and Vikram, Sharad and
               Lebron, Federico and Choy, Peter and Ramasesh, Vinay and
               Webson, Albert and Pope, Reiner},
  publisher = {Google DeepMind},
  year      = {2025},
  note      = {https://jax-ml.github.io/scaling-book/}
}
```
