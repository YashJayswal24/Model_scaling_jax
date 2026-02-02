# Section 0: Introduction to Scaling LLMs

**From**: [How to Scale Your Model - A Systems View of LLMs on TPUs](https://jax-ml.github.io/scaling-book/index)

## 🎯 Key Concepts

This introductory section sets the stage for understanding **Strong Scaling** and the fundamental trade-offs when parallelizing LLMs across multiple chips (TPUs/GPUs).

---

## 📝 Questions & Insights

### Round 1: Understanding the Fundamentals

#### Q1: The "Strong Scaling" Trade-off
**Question**: Why does adding more chips not always result in a linear increase in speed, and what "cost" begins to outweigh the benefits?

**My Answer**: Strong scaling means adding chips to handle proportional computation/inference. However, as more chips are added, **communication becomes the bottleneck**.

**Key Takeaway**: 
- **Weak Scaling**: Handling more users/data by adding chips.
- **Strong Scaling**: Reducing time for a **fixed problem size** by adding chips.
- As you shard a model further, the **Compute-to-Communication ratio** drops, and eventually communication dominates.

---

#### Q2: The Bottleneck Theory
**Question**: What are the three primary factors that form the "roofline" and limit a model's ability to scale?

**My Answer**: Communication latency between components, computation efficiency, and memory bandwidth.

**Key Takeaway**: The three pillars of the "roofline model":
1. **Peak FLOPs** (compute capacity)
2. **HBM Bandwidth** (memory bandwidth)
3. **Interconnect Bandwidth** (communication between chips)

---

#### Q3: Architecture vs. Hardware
**Question**: Why is it more important for a researcher to understand interconnects and bandwidth today than it was five years ago?

**My Answer**: Algorithms have consolidated into **Transformers**, which heavily depend on matrix multiplications—the exact thing TPUs and GPUs are specialized for. At high scale, any "slow link" or latency adds up and can slow down the entire process.

**Key Takeaway**: 
- Modern research involves trading **"algorithmic cleverness"** for **"hardware efficiency"**.
- The hardware's physical limits (the **"Memory Wall"**) are now a primary design constraint.

---

### Round 2: Deeper Systems Thinking

#### Q4: The "Alchemy" vs. "Science" Distinction
**Question**: What distinguishes ML "alchemy" from ML "science"?

**My Answer**: 
- **Science** = The ability to predict performance and resource usage accurately.
- **Alchemy** = Relying on intuition or "vibes" where things just happen to work.

**Key Takeaway**: Understanding the systems view lets us move from black-box experimentation to **principled design**.

---

#### Q5: The "Communication-Bound" Nightmare
**Question**: Why can doubling chips sometimes slow down training?

**My Answer**: The bottleneck shifts to the **ICI (Inter-Chip Interconnects)**. The bandwidth limits add latency to the sequential steps, breaking Strong Scaling.

**Key Takeaway**: 
- Adding chips increases **communication overhead**.
- If communication takes longer than computation, we become **communication-bound**.

---

#### Q6: Hardware-Driven Design
**Question**: Why did Transformers "win" over other architectures?

**My Answer**: Transformers were designed for **parallelism**. Their math (matrix representation) perfectly matches **hardware compatibility**, allowing them to grow with compute rather than just "understanding humans."

**Key Takeaway**: 
- The Transformer architecture aligns perfectly with TPU/GPU capabilities (massive matrix multiplications).
- This isn't just about better language understanding—it's about **scaling efficiently** on modern hardware.

---

## 🧠 Personal Reflection

This section reminded me that at Samsung, when we talk about "optimizing models," we're really talking about understanding these three constraints:
1. Are we **compute-bound**? (Waiting for matrix ops to finish)
2. Are we **memory-bound**? (Waiting for data to load)
3. Are we **communication-bound**? (Waiting for chips to talk)

The days of "just throw more GPUs at it" are over. If you don't account for these bottlenecks, you're just burning money.

---

**Next**: Section 1 - Roofline Analysis
