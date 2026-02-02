# Model Scaling with JAX 🚀

**A Systems View of LLMs on TPUs**

This repository documents my journey mastering the concepts from the book **["How to Scale Your Model"](https://jax-ml.github.io/scaling-book/)**.

## 📚 Objective
To understand and implement the underlying systems engineering concepts required to train Large Language Models (LLMs) at scale using **JAX** and **TPUs**.

## 🛠️ Tech Stack
- **Framework**: [JAX](https://github.com/google/jax)
- ** Hardware Focus**: Cloud TPUs (v4/v5e)
- **Key Concepts**:
    - Data Parallelism (DP)
    - Tensor Parallelism (TP)
    - Fully Sharded Data Parallelism (FSDP)
    - Pipeline Parallelism (PP)
    - `jax.pjit` and `jax.sharding`

## 📂 Structure
The project is organized by the sections of the book. Each folder contains:
- **Implementation Code**: JAX scripts demonstrating the concept.
- **Documentation**: Detailed `README` with mathematical intuition and TPU-specific notes.

## 📝 Blog Posts
Detailed walkthroughs of these implementations are published on my [Portfolio](https://yashjayswal24.github.io/portfolio/).

---
*Learning by building.*
