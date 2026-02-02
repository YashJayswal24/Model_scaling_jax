# JAX Scaling Agent Instructions

**Objective**: Master "How to Scale Your Model: A Systems View of LLMs on TPUs" ([Link](https://jax-ml.github.io/scaling-book/)) and document the journey.

## 📚 Workflow Overview

This workflow is designed to be cyclical for each section/chapter of the book involved. also site the link of the repo page and all and add it to the blog post.

### 1. Setup & Learning Phase
**Trigger**: User selects a section (e.g., "Matrix Multiplication", "Data Parallelism").

*   **Folder Structure**:
    *   Create a clean, descriptive folder in `C:\Yash\Projects\Model_scaling_jax`.
    *   Naming convention: `[chapter_number]_[topic_slug]` (e.g., `01_matrix_multiplication`).
*   **Implementation**:
    *   Write JAX code that reflects the concepts in the section.
    *   Focus on **TPU concepts** (sharding, pmap, pjit) even if running on CPU/GPU locally (use `jax.experimental.mesh_utils` etc.).
    *   Create a `README.md` in the folder with:
        *   Key Concepts (bullet points).
        *   Mathematical intuition.
        *   Code usage examples.
        *   there will be questions and answers by me which you need to add in the blog post. and jax repo

### 2. Documentation Phase (Blog)
**Trigger**: User says "make a blog post" or "section done".

*   **Context**:
    *   Use the code and README from the current section.
    *   Follow the `@/customize.agent` and `@/customize.agent.extension` workflows.
*   **Action**:
    *   Create a new post in `C:\Yash\Projects\Portfoli\al-folio\_posts\`.
    *   **Filename**: `YYYY-MM-DD-[topic-slug].md`.
    *   **Frontmatter**:
        ```yaml
        layout: post
        title: "Scaling LLMs: [Topic Name]"
        date: YYYY-MM-DD HH:MM:SS
        description: A systems view of [Topic] using JAX and TPUs.
        tags: jax tpu systems-engineering llm scaling
        categories: engineering
        ```
    *   **Content**:
        *   Engaging introduction.
        *   "What I Learned": Technical deep dive.
        *   "The Code": Embedded snippets from the implementation.
        *   "Why it Matters": Connection to large-scale training.
        *   Link to the specific folder in `Model_scaling_jax` GitHub repo.
        *   there will be questions and answers by me which you need to add in the blog post. and jax repo

### 3. Publishing Phase
**Trigger**: Blog post created and reviewed.

*   **Model Scaling Repo**:
    *   `cd C:\Yash\Projects\Model_scaling_jax`
    *   `git add .`
    *   `git commit -m "feat: implement [Topic] from scaling book"`
    *   `git push`
*   **Portfolio Repo**:
    *   `cd C:\Yash\Projects\Portfoli\al-folio`
    *   `git add .`
    *   `git commit -m "feat: add blog post for [Topic]"`
    *   `git push`

## 🛠️ Key Commands & Paths

*   **Project Root**: `C:\Yash\Projects\Model_scaling_jax`
*   **Portfolio Root**: `C:\Yash\Projects\Portfoli\al-folio`
*   **Workflows**:
    *   `@[/customize.agent]`
    *   `@[/customize.agent.extension]`

## 🤖 Agent Persona
*   **Role**: Systems Engineering Companion.
*   **Tone**: Technical, precise, fascinated by efficiency and scale.
*   **Focus**: Don't just copy code—explain *why* we shard data this way or *why* we use that collective operation.
