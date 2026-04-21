# AfriLION: Infrastructure, Skills, and API Roadmap

This document outlines the critical technical components, necessary expertise, and API integrations required to successfully execute the AfriLION project using TPU resources and Claude Code CLI.

## 1. Core API Integrations

### **Machine Learning & Model Management**
*   **Hugging Face Hub API**: 
    *   *Purpose*: Hosting models, datasets, and Spaces.
    *   *Usage*: Programmatic uploading of model checkpoints and versioning of the `AfriCorpus` dataset.
*   **Weights & Biases (W&B) API**:
    *   *Purpose*: Experiment tracking, loss visualization, and hardware monitoring.
    *   *Usage*: Integrated into `training/tpu_train.py` to monitor long-running TPU jobs.

### **Infrastructure & Compute**
*   **Google Cloud SDK / TPU API**:
    *   *Purpose*: Provisioning and managing Cloud TPU VMs (v4/v5e/v6e).
    *   *Usage*: Handled via `training/tpu_setup.sh` and `gcloud` CLI.
*   **Google Cloud Storage (GCS) API**:
    *   *Purpose*: Large-scale artifact storage.
    *   *Usage*: Storing training logs and multi-gigabyte checkpoints that exceed GitHub limits.

### **Data Pipeline & Quality**
*   **Label Studio API**:
    *   *Purpose*: Human-in-the-loop data auditing and correction.
    *   *Usage*: Exporting cleaned data for model fine-tuning.
*   **Hugging Face Datasets library**:
    *   *Purpose*: Efficient data streaming to TPUs.
    *   *Usage*: `load_dataset(streaming=True)` to handle TB-scale corpora without local disk bottlenecks.

## 2. Necessary Skills & Expertise

### **Distributed Machine Learning**
*   **JAX/Flax Mastery**: Essential for maximizing TPU throughput. Understanding `jax.jit`, `jax.pmap`, and `jax.sharding` is non-negotiable for v4 clusters.
*   **Sharding Strategies**: Knowledge of Data Parallelism (DP) and Fully Sharded Data Parallel (FSDP) to fit large models across multiple TPU chips.

### **Multilingual NLP & Linguistics**
*   **African Language Tokenization**: Specialized knowledge in Ge'ez (Amharic/Tigrinya) script handling and morphologically rich languages (Wolof/Swahili).
*   **Data Quality Auditing**: Ability to build automated "smoke tests" for data poisoning, code-switching detection, and toxicity filtering in low-resource contexts.

### **DevOps & Security**
*   **Claude Code CLI Workflow**: Expert usage of CLI-driven development to iterate on training scripts without leaving the terminal.
*   **Secret Management**: Hardening of CI/CD pipelines and local environments (using `.gitignore`, `.claudeignore`, and `.env` files) to prevent credential leaks.

## 3. Project Roadmap Components (MPCs)

*   **Tokenizer Infrastructure**: (Phase 0) SentencePiece BPE with 0.9999 coverage.
*   **Pre-training Pipeline**: (Phase 1) Streaming JAX-based Causal LM training.
*   **Evaluation Framework**: (Phase 2) Zero-shot and few-shot benchmarking on African-specific tasks (e.g., AfriSenti, MasakhaNER).
*   **Community Engagement**: Open-source publishing of results, blog posts, and cleaned subsets to build the LocaleNLP brand.

---
*Last updated: April 22, 2026*
