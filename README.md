# AfriLION

**Large-scale African Language Intelligence & Open NLP**

AfriLION is a monorepo dedicated to building state-of-the-art language models for African languages. Hosted by [LocaleNLP](https://github.com/LocaleNLP), this project focuses on data scarcity solutions, production-grade tokenization, and distributed training on TPU infrastructure.

## 🌍 Mission
To bridge the digital divide for African languages by providing the infrastructure, datasets, and models required for high-fidelity natural language processing in low-resource contexts.

## 📂 Monorepo Structure
- `data/`: Ingestion, filtering, and auditing for multilingual corpora (e.g., CC-100 subsets).
- `tokenizer/`: Custom SentencePiece and Llama-3 extensions with Ge'ez and West African character coverage.
- `training/`: TPU-optimized training loops using JAX and Flax.
- `eval/`: Benchmarking pipelines and fertility score analysis.
- `docs/`: Strategic planning, roadmap, and university outreach templates.

## 🚀 Phase 0: Foundations (In Progress)
We are currently in Phase 0, establishing the core infrastructure:
- **Compute**: Secured via TPU Research Cloud (TRC).
- **Data**: Target of 50GB clean multilingual corpus for top 5 languages.
- **Tokenization**: Fertility score optimization (< 2.0).
- **Sustainability**: Launching the \"Wolof Pattern\" for community-driven data creation.

## 🛠 Key Commands
Refer to `CLAUDE.md` for full development guidelines.
- **TPU Setup**: `bash training/tpu_setup.sh help`
- **Smoke Tests**: `pytest eval/afriberta_smoke_test.py`
- **Data Audit**: `python data/audit_cc100.py`
- **Linting**: `ruff check .`

## 🤝 Getting Involved
We are recruiting **Regional Coordinators** in Dakar, Casablanca, Nairobi, Lagos, and Addis Ababa. See `docs/coordinator-role-spec.md` for details.

---
© 2026 LocaleNLP. Built IN Africa.
