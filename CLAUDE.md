# AfriLION Project Guidelines (Claude Code CLI)

Welcome to the AfriLION monorepo. This file provides instructions for Claude Code CLI to ensure consistent development and adherence to project standards.

## Project Overview
AfriLION (African Language Intelligence & Open NLP) is building large-scale language models for African languages under the LocaleNLP organization.

## Key Build & Run Commands
*   **TPU Setup**: `bash training/tpu_setup.sh help`
*   **Smoke Tests**: `pytest eval/afriberta_smoke_test.py`
*   **Data Audit**: `python data/audit_cc100.py`
*   **Tokenizer Extension**: `python tokenizer/extend_llama3_tokenizer.py`
*   **Training (JAX)**: `python training/tpu_train.py`

## Code Style & Standards
*   **Linting**: Use `ruff check .` for all Python code.
*   **JAX/Flax**: Prefer JAX/Flax for any training-related code to ensure TPU compatibility.
*   **Security**: NEVER push secrets. Use `.env.example` as a template. Claude CLI must respect `.claudeignore`.
*   **Documentation**: All new features must be documented in the `docs/` directory.

## Monorepo Structure
*   `data/`: Data ingestion, filtering, and auditing scripts.
*   `tokenizer/`: SentencePiece and Llama-3 extension logic.
*   `training/`: TPU provisioning and JAX training loops.
*   `eval/`: Model benchmarking and smoke tests.
*   `docs/`: Strategy, roadmap, and community outreach.

## Security Reminders
*   Claude Code CLI should use the system's Anthropic credentials.
*   If you see an "Auth conflict" warning regarding `ANTHROPIC_AUTH_TOKEN` and `ANTHROPIC_API_KEY`, resolve it by unsetting one of the variables as per the error message.
*   Always check `.claudeignore` before performing any codebase-wide indexing.
