# AfriLION Phase 0: Critical Action Plan

**Status**: In Progress
**Timeline**: Days 1-30 (Weeks 1-4)
**Location**: Casablanca, Morocco
**Date Started**: April 17, 2026
**Last Updated**: April 22, 2026

---

## 🎯 Mission-Critical Objectives

Phase 0 focuses on establishing the foundational infrastructure required for AfriLION's success:

1. **Secure compute resources** (TPU Research Cloud) - _In Progress_
2. **Build production tokenizer** (LLaMA-3 extended + custom) - _In Progress_
3. **Launch data pipeline** (CC-100 African languages) - _In Progress_
4. **Establish research partnerships** (African university labs) - _Outreach Sent_
5. **Validate technical approach** (before committing to full training) - _Ongoing_

---

## 🔥 CRITICAL: TPU Research Cloud Application

### 🚨 DEADLINE: APPLY TODAY (April 17, 2026)

**Why This Is Urgent**:
- Processing time: **2-4 weeks**
- Rolling admissions: competitive
- Required for Phase 1 training
- Free access to TPU v2/v3/v4 (up to 275 teraflops per chip)

### Application Requirements

1. **Active Google Cloud Project** ✅
   - Project ID: `afrilion-493616`
   - Project Number: `807524208601`
   - Already created and active

2. **Research Plan** (REQUIRED)

   **a) Project Overview**
   ```
   Title: AfriLION - Multilingual Language Model for 20+ African Languages

   Goal: Train a RoBERTa-base model (125M parameters) on high-quality
   African language corpora.

   Expected compute: 10M documents across 20 languages. Training for 10 epochs.

   Framework: PyTorch/XLA (or JAX)
   ```

3. **Apply at**: `sites.research.google/trc`

### TPU Setup (see `docs/infrastructure/tpu-setup-guide.md`)
- Environment configured via `training/tpu_setup.sh`
- Training loop: `training/tpu_train.py` (JAX/Flax-based)
- TPU requirements: `training/requirements_tpu.txt`

---

## ⚡ Tokenizer Strategy

### 1. Extended LLaMA-3 Tokenizer
- **Base**: LLaMA-3 (128K vocab)
- **Extension**: Add ~30K tokens for African scripts (Ge'ez, N'Ko, Tifinagh)
- **Benefit**: Zero-shot transfer from LLaMA-3, reducing training compute by 70-80%
- **Script**: `tokenizer/extend_llama3_tokenizer.py`
- **Benchmark**: `tokenizer/llama3_extension_benchmark.py`

### 2. Custom AfriLION Tokenizer
- **Algorithm**: SentencePiece BPE
- **Vocab**: 128,000 tokens
- **Training**: Equal weighting across 10 major languages (Phase 0 focus)
- **Goal**: Token fertility < 2.0 (mBERT fertility is 5.5 for Wolof)
- **Script**: `tokenizer/train_sentencepiece.py`

### Tokenizer Evaluation
- **Baseline**: mBERT fertility benchmarked via `eval/benchmark_mbert.py`
- **Target languages**: Wolof, Swahili, Hausa, Yoruba, Amharic (Phase 0)
- **Fertility target**: < 2.0 tokens/word on native text

---

## 📊 Data Pipeline: CC-100 Focus

### Initial Audit Targets
| Language | CC-100 Size | Priority |
|----------|------------|----------|
| Swahili  | 6.6 GB     | HIGH     |
| Amharic  | 862 MB     | HIGH     |
| Hausa    | 318 MB     | MEDIUM   |
| Yoruba   | 145 MB     | MEDIUM   |
| Wolof    | ~40 MB     | HIGH (low-resource focus) |

### Quality Controls
- Language detection confidence: `langdetect > 0.9`
- Deduplication: MinHash LSH (Jaccard threshold 0.85)
- Length filtering: 50-10,000 characters per document
- Script validation: Unicode block matching per language

### Coordinator-Sourced Data (Wolof Pattern)
- Budget: ~1,000 original sentences/month per coordinator
- Topics: Technical, Conversational, Narrative
- This "Seed Corpus" is the foundation for synthetic upsampling
- See `docs/submission-guide.md` for contributor workflow

---

## 🤝 University Lab Partnerships

### Target Institutions
| Institution | Country | Focus Area | Status |
|------------|---------|-----------|--------|
| Makerere AI Lab | Uganda | Luganda, Swahili | Outreach sent |
| Wits NLP Group | South Africa | Zulu, Xhosa | Outreach sent |
| HILT Lab / EthioNLP | Ethiopia | Amharic, Tigrinya | Outreach sent |
| University of Dakar | Senegal | Wolof, French | Relationship active |

### Partnership Goals
- Shared dataset annotation and quality review
- Co-authorship on benchmark paper
- Regional coordinator recruitment pipeline
- Letter of support for grant applications

See `docs/outreach-university-labs.md` for full email templates.

---

## 🚀 IMMEDIATE ACTION CHECKLIST

### Week 1 (April 17-23, 2026)
- [x] Create GitHub repository structure
- [x] Set up tokenizer scripts (`tokenizer/`)
- [x] Set up eval scripts (`eval/`)
- [x] Add TPU training infrastructure (`training/`)
- [x] Add TPU setup guide (`docs/infrastructure/tpu-setup-guide.md`)
- [x] Draft university outreach emails
- [x] Add recruitment channels doc
- [ ] Submit TPU Research Cloud application at `sites.research.google/trc`
- [ ] Fork Masakhane repositories (masakhane-io/masakhane-mt, masakhane-io/masakhane-ner)

### Week 2 (April 24-30, 2026)
- [ ] Run mBERT fertility benchmark on CC-100 Swahili + Wolof samples
- [ ] Download and audit CC-100 subsets (Swahili, Hausa, Yoruba, Amharic)
- [ ] Train initial SentencePiece tokenizer on CC-100 samples
- [ ] Set up Hugging Face organization (LocaleNLP)
- [ ] Push tokenizer artifacts to HF Hub

### Week 3 (May 1-7, 2026)
- [ ] Receive TPU TRC decision (or follow up)
- [ ] Run AfriBERTa smoke test (`eval/afriberta_smoke_test.py`)
- [ ] Begin SFT fine-tuning pipeline (`training/requirements_sft.txt`)
- [ ] Confirm at least 1 university partnership

### Week 4 (May 8-14, 2026)
- [ ] Full data pipeline audit complete
- [ ] Tokenizer fertility < 2.0 on 5 languages
- [ ] Baseline dataset uploaded to Hugging Face
- [ ] Phase 1 kickoff: begin model pre-training

---

## 📈 Phase 0 KPIs (30-Day Goal)

| KPI | Target | Status |
|-----|--------|--------|
| Compute | TPU access approved | Pending |
| Data | 50GB clean corpus | In Progress |
| Tokenization | Fertility < 2.0 (top 5 langs) | In Progress |
| Partnerships | 2 signed agreements | Outreach sent |
| Public | Baseline dataset on Hugging Face | Pending |
| Repository | All Phase 0 scripts committed | ✅ |

---

## 📁 Repository Structure (Phase 0)

```
afrilion/
├── docs/
│   ├── infrastructure/
│   │   └── tpu-setup-guide.md
│   ├── phase0-action-plan.md       # This file
│   ├── phase0-expansion-strategy.md
│   ├── outreach-university-labs.md
│   ├── recruitment-channels.md
│   ├── submission-guide.md
│   ├── coordinator-role-spec.md
│   ├── grant-master-doc-template.md
│   ├── infrastructure-and-skills.md
│   └── announcing-afrilion.md
├── tokenizer/
│   ├── extend_llama3_tokenizer.py
│   ├── llama3_extension_benchmark.py
│   └── train_sentencepiece.py
├── eval/
│   ├── benchmark_mbert.py
│   └── afriberta_smoke_test.py
├── training/
│   ├── tpu_train.py
│   ├── tpu_setup.sh
│   ├── requirements_tpu.txt
│   └── requirements_sft.txt
└── data/
       ├── README.md
    ├── pipeline.py
    └── audit_cc100.py
```

---

*Authored by AfriLION Team - Casablanca, Morocco*
*Last updated: April 22, 2026*
