# AfriLION Phase 0: Critical Action Plan

**Status**: In Progress  
**Timeline**: Days 1-30 (Weeks 1-4)  
**Location**: Casablanca, Morocco  
**Date Started**: April 17, 2026

## 🎯 Mission-Critical Objectives

Phase 0 focuses on establishing the foundational infrastructure required for AfriLION's success:

1. **Secure compute resources** (TPU Research Cloud)
2. **Build production tokenizer** (LLaMA-3 extended + custom)
3. **Launch data pipeline** (CC-100 African languages)
4. **Establish research partnerships** (African university labs)
5. **Validate technical approach** (before committing to full training)

---

## 🔥 CRITICAL: TPU Research Cloud Application

### ⏰ DEADLINE: APPLY TODAY (April 17, 2026)

**Why This Is Urgent**: [web:5][web:8]
- Processing time: **2-4 weeks**
- Rolling admissions: competitive
- Required for Phase 1 training
- Free access to TPU v2/v3/v4 (up to 275 teraflops per chip)

### Application Requirements [web:5][web:8]

1. **Active Google Cloud Project** ✅
   - Project ID: `afrilion-493616`
   - Project Number: `807524208601`
   - Already created and active

2. **Research Plan** (REQUIRED)
   You need to prepare:
   
   **a) Project Overview**
   ```
   Title: AfriLION - Multilingual Language Model for 20+ African Languages
   
   Goal: Train a RoBERTa-base model (125M parameters) on high-qualityAfrican language corpora.
   
   Expected compute: 10M documents across 20 languages. Training for 10 epochs.
   
   Framework: PyTorch/XLA (or JAX)
   ```

---

## 🛠️ Tokenizer Strategy

### 1. Extended LLaMA-3 Tokenizer
- **Base**: LLaMA-3 (128K vocab)
- **Extension**: Add ~30K tokens for African scripts (Ge'ez, N'Ko, Tifinagh)
- **Benefit**: Zero-shot transfer from LLaMA-3, reducing training compute by 70-80% [screenshot:1]

### 2. Custom AfriLION Tokenizer
- **Algorithm**: SentencePiece BPE [screenshot:3][screenshot:4]
- **Vocab**: 128,000 tokens
- **Training**: Equal weighting across 10 major languages (Phase 0 focus)
- **Goal**: Token fertility < 2.0 (mBERT fertility is 5.5 for Wolof)

---

## 📊 Data Pipeline Skeleton (CC-100 Focus) [screenshot:9][screenshot:10]

### Initial Audit Targets
- **Swahili (sw)**: 6.6 GB
- **Hausa (ha)**: 318 MB
- **Yoruba (yo)**: 145 MB
- **Amharic (am)**: 862 MB

### Quality Controls
- `langdetect` confidence > 0.9
- MinHash LSH deduplication
- Length filtering (20-2048 tokens)

---

## 🤝 University Lab Partnerships [screenshot:18]

Establishing academic links is critical for:
- Data quality review
- Recruitment of language coordinators
- Grant application credibility

**Target Labs**:
1. **Makerere AI Lab** (Uganda)
2. **Wits University** (South Africa)
3. **HILT Lab** (Ethiopia)

---

## 🚀 IMMEDIATE ACTION CHECKLIST

1. [ ] **Submit TPU Research Cloud application** at `sites.research.google/trc`
2. [ ] **Fork Masakhane repositories** (masakhane-mt, masakhane-ner)
3. [ ] **Benchmark mBERT fertility** on your collected CC-100 data
4. [ ] **Draft university outreach emails** (Template available in `/docs`)

---

## 📈 Phase 0 KPIs (30-Day Goal)

- **Compute**: TPU access approved
- **Data**: 50GB clean multilingual corpus collected
- **Tokenization**: Fertilty scores < 2.0 across top 5 languages
- **Partnerships**: 2 signed research agreements
- **Public**: Baseline dataset published on Hugging Face Hub

---

*Authored by AfriLION AI Assistant - April 17, 2026*
