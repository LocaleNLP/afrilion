# AfriLION Phase 0: Expansion & Sustainability Strategy

This document details the critical success factors and strategic pivots necessary to transition from initial infrastructure to a sustainable, community-driven NLP project.

## 1. The Data Scarcity Pivot (The "Wolof Pattern")
Automated filtering and scraping (CC-100, Common Crawl) have hard limits for low-resource languages. 
*   **The Problem**: Wolof has only ~40MB in CC-100.
*   **The Solution**: Shift from "Reviewers" to "Creators."
*   **Execution**: Budget for ~1,000 original sentences per month per coordinator across varied topics (Technical, Conversation, Narrative). This "Seed Corpus" is the foundation for synthetic upsampling.
*   **Scaling**: Apply this template to Tigrinya, Twi, and Shona in Phase 2.

## 2. Fundraising Infrastructure: The Grant Master Document
The "Master Doc" is the most valuable project artifact.
*   **Design**: A 15-page deep-dive answering the Core Five: Problem, Solution, Team, Metrics, and Financial Stewardship.
*   **Audience Strategy**: Write for Program Officers (Policy/Social Impact), not just ML peers.
*   **Efficiency**: A polished master doc reduces application adaptation time from 20 hours to 4 hours.

## 3. Funding Roadmap: Lacuna First
Strategic sequence for cash grants:
1.  **Lacuna Fund**: Primary target. Directly funds the data collection and coordination work happening NOW.
2.  **Mozilla / Google.org**: Secondary targets for higher amounts once team stability is proven via Lacuna-funded stipends.

## 4. The Geopolitics of Identity: Built in Africa
*   **Asset**: Our Casablanca base at the intersection of North, West, and Francophone Africa.
*   **Framing**: AfriLION is **Built IN Africa**, not *For* Africa. This distinction is critical for recruiting elite coordinators and attracting regional institutional partners (AfDB, etc.).

## 5. Discoverability: The "200 Downloads" Milestone
Hugging Face (HF) is our primary marketing channel.
*   **Goal**: Cross 200 downloads in 6 months to trigger search algorithm surfacing.
*   **Checklist for each Dataset/Model**:
    *   [ ] Professional Dataset Card (YAML metadata + loading examples).
    *   [ ] Interactive Demo Space (Gradio/Streamlit).
    *   [ ] "Friday Tweet" strategy targeting the Masakhane and AfricanNLP communities.

---
*Reference: Claude Strategy Session - April 2026*


---

## 6. Community Building: The MasakhaNLP Network Effect

The Masakhane community is the single most important external asset for AfriLION.

- **Why**: Pre-existing trust, quality benchmarks (MasakhaNER, MasakhaPOS, MasakhaNEWS), and 300+ African NLP contributors
- **Strategy**: Fork key repositories, contribute improvements, co-author evaluations
- **Key Repos to Fork**:
  - `masakhane-io/masakhane-mt` (Machine Translation)
  - `masakhane-io/masakhane-ner` (Named Entity Recognition)
  - `masakhane-io/masakhane-news` (News classification)
- **Contribution Plan**:
  - Add Wolof to existing benchmarks (high-impact, low-competition)
  - Submit pull requests with quality data for Tifinagh script languages
  - Attend monthly MasakhaNLP virtual meetups

---

## 7. Technical Risk Mitigation

### Risk 1: TPU Application Rejected
- **Mitigation A**: Apply simultaneously to Lambda Labs ($1.10/hr A100 GPU)
- **Mitigation B**: Use Google Colab TPU (free v2-8, limited to 3hr sessions)
- **Mitigation C**: Downscale to BERT-base (110M params) on CPU for proof-of-concept
- **Fallback**: Train tokenizer only (no GPU needed), upload to HF Hub as deliverable

### Risk 2: CC-100 Data Quality Too Low
- **Mitigation**: Switch to OPUS corpus for Swahili (higher quality, 2.3GB clean)
- **Fallback**: Use FLORES-200 evaluation set as training seed (1,000 sentences/language)
- **Long-term**: Coordinator-sourced data (Wolof Pattern) is the permanent solution

### Risk 3: Coordinator Recruitment Stalls
- **Mitigation**: Partner with university labs (Makerere, Wits) for student coordinators
- **Incentive Structure**: Academic credit + co-authorship on dataset paper
- **Minimum Viable**: 3 coordinators (Wolof, Swahili, Hausa) to launch Phase 1

---

## 8. Phase 1 Readiness Checklist

Phase 0 is complete when ALL of the following are true:

- [ ] TPU access confirmed (or viable compute alternative secured)
- [ ] Tokenizer trained: fertility < 2.0 on Swahili, Wolof, Hausa, Yoruba, Amharic
- [ ] Clean corpus: minimum 10GB across 5 languages
- [ ] At least 2 coordinators active and submitting sentences
- [ ] AfriBERTa smoke test passing (`eval/afriberta_smoke_test.py`)
- [ ] HF organization `LocaleNLP` created with at least 1 dataset published
- [ ] University partnership: at least 1 signed MOU or data-sharing agreement

---

*Reference: Claude Strategy Session - April 2026*
*Last updated: April 22, 2026*

<!-- trigger ci run -->
