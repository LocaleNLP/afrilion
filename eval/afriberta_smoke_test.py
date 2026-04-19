#!/usr/bin/env python3
"""
AfriBERTa Smoke Test — AfriLION End-to-End Pipeline Debugger
=============================================================
This is your CHEAPEST debugging tool. A 3-epoch fine-tune on a Colab T4 GPU
acts as an end-to-end smoke test of your ENTIRE data pipeline:
    download -> detect -> filter -> dedup -> tokenize

If perplexity does NOT drop 25-30% after epoch 1, something is wrong upstream.
Common causes:
    - Language detection threshold too loose (langdetect noise in corpus)
    - character_coverage too low (Ge'ez byte-fallback tokens in Amharic)
    - Upsampling weights off (Wolof starved of vocab budget)
    - Wrong model_type (unigram instead of BPE)

NOTE: Do NOT use mBERT's WordPiece tokenizer as a base. WordPiece is
encoder-only and incompatible with causal LM training. AfriBERTa uses
RoBERTa architecture (encoder only) which is fine for classification
debugging, but the final AfriLION model uses LLaMA-3 BPE.

Usage:
    pip install transformers datasets torch sentencepiece
    python afriberta_smoke_test.py \\
        --data_dir ./processed/shards \\
        --tokenizer_model ../tokenizer/afrilion_tokenizer.model \\
        --lang am sw wo  # Test at least one Ge'ez, one high-resource, one low-resource
"""
import argparse
import json
import random
import sys
from pathlib import Path
from typing import List

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
AFRIBERTA_MODEL = "castorini/afriberta_base"  # Public model on HF Hub
MAX_SEQ_LEN = 256
N_SAMPLES_PER_LANG = 5_000  # Keep it small for a quick smoke test
EPOCHS = 3
BATCH_SIZE = 16
LEARNING_RATE = 2e-5

# Expected perplexity drop thresholds
EXPECTED_PPL_DROP_EPOCH1 = 0.20   # >= 20% reduction from baseline
WARNING_PPL_DROP_EPOCH1  = 0.10   # < 10% = something is definitely wrong


# ---------------------------------------------------------------------------
# Data loading with random QA sampling
# ---------------------------------------------------------------------------
class AfriCorpusDataset(Dataset):
    """
    Loads sentences from JSONL shards. Automatically does random document
    sampling (10 docs per language) for manual quality inspection.
    The most important debugging step: READ the actual documents before training.
    """
    def __init__(self, data_dir: Path, langs: List[str], tokenizer, max_len: int = MAX_SEQ_LEN):
        self.samples = []
        self.tokenizer = tokenizer
        self.max_len = max_len

        for lang in langs:
            lang_dir = data_dir / "shards" / lang
            if not lang_dir.exists():
                print(f"[WARN] No shards for lang={lang}, skipping.")
                continue

            lines = []
            for shard in sorted(lang_dir.glob("*.jsonl")):
                with open(shard, encoding="utf-8") as f:
                    for line in f:
                        obj = json.loads(line)
                        lines.append(obj["text"])

            if not lines:
                print(f"[WARN] No data found for lang={lang}")
                continue

            # === RANDOM QA SAMPLING ===
            # Critical: sample 10 random docs and print them BEFORE training.
            # If you see mixed-language content, HTML artifacts, or repetitive
            # filler text, your langdetect threshold is too loose.
            print(f"\n{'='*60}")
            print(f"[QA] Random document sample for lang={lang} ({len(lines):,} docs total)")
            print(f"{'='*60}")
            qa_samples = random.sample(lines, min(10, len(lines)))
            for i, doc in enumerate(qa_samples):
                print(f"  [{i+1}] {doc[:200]}")
            print()

            # Subsample for speed
            random.shuffle(lines)
            selected = lines[:N_SAMPLES_PER_LANG]
            self.samples.extend([(text, lang) for text in selected])
            print(f"[INFO] Loaded {len(selected):,} samples for {lang}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text, lang = self.samples[idx]
        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {k: v.squeeze(0) for k, v in encoding.items()}


# ---------------------------------------------------------------------------
# Fertility check before training
# ---------------------------------------------------------------------------
def check_fertility(tokenizer, langs: List[str], test_sentences: dict) -> dict:
    """
    Check fertility (tokens per word) before training.
    This is the first diagnostic: if Wolof fertility > 3.0, your tokenizer
    is not properly trained or the vocab budget allocation is wrong.

    Expected targets after AfriLION tokenizer:
        English:  ~1.2x
        Swahili:  ~1.4x
        Wolof:    ~1.5x  (was 4.0+ with base LLaMA-3)
        Amharic:  ~1.8x  (was 4.2+ with base LLaMA-3 / mBERT)
    """
    print("\n--- Fertility Check (tokens per whitespace word) ---")
    results = {}
    for lang, text in test_sentences.items():
        words = text.split()
        tokens = tokenizer.encode(text)
        fertility = len(tokens) / max(len(words), 1)
        status = "OK" if fertility < 2.5 else "WARN" if fertility < 4.0 else "FAIL"
        print(f"  [{status}] {lang:10s}: {fertility:.2f}x  ({len(words)} words -> {len(tokens)} tokens)")
        results[lang] = fertility
    return results


# ---------------------------------------------------------------------------
# Perplexity tracker
# ---------------------------------------------------------------------------
def compute_perplexity(eval_loss: float) -> float:
    import math
    return math.exp(eval_loss)


class PplCallback:
    """Track perplexity drop per epoch and warn if insufficient."""
    def __init__(self):
        self.epoch_ppls = []
        self.baseline_ppl = None

    def on_epoch_end(self, epoch: int, eval_loss: float):
        ppl = compute_perplexity(eval_loss)
        self.epoch_ppls.append(ppl)
        if self.baseline_ppl is None:
            self.baseline_ppl = ppl
            print(f"  [Epoch {epoch}] Baseline PPL: {ppl:.2f}")
        else:
            drop = (self.baseline_ppl - ppl) / self.baseline_ppl
            if drop < WARNING_PPL_DROP_EPOCH1 and epoch == 1:
                print(f"  [CRITICAL][Epoch {epoch}] PPL drop only {drop*100:.1f}% (expected >=20%)")
                print("  [CRITICAL] Likely causes:")
                print("  [CRITICAL]   1. Language detection threshold too loose -> noise in corpus")
                print("  [CRITICAL]   2. character_coverage too low -> Ge'ez byte-fallbacks")
                print("  [CRITICAL]   3. Inspect random document samples printed above")
            elif drop < EXPECTED_PPL_DROP_EPOCH1 and epoch == 1:
                print(f"  [WARN][Epoch {epoch}] PPL drop {drop*100:.1f}% (expected >=20%, got <25%)")
            else:
                print(f"  [OK][Epoch {epoch}] PPL: {ppl:.2f} (drop: {drop*100:.1f}%)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="AfriLION AfriBERTa smoke test")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--langs", nargs="+", default=["am", "sw", "wo"])
    parser.add_argument("--output_dir", type=str, default="./smoke_test_output")
    args = parser.parse_args()

    print("\n" + "="*60)
    print("AfriLION AfriBERTa Smoke Test")
    print("="*60)
    print(f"Model: {AFRIBERTA_MODEL}")
    print(f"Languages: {args.langs}")
    print(f"Samples per lang: {N_SAMPLES_PER_LANG:,}")
    print(f"Epochs: {EPOCHS}")

    # Load AfriBERTa tokenizer (for this smoke test only)
    # NOTE: AfriBERTa uses its own tokenizer which covers African langs well.
    # The final AfriLION training uses our custom SentencePiece BPE tokenizer.
    print(f"\nLoading {AFRIBERTA_MODEL}...")
    tokenizer = AutoTokenizer.from_pretrained(AFRIBERTA_MODEL)
    model = AutoModelForMaskedLM.from_pretrained(AFRIBERTA_MODEL)

    # Fertility check
    test_sentences = {
        "English":  "The quick brown fox jumps over the lazy dog.",
        "Swahili":  "Habari yako? Ninafurahi kukuona tena leo.",
        "Wolof":    "Nanga def, naka sa fan bi? Mangi fi rekk.",
        "Amharic":  "\u1230\u120b\u121d \u12d3\u1208\u121d \u12d0\u1298\u1235\u1275 \u12a5\u12a8\u120d\u12a8\u1208",
        "Hausa":    "Sannu da zuwa. Yaya kuke?",
    }
    check_fertility(tokenizer, args.langs, test_sentences)

    # Load dataset
    data_dir = Path(args.data_dir)
    dataset = AfriCorpusDataset(data_dir, args.langs, tokenizer)

    if len(dataset) == 0:
        print("[CRITICAL] No data loaded. Check --data_dir path.")
        sys.exit(1)

    # Split train/eval
    split = int(len(dataset) * 0.9)
    train_data = torch.utils.data.Subset(dataset, range(split))
    eval_data  = torch.utils.data.Subset(dataset, range(split, len(dataset)))

    # Data collator for MLM
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )

    # Training
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        evaluation_strategy="epoch",
        logging_strategy="epoch",
        save_strategy="no",
        report_to="none",
        fp16=torch.cuda.is_available(),
    )

    ppl_tracker = PplCallback()

    class PplTrainer(Trainer):
        def on_evaluate(self, args, state, control, metrics=None, **kwargs):
            if metrics and "eval_loss" in metrics:
                ppl_tracker.on_epoch_end(state.epoch, metrics["eval_loss"])

    trainer = PplTrainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=eval_data,
        data_collator=data_collator,
    )

    print("\nStarting 3-epoch smoke test...")
    trainer.train()

    print("\n" + "="*60)
    print("SMOKE TEST COMPLETE")
    print(f"Final PPL trajectory: {[f'{p:.2f}' for p in ppl_tracker.epoch_ppls]}")
    if len(ppl_tracker.epoch_ppls) >= 2:
        final_drop = (ppl_tracker.epoch_ppls[0] - ppl_tracker.epoch_ppls[-1]) / ppl_tracker.epoch_ppls[0]
        print(f"Total PPL reduction: {final_drop*100:.1f}%")
        if final_drop >= 0.25:
            print("[PASS] Pipeline looks clean. Proceed to full training.")
        else:
            print("[FAIL] Insufficient PPL drop. Inspect random doc samples above.")
    print("="*60)


if __name__ == "__main__":
    main()
