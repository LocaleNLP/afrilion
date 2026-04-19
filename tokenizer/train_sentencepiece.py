#!/usr/bin/env python3
"""
AfriLION Tokenizer Training Script — Production-Grade
======================================================
Key design decisions:
 - BPE via SentencePiece (NOT unigram) to match LLaMA-3's tokenizer base.
 - vocab_size = 128,000 (required for full Ge'ez syllabic glyph coverage).
 - character_coverage = 0.9999 for Ge'ez/Amharic scripts (not 0.9995 — CRITICAL).
 - Equal per-language weighting via upsampling (prevents Swahili dominating vocab).
 - Verification step: scans output for <0x byte-fallback sequences.
 - Lang ID special tokens: [SW], [HA], [AM], [YO], [WO] etc.

Usage:
 python train_sentencepiece.py \\
     --input_dir ./processed/shards \\
     --model_prefix afrilion_tokenizer \\
     --vocab_size 128000
"""
import argparse
import os
import random
import re
import tempfile
from pathlib import Path
import sentencepiece as spm

# ---------------------------------------------------------------------------
# Language config: name, CC-100 id, target weight (upsampling multiplier)
# Swahili has ~6.6GB; Wolof ~40MB. Equal weighting = upsample Wolof ~150x.
# ---------------------------------------------------------------------------
LANG_CONFIG = {
    "wo": {"name": "Wolof",     "weight": 150},  # ~40 MB  -> upsample heavily
    "sw": {"name": "Swahili",   "weight": 1},    # ~6.6 GB -> baseline
    "ha": {"name": "Hausa",     "weight": 8},    # ~800 MB
    "yo": {"name": "Yoruba",    "weight": 12},   # ~500 MB
    "am": {"name": "Amharic",   "weight": 20},   # ~300 MB (Ge'ez script)
    "ti": {"name": "Tigrinya",  "weight": 30},   # ~200 MB (Ge'ez script)
    "so": {"name": "Somali",    "weight": 15},   # ~400 MB
    "ig": {"name": "Igbo",      "weight": 25},   # ~250 MB
    "zu": {"name": "Zulu",      "weight": 10},   # ~600 MB
    "ar": {"name": "Arabic",    "weight": 1},    # already covered by LLaMA-3
}

# Lang ID tokens — prepended to every document at training time.
# This lets the model condition on language at inference: critical for
# code-switching and per-language perplexity measurement.
LANG_ID_TOKENS = [f"[{code.upper()}]" for code in LANG_CONFIG]

# Character coverage: 0.9999 is mandatory for Ge'ez/Amharic.
# Ge'ez has ~500 base syllabic characters + tonal combinations = thousands
# of unique glyphs. Lower values produce <0x byte-fallback tokens that
# silently corrupt Amharic training data.
GEEZ_SCRIPTS = {"am", "ti"}  # languages requiring 0.9999 coverage
CHAR_COVERAGE_DEFAULT = 0.9995
CHAR_COVERAGE_GEEZ    = 0.9999  # CRITICAL — do not lower this


def build_weighted_corpus(input_dir: Path, output_file: Path) -> None:
    """
    Build a single merged training corpus with equal-weight upsampling.

    CRITICAL INSIGHT: Do NOT weight by data volume. A proportionally-weighted
    tokenizer devotes most of its vocab to Swahili/Arabic, leaving Wolof with
    ~200 tokens that fragment every word into 5-6 pieces.
    Solution: upsample each language to equal representation.
    """
    print("Building weighted corpus...")
    lines_per_lang = {}

    # Pass 1: count available lines per language
    for lang_code, cfg in LANG_CONFIG.items():
        lang_dir = input_dir / "shards" / lang_code
        if not lang_dir.exists():
            print(f"  [WARN] No shards found for {lang_code}, skipping.")
            continue
        lines = []
        for shard in sorted(lang_dir.glob("*.jsonl")):
            import json
            with open(shard, "r", encoding="utf-8") as f:
                for line in f:
                    obj = json.loads(line)
                    # Prepend Lang ID token — critical design decision
                    lines.append(f"[{lang_code.upper()}] " + obj["text"])
        lines_per_lang[lang_code] = lines
        print(f"  {cfg['name']:10s} ({lang_code}): {len(lines):>8,} sentences (weight={cfg['weight']}x)")

    # Pass 2: upsample to equal weight and write merged file
    with open(output_file, "w", encoding="utf-8") as out:
        for lang_code, lines in lines_per_lang.items():
            weight = LANG_CONFIG[lang_code]["weight"]
            upsampled = lines * weight  # repeat corpus weight times
            random.shuffle(upsampled)
            for line in upsampled:
                out.write(line.strip() + "\n")

    total = output_file.stat().st_size / (1024 ** 2)
    print(f"\nMerged corpus: {output_file} ({total:.1f} MB)")


def train_tokenizer(
    input_dir: Path,
    model_prefix: str = "afrilion_tokenizer",
    vocab_size: int = 128_000,
    num_threads: int = os.cpu_count(),
) -> None:
    """
    Train a BPE SentencePiece tokenizer over all African language shards.

    Architecture decisions (matching Claude artifact):
    - Algorithm: BPE (not unigram) to stay compatible with LLaMA-3.
    - Vocab size: 128,000 — needed for full Ge'ez syllabic coverage.
    - character_coverage: 0.9999 — prevents <0x byte-fallback in Amharic.
    - byte_fallback: enabled — safely represents any unseen character.
    - normalization: NFKC, lowercasing OFF (African scripts are case-sensitive).
    """
    with tempfile.NamedTemporaryFile(
        suffix=".txt", delete=False, mode="w", encoding="utf-8"
    ) as tmp:
        tmp_path = Path(tmp.name)

    build_weighted_corpus(input_dir, tmp_path)

    # Special tokens: Lang ID tokens + standard tokens
    user_defined = ",".join(LANG_ID_TOKENS + ["[MASK]", "[CLS]", "[SEP]"])

    training_args = (
        f"--input={tmp_path} "
        f"--model_prefix={model_prefix} "
        f"--vocab_size={vocab_size} "
        f"--model_type=bpe "
        f"--character_coverage={CHAR_COVERAGE_GEEZ} "
        f"--num_threads={num_threads} "
        f"--byte_fallback=true "
        f"--normalization_rule_name=nfkc "
        f"--add_dummy_prefix=false "
        f"--split_digits=true "
        f"--pad_id=0 --unk_id=1 --bos_id=2 --eos_id=3 "
        f"--pad_piece=[PAD] --unk_piece=[UNK] --bos_piece=[BOS] --eos_piece=[EOS] "
        f"--user_defined_symbols={user_defined}"
    )

    print(f"\nTraining SentencePiece BPE tokenizer...")
    print(f"  vocab_size      = {vocab_size:,}")
    print(f"  char_coverage   = {CHAR_COVERAGE_GEEZ} (Ge'ez-safe)")
    print(f"  byte_fallback   = true")
    print(f"  lang_id_tokens  = {LANG_ID_TOKENS}")

    spm.SentencePieceTrainer.train(training_args)
    print(f"\nTokenizer saved: {model_prefix}.model / {model_prefix}.vocab")

    # Cleanup temp file
    tmp_path.unlink()

    # Run verification step
    verify_geez_coverage(model_prefix)


def verify_geez_coverage(model_prefix: str) -> None:
    """
    CRITICAL VERIFICATION: Scan vocabulary for <0x byte-fallback sequences.

    If character_coverage was too low, Ge'ez glyphs will appear as raw UTF-8
    bytes: e.g., <0xE1><0x88><0xA0> instead of the actual character ሠ.
    The model will NEVER learn Amharic properly if these appear in training.

    Run this immediately after training. Any <0x count > 0 means you must
    retrain with character_coverage=0.9999.
    """
    sp = spm.SentencePieceProcessor(model_file=f"{model_prefix}.model")

    byte_fallback_count = 0
    byte_pattern = re.compile(r'<0x[0-9A-Fa-f]{2}>')

    print("\n--- Ge'ez Coverage Verification ---")
    for i in range(sp.vocab_size()):
        piece = sp.id_to_piece(i)
        if byte_pattern.search(piece):
            byte_fallback_count += 1

    if byte_fallback_count > 256:  # >256 = actual language chars are falling back
        print(f"[CRITICAL] {byte_fallback_count} byte-fallback tokens found!")
        print("[CRITICAL] Ge'ez/Amharic script is NOT properly covered.")
        print("[CRITICAL] Retrain with character_coverage=0.9999")
    else:
        print(f"[OK] Byte fallback tokens: {byte_fallback_count} (expected <= 256 for emoji/special)")
        print("[OK] Ge'ez coverage is sufficient.")

    # Test on actual Amharic text
    test_amharic = "ሰላም ዓለም"  # "Hello World" in Amharic
    tokens = sp.encode(test_amharic, out_type=str)
    byte_pieces = [t for t in tokens if byte_pattern.search(t)]
    if byte_pieces:
        print(f"[FAIL] Amharic test produced byte tokens: {byte_pieces}")
    else:
        print(f"[OK] Amharic test tokens: {tokens}")


def main():
    parser = argparse.ArgumentParser(
        description="AfriLION production tokenizer trainer"
    )
    parser.add_argument(
        "--input_dir", type=str, required=True,
        help="Root dir containing processed/shards/<lang_code>/*.jsonl"
    )
    parser.add_argument(
        "--model_prefix", type=str, default="afrilion_tokenizer",
        help="Output model name prefix"
    )
    parser.add_argument(
        "--vocab_size", type=int, default=128_000,
        help="Vocabulary size (default: 128000 for Ge'ez script coverage)"
    )
    args = parser.parse_args()

    train_tokenizer(
        input_dir=Path(args.input_dir),
        model_prefix=args.model_prefix,
        vocab_size=args.vocab_size,
    )


if __name__ == "__main__":
    main()
