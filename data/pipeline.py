#!/usr/bin/env python3
"""
AfriLION Data Pipeline
======================
Stage 1 : Download CC-100 African language subsets
Stage 2 : Language-ID filter  (langdetect confidence > 0.9)
Stage 3 : Text cleaning
Stage 4 : Deduplication via MinHash LSH
Stage 5 : Length filter (20–2048 tokens by whitespace)
Stage 6 : Shard to JSONL
Stage 7 : Upload to Hugging Face Datasets

Usage:
    pip install datasets langdetect datasketch tqdm huggingface_hub
    python pipeline.py --langs wo sw ha yo am --output_dir ./processed
"""

import argparse
import gzip
import hashlib
import json
import logging
import os
import re
import unicodedata
from pathlib import Path
from typing import Iterator, List

import requests
from datasets import Dataset, DatasetDict
from datasketch import MinHash, MinHashLSH
from langdetect import detect_langs
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CC100_BASE = "https://data.statmt.org/cc-100"

LANG_META = {
    "wo": {"name": "Wolof",   "cc100_id": "wo"},
    "sw": {"name": "Swahili", "cc100_id": "sw"},
    "ha": {"name": "Hausa",   "cc100_id": "ha"},
    "yo": {"name": "Yoruba",  "cc100_id": "yo"},
    "am": {"name": "Amharic", "cc100_id": "am"},
}

MIN_TOKENS   = 20
MAX_TOKENS   = 2048
LANGDET_CONF = 0.9
MINHASH_THRESHOLD = 0.85
MINHASH_PERMS     = 128
SHARD_SIZE        = 100_000        # lines per JSONL shard
HF_REPO_ID        = "AfriLION/afrilion-corpus"


# ---------------------------------------------------------------------------
# Stage 1: Download
# ---------------------------------------------------------------------------
def download_cc100(lang_id: str, output_dir: Path) -> Path:
    """Download CC-100 corpus for a language; returns local .txt.xz path."""
    fname = f"{lang_id}.txt.xz"
    url   = f"{CC100_BASE}/{fname}"
    dest  = output_dir / "raw" / fname
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists():
        log.info(f"[{lang_id}] Already downloaded: {dest}")
        return dest

    log.info(f"[{lang_id}] Downloading {url} ...")
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(dest, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc=lang_id
        ) as pbar:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                pbar.update(len(chunk))
    return dest


def iter_lines(path: Path) -> Iterator[str]:
    """Yield lines from .txt.xz or plain .txt file."""
    import lzma
    opener = lzma.open if str(path).endswith(".xz") else open
    with opener(path, "rt", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if line:
                yield line


# ---------------------------------------------------------------------------
# Stage 2: Language-ID filter
# ---------------------------------------------------------------------------
def langid_ok(text: str, target_lang: str) -> bool:
    """Return True if langdetect gives target_lang with confidence > LANGDET_CONF."""
    try:
        langs = detect_langs(text)
        for l in langs:
            if l.lang == target_lang and l.prob >= LANGDET_CONF:
                return True
    except Exception:
        pass
    return False


# ---------------------------------------------------------------------------
# Stage 3: Text cleaning
# ---------------------------------------------------------------------------
URL_RE    = re.compile(r"https?://\S+|www\.\S+")
HTML_RE   = re.compile(r"<[^>]+>")
WS_RE     = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Remove URLs, HTML tags, normalize whitespace, strip control chars."""
    text = URL_RE.sub(" ", text)
    text = HTML_RE.sub(" ", text)
    # Remove control characters except newlines
    text = "".join(
        ch for ch in text
        if unicodedata.category(ch)[0] != "C" or ch == "\n"
    )
    text = WS_RE.sub(" ", text).strip()
    return text


# ---------------------------------------------------------------------------
# Stage 4 + 5: MinHash dedup + length filter
# ---------------------------------------------------------------------------
def tokenize_simple(text: str) -> List[str]:
    """Whitespace tokenize for length counting and shingling."""
    return text.split()


def make_minhash(tokens: List[str], num_perm: int = MINHASH_PERMS) -> MinHash:
    m = MinHash(num_perm=num_perm)
    for shingle in _shingles(tokens, k=5):
        m.update(shingle.encode("utf-8"))
    return m


def _shingles(tokens: List[str], k: int = 5) -> List[str]:
    return [" ".join(tokens[i:i+k]) for i in range(max(1, len(tokens) - k + 1))]


def process_language(
    lang_id: str,
    raw_path: Path,
    output_dir: Path,
    lsh: MinHashLSH,
) -> List[dict]:
    """
    Full per-language pipeline:
      download → detect → clean → dedup → length-filter → collect records
    """
    records = []
    seen_hashes: set = set()
    total = filtered_lang = filtered_len = filtered_dup = 0

    for raw_line in iter_lines(raw_path):
        total += 1

        # --- langdetect ---
        if not langid_ok(raw_line, lang_id):
            filtered_lang += 1
            continue

        # --- clean ---
        text = clean_text(raw_line)
        if not text:
            continue

        # --- length filter ---
        tokens = tokenize_simple(text)
        if not (MIN_TOKENS <= len(tokens) <= MAX_TOKENS):
            filtered_len += 1
            continue

        # --- MinHash dedup ---
        mh = make_minhash(tokens)
        key = hashlib.md5(text.encode()).hexdigest()
        if key in seen_hashes:
            filtered_dup += 1
            continue
        dupes = lsh.query(mh)
        if dupes:
            filtered_dup += 1
            continue
        lsh.insert(key, mh)
        seen_hashes.add(key)

        records.append({
            "text": text,
            "lang": lang_id,
            "lang_name": LANG_META[lang_id]["name"],
            "token_count": len(tokens),
            "source": "cc100",
        })

    log.info(
        f"[{lang_id}] total={total:,} "
        f"langid_dropped={filtered_lang:,} "
        f"len_dropped={filtered_len:,} "
        f"dup_dropped={filtered_dup:,} "
        f"kept={len(records):,}"
    )
    return records


# ---------------------------------------------------------------------------
# Stage 6: Write JSONL shards
# ---------------------------------------------------------------------------
def write_jsonl_shards(records: List[dict], lang_id: str, output_dir: Path) -> List[Path]:
    shard_dir = output_dir / "shards" / lang_id
    shard_dir.mkdir(parents=True, exist_ok=True)
    shards = []
    for i in range(0, len(records), SHARD_SIZE):
        shard_path = shard_dir / f"{lang_id}_{i // SHARD_SIZE:04d}.jsonl"
        with open(shard_path, "w", encoding="utf-8") as f:
            for rec in records[i:i + SHARD_SIZE]:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        shards.append(shard_path)
        log.info(f"[{lang_id}] Wrote shard {shard_path} ({min(SHARD_SIZE, len(records)-i):,} lines)")
    return shards


# ---------------------------------------------------------------------------
# Stage 7: Upload to Hugging Face
# ---------------------------------------------------------------------------
def upload_to_hf(all_records: dict, repo_id: str = HF_REPO_ID):
    """Build a DatasetDict split by language and push to HF Hub."""
    from huggingface_hub import HfApi
    ds_dict = {}
    for lang_id, records in all_records.items():
        if records:
            ds_dict[lang_id] = Dataset.from_list(records)
    dataset = DatasetDict(ds_dict)
    log.info(f"Pushing to HF Hub: {repo_id} ...")
    dataset.push_to_hub(repo_id, private=False)
    log.info("Upload complete.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="AfriLION CC-100 Data Pipeline")
    parser.add_argument(
        "--langs", nargs="+", default=list(LANG_META.keys()),
        help="Language codes to process (default: all)"
    )
    parser.add_argument(
        "--output_dir", default="./processed",
        help="Root output directory"
    )
    parser.add_argument(
        "--skip_download", action="store_true",
        help="Skip download step (use cached files)"
    )
    parser.add_argument(
        "--upload_hf", action="store_true",
        help="Upload cleaned dataset to Hugging Face Hub"
    )
    parser.add_argument(
        "--no_langid", action="store_true",
        help="Skip langdetect filtering (faster, less precise)"
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Shared LSH across all languages to catch cross-lingual dupes
    lsh = MinHashLSH(threshold=MINHASH_THRESHOLD, num_perm=MINHASH_PERMS)

    all_records = {}
    for lang_id in args.langs:
        if lang_id not in LANG_META:
            log.warning(f"Unknown lang: {lang_id}, skipping.")
            continue

        log.info(f"\n{'='*60}\nProcessing: {LANG_META[lang_id]['name']} ({lang_id})\n{'='*60}")

        # Stage 1: Download
        if not args.skip_download:
            raw_path = download_cc100(lang_id, output_dir)
        else:
            raw_path = output_dir / "raw" / f"{lang_id}.txt.xz"

        # Stages 2-5
        records = process_language(lang_id, raw_path, output_dir, lsh)

        # Stage 6: Write shards
        write_jsonl_shards(records, lang_id, output_dir)
        all_records[lang_id] = records

    # Stage 7: Upload
    if args.upload_hf:
        upload_to_hf(all_records)

    # Summary
    log.info("\n" + "="*60)
    log.info("PIPELINE SUMMARY")
    log.info("="*60)
    for lang_id, records in all_records.items():
        log.info(f"  {LANG_META[lang_id]['name']:10s} ({lang_id}): {len(records):>8,} clean sentences")
    total = sum(len(v) for v in all_records.values())
    log.info(f"  {'TOTAL':>12s}        : {total:>8,} clean sentences")
    log.info("="*60)


if __name__ == "__main__":
    main()
