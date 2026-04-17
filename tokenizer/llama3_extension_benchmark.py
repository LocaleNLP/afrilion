"""Benchmark LLaMA-3 tokenizer fertility on African languages and prepare for extension.

Fertility is defined as (number of tokens) / (number of words).
Lower is better (closer to 1.0).
"""

import argparse
from transformers import AutoTokenizer
import numpy as np


def calculate_fertility(tokenizer, text):
    """Calculate fertility of a tokenizer on a given text."""
    words = text.split()
    if not words:
        return 0
    
    tokens = tokenizer.tokenize(text)
    return len(tokens) / len(words)


def run_benchmark(model_id, texts_dict):
    """Run benchmark for a specific tokenizer."""
    print(f"
Benchmarking {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    results = {}
    for lang, text in texts_dict.items():
        fertility = calculate_fertility(tokenizer, text)
        results[lang] = fertility
        print(f"  {lang}: {fertility:.2f}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="LLaMA-3 Tokenizer Fertility Benchmark")
    parser.add_argument("--lang_data", type=str, help="Path to sample data for languages")
    
    # Sample texts for demonstration (in production, load from CC-100)
    samples = {
        "Wolof": "Wolof nag làkk la buy nuyoo ci réewum Senegaal, Gàmbi, ak Mauritanie.",
        "Swahili": "Kiswahili ni lugha ya Kibantu inayozungumzwa na watu wengi katika Afrika Mashariki.",
        "Amharic": "አማርኛ የኢትዮጵያ ይፋዊ የሥራ ቋንቋ ነው።",
        "Yoruba": "Èdè Yorùbá jẹ́ èdè kan tí a ń sọ ní apá ìwọ̀-oòrùn ilẹ̀ Áfíríkà.",
        "English": "The quick brown fox jumps over the lazy dog."
    }
    
    # Compare LLaMA-3 with mBERT (the current "gold standard" for multilingual)
    llama3_results = run_benchmark("meta-llama/Meta-Llama-3-8B", samples)
    mbert_results = run_benchmark("google-bert/bert-base-multilingual-cased", samples)
    
    print("
" + "="*40)
    print(f"{'Language':<12} | {'LLaMA-3':<10} | {'mBERT':<10} | {'Diff'}")
    print("-" * 40)
    for lang in samples.keys():
        diff = llama3_results[lang] - mbert_results[lang]
        print(f"{lang:<12} | {llama3_results[lang]:<10.2f} | {mbert_results[lang]:<10.2f} | {diff:+.2f}")


if __name__ == "__main__":
    main()
