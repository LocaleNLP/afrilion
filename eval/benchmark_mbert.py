import argparse
from transformers import AutoTokenizer
import numpy as np

def calculate_fertility(tokenizer_id, text_samples):
    \"\"\"Calculate fertility of a tokenizer on a given list of texts.\"\"\"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
    fertility_scores = []
    for text in text_samples:
        words = text.split()
        if not words: continue
        tokens = tokenizer.tokenize(text)
        fertility_scores.append(len(tokens) / len(words))
    return np.mean(fertility_scores)

if __name__ == \"__main__\":
    # Sample texts for Wolof (wo), Swahili (sw), Amharic (am), Hausa (ha), Yoruba (yo)
    samples = {
        \"sw\": [\"Habari ya asubuhi, rafiki yangu?\", \"Jina langu ni AfriLION.\"],
        \"wo\": [\"Naka nga fanaane?\", \"AfriLION mooy mbooloom xam-xam.\"],
        \"am\": [\"እንዴት አደሩ?\", \"ስሜ አፍሪሊዮን ይባላል።\"],
        \"ha\": [\"Ina kwana?\", \"Sunana AfriLION.\"],
        \"yo\": [\"Bawo ni?\", \"Oruko mi ni AfriLION.\"]
    }
    
    model_id = \"bert-base-multilingual-cased\"
    print(f\"Benchmarking mBERT ({model_id}) fertility...\")
    
    for lang, texts in samples.items():
        score = calculate_fertility(model_id, texts)
        print(f\"{lang}: {score:.2f}\")
