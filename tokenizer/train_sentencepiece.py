"""Train SentencePiece tokenizer for AfriLION.

This script trains a SentencePiece tokenizer on African language data.
Usage: python train_sentencepiece.py --input_file data.txt --vocab_size 32000
"""

import argparse
import sentencepiece as spm
import os
from pathlib import Path


def train_tokenizer(
    input_files,
    model_prefix="afrilion_tokenizer",
    vocab_size=32000,
    model_type="unigram",
    character_coverage=0.9995,
    num_threads=os.cpu_count(),
):
    """
    Train a SentencePiece tokenizer.
    
    Args:
        input_files: List of input text files or single file path
        model_prefix: Output model name prefix
        vocab_size: Size of vocabulary
        model_type: Type of tokenizer (unigram, bpe, char, word)
        character_coverage: Character coverage for non-alphabetic languages
        num_threads: Number of threads for training
    """
    # Ensure input files exist
    if isinstance(input_files, str):
        input_files = [input_files]
    
    for file in input_files:
        if not Path(file).exists():
            raise FileNotFoundError(f"Input file not found: {file}")
    
    # Join input files with comma
    input_arg = ",".join(input_files)
    
    # Training arguments
    training_args = (
        f"--input={input_arg} "
        f"--model_prefix={model_prefix} "
        f"--vocab_size={vocab_size} "
        f"--model_type={model_type} "
        f"--character_coverage={character_coverage} "
        f"--num_threads={num_threads} "
        f"--pad_id=0 "
        f"--unk_id=1 "
        f"--bos_id=2 "
        f"--eos_id=3 "
        f"--pad_piece=[PAD] "
        f"--unk_piece=[UNK] "
        f"--bos_piece=[BOS] "
        f"--eos_piece=[EOS] "
        f"--user_defined_symbols=[MASK],[CLS],[SEP]"
    )
    
    print(f"Training tokenizer with vocab size {vocab_size}...")
    print(f"Input files: {input_files}")
    print(f"Model type: {model_type}")
    
    spm.SentencePieceTrainer.train(training_args)
    
    print(f"\nTokenizer saved to {model_prefix}.model and {model_prefix}.vocab")
    
    # Test the tokenizer
    sp = spm.SentencePieceProcessor(model_file=f"{model_prefix}.model")
    test_text = "This is a test sentence for African languages."
    encoded = sp.encode(test_text, out_type=str)
    print(f"\nTest encoding: {test_text}")
    print(f"Tokens: {encoded}")
    print(f"Vocab size: {sp.vocab_size()}")


def main():
    parser = argparse.ArgumentParser(
        description="Train SentencePiece tokenizer for AfriLION"
    )
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Input text file(s) for training (comma-separated for multiple)",
    )
    parser.add_argument(
        "--model_prefix",
        type=str,
        default="afrilion_tokenizer",
        help="Output model name prefix",
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=32000,
        help="Vocabulary size",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="unigram",
        choices=["unigram", "bpe", "char", "word"],
        help="Tokenizer model type",
    )
    parser.add_argument(
        "--character_coverage",
        type=float,
        default=0.9995,
        help="Character coverage",
    )
    
    args = parser.parse_args()
    
    # Split input files if comma-separated
    input_files = args.input_file.split(",")
    
    train_tokenizer(
        input_files=input_files,
        model_prefix=args.model_prefix,
        vocab_size=args.vocab_size,
        model_type=args.model_type,
        character_coverage=args.character_coverage,
    )


if __name__ == "__main__":
    main()
