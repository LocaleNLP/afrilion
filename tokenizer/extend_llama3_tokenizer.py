"""Extend LLaMA-3 tokenizer with African language tokens.

This script merges a new SentencePiece model (trained on African data) 
into the LLaMA-3 tokenizer.
"""

import argparse
import os
from transformers import AutoTokenizer
import sentencepiece as spm


def extend_tokenizer(
    base_model_id="meta-llama/Meta-Llama-3-8B",
    new_sp_model_path="afrilion_tokenizer.model",
    output_dir="extended_llama3_tokenizer",
):
    """
    Merge new tokens from a SentencePiece model into LLaMA-3.
    """
    print(f"Loading base tokenizer: {base_model_id}")
    base_tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    
    print(f"Loading new SentencePiece model: {new_sp_model_path}")
    new_sp = spm.SentencePieceProcessor()
    new_sp.load(new_sp_model_path)
    
    # Extract tokens from the new SentencePiece model
    new_tokens = [new_sp.id_to_piece(i) for i in range(new_sp.get_piece_size())]
    
    # Filter out tokens that are already in the base tokenizer
    existing_vocab = base_tokenizer.get_vocab()
    tokens_to_add = [t for t in new_tokens if t not in existing_vocab]
    
    print(f"Adding {len(tokens_to_add)} new tokens to the tokenizer...")
    base_tokenizer.add_tokens(tokens_to_add)
    
    # Save the extended tokenizer
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    base_tokenizer.save_pretrained(output_dir)
    print(f"Extended tokenizer saved to: {output_dir}")
    print(f"New vocabulary size: {len(base_tokenizer)}")
    
    # Note on Model Resizing:
    print("
IMPORTANT: After loading this tokenizer with a model, you MUST resize")
    print("the model's token embeddings to match the new tokenizer size:")
    print("  model.resize_token_embeddings(len(tokenizer))")


def main():
    parser = argparse.ArgumentParser(description="Extend LLaMA-3 Tokenizer")
    parser.add_argument(
        "--base_model", 
        type=str, 
        default="meta-llama/Meta-Llama-3-8B",
        help="HF model ID for LLaMA-3"
    )
    parser.add_argument(
        "--new_sp_model", 
        type=str, 
        required=True,
        help="Path to the new SentencePiece .model file"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="extended_llama3_tokenizer",
        help="Directory to save the extended tokenizer"
    )
    
    args = parser.parse_args()
    extend_tokenizer(args.base_model, args.new_sp_model, args.output_dir)


if __name__ == "__main__":
    main()
