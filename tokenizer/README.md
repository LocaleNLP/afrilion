# AfriLION Tokenizer

This directory contains scripts for training and managing tokenizers for the AfriLION (African Language Integrated Operation Network) project.

## Overview

The AfriLION tokenizer is designed to handle multiple African languages with diverse scripts and linguistic features. We use SentencePiece for subword tokenization, which provides robust handling of:

- Multiple writing systems (Latin, Arabic, Ge'ez, N'Ko, Tifinagh, etc.)
- Rich morphology common in African languages
- Low-resource language scenarios
- Code-switching between languages

## Quick Start

### Installation

```bash
pip install sentencepiece
```

### Training a Tokenizer

Basic usage:

```bash
python train_sentencepiece.py \
  --input_file /path/to/training/data.txt \
  --model_prefix afrilion_tokenizer \
  --vocab_size 32000
```

### Training with Multiple Files

```bash
python train_sentencepiece.py \
  --input_file data1.txt,data2.txt,data3.txt \
  --model_prefix afrilion_multilingual \
  --vocab_size 50000 \
  --model_type unigram
```

## Configuration Options

### Vocabulary Size

- **Default**: 32,000 tokens
- **Recommended for multilingual**: 50,000-100,000 tokens
- **For single language**: 16,000-32,000 tokens

```bash
--vocab_size 50000
```

### Model Type

Choose between different tokenization algorithms:

- **unigram** (default): Best for morphologically rich languages
- **bpe**: Byte Pair Encoding, good balance
- **char**: Character-level tokenization
- **word**: Word-level tokenization

```bash
--model_type unigram
```

### Character Coverage

- **Default**: 0.9995 (99.95%)
- Ensures rare characters in African scripts are covered
- Higher values = more comprehensive coverage

```bash
--character_coverage 0.9995
```

## Training Data Requirements

### Data Format

- Plain text files (UTF-8 encoding)
- One sentence per line (recommended)
- Diverse corpus covering multiple domains

### Recommended Data Size

- Minimum: 10MB per language
- Optimal: 100MB-1GB per language
- For low-resource languages: Supplement with multilingual data

### Data Sources

Recommended sources for African language data:

1. **OSCAR**: Multilingual corpus with African languages
2. **CC100**: CommonCrawl-based monolingual datasets
3. **JW300**: Parallel corpus from religious texts
4. **Wikipedia dumps**: For major African languages
5. **MasakhaNER**: Named entity recognition datasets
6. **AfriSenti**: Sentiment analysis datasets

## Output Files

Training produces two files:

1. `{model_prefix}.model` - SentencePiece model file
2. `{model_prefix}.vocab` - Vocabulary file (human-readable)

## Usage Examples

### For Swahili

```bash
python train_sentencepiece.py \
  --input_file swahili_corpus.txt \
  --model_prefix swahili_tokenizer \
  --vocab_size 24000
```

### For Hausa (Arabic script)

```bash
python train_sentencepiece.py \
  --input_file hausa_corpus.txt \
  --model_prefix hausa_tokenizer \
  --vocab_size 24000 \
  --character_coverage 0.9998
```

### For Multilingual Model (10+ languages)

```bash
python train_sentencepiece.py \
  --input_file swahili.txt,hausa.txt,yoruba.txt,zulu.txt,amharic.txt \
  --model_prefix afrilion_multilingual \
  --vocab_size 64000 \
  --character_coverage 0.9998
```

## Integration with Transformers

To use the trained tokenizer with Hugging Face Transformers:

```python
from transformers import PreTrainedTokenizerFast
import sentencepiece as spm

# Load the SentencePiece model
sp = spm.SentencePieceProcessor()
sp.load('afrilion_tokenizer.model')

# Create a Hugging Face tokenizer
tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=sp,
    bos_token="[BOS]",
    eos_token="[EOS]",
    unk_token="[UNK]",
    pad_token="[PAD]",
    mask_token="[MASK]",
    cls_token="[CLS]",
    sep_token="[SEP]",
)
```

## Special Tokens

The tokenizer includes the following special tokens:

- `[PAD]` (ID: 0) - Padding token
- `[UNK]` (ID: 1) - Unknown token
- `[BOS]` (ID: 2) - Beginning of sequence
- `[EOS]` (ID: 3) - End of sequence
- `[MASK]` - Masking token for MLM
- `[CLS]` - Classification token
- `[SEP]` - Separator token

## Best Practices

### For Low-Resource Languages

1. Use smaller vocabulary (16k-24k)
2. Include related language data
3. Use higher character coverage (0.9998+)
4. Consider char or unigram models

### For High-Resource Languages

1. Use larger vocabulary (32k-50k)
2. Include diverse domains (news, social, web)
3. Balance formal and informal text
4. Consider BPE for efficiency

### For Code-Switching

1. Include mixed-language samples
2. Use larger vocabulary
3. Ensure all script coverage
4. Test on bilingual data

## Troubleshooting

### Out of Memory

- Reduce `num_threads` parameter
- Process data in batches
- Use a machine with more RAM

### Poor Tokenization Quality

- Increase training data size
- Adjust character coverage
- Try different model types
- Clean and normalize input data

### Script Coverage Issues

- Set character_coverage to 0.9998 or higher
- Include representative samples of all scripts
- Check input data encoding (must be UTF-8)

## Contributing

When adding new language support:

1. Prepare clean training corpus
2. Document any script-specific considerations
3. Test tokenization quality
4. Share model on Hugging Face Hub

## License

See LICENSE file in repository root.

## References

- [SentencePiece Paper](https://arxiv.org/abs/1808.06226)
- [African NLP Resources](https://github.com/masakhane-io/masakhane)
- [Hugging Face Tokenizers](https://huggingface.co/docs/tokenizers/)

## Contact

For questions or issues, please open an issue on the GitHub repository.
