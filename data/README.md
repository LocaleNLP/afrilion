# AfriLION Data Pipeline

This directory contains the data collection and preprocessing pipeline for the AfriLION project.

## Overview

The AfriLION data pipeline processes multilingual African language text data from the CC-100 corpus. The pipeline applies rigorous quality filtering, deduplication, and preprocessing to create high-quality training data for African language models.

## Pipeline Architecture

The pipeline consists of 7 sequential stages:

### Stage 1: Download CC-100 African Language Subsets
- Downloads compressed `.txt.xz` files from StatMT CC-100 repository
- Supports 20+ African languages
- Handles large files with streaming and progress bars
- Automatic retry on failure

### Stage 2: Language Detection Filter
- Uses `langdetect` library for language identification
- Filters out texts with confidence < 0.9
- Removes code-switched or misidentified content
- Ensures data purity per language

### Stage 3: Text Cleaning
- Unicode normalization (NFC)
- Removes excessive whitespace
- Filters out non-textual content
- Preserves linguistic features of African languages

### Stage 4: Deduplication via MinHash LSH
- Uses MinHash with Locality-Sensitive Hashing (LSH)
- Threshold: 0.85 similarity
- 128 permutations for hash generation
- Removes near-duplicate documents
- Memory-efficient streaming implementation

### Stage 5: Length Filter
- Minimum: 20 tokens (by whitespace)
- Maximum: 2048 tokens
- Removes very short and very long documents
- Optimizes for transformer context windows

### Stage 6: Shard to JSONL
- Splits data into manageable shards
- Default: 100,000 lines per shard
- JSONL format for efficient loading
- Preserves language metadata

### Stage 7: Upload to Hugging Face Datasets
- Uploads to `AfriLION/afrilion-corpus` repository
- Creates dataset cards with metadata
- Enables easy access for training
- Supports streaming for large datasets

## Supported Languages

The pipeline currently supports African language subsets from CC-100 including:

### West Africa
- Wolof (wo)
- Hausa (ha)
- Yoruba (yo)
- Igbo (ig)

### East Africa
- Swahili (sw)
- Somali (so)
- Amharic (am)
- Oromo (om)

### Southern Africa
- Zulu (zu)
- Xhosa (xh)
- Shona (sn)
- Sesotho (st)

### North Africa
- Arabic dialects
- Berber/Amazigh

*Note: Language availability depends on CC-100 coverage*

## Usage

### Installation

```bash
pip install datasets langdetect datasketch tqdm huggingface_hub
```

### Basic Usage

```bash
python pipeline.py --langs wo sw ha yo am --output_dir ./processed
```

### Advanced Usage

```bash
python pipeline.py \
  --langs wo sw ha yo am zu xh sn \
  --output_dir ./processed \
  --min_tokens 20 \
  --max_tokens 2048 \
  --langdet_conf 0.9 \
  --minhash_threshold 0.85 \
  --shard_size 100000
```

### Processing All Available Languages

```bash
# Process all CC-100 African language subsets
python pipeline.py --langs all --output_dir ./processed
```

## Configuration

### Pipeline Parameters

```python
MIN_TOKENS = 20              # Minimum document length
MAX_TOKENS = 2048            # Maximum document length  
LANGDET_CONF = 0.9           # Language detection confidence threshold
MINHASH_THRESHOLD = 0.85     # Deduplication similarity threshold
MINHASH_PERMS = 128          # MinHash permutations
SHARD_SIZE = 100_000         # Lines per JSONL shard
HF_REPO_ID = "AfriLION/afrilion-corpus"  # Hugging Face repository
```

### Language Metadata

Each language can have custom metadata:

```python
LANG_META = {
    "wo": {"name": "Wolof", "script": "Latin", "region": "West Africa"},
    "sw": {"name": "Swahili", "script": "Latin", "region": "East Africa"},
    "am": {"name": "Amharic", "script": "Ge'ez", "region": "East Africa"},
    # ... more languages
}
```

## Data Quality

### Quality Assurance Measures

1. **Language Purity**: 90%+ confidence threshold ensures linguistic consistency
2. **Deduplication**: MinHash LSH removes near-duplicates at 85% similarity
3. **Length Filtering**: Focuses on substantive documents (20-2048 tokens)
4. **Text Cleaning**: Normalizes Unicode and removes artifacts
5. **Format Validation**: JSONL format with schema validation

### Expected Output Quality

- **Precision**: High-quality, linguistically pure documents
- **Recall**: Balances coverage with quality
- **Diversity**: Removes duplicates while preserving linguistic variety
- **Size**: Typically 60-80% retention after all filtering stages

## Output Format

Processed data is stored in JSONL format:

```jsonl
{"text": "...", "lang": "wo", "source": "cc100", "id": "wo_000001"}
{"text": "...", "lang": "wo", "source": "cc100", "id": "wo_000002"}
```

### Directory Structure

```
processed/
├── wo/
│   ├── shard_000.jsonl
│   ├── shard_001.jsonl
│   └── ...
├── sw/
│   ├── shard_000.jsonl
│   └── ...
└── ha/
    └── ...
```

## Performance

### Processing Speed

- **Download**: Dependent on network speed (~10-50 MB/s)
- **Language Detection**: ~1000 docs/second
- **Deduplication**: ~500 docs/second (memory-dependent)
- **Overall**: ~2-5 hours for 1M documents per language

### Resource Requirements

- **CPU**: Multi-core recommended (uses threading)
- **RAM**: 8-16 GB recommended for large datasets
- **Storage**: ~2-5 GB per language (compressed)
- **Network**: Stable connection for downloading

## Extending the Pipeline

### Adding New Languages

1. Check CC-100 availability at `https://data.statmt.org/cc-100/`
2. Add language code to `LANG_META` dictionary
3. Run pipeline with new language codes

### Adding Custom Filters

```python
def custom_filter(text: str) -> bool:
    """Custom filtering logic"""
    # Add your logic here
    return True

# Insert into pipeline after Stage 3
```

### Adding New Data Sources

Beyond CC-100, you can integrate:

1. **OSCAR**: Multilingual web crawl
2. **mC4**: Multilingual C4 dataset
3. **Wikipedia**: Language-specific dumps
4. **Local corpora**: Government, news, literature

## Monitoring and Logging

The pipeline logs detailed information:

```
[2025-01-23 10:30:45] INFO: Starting pipeline for language: wo
[2025-01-23 10:31:12] INFO: Downloaded 523 MB
[2025-01-23 10:45:30] INFO: Language detection: 89,234 / 100,000 passed
[2025-01-23 11:15:42] INFO: Deduplication: 67,891 unique documents
[2025-01-23 11:20:15] INFO: Length filter: 65,432 documents retained
[2025-01-23 11:25:00] INFO: Sharded into 7 JSONL files
```

## Troubleshooting

### Common Issues

#### Download Failures
- **Issue**: Network timeout or connection error
- **Solution**: Pipeline auto-retries; check internet connection

#### Memory Errors
- **Issue**: Out of memory during deduplication
- **Solution**: Reduce `SHARD_SIZE` or increase system RAM

#### Low Retention Rate
- **Issue**: < 50% documents retained after filtering
- **Solution**: Check `LANGDET_CONF` threshold; may be too strict

#### Slow Processing
- **Issue**: Pipeline takes very long
- **Solution**: Use SSD for storage; increase CPU cores

## Best Practices

### For Production Use

1. **Monitor Quality**: Sample and review filtered data regularly
2. **Version Control**: Tag datasets with version numbers
3. **Incremental Updates**: Process new data separately from existing
4. **Backup**: Keep raw downloaded data before processing
5. **Documentation**: Log parameters and timestamps for reproducibility

### For Research Use

1. **Reproducibility**: Fix random seeds and document all parameters
2. **Ablation Studies**: Test different threshold values
3. **Quality Analysis**: Measure retention rates per stage
4. **Error Analysis**: Sample rejected documents to understand filters

## Future Enhancements

- [ ] Parallel processing across multiple languages
- [ ] Advanced text normalization for specific scripts
- [ ] Toxicity and bias filtering
- [ ] Quality scoring beyond length
- [ ] Integration with additional data sources
- [ ] Automated quality reports and statistics

## Contributing

When adding features or improvements:

1. Test on a small sample first
2. Document configuration changes
3. Update this README
4. Measure impact on quality and performance

## License

See LICENSE file in repository root.

## References

- [CC-100 Dataset](https://data.statmt.org/cc-100/)
- [MinHash LSH for Deduplication](https://ekzhu.com/datasketch/lsh.html)
- [Hugging Face Datasets](https://huggingface.co/docs/datasets/)
- [Masakhane NLP Community](https://www.masakhane.io/)

## Contact

For issues or questions about the data pipeline, please open an issue on GitHub.
