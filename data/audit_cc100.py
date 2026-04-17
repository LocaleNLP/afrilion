"""Audit CC-100 data for African languages.

This script audits the available CC-100 data for 4 initial languages:
Swahili, Hausa, Yoruba, and Amharic.
"""

import argparse
import requests
from tqdm import tqdm


def audit_cc100(languages):
    """
    Check size and accessibility of CC-100 data subsets.
    """
    base_url = "https://data.statmt.org/cc-100/"
    
    print(f"Auditing CC-100 subsets for: {languages}")
    print("-" * 50)
    print(f"{'Lang':<10} | {'Status':<10} | {'Size (Est)':<15} | {'URL'}")
    print("-" * 50)
    
    # Metadata for initial languages
    meta = {
        "sw": "Swahili",
        "ha": "Hausa",
        "yo": "Yoruba",
        "am": "Amharic"
    }
    
    for lang in languages:
        url = f"{base_url}{lang}.txt.xz"
        try:
            response = requests.head(url, timeout=10)
            if response.status_code == 200:
                size_bytes = int(response.headers.get('content-length', 0))
                size_mb = size_bytes / (1024 * 1024)
                status = "✅ Found"
                size_str = f"{size_mb:.2f} MB"
            else:
                status = "❌ Missing"
                size_str = "N/A"
        except Exception as e:
            status = "⚠️ Error"
            size_str = "N/A"
        
        print(f"{lang:<10} | {status:<10} | {size_str:<15} | {url}")


def main():
    parser = argparse.ArgumentParser(description="Audit CC-100 African Language Data")
    parser.add_argument(
        "--langs", 
        type=str, 
        default="sw,ha,yo,am",
        help="Comma-separated language codes to audit"
    )
    
    args = parser.parse_args()
    languages = args.langs.split(",")
    
    audit_cc100(languages)


if __name__ == "__main__":
    main()
