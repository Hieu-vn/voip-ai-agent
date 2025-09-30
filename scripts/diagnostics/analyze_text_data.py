#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import logging
from pathlib import Path
import pyarrow.parquet as pq
from collections import Counter
import re
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

# Vietnamese alphabet and numbers
VIETNAMESE_CHARS = "aáàảãạăắằẳẵặâấầẩẫậbcdđeéèẻẽẹêếềểễệfghiíìỉĩịjklmnoóòỏõọôốồổỗộơớờởỡợpqrstuúùủũụưứừửữựvxyýỳỷỹỵz "
ALLOWED_CHARS = set(VIETNAMESE_CHARS + VIETNAMESE_CHARS.upper() + "0123456789")

def analyze_text_data(data_dir: Path, file_limit: int):
    """Scans Parquet files in a directory and provides statistics on the 'text' column."""
    parquet_files = sorted(list(data_dir.glob("*.parquet")))
    if not parquet_files:
        logger.error(f"No .parquet files found in {data_dir}")
        return

    if file_limit and file_limit < len(parquet_files):
        logger.warning(f"Limiting analysis to the first {file_limit} files.")
        parquet_files = parquet_files[:file_limit]

    # --- Statistics counters ---
    total_records = 0
    leading_trailing_space = 0
    double_space = 0
    empty_or_whitespace = 0
    special_char_counter = Counter()

    logger.info(f"Starting analysis on {len(parquet_files)} Parquet files...")

    for file_path in tqdm(parquet_files, desc="Analyzing files"):
        try:
            pf = pq.ParquetFile(str(file_path))
            if 'text' not in pf.schema_arrow.names:
                logger.warning(f"Skipping {file_path.name}: no 'text' column found.")
                continue

            for batch in pf.iter_batches(columns=['text'], batch_size=1024):
                texts = batch.to_pylist()
                for item in texts:
                    text = item['text']
                    if text is None:
                        continue
                    
                    total_records += 1

                    # Check for whitespace issues
                    if text != text.strip():
                        leading_trailing_space += 1
                    if '  ' in text:
                        double_space += 1
                    if not text.strip():
                        empty_or_whitespace += 1

                    # Check for special characters
                    for char in text:
                        if char not in ALLOWED_CHARS:
                            special_char_counter[char] += 1
        except Exception as e:
            logger.error(f"Failed to process {file_path.name}: {e}")

    # --- Print Report ---
    print("\n" + "="*50)
    print("Text Data Analysis Report")
    print("="*50)
    print(f"Total records analyzed: {total_records:,}")
    print("\n--- Whitespace Issues ---")
    print(f"Records with leading/trailing whitespace: {leading_trailing_space:,}")
    print(f"Records with double spaces:              {double_space:,}")
    print(f"Records with empty/whitespace-only text: {empty_or_whitespace:,}")
    print("\n--- Special Character Analysis ---")
    if not special_char_counter:
        print("No special characters found. The dataset is clean.")
    else:
        print("Found the following special characters (char: count):")
        # Sort by count descending
        sorted_chars = sorted(special_char_counter.items(), key=lambda item: item[1], reverse=True)
        for char, count in sorted_chars:
            # Use repr() to make non-printable characters visible
            print(f"  - {repr(char)}: {count:,}")
    print("\n" + "="*50)

def main():
    parser = argparse.ArgumentParser(description="Analyze text data in Parquet files.")
    parser.add_argument("--dir", required=True, help="Directory containing Parquet files to analyze.")
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of files to process (for faster analysis).")
    args = parser.parse_args()

    analyze_text_data(Path(args.dir), args.limit)

if __name__ == "__main__":
    main()
