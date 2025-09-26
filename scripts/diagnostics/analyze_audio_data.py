#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import logging
from pathlib import Path
import numpy as np
import soundfile as sf
import pyloudnorm as pyln
import matplotlib.pyplot as plt
import pyarrow.parquet as pq
from tqdm import tqdm
import io
from typing import Tuple, Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

# --- Helper function to read audio ---
def _read_audio_from_any(obj: Any, fallback_sr: int) -> Tuple[np.ndarray, int]:
    if isinstance(obj, dict) and "bytes" in obj:
        b = obj["bytes"]
        with io.BytesIO(b) as bio:
            y, sr = sf.read(bio, dtype="float32", always_2d=False)
    elif isinstance(obj, (bytes, bytearray, memoryview)):
        with io.BytesIO(bytes(obj)) as bio:
            y, sr = sf.read(bio, dtype="float32", always_2d=False)
    elif isinstance(obj, dict) and "array" in obj:
        y = np.asarray(obj["array"], dtype=np.float32)
        sr = int(obj.get("sampling_rate", fallback_sr))
    else:
        raise ValueError("Unsupported audio object format")
    
    if isinstance(sr, (list, tuple)): sr = sr[0]
    if y.ndim > 1: y = y.mean(axis=1)
    return y.astype(np.float32), int(sr or fallback_sr)

def analyze_audio_data(data_dir: Path, output_dir: Path, file_limit: int):
    """Scans Parquet files and provides statistics on the audio data."""
    parquet_files = sorted(list(data_dir.glob("*.parquet")))
    if not parquet_files:
        logger.error(f"No .parquet files found in {data_dir}")
        return

    if file_limit and file_limit < len(parquet_files):
        logger.warning(f"Limiting analysis to the first {file_limit} files.")
        parquet_files = parquet_files[:file_limit]

    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Statistics lists ---
    durations = []
    lufs_values = []
    clipping_ratios = []
    sample_rates = []

    logger.info(f"Starting audio analysis on {len(parquet_files)} Parquet files...")

    for file_path in tqdm(parquet_files, desc="Analyzing audio"):
        try:
            pf = pq.ParquetFile(str(file_path))
            if 'audio' not in pf.schema_arrow.names:
                continue

            for batch in pf.iter_batches(columns=['audio'], batch_size=256):
                records = batch.to_pylist()
                for item in records:
                    try:
                        y, sr = _read_audio_from_any(item['audio'], 16000)
                        if y.size == 0:
                            continue

                        # Duration
                        duration = len(y) / sr
                        durations.append(duration)
                        sample_rates.append(sr)

                        # Loudness (LUFS)
                        meter = pyln.Meter(sr) # create BS.1770 meter
                        loudness = meter.integrated_loudness(y)
                        if loudness != -float('inf'): # Avoid -inf for silent clips
                            lufs_values.append(loudness)

                        # Clipping
                        clipped_samples = np.sum(np.abs(y) >= 0.999)
                        clipping_ratio = clipped_samples / len(y)
                        clipping_ratios.append(clipping_ratio)

                    except Exception as e:
                        logger.debug(f"Skipping a record due to error: {e}")
        except Exception as e:
            logger.error(f"Failed to process {file_path.name}: {e}")

    if not durations:
        logger.error("No valid audio data was analyzed.")
        return

    # --- Generate and Save Plots ---
    plt.style.use('ggplot')

    # Duration plot
    plt.figure(figsize=(10, 6))
    plt.hist(durations, bins=50, color='skyblue', edgecolor='black')
    plt.title('Audio Duration Distribution')
    plt.xlabel('Duration (seconds)')
    plt.ylabel('Frequency')
    duration_plot_path = output_dir / "duration_distribution.png"
    plt.savefig(duration_plot_path)
    logger.info(f"Saved duration distribution plot to {duration_plot_path}")

    # LUFS plot
    plt.figure(figsize=(10, 6))
    plt.hist(lufs_values, bins=50, color='salmon', edgecolor='black')
    plt.title('Loudness (LUFS) Distribution')
    plt.xlabel('Integrated Loudness (LUFS)')
    plt.ylabel('Frequency')
    lufs_plot_path = output_dir / "lufs_distribution.png"
    plt.savefig(lufs_plot_path)
    logger.info(f"Saved LUFS distribution plot to {lufs_plot_path}")

    # --- Print Report ---
    df = pd.DataFrame({
        'duration': durations,
        'lufs': lufs_values,
        'clipping_ratio': clipping_ratios
    })

    print("\n" + "="*50)
    print("Audio Data Analysis Report")
    print("="*50)
    print(f"Total audio clips analyzed: {len(durations):,}")
    print(f"Sample rates found: {set(sample_rates)}")
    print("\n--- Duration (seconds) ---")
    print(df['duration'].describe().to_string())
    print("\n--- Loudness (LUFS) ---")
    print(df['lufs'].describe().to_string())
    print("\n--- Clipping ---")
    clipped_files_count = np.sum(np.array(clipping_ratios) > 0)
    print(f"Number of files with clipping: {clipped_files_count:,} ({clipped_files_count/len(durations):.2%})")
    print("\n" + "="*50)

def main():
    parser = argparse.ArgumentParser(description="Analyze audio data in Parquet files.")
    parser.add_argument("--dir", required=True, help="Directory containing Parquet files.")
    parser.add_argument("--output_dir", required=True, help="Directory to save output plots.")
    parser.add_argument("--limit", type=int, default=None, help="Limit the number of files to process.")
    args = parser.parse_args()

    analyze_audio_data(Path(args.dir), Path(args.output_dir), args.limit)

if __name__ == "__main__":
    # Need to import pandas for the final report
    try:
        import pandas as pd
    except ImportError:
        print("Pandas is not installed. Please install it using: pip install pandas")
        exit(1)
    main()
