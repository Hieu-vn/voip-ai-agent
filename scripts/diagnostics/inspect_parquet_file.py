#!/usr/bin/env python3
import argparse
import io
from pathlib import Path
import pyarrow.parquet as pq
import soundfile as sf

def inspect_parquet(file_path: Path, num_records: int):
    """Reads and inspects the first few records of a Parquet file."""
    if not file_path.exists():
        print(f"Error: File not found at {file_path}")
        return

    print(f"--- Inspecting file: {file_path.name} ---")

    pf = pq.ParquetFile(str(file_path))

    # 1. Print Schema
    print("\n[+] File Schema:")
    print(pf.schema_arrow.to_string())

    # 2. Inspect first N records
    print(f"\n[+] Inspecting first {num_records} records:")
    batch_iter = pf.iter_batches(batch_size=num_records)
    try:
        first_batch = next(batch_iter)
        records = first_batch.to_pylist()
    except StopIteration:
        print("File is empty or has fewer records than requested.")
        return

    for i, rec in enumerate(records):
        print(f"\n--- Record #{i+1} ---")
        for col_name, value in rec.items():
            if col_name == 'audio' and isinstance(value, dict) and 'bytes' in value:
                audio_bytes = value['bytes']
                if audio_bytes:
                    try:
                        with io.BytesIO(audio_bytes) as bio:
                            info = sf.info(bio)
                            duration = info.duration
                            sample_rate = info.samplerate
                            channels = info.channels
                            print(f"  - {col_name}:")
                            print(f"    - duration:    {duration:.2f} seconds")
                            print(f"    - sample_rate: {sample_rate} Hz")
                            print(f"    - channels:    {channels}")
                    except Exception as e:
                        print(f"  - {col_name}: Could not decode audio - {e}")
                else:
                    print(f"  - {col_name}: Empty audio bytes.")
            else:
                # Truncate long text for readability
                display_value = str(value)
                if len(display_value) > 200:
                    display_value = display_value[:200] + "..."
                print(f"  - {col_name}: {display_value}")

def main():
    parser = argparse.ArgumentParser(description="Inspect records in a Parquet file.")
    parser.add_argument("--file", required=True, help="Path to the Parquet file.")
    parser.add_argument("--num_records", type=int, default=3, help="Number of records to inspect.")
    args = parser.parse_args()

    inspect_parquet(Path(args.file), args.num_records)

if __name__ == "__main__":
    main()
