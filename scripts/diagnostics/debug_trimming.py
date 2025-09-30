#!/usr/bin/env python3
import argparse
import io
from pathlib import Path
import numpy as np
import librosa
import soundfile as sf
import pyarrow.parquet as pq
from typing import Tuple, Any

# --- Copied from prepare_audio_parquet.py for self-containment ---
def _read_audio_from_any(obj: Any, fallback_sr: int) -> Tuple[np.ndarray, int, bytes]:
    if isinstance(obj, dict) and "bytes" in obj:
        b = obj["bytes"]
        if not isinstance(b, (bytes, bytearray, memoryview)):
            raise ValueError("audio.bytes must be bytes-like")
        with io.BytesIO(b) as bio:
            y, sr = sf.read(bio, dtype="float32", always_2d=False)
        if isinstance(sr, (list, tuple)): sr = sr[0]
        if y.ndim > 1: y = y.mean(axis=1)
        return y.astype(np.float32), int(sr or fallback_sr), bytes(b)
    elif isinstance(obj, (bytes, bytearray, memoryview)):
        b = bytes(obj)
        with io.BytesIO(b) as bio:
            y, sr = sf.read(bio, dtype="float32", always_2d=False)
        if isinstance(sr, (list, tuple)): sr = sr[0]
        if y.ndim > 1: y = y.mean(axis=1)
        return y.astype(np.float32), int(sr or fallback_sr), b
    elif isinstance(obj, dict) and "array" in obj:
        arr = np.asarray(obj["array"], dtype=np.float32)
        if arr.ndim > 1:
            arr = arr.mean(axis=1).astype(np.float32)
        sr = int(obj.get("sampling_rate", fallback_sr))
        b = sr.to_bytes(4, "little", signed=False) + arr.tobytes()
        return arr, sr, b
    else:
        raise ValueError("Unsupported audio object format")

def _trim_silence(y: np.ndarray, sr: int, top_db: float = 35.0):
    y_trim, _ = librosa.effects.trim(y, top_db=top_db, frame_length=1024, hop_length=256)
    return y_trim

def main():
    parser = argparse.ArgumentParser(description="Debug trimming parameters on a Parquet file.")
    parser.add_argument("--file", required=True, help="Path to the Parquet file to diagnose.")
    parser.add_argument("--min_long_dur", type=float, default=9.0, help="Minimum duration to consider a 'long' file for testing.")
    args = parser.parse_args()

    test_top_dbs = [30, 35, 40, 50, 60, 70]
    file_path = Path(args.file)

    if not file_path.exists():
        print(f"Error: File not found at {file_path}")
        return

    print(f"--- Diagnosing {file_path.name} ---")
    print(f"Searching for the first record longer than {args.min_long_dur}s to test with top_db values: {test_top_dbs}")

    pf = pq.ParquetFile(str(file_path))
    
    found_long_record = False
    for i, batch in enumerate(pf.iter_batches(batch_size=100)):
        if found_long_record:
            break
        records = batch.to_pylist()
        for j, rec in enumerate(records):
            try:
                y, sr, _ = _read_audio_from_any(rec["audio"], 22050)
                original_dur = len(y) / sr

                if original_dur > args.min_long_dur:
                    print(f"\nFound long record at batch #{i+1}, record #{j+1} (Original index: {i*100+j})")
                    header = f"{ 'Original Dur':<15} | " + " | ".join([f'Trimmed (db={db})' for db in test_top_dbs])
                    print(header)
                    print("-" * len(header))
                    
                    results = []
                    for db in test_top_dbs:
                        y_trimmed = _trim_silence(y, sr, top_db=db)
                        trimmed_dur = len(y_trimmed) / sr
                        results.append(f"{trimmed_dur:<15.2f}s")
                    
                    print(f"{original_dur:<15.2f}s | " + " | ".join(results))
                    found_long_record = True
                    break # Exit inner loop
            except Exception as e:
                # Ignore records that can't be processed
                continue
    
    if not found_long_record:
        print(f"\nCould not find any records longer than {args.min_long_dur}s in the file.")

if __name__ == "__main__":
    main()