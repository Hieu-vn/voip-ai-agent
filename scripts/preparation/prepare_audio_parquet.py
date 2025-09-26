#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prepare_audio_parquet.py — memory-safe, idempotent, sharded output
- Stream Parquet by batches (PyArrow)
- Deterministic hashing: hash(audio_bytes + parquet_tag + speaker)
- Sharded wav dirs: wavs/<hash[:2]>/
- Filter chain: aresample -> loudnorm -> aformat
- Append processed.csv in flush batches with fixed column order
"""
import argparse, io, logging, os, tempfile, subprocess, sys
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, Iterable, List
import concurrent.futures
import hashlib

import numpy as np
import soundfile as sf
import librosa
from tqdm import tqdm

import pyarrow.parquet as pq
import pyarrow as pa

import yaml
from shutil import which

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger("prepare_audio_parquet")

VI_DIGIT = {
    "0":"không","1":"một","2":"hai","3":"ba","4":"bốn","5":"năm","6":"sáu","7":"bảy","8":"tám","9":"chín"
}

CSV_COLS = ["audio_filepath", "text", "speaker", "duration"]

# ---------- helpers ----------
def _guess_columns_from_schema(schema: pa.Schema) -> Tuple[Optional[str], Optional[str], Optional[str], Optional[str]]:
    names = [n.lower() for n in schema.names]
    def pick(cands):
        for c in schema.names:
            if c.lower() in cands:
                return c
        return None
    audio_col = pick({"audio","audio_bytes","audio_wav","wav","sound"})
    text_col  = pick({"text","sentence","transcript","normalized_text"})
    spk_col   = pick({"speaker","speaker_id","spk","spkid"})
    sr_col    = pick({"sampling_rate","sample_rate","sr"})
    return audio_col, text_col, spk_col, sr_col

def _read_audio_from_any(obj: Any, fallback_sr: int) -> Tuple[np.ndarray, int, bytes]:
    """
    Return mono float32 waveform, sr, and raw-bytes used for hashing (original domain).
    Supports:
      - dict with "bytes"
      - dict with "array" (+ optional "sampling_rate")
      - raw bytes (wav/flac)
    """
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
        # produce a stable raw-bytes for hashing from float array + sr
        b = sr.to_bytes(4, "little", signed=False) + arr.tobytes()
        return arr, sr, b
    else:
        raise ValueError("Unsupported audio object format")

def _trim_silence(y: np.ndarray, sr: int, top_db: float = 35.0, pad_sec: float = 0.15):
    y_trim, _ = librosa.effects.trim(y, top_db=top_db, frame_length=1024, hop_length=256)
    pad = int(pad_sec * sr)
    if pad > 0:
        y_trim = np.concatenate([np.zeros(pad, dtype=y.dtype), y_trim, np.zeros(pad, dtype=y.dtype)])
    return y_trim

def _basic_vi_text_norm(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.strip()
    out = []
    for ch in s:
        out.append(VI_DIGIT.get(ch, ch) if ch.isdigit() else ch)
    s = "".join(out)
    s = s.replace("	"," ").replace(""," ")
    return " ".join(s.split())

def _ffmpeg_resample_loudnorm_from_mem(in_audio_bytes: bytes, in_sr: int, out_wav_path: Path, target_sr: int, target_lufs: float, peak_dbfs: float):
    """
    Filter chain: aresample (soxr) -> loudnorm at target SR -> aformat mono/s16. Reads from stdin.
    """
    filter_chain = (
        f"aresample={target_sr}:resampler=soxr,"
        f"loudnorm=I={target_lufs}:TP={peak_dbfs}:LRA=11:linear=true,"
        f"aformat=channel_layouts=mono:sample_rates={target_sr}:sample_fmts=s16"
    )
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        # Input stream format from stdin
        "-f", "s16le",
        "-ar", str(in_sr),
        "-ac", "1",
        "-i", "pipe:0",
        # Filters
        "-filter:a", filter_chain,
        # Output format
        "-y",
        str(out_wav_path),
    ]
    subprocess.run(cmd, input=in_audio_bytes, check=True)

def _make_out_path(base_dir: Path, spk: str, utt_hex: str) -> Path:
    sub = utt_hex[:2]
    subdir = base_dir / "wavs" / sub
    subdir.mkdir(parents=True, exist_ok=True)
    safe_spk = spk if spk else "spk_unk"
    return subdir / f"{safe_spk}_{utt_hex}.wav"

def _hash_for_filename(audio_raw_bytes: bytes, parquet_tag: str, speaker: str) -> str:
    h = hashlib.blake2b(digest_size=8)
    h.update(parquet_tag.encode("utf-8"))
    h.update(b"|")
    h.update((speaker or "spk_unk").encode("utf-8"))
    h.update(b"|")
    h.update(audio_raw_bytes)
    return h.hexdigest()

# ---------- core processing ----------
def process_parquet_file_stream(
    parquet_file_path: Path,
    out_dir: Path,
    target_sr: int,
    target_lufs: float,
    peak_dbfs: float,
    min_dur: float,
    max_dur: float,
    top_db: float,
    flush_every: int = 50_000
) -> List[Dict[str, Any]]:
    """
    Stream rows via PyArrow to keep memory low. Returns list of rows (may be large for small files),
    but we still flush in caller after merging futures; however, we add an inner flush safeguard too.
    """
    parquet_tag = parquet_file_path.name
    pf = pq.ParquetFile(str(parquet_file_path))
    audio_col, text_col, spk_col, sr_col = _guess_columns_from_schema(pf.schema_arrow)
    if audio_col is None or text_col is None:
        logger.error("Thiếu cột audio/text trong %s", parquet_file_path.name)
        return []

    cols = [c for c in [audio_col, text_col, spk_col, sr_col] if c is not None]
    local_rows: List[Dict[str, Any]] = []
    total = pf.metadata.num_rows if pf.metadata else None
    batch_iter = pf.iter_batches(columns=cols, batch_size=512)

    pbar = tqdm(batch_iter, desc=f"parse {parquet_file_path.name}", total=None, mininterval=0.5)
    for batch in pbar:
        # to_pylist returns list of dicts with selected columns
        records: List[Dict[str, Any]] = batch.to_pylist()
        for rec in records:
            try:
                audio_obj = rec[audio_col]
                y, sr, raw_b = _read_audio_from_any(audio_obj, int(rec.get(sr_col, 22050)) if sr_col else 22050)
                if y.ndim > 1:
                    y = y.mean(axis=1).astype(np.float32)

                y = _trim_silence(y, sr, top_db=top_db, pad_sec=0.15)
                if y.size == 0:
                    raise ValueError("empty after trim")

                txt = _basic_vi_text_norm(str(rec[text_col]))
                spk = str(rec[spk_col]) if spk_col and rec.get(spk_col) is not None else "spk_unk"

                utt_hex = _hash_for_filename(raw_b, parquet_tag, spk)
                out_path = _make_out_path(out_dir, spk, utt_hex)

                # idempotency
                if out_path.exists():
                    try:
                        # If file exists, we must still generate its metadata row for the CSV.
                        y2, sr2 = sf.read(out_path, dtype="float32", always_2d=False)
                        dur = float(len(y2) / sr2) if isinstance(y2, np.ndarray) else 0.0
                        if dur >= min_dur and dur <= max_dur:
                            row = {
                                "audio_filepath": str(out_path.resolve()),
                                "text": txt,
                                "speaker": spk,
                                "duration": round(dur, 3),
                            }
                            local_rows.append(row)
                    except Exception as e:
                        logger.warning(f"Could not read existing file {out_path}, will attempt to re-process. Error: {e}")
                    else:
                        # If we successfully read the existing file, we can skip the rest of the processing for this record.
                        continue

                # Convert numpy array to raw PCM bytes in memory for piping
                with io.BytesIO() as buffer:
                    # soundfile handles float to PCM_16 conversion automatically
                    sf.write(buffer, y, sr, format='RAW', subtype='PCM_16')
                    audio_bytes = buffer.getvalue()

                # Process from memory
                _ffmpeg_resample_loudnorm_from_mem(audio_bytes, sr, out_path, target_sr, target_lufs, peak_dbfs)

                y2, sr2 = sf.read(out_path, dtype="float32", always_2d=False)
                if isinstance(y2, np.ndarray) and y2.ndim > 1:
                    y2 = y2.mean(axis=1)
                dur = float(len(y2) / sr2) if isinstance(y2, np.ndarray) else 0.0
                if dur < min_dur or dur > max_dur:
                    # remove invalid
                    try:
                        Path(out_path).unlink(missing_ok=True)
                    except Exception:
                        pass
                    raise ValueError(f"duration_out_of_range={dur:.2f}")

                row = {
                    "audio_filepath": str(Path(out_path).resolve()),
                    "text": txt,
                    "speaker": spk,
                    "duration": round(dur, 3),
                }
                local_rows.append(row)

                # Soft flush inside worker to constrain RAM on huge files
                if len(local_rows) >= flush_every:
                    yield from local_rows
                    local_rows.clear()

            except Exception as e:
                logger.warning("Skip %s: %s", parquet_file_path.name, str(e))
                continue

    if local_rows:
        yield from local_rows

def process_parquet(
    in_dir: Path,
    out_dir: Path,
    target_sr: int = 22050,
    target_lufs: float = -23.0,
    peak_dbfs: float = -2.0,
    min_dur: float = 1.5,
    max_dur: float = 20.0,
    top_db: float = 35.0,
    num_workers: int = max(1, os.cpu_count() // 2),
    limit: Optional[int] = None,
    flush_every_csv: int = 50_000
):
    out_dir.mkdir(parents=True, exist_ok=True)
    out_meta = out_dir / "processed.csv"
    parquet_files = sorted(list(in_dir.glob("*.parquet")))
    if not parquet_files:
        logger.error("Không tìm thấy file .parquet trong %s", in_dir)
        return

    if limit:
        parquet_files = parquet_files[:limit]
        logger.warning(f"--- LIMIT MODE: processing only first {len(parquet_files)} parquet files ---")

    logger.info("Bắt đầu xử lý %d tệp parquet với %d workers...", len(parquet_files), num_workers)

    # CSV writer state
    def append_rows(rows: List[Dict[str, Any]]):
        import pandas as pd
        if not rows:
            return
        df_new = pd.DataFrame(rows)
        # enforce column order
        for c in CSV_COLS:
            if c not in df_new.columns:
                df_new[c] = np.nan
        df_new = df_new[CSV_COLS]
        header = not out_meta.exists()
        df_new.to_csv(out_meta, mode="a", header=header, index=False)

    buffer_rows: List[Dict[str, Any]] = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as ex:
        futures = []
        for pq_path in parquet_files:
            futures.append(ex.submit(
                _collect_worker_results,
                pq_path, out_dir, target_sr, target_lufs, peak_dbfs, min_dur, max_dur, top_db
            ))

        for fut in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Tổng hợp kết quả"):
            worker_chunk = fut.result()
            if worker_chunk:
                buffer_rows.extend(worker_chunk)
                if len(buffer_rows) >= flush_every_csv:
                    append_rows(buffer_rows)
                    logger.info("Flushed %d rows to %s", len(buffer_rows), out_meta)
                    buffer_rows.clear()

    # final flush
    if buffer_rows:
        append_rows(buffer_rows)
        logger.info("Final flush %d rows to %s", len(buffer_rows), out_meta)

def _collect_worker_results(
    parquet_file_path: Path,
    out_dir: Path,
    target_sr: int,
    target_lufs: float,
    peak_dbfs: float,
    min_dur: float,
    max_dur: float,
    top_db: float
) -> List[Dict[str, Any]]:
    """Run in worker; consume generator to a list (bounded by inner yields)."""
    out: List[Dict[str, Any]] = []
    try:
        for row in process_parquet_file_stream(
            parquet_file_path, out_dir,
            target_sr, target_lufs, peak_dbfs,
            min_dur, max_dur, top_db,
            flush_every=25_000
        ):
            out.append(row)
    except Exception as e:
        logger.error("Worker failed on %s: %s", parquet_file_path.name, str(e))
    return out

def main():
    ap = argparse.ArgumentParser(description="Prepare audio data from Parquet files.")
    ap.add_argument("--in_dir", required=True, help="Directory containing input Parquet files.")
    ap.add_argument("--out_dir", required=True, help="Output directory for processed wavs and CSV.")
    ap.add_argument("--config", default="config/data_processing_config.yaml", help="Path to the data processing YAML config file.")
    ap.add_argument("--num_workers", type=int, default=None, help="Number of worker processes. Overrides config if set.")
    ap.add_argument("--limit", type=int, default=None, help="Limit the number of Parquet files to process (for testing).")
    args = ap.parse_args()

    # Load config from YAML
    try:
        with open(args.config, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    except FileNotFoundError:
        logger.error(f"Config file not found at: {args.config}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error loading or parsing config file: {e}")
        sys.exit(1)

    # Extract parameters from config, providing defaults
    audio_cfg = config.get("audio", {})
    vad_cfg = config.get("vad", {})
    filter_cfg = config.get("filtering", {})

    # Determine number of workers (command line overrides default)
    num_workers = args.num_workers if args.num_workers is not None else max(1, os.cpu_count() // 2)

    # Quick sanity check for ffmpeg
    if not which("ffmpeg"):
        logger.error("FFmpeg not found in PATH. Please install ffmpeg and try again.")
        sys.exit(1)

    logger.info(f"Loaded configuration from {args.config}")
    process_parquet(
        in_dir=Path(args.in_dir),
        out_dir=Path(args.out_dir),
        target_sr=audio_cfg.get("target_sr", 22050),
        target_lufs=audio_cfg.get("target_lufs", -23.0),
        peak_dbfs=audio_cfg.get("peak_dbfs", -2.0),
        min_dur=filter_cfg.get("min_dur", 1.5),
        max_dur=filter_cfg.get("max_dur", 22.0),
        top_db=vad_cfg.get("top_db", 35.0),
        num_workers=num_workers,
        limit=args.limit,
        flush_every_csv=filter_cfg.get("flush_every_csv", 50000)
    )

if __name__ == "__main__":
    main()