
#!/usr/bin/env python3
import argparse, json, logging, yaml
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
import re

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

class TextNormalizer:
    def __init__(self, rules_path: Path):
        with rules_path.open('r', encoding='utf-8') as f:
            rules = yaml.safe_load(f)
        self.digit_map = rules.get("digit_map", {})
        self.regex_rules = rules.get("regex_rules", [])
        self.max_length = rules.get("max_length", 250)
        logging.info(f"Tải {len(self.regex_rules)} quy tắc chuẩn hoá từ {rules_path}")

    def normalize(self, s: str) -> str:
        if not isinstance(s, str):
            return ""
        s = s.strip()
        
        # 1. Thay thế số thành chữ
        def _digits_to_words(m):
            return " ".join(self.digit_map.get(ch, ch) for ch in m.group(0))
        s = re.sub(r"\d+", _digits_to_words, s)

        # 2. Áp dụng các quy tắc regex
        for rule in self.regex_rules:
            s = re.sub(rule['pattern'], rule['replace'], s)
        
        s = s.strip()

        # 3. Cắt độ dài
        if len(s) > self.max_length:
            s = s[:self.max_length]
        
        return s

def to_jsonl(df: pd.DataFrame, out_path: Path):
    with out_path.open("w", encoding="utf-8") as f:
        for _, r in df.iterrows():
            obj = {
                "audio_filepath": r["audio_filepath"],
                "text": r["text"],
                "duration": float(r["duration"]),
                "speaker": str(r["speaker"])
            }
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed_csv", required=True, help="CSV từ bước 1")
    parser.add_argument("--out_dir", required=True, help="Thư mục output manifest")
    parser.add_argument("--rules_path", default="config/normalization_rules.yaml", help="File YAML chứa quy tắc chuẩn hoá")
    parser.add_argument("--val_ratio", type=float, default=0.05, help="Tỷ lệ cho tập validation")
    parser.add_argument("--test_ratio", type=float, default=0.05, help="Tỷ lệ cho tập test")
    parser.add_argument("--min_duration", type=float, default=1.5, help="Độ dài tối thiểu (giây)")
    parser.add_argument("--max_duration", type=float, default=20.0, help="Độ dài tối đa (giây)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Tải quy tắc và chuẩn hoá văn bản
    normalizer = TextNormalizer(Path(args.rules_path))
    df = pd.read_csv(args.processed_csv)
    initial_count = len(df)
    logging.info(f"Đọc {initial_count} dòng từ {args.processed_csv}")

    df["text"] = df["text"].astype(str).map(normalizer.normalize)

    # 2. Lọc dữ liệu theo "data contract"
    df_filtered = df[
        (df["text"].str.len() > 1) &
        (df["duration"] >= args.min_duration) &
        (df["duration"] <= args.max_duration)
    ].copy()
    
    final_count = len(df_filtered)
    rejected_count = initial_count - final_count
    logging.info(f"Lọc dữ liệu: giữ lại {final_count}, loại bỏ {rejected_count} dòng.")
    logging.info(f"  - Điều kiện: text length > 1, duration in [{args.min_duration}, {args.max_duration}]s")

    # 3. Stratified split theo speaker (Train, Val, Test)
    speakers = df_filtered["speaker"].astype(str)
    
    # Tách tập test ra trước
    train_val_idx, test_idx = train_test_split(
        df_filtered.index,
        test_size=args.test_ratio,
        random_state=42,
        stratify=df_filtered["speaker"]
    )
    df_test = df_filtered.loc[test_idx].reset_index(drop=True)
    df_train_val = df_filtered.loc[train_val_idx]

    # Tách train và val từ phần còn lại
    # Cần tính lại tỷ lệ val trên tập train_val
    val_ratio_adjusted = args.val_ratio / (1 - args.test_ratio)
    train_idx, val_idx = train_test_split(
        df_train_val.index,
        test_size=val_ratio_adjusted,
        random_state=42,
        stratify=df_train_val["speaker"]
    )
    df_train = df_filtered.loc[train_idx].reset_index(drop=True)
    df_val = df_filtered.loc[val_idx].reset_index(drop=True)

    # 4. Ghi manifest
    train_manifest = out_dir / "train_manifest.json"
    val_manifest   = out_dir / "val_manifest.json"
    test_manifest  = out_dir / "test_manifest.json"
    
    to_jsonl(df_train, train_manifest)
    to_jsonl(df_val, val_manifest)
    to_jsonl(df_test, test_manifest)

    logging.info("Đã sinh manifest:")
    logging.info("  %s (%d dòng)", train_manifest, len(df_train))
    logging.info("  %s (%d dòng)", val_manifest, len(df_val))
    logging.info("  %s (%d dòng)", test_manifest, len(df_test))
    
    total_hours = df_filtered['duration'].sum() / 3600
    logging.info(f"Tổng thời lượng dữ liệu hợp lệ: {total_hours:.2f} giờ.")

if __name__ == "__main__":
    main()
