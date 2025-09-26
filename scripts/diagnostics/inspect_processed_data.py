
import argparse
import logging
import re
from pathlib import Path
import pandas as pd
import soundfile as sf
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

def analyze_processed_data(wavs_dir: Path):
    logger.info(f"Bắt đầu phân tích thư mục: {wavs_dir}")
    
    wav_files = list(wavs_dir.glob("**/*.wav"))
    
    if not wav_files:
        logger.warning("Không tìm thấy file .wav nào trong thư mục.")
        return

    data = []
    # Regex để trích xuất speaker từ tên file, ví dụ: '50_Nghệ_Sĩ-01-Quyen Linh_1bce1ed8e9ddb62b.wav'
    # Speaker ID sẽ là '50_Nghệ_Sĩ-01-Quyen Linh'
    speaker_pattern = re.compile(r"(.+)_([0-9a-f]{16})\.wav")

    for wav_file in tqdm(wav_files, desc="Đang phân tích các file audio"):
        try:
            info = sf.info(wav_file)
            duration = info.duration
            
            match = speaker_pattern.match(wav_file.name)
            if match:
                speaker = match.group(1)
            else:
                speaker = "unknown"

            data.append({
                "filepath": str(wav_file),
                "speaker": speaker,
                "duration_sec": duration
            })
        except Exception as e:
            logger.warning(f"Lỗi khi xử lý file {wav_file}: {e}")

    if not data:
        logger.error("Không có dữ liệu audio nào được phân tích thành công.")
        return

    df = pd.DataFrame(data)

    # --- Hiển thị thống kê ---
    total_files = len(df)
    total_duration_hours = df["duration_sec"].sum() / 3600
    unique_speakers = df["speaker"].nunique()

    print("\n" + "="*50)
    print("          BÁO CÁO THỐNG KÊ DỮ LIỆU AUDIO ĐÃ XỬ LÝ")
    print("="*50)
    print(f"Tổng số file audio: {total_files:,}")
    print(f"Tổng thời lượng: {total_duration_hours:.2f} giờ")
    print(f"Số lượng người nói (speakers) độc nhất: {unique_speakers:,}")
    print("-"*50)
    
    duration_stats = df["duration_sec"].describe()
    print("Thống kê thời lượng mỗi clip (giây):")
    print(f"  - Ngắn nhất (min): {duration_stats['min']:.2f}s")
    print(f"  - Dài nhất (max): {duration_stats['max']:.2f}s")
    print(f"  - Trung bình (mean): {duration_stats['mean']:.2f}s")
    print(f"  - Trung vị (50%): {duration_stats['50%']:.2f}s")
    print("-"*50)

    speaker_counts = df["speaker"].value_counts()
    print("Thống kê số lượng clip mỗi người nói:")
    print("Top 5 người nói có nhiều clip nhất:")
    print(speaker_counts.head(5).to_string())
    print("\nTop 5 người nói có ít clip nhất:")
    print(speaker_counts.tail(5).to_string())
    print("="*50 + "\n")

def main():
    parser = argparse.ArgumentParser(description="Phân tích và thống kê các file audio đã xử lý.")
    parser.add_argument(
        "--wavs_dir",
        type=Path,
        default=Path("data/processed/phoaudiobook/wavs"),
        help="Thư mục chứa các file .wav đã xử lý."
    )
    args = parser.parse_args()

    if not args.wavs_dir.exists() or not args.wavs_dir.is_dir():
        logger.error(f"Thư mục không tồn tại: {args.wavs_dir}")
        return

    analyze_processed_data(args.wavs_dir)

if __name__ == "__main__":
    main()
