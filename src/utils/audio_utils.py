import logging
from pydub import AudioSegment
import io

def transcode_to_pcm(
    audio_chunk: bytes, 
    source_format: str, 
    source_rate: int, 
    target_rate: int,
    target_channels: int = 1,
    target_width: int = 2 # 2 bytes for 16-bit
) -> bytes:
    """
    Chuyển mã một đoạn audio từ định dạng nguồn (ví dụ: ulaw, alaw) sang LINEAR16 (raw PCM).

    :param audio_chunk: Dữ liệu audio chunk (bytes).
    :param source_format: Định dạng của pydub ('ulaw', 'alaw').
    :param source_rate: Tần số lấy mẫu nguồn (ví dụ: 8000).
    :param target_rate: Tần số lấy mẫu đích (ví dụ: 16000).
    :param target_channels: Số kênh đích (1 cho mono).
    :param target_width: Độ rộng mẫu đích (2 cho 16-bit).
    :return: Dữ liệu audio đã được chuyển mã sang raw 16-bit PCM, hoặc bytes rỗng nếu lỗi.
    """
    if not audio_chunk:
        return b''

    try:
        # Sử dụng BytesIO để pydub có thể đọc chunk audio trong bộ nhớ như một file
        audio_stream = io.BytesIO(audio_chunk)
        
        # Đọc audio từ stream với thông tin định dạng được cung cấp
        # G.711 (ulaw/alaw) là 8-bit, mono
        audio_segment = AudioSegment.from_file(
            audio_stream,
            format=source_format,
            frame_rate=source_rate,
            channels=1,
            sample_width=1 
        )

        # Thực hiện chuyển đổi nếu các tham số đích khác với nguồn
        if audio_segment.frame_rate != target_rate:
            audio_segment = audio_segment.set_frame_rate(target_rate)
        
        if audio_segment.channels != target_channels:
            audio_segment = audio_segment.set_channels(target_channels)

        if audio_segment.sample_width != target_width:
            audio_segment = audio_segment.set_sample_width(target_width)

        # Trả về dữ liệu PCM rå
        return audio_segment.raw_data
        
    except Exception as e:
        logging.error(f"Audio Utils: Lỗi khi chuyển mã audio từ '{source_format}': {e}", exc_info=True)
        return b''
