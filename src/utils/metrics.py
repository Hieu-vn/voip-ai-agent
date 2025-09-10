from prometheus_client import Histogram, Counter

# --- Latency Metrics (Histogram) ---
STT_LATENCY_SECONDS = Histogram(
    'stt_latency_seconds',
    'Độ trễ của một lần xử lý STT (từ lúc bắt đầu nghe đến khi có transcript cuối cùng)'
)

NLP_LATENCY_SECONDS = Histogram(
    'nlp_latency_seconds',
    'Độ trễ của một lần xử lý NLP Agent (từ lúc nhận text đến khi có phản hồi cuối cùng)'
)

TTS_LATENCY_SECONDS = Histogram(
    'tts_latency_seconds',
    'Độ trễ của một lần tổng hợp giọng nói (bao gồm cả cache lookup và gọi API)'
)

# --- Error & Event Metrics (Counter) ---
STT_ERRORS_TOTAL = Counter(
    'stt_errors_total',
    'Tổng số lỗi xảy ra trong quá trình STT',
    ['type']  # Phân loại lỗi, ví dụ: 'api_error', 'timeout'
)

LLM_ERRORS_TOTAL = Counter(
    'llm_errors_total',
    'Tổng số lỗi xảy ra khi gọi LLM',
    ['type'] # Phân loại lỗi, ví dụ: 'timeout', 'client_error', 'api_error'
)

TTS_ERRORS_TOTAL = Counter(
    'tts_errors_total',
    'Tổng số lỗi xảy ra trong quá trình tổng hợp giọng nói'
)

BARGEIN_COUNT_TOTAL = Counter(
    'bargein_count_total',
    'Tổng số lần người dùng ngắt lời (barge-in)'
)
