import logging
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource

def initialize_tracer(service_name: str) -> TracerProvider:
    """
    Cấu hình và khởi tạo OpenTelemetry Tracer.
    Hàm này nên được gọi một lần khi ứng dụng khởi động.
    """
    # Tạo một resource để định danh service này trong hệ thống tracing (ví dụ: Jaeger, Zipkin)
    resource = Resource(attributes={
        "service.name": service_name
    })

    # Cấu hình provider, đây là trái tim của OTel SDK
    provider = TracerProvider(resource=resource)

    # Cấu hình exporter để gửi dữ liệu đến một OpenTelemetry Collector.
    # Mặc định, nó sẽ gửi đến OTel Collector tại http://localhost:4317
    # Bạn cần phải chạy một OTel Collector riêng để nhận dữ liệu này.
    exporter = OTLPSpanExporter() 

    # Sử dụng BatchSpanProcessor để gửi các span theo từng batch, hiệu quả hơn cho production
    processor = BatchSpanProcessor(exporter)
    provider.add_span_processor(processor)

    # Set provider này làm provider toàn cục cho cả ứng dụng
    trace.set_tracer_provider(provider)
    
    logging.info(f"OpenTelemetry Tracer đã được khởi tạo cho service '{service_name}'.")
    return provider

# Lấy một tracer object toàn cục mà các module khác có thể import và sử dụng
# Tên của tracer thường là tên của module hoặc thư viện instrument
tracer = trace.get_tracer("voip.ai.agent.tracer")
