# src/tracing.py
import os
from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider, BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

# Resource identifies this service in Jaeger
resource = Resource(attributes={
    "service.name": "citadel-quantum-trader",
    "service.instance.id": os.getenv("HOSTNAME", "unknown"),
    "service.version": "1.0.0",
})

provider = TracerProvider(resource=resource)
trace.set_tracer_provider(provider)

# Export spans via OTLP over HTTP (default port 4318)
otlp_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://tempo:4318")
otlp_exporter = OTLPSpanExporter(endpoint=otlp_endpoint)
span_processor = BatchSpanProcessor(otlp_exporter)
provider.add_span_processor(span_processor)

tracer = trace.get_tracer(__name__)
