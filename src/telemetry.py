# src/telemetry.py
import os
from opentelemetry import trace, propagators
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider, BatchSpanProcessor
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.instrumentation.logging import LoggingInstrumentor
from opentelemetry.instrumentation.asyncio import AsyncioInstrumentor

# -------------------------------------------------
# 1️⃣ Configure the resource (service name, version, etc.)
# -------------------------------------------------
resource = Resource.create(
    attributes={
        "service.name": "citadel-quantum-trader",
        "service.version": "1.0.0",
        "deployment.environment": os.getenv("DEPLOYMENT_ENV", "dev"),
    }
)

# -------------------------------------------------
# 2️⃣ Set up the tracer provider
# -------------------------------------------------
provider = TracerProvider(resource=resource)
trace.set_tracer_provider(provider)

# -------------------------------------------------
# 3️⃣ Exporter – Jaeger (UDP) – can be swapped for OTLP
# -------------------------------------------------
jaeger_host = os.getenv("JAEGER_HOST", "jaeger")
jaeger_port = int(os.getenv("JAEGER_PORT", "6831"))  # UDP port

jaeger_exporter = JaegerExporter(
    agent_host_name=jaeger_host,
    agent_port=jaeger_port,
)

# You can also use OTLP over HTTP:
# from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
# otlp_exporter = OTLPSpanExporter(endpoint=os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT"))

# -------------------------------------------------
# 4️⃣ Span processor (batch is efficient)
# -------------------------------------------------
provider.add_span_processor(BatchSpanProcessor(jaeger_exporter))

# -------------------------------------------------
# 5️⃣ Optional: instrument standard library logging
# -------------------------------------------------
LoggingInstrumentor().instrument(set_logging_format=True)

# -------------------------------------------------
# 6️⃣ Optional: instrument asyncio (adds context propagation for tasks)
# -------------------------------------------------
AsyncioInstrumentor().instrument()
