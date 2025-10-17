from opentelemetry import trace as otel_trace

import sglang.srt.tracing.trace as trace_module
from sglang.srt.tracing.trace import (
    process_tracing_init,
    trace_req_finish,
    trace_req_start,
)


def test_trace_req_start_auto_registers_thread():
    # Ensure clean state
    trace_module.threads_info.clear()
    trace_module.reqs_context.clear()
    trace_module.tracing_enabled = False

    process_tracing_init("localhost:4317", "sglang-test")

    rid = "req-auto-register"
    try:
        trace_req_start(rid)
        assert rid in trace_module.reqs_context
    finally:
        trace_req_finish(rid)
        provider = otel_trace.get_tracer_provider()
        shutdown = getattr(provider, "shutdown", None)
        if callable(shutdown):
            shutdown()

    trace_module.threads_info.clear()
    trace_module.reqs_context.clear()
    trace_module.tracing_enabled = False
