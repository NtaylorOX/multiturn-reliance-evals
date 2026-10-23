"""Span export: a JSONL file for local analysis, plus OTLP when an endpoint is configured."""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path

from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult

from observability.tracing import Telemetry

DEFAULT_TRACE_FILE = "logs/traces/spans.jsonl"


class JsonlSpanExporter(SpanExporter):
    """Append one JSON object per finished span: ids, name, timing and attributes."""

    def __init__(self, path: str | Path) -> None:
        """Create the output folder if needed; spans are appended to `path`."""
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def export(self, spans) -> SpanExportResult:
        lines = []
        for sp in spans:
            ctx = sp.get_span_context()
            lines.append(json.dumps({
                "trace_id": f"{ctx.trace_id:032x}",
                "span_id": f"{ctx.span_id:016x}",
                "parent_id": f"{sp.parent.span_id:016x}" if sp.parent else None,
                "name": sp.name,
                "start_ns": sp.start_time,
                "end_ns": sp.end_time,
                "attributes": dict(sp.attributes or {}),
            }, default=list))
        with self._lock, self.path.open("a", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        pass


_telemetry: Telemetry | None = None


def get_telemetry() -> Telemetry:
    """Process-wide Telemetry writing to RELIANCE_TRACE_FILE (and OTLP if OTEL_EXPORTER_OTLP_ENDPOINT is set)."""
    global _telemetry
    if _telemetry is None:
        _telemetry = Telemetry(exporter=JsonlSpanExporter(os.getenv("RELIANCE_TRACE_FILE", DEFAULT_TRACE_FILE)),
                               otlp_endpoint=os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT") or None)
    return _telemetry


def load_spans(path: str | Path = DEFAULT_TRACE_FILE) -> list[dict]:
    """Read a spans JSONL file back as a list of dicts."""
    with Path(path).open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
