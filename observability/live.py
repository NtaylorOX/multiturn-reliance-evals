"""Local live event stream for the monitor.

Unlike the OpenTelemetry export, this carries full message text so a person can watch conversations
as they happen. It is a plain JSONL file under the run's log directory (which is gitignored) and
never leaves the machine.
"""
from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

DEFAULT_LIVE_FILE = "logs/live/events.jsonl"
_lock = threading.Lock()


def live_path() -> Path:
    """Where live events go: `RELIANCE_LIVE_FILE`, else `logs/live/events.jsonl`."""
    return Path(os.getenv("RELIANCE_LIVE_FILE", DEFAULT_LIVE_FILE))


def emit(kind: str, **fields) -> None:
    """Append one event. Never raises: monitoring must not be able to break a run."""
    try:
        path = live_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps({"ts": time.time(), "kind": kind, **fields}, default=str)
        with _lock, path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:  # noqa: BLE001, S110 - deliberate: a monitor failure must not stop a run
        pass
