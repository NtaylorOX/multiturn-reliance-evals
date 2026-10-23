"""Local monitor for a running study grid: a small API plus one React page (no build step).

    uv run python -m monitor.server logs/pilot            # then open http://localhost:8765

    GET /                 the page
    GET /api/progress     finished cells per model, read from the Inspect logs
    GET /api/stream       server-sent events: replays the run's live.jsonl, then follows it
"""
from __future__ import annotations

import argparse
import asyncio
import time
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse

from evals.models import MODELS
from evals.progress import _finished

STATIC = Path(__file__).parent / "static"


def read_new_lines(path: Path, offset: int) -> tuple[list[str], int]:
    """Complete lines appended since `offset`, and the new offset. A partial last line is left for next time."""
    if not path.exists():
        return [], offset
    with path.open("rb") as f:
        f.seek(offset)
        chunk = f.read()
    end = chunk.rfind(b"\n")
    if end < 0:
        return [], offset
    lines = [ln for ln in chunk[: end + 1].decode("utf-8", "replace").splitlines() if ln.strip()]
    return lines, offset + end + 1


def create_app(log_dir: str | Path, cells_per_model: int = 16) -> FastAPI:
    log_dir = Path(log_dir)
    app = FastAPI(title="reliance-evals monitor")
    cache: dict = {"at": 0.0, "value": None}

    @app.get("/api/progress")
    def progress():
        # Reading every log header is not free, so cache briefly.
        if time.time() - cache["at"] > 5:
            models = {}
            for m in MODELS:
                models[m] = {"done": len(_finished(log_dir / m)), "total": cells_per_model}
            cache.update(at=time.time(), value={"models": models, "log_dir": str(log_dir)})
        return cache["value"]

    @app.get("/api/stream")
    async def stream():
        live = log_dir / "live.jsonl"

        async def gen():
            offset = 0
            idle = 0
            while True:
                lines, offset = read_new_lines(live, offset)
                for ln in lines:
                    yield f"data: {ln}\n\n"
                idle = 0 if lines else idle + 1
                if idle and idle % 20 == 0:
                    yield ": keepalive\n\n"
                await asyncio.sleep(0.5)

        return StreamingResponse(gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache"})

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    return app


def main() -> None:
    import uvicorn

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log_dir")
    ap.add_argument("--cells-per-model", type=int, default=16)
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    uvicorn.run(create_app(args.log_dir, args.cells_per_model), host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
