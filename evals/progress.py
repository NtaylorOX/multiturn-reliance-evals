"""Terminal progress view for a running study grid. Reads only the log directory, so it is safe to run alongside a run.

    uv run python -m evals.progress logs/pilot --cells-per-model 16 --watch 10
    uv run python -m evals.progress logs/pilot --run-output logs/pilot_run.txt   # also shows HTTP retries

A cell is one (model, persona, pressure) Inspect log; it appears once the cell has finished.
"""
from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

from evals.models import MODELS


def _finished(model_dir: Path) -> list[Path]:
    """Logs whose status is success. Inspect creates a log when a cell starts, so file count alone overstates progress."""
    from inspect_ai.log import read_eval_log

    done = []
    for f in model_dir.glob("*.eval") if model_dir.exists() else []:
        try:
            if read_eval_log(str(f), header_only=True).status == "success":
                done.append(f)
        except Exception:  # noqa: BLE001, S110 - a log still being written can be unreadable; count it as not done
            pass
    return done


def snapshot(log_dir: Path, cells_per_model: int, run_output: Path | None = None) -> str:
    """A text table of finished cells per model, time since the last finish, a rough ETA and rate-limit retries."""
    now = time.time()
    lines = [f"{'model':<14}{'cells done':>12}{'last finished':>16}"]
    stamps: list[float] = []
    done_total = 0
    for model in MODELS:
        files = sorted(_finished(log_dir / model), key=lambda p: p.stat().st_mtime)
        done_total += len(files)
        stamps += [f.stat().st_mtime for f in files]
        last = f"{int(now - files[-1].stat().st_mtime)}s ago" if files else "-"
        lines.append(f"{model:<14}{f'{len(files)}/{cells_per_model}':>12}{last:>16}")
    total = cells_per_model * len(MODELS)
    lines.append(f"{'total':<14}{f'{done_total}/{total}':>12}")
    if len(stamps) >= 2 and done_total < total:
        rate = (done_total - 1) / (max(stamps) - min(stamps))  # cells per second between first and last finish
        if rate > 0:
            lines.append(f"rough ETA: {int((total - done_total) / rate / 60)} min at the current pace")
    if run_output and run_output.exists():
        m = [int(x) for x in re.findall(r"HTTP retries:\s*(\d+)", run_output.read_text(errors="ignore"))]
        if m:
            # Inspect prints this per task, and it restarts for each new cell, so it is not a running total.
            lines.append(f"HTTP retries in the current cell: {m[-1]} (worst cell so far: {max(m)})")
    return "\n".join(lines)


def main() -> None:
    """Command-line entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log_dir")
    ap.add_argument("--cells-per-model", type=int, default=16)
    ap.add_argument("--run-output", default=None)
    ap.add_argument("--watch", type=int, default=0, help="refresh every N seconds (0 = print once)")
    args = ap.parse_args()
    run_output = Path(args.run_output) if args.run_output else None
    while True:
        out = snapshot(Path(args.log_dir), args.cells_per_model, run_output)
        print(("\033[2J\033[H" if args.watch else "") + out, flush=True)
        if not args.watch:
            break
        time.sleep(args.watch)


if __name__ == "__main__":
    main()
