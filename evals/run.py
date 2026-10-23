"""Run the study grid: models x personas x pressures, `n` seeds each.

    uv run python -m evals.run --models llama31-8b,deepseek-r1,nemotron --n 1 --log-dir logs/pilot

Each (model, persona, pressure) cell is one Inspect task, so every cell gets its own log file.
Finished cells are skipped on a rerun (Inspect's eval_set), so an interrupted run resumes.
"""
from __future__ import annotations

import argparse
import os

from dotenv import load_dotenv
from inspect_ai import eval_set

from evals.conversation import reliance_eval
from evals.models import MODELS
from redteam.personas import PERSONAS
from redteam.pressure import PRESSURES


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--personas", default=",".join(PERSONAS))
    ap.add_argument("--pressures", default=",".join(PRESSURES))
    ap.add_argument("--n", type=int, default=1, help="conversations (seeds) per cell")
    ap.add_argument("--turns", type=int, default=10)
    # NVIDIA's free tier allows roughly 40 requests a minute, shared by the simulator, judge and target.
    # Total in-flight requests is about max-tasks x max-connections, so keep both small.
    ap.add_argument("--max-connections", type=int, default=3)
    ap.add_argument("--max-tasks", type=int, default=2, help="cells run in parallel")
    ap.add_argument("--temperature", type=float, default=0.7, help="target model temperature")
    ap.add_argument("--log-dir", required=True)
    args = ap.parse_args()

    load_dotenv()
    # Everything a run writes beside its logs: live events for the monitor, and the OpenTelemetry spans.
    os.environ["RELIANCE_LIVE_FILE"] = f"{args.log_dir}/live.jsonl"
    os.environ["RELIANCE_TRACE_FILE"] = f"{args.log_dir}/spans.jsonl"
    tasks = [reliance_eval(persona=p, pressure=q, n=args.n, turns=args.turns)
             for p in args.personas.split(",") for q in args.pressures.split(",")]
    for name in args.models.split(","):
        eval_set(tasks, model=MODELS[name], log_dir=f"{args.log_dir}/{name}", max_connections=args.max_connections,
                 max_tasks=args.max_tasks, temperature=args.temperature, retry_attempts=2, display="plain")


if __name__ == "__main__":
    main()
