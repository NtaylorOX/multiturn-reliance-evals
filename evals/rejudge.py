"""Re-score the turns in existing logs with a different rubric or judge, without rerunning any conversation.

    uv run python -m evals.rejudge logs/pilot2 --rubric validation/rubrics/tuned.json --out results/pilot/rescored.jsonl

The conversations stay exactly as they were; only the judge's scores change. Output is one JSON line per turn
(`log`, `sample`, `turn`, `scores`). Pass it to the plots with `--rescored`, or to `evals.logs.iter_turn_records`.
An interrupted run resumes: turns already in the output file are skipped.
"""
from __future__ import annotations

import argparse
import asyncio
import json
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv

from evals.conversation import _transcript
from evals.judge import judge_reply
from evals.logs import iter_turn_records
from evals.models import JUDGE
from evals.rubric import load_rubric


async def rejudge(log_dir: str | Path, out: Path, judge: str = JUDGE, rubric_path: str | None = None,
                  concurrency: int = 2) -> int:
    """Re-score every turn under `log_dir`, appending to `out`. Returns the number of turns scored this call."""
    rubric = load_rubric(rubric_path)
    done = set()
    if out.exists():
        for line in out.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            done.add((r["log"], r["sample"], r["turn"]))
    convs: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in iter_turn_records(log_dir):
        convs[(r["log"], r["sample"])].append(r)
    out.parent.mkdir(parents=True, exist_ok=True)
    sem = asyncio.Semaphore(concurrency)
    count = 0

    async def one(turns: list[dict], idx: int, fh) -> None:
        nonlocal count
        r = turns[idx]
        if (r["log"], r["sample"], r["turn"]) in done:
            return
        async with sem:
            scores = await judge_reply(judge, _transcript(turns[:idx]), r["user"], r["assistant"], rubric)
        fh.write(json.dumps({"log": r["log"], "sample": r["sample"], "turn": r["turn"], "scores": scores}) + "\n")
        fh.flush()
        count += 1

    with out.open("a", encoding="utf-8") as fh:
        await asyncio.gather(*(one(sorted(t, key=lambda x: x["turn"]), i, fh)
                               for t in convs.values() for i in range(len(t))))
    return count


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log_dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--rubric", default=None, help="rubric JSON (default: the built-in rubric)")
    ap.add_argument("--judge", default=JUDGE)
    args = ap.parse_args()
    load_dotenv()
    n = asyncio.run(rejudge(args.log_dir, Path(args.out), args.judge, args.rubric))
    print(f"scored {n} turns -> {args.out}")


if __name__ == "__main__":
    main()
