"""Draw a blind sample of judged turns for a human to label.

    uv run python -m validation.sample logs/pilot2 --n 30 --out validation

Writes two files:

* `label_sheet.csv`: what the human sees. The conversation context, the user's message, the assistant's
  reply, and empty columns for the five rubric dimensions. It deliberately omits the model name and the
  judge's scores so the labels are independent of both.
* `label_key.csv`: the judge's scores and the turn's origin, joined back on `turn_id` by `validation.kappa`.

Label each dimension 0 (absent), 1 (mild or ambiguous) or 2 (clear and strong), using the rubric in
`evals/rubric.py`.
"""
from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path

from evals.logs import iter_turn_records
from observability.tracing import DIMENSIONS


def stratified_sample(records: list[dict], n: int, seed: int = 0) -> list[dict]:
    """Pick `n` turns, spread as evenly as possible over pressure conditions, then personas.

    Only turns the judge scored are eligible. Stratifying stops the sample being dominated by the unpressured
    baseline, where almost every turn scores zero and agreement would be trivially high.
    """
    rng = random.Random(seed)
    strata: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in records:
        if r["scores"]:
            strata[(r["pressure"], r["persona"])].append(r)
    for pool in strata.values():
        rng.shuffle(pool)
    keys = sorted(strata)
    rng.shuffle(keys)
    picked: list[dict] = []
    while len(picked) < n and any(strata.values()):
        for key in keys:
            if strata[key] and len(picked) < n:
                picked.append(strata[key].pop())
    return picked


def _context(records: list[dict], rec: dict) -> str:
    """The previous exchange in the same conversation, so a rater can judge a turn in context."""
    prev = [r for r in records if r["sample"] == rec["sample"] and r["model"] == rec["model"]
            and r["turn"] == rec["turn"] - 1]
    return f"USER: {prev[0]['user']}\nASSISTANT: {prev[0]['assistant']}" if prev else ""


def write_sheets(records: list[dict], picked: list[dict], out_dir: Path) -> None:
    """Write the blind `label_sheet.csv` and the `label_key.csv` that unblinds it."""
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(1)
    order = list(range(len(picked)))
    rng.shuffle(order)  # so the sheet is not grouped by condition
    with (out_dir / "label_sheet.csv").open("w", newline="", encoding="utf-8") as sheet, \
            (out_dir / "label_key.csv").open("w", newline="", encoding="utf-8") as key:
        sw = csv.writer(sheet)
        kw = csv.writer(key)
        sw.writerow(["turn_id", "persona", "context", "user_message", "assistant_reply", *DIMENSIONS, "notes"])
        kw.writerow(["turn_id", "model", "persona", "pressure", "sample", "turn", "log",
                     *[f"judge_{d}" for d in DIMENSIONS]])
        for turn_id, idx in enumerate(order, start=1):
            r = picked[idx]
            sw.writerow([turn_id, r["persona"], _context(records, r), r["user"], r["assistant"],
                         *[""] * len(DIMENSIONS), ""])
            kw.writerow([turn_id, r["model"], r["persona"], r["pressure"], r["sample"], r["turn"], r["log"],
                         *[r["scores"][d] for d in DIMENSIONS]])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log_dir")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="validation")
    args = ap.parse_args()
    records = list(iter_turn_records(args.log_dir))
    picked = stratified_sample(records, args.n, args.seed)
    write_sheets(records, picked, Path(args.out))
    print(f"wrote {len(picked)} turns to {args.out}/label_sheet.csv (key: label_key.csv)")


if __name__ == "__main__":
    main()
