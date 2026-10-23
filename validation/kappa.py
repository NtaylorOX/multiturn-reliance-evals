"""Agreement between the LLM judge and a human labeller.

    uv run python -m validation.kappa validation/label_sheet.csv validation/label_key.csv

Reports Cohen's kappa for each rubric dimension in two forms:

* `weighted_kappa`: quadratic-weighted over the 0/1/2 scale, so a 0-vs-2 disagreement costs more than 0-vs-1.
* `binary_kappa`: whether the sign is present at all (score >= 1), which is the more forgiving question.

Kappa is undefined when either rater gives one constant label, which is common for rare dimensions on a small
sample. In that case the value is reported as NaN and `raw_agreement` is the number to read.
"""
from __future__ import annotations

import argparse
import csv
import math
import warnings
from pathlib import Path

from sklearn.metrics import cohen_kappa_score

from observability.tracing import DIMENSIONS


def kappa(a: list[int], b: list[int], weights: str | None = None) -> float:
    """Cohen's kappa between two label lists, NaN when it is undefined (a rater used a single label)."""
    if len(set(a)) < 2 or len(set(b)) < 2:
        return math.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(cohen_kappa_score(a, b, weights=weights))


def load_pairs(sheet_path: str | Path, key_path: str | Path) -> dict[str, tuple[list[int], list[int]]]:
    """Join the human sheet to the judge's scores. Returns dimension -> (human labels, judge labels), both 0..2.

    Rows the human has not labelled yet are skipped, so partial progress can be scored.
    """
    with open(key_path, newline="", encoding="utf-8") as f:
        key = {row["turn_id"]: row for row in csv.DictReader(f)}
    pairs: dict[str, tuple[list[int], list[int]]] = {d: ([], []) for d in DIMENSIONS}
    with open(sheet_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if any(row[d].strip() == "" for d in DIMENSIONS):
                continue
            judged = key[row["turn_id"]]
            for d in DIMENSIONS:
                human = int(row[d])
                if human not in (0, 1, 2):
                    raise ValueError(f"turn {row['turn_id']}, {d}: label must be 0, 1 or 2, got {human}")
                pairs[d][0].append(human)
                pairs[d][1].append(round(float(judged[f"judge_{d}"]) * 2))
    return pairs


def agreement(pairs: dict[str, tuple[list[int], list[int]]]) -> list[dict]:
    """One summary row per dimension: sample size, raw agreement and both kappas."""
    rows = []
    for d, (human, judge) in pairs.items():
        n = len(human)
        rows.append({
            "dimension": d,
            "n": n,
            "raw_agreement": round(sum(h == j for h, j in zip(human, judge)) / n, 3) if n else math.nan,
            "weighted_kappa": round(kappa(human, judge, "quadratic"), 3) if n else math.nan,
            "binary_kappa": round(kappa([h >= 1 for h in human], [j >= 1 for j in judge]), 3) if n else math.nan,
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sheet")
    ap.add_argument("key")
    ap.add_argument("--out", default=None, help="optional CSV path for the summary")
    args = ap.parse_args()
    rows = agreement(load_pairs(args.sheet, args.key))
    print(f"{'dimension':<30}{'n':>4}{'raw':>8}{'w-kappa':>10}{'bin-kappa':>11}")
    for r in rows:
        print(f"{r['dimension']:<30}{r['n']:>4}{r['raw_agreement']:>8}{r['weighted_kappa']:>10}{r['binary_kappa']:>11}")
    if args.out:
        with open(args.out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
