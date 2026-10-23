"""Agreement between the LLM judge and a labeller (a human, or a reference model).

    uv run python -m validation.kappa validation/labels/claude.json validation/label_key.csv

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
from validation.labels import labeller_kind, load_labels


def kappa(a: list[int], b: list[int], weights: str | None = None) -> float:
    """Cohen's kappa between two label lists, NaN when it is undefined (a rater used a single label)."""
    if len(set(a)) < 2 or len(set(b)) < 2:
        return math.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(cohen_kappa_score(a, b, weights=weights))


def judge_scores(key_path: str | Path) -> dict[str, dict[str, int]]:
    """The judge's scores from `label_key.csv` as `turn_id -> {dimension: 0|1|2}`."""
    with open(key_path, newline="", encoding="utf-8") as f:
        return {row["turn_id"]: {d: round(float(row[f"judge_{d}"]) * 2) for d in DIMENSIONS}
                for row in csv.DictReader(f)}


def pairs_from(labels: dict[str, dict[str, int]],
               judge: dict[str, dict[str, int]]) -> dict[str, tuple[list[int], list[int]]]:
    """Line labels up with judge scores. Returns dimension -> (labeller values, judge values) over shared turns."""
    shared = sorted(set(labels) & set(judge), key=int)
    return {d: ([labels[t][d] for t in shared], [judge[t][d] for t in shared]) for d in DIMENSIONS}


def load_pairs(labels_path: str | Path, key_path: str | Path) -> dict[str, tuple[list[int], list[int]]]:
    """Join a labels file (JSON, or a filled-in sheet CSV) to the judge's scores.

    Turns the labeller has not fully labelled yet are skipped, so partial progress can be scored.
    """
    return pairs_from(load_labels(labels_path), judge_scores(key_path))


def agreement(pairs: dict[str, tuple[list[int], list[int]]]) -> list[dict]:
    """One summary row per dimension: sample size, raw agreement and both kappas."""
    rows = []
    for d, (human, judge) in pairs.items():
        n = len(human)
        rows.append({
            "dimension": d,
            "n": n,
            "raw_agreement": round(sum(h == j for h, j in zip(human, judge, strict=False)) / n, 3) if n else math.nan,
            "weighted_kappa": round(kappa(human, judge, "quadratic"), 3) if n else math.nan,
            "binary_kappa": round(kappa([h >= 1 for h in human], [j >= 1 for j in judge]), 3) if n else math.nan,
        })
    return rows


def main() -> None:
    """Command-line entry point."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("labels", help="labels JSON, or a filled-in label_sheet.csv")
    ap.add_argument("key")
    ap.add_argument("--out", default=None, help="optional CSV path for the summary")
    args = ap.parse_args()
    rows = agreement(load_pairs(args.labels, args.key))
    print(f"labeller kind: {labeller_kind(args.labels)}")
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
