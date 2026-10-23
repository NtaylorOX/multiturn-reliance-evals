"""Reading and writing human (or reference) labels for the blind sample.

Labels live in a small JSON file per labeller, so several people can label the same sheet and the monitor UI can
save to it:

    {"labeller": "alice", "kind": "human",
     "labels": {"1": {"exclusivity": 2, "sycophancy": 0, ...}, "2": {...}}}

A turn that is only partly labelled is ignored until all five dimensions are set. `kind` records who produced the
labels ("human" or "llm"), which any report of agreement should state.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

from observability.tracing import DIMENSIONS


def load_labels(path: str | Path) -> dict[str, dict[str, int]]:
    """Return `turn_id -> {dimension: 0|1|2}` from a labels JSON file or a filled-in `label_sheet.csv`.

    Only fully labelled turns are returned. Raises ValueError for a label outside 0..2.
    """
    path = Path(path)
    if path.suffix == ".json":
        raw = json.loads(path.read_text(encoding="utf-8"))["labels"]
    else:
        with path.open(newline="", encoding="utf-8") as f:
            raw = {row["turn_id"]: {d: row[d] for d in DIMENSIONS} for row in csv.DictReader(f)}
    out: dict[str, dict[str, int]] = {}
    for turn_id, dims in raw.items():
        if any(str(dims.get(d, "")).strip() == "" for d in DIMENSIONS):
            continue
        labels = {d: int(dims[d]) for d in DIMENSIONS}
        if any(v not in (0, 1, 2) for v in labels.values()):
            raise ValueError(f"turn {turn_id}: labels must be 0, 1 or 2, got {labels}")
        out[str(turn_id)] = labels
    return out


def save_labels(path: str | Path, labels: dict[str, dict[str, int]], labeller: str, kind: str = "human") -> None:
    """Write labels in the JSON format above, creating the folder if needed."""
    if kind not in ("human", "llm"):
        raise ValueError("kind must be 'human' or 'llm'")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"labeller": labeller, "kind": kind, "labels": labels}
    tmp = path.with_suffix(".tmp")  # write beside the target, then swap, so a crash never leaves a half-written file
    tmp.write_text(json.dumps(payload, indent=1, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def labeller_kind(path: str | Path) -> str:
    """Who produced a labels file: "human", "llm", or "human" for a CSV sheet (filled in by hand)."""
    path = Path(path)
    if path.suffix != ".json":
        return "human"
    return json.loads(path.read_text(encoding="utf-8")).get("kind", "human")
