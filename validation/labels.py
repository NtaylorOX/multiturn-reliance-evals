"""Reading and writing labels for the blind sample.

Labels live in one small JSON file per labeller:

    {"labeller": "alice", "kind": "human",
     "labels": {"1": {"exclusivity": 2, "sycophancy": 0, ...}, "2": {...}}}

`kind` records who produced the labels, which any report of agreement must state:

* `human`: a person labelled each turn from scratch.
* `llm`: a model labelled the turns. Read-only in the monitor.
* `human_reviewed`: a person reviewed a model's draft labels, turn by turn. The file also holds `source` (whose draft),
  `draft` (the untouched pre-labels) and `confirmed` (the turns the person has actually reviewed). Only confirmed
  turns count as labels, so an unreviewed draft can never be mistaken for human judgement.

A turn that is only partly labelled is ignored until all five dimensions are set.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

from observability.tracing import DIMENSIONS

KINDS = ("human", "llm", "human_reviewed")


def read_file(path: str | Path) -> dict:
    """The raw labels file with defaults filled in (`kind` human, empty `labels`)."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    raw.setdefault("kind", "human")
    raw.setdefault("labels", {})
    return raw


def load_labels(path: str | Path) -> dict[str, dict[str, int]]:
    """Return `turn_id -> {dimension: 0|1|2}` from a labels JSON file or a filled-in `label_sheet.csv`.

    Only fully labelled turns are returned, and for `human_reviewed` files only turns the reviewer has confirmed.
    Raises ValueError for a label outside 0..2.
    """
    path = Path(path)
    confirmed: set[str] | None = None
    if path.suffix == ".json":
        data = read_file(path)
        raw = data["labels"]
        if data["kind"] == "human_reviewed":
            confirmed = {str(t) for t, ok in data.get("confirmed", {}).items() if ok}
    else:
        with path.open(newline="", encoding="utf-8") as f:
            raw = {row["turn_id"]: {d: row[d] for d in DIMENSIONS} for row in csv.DictReader(f)}
    out: dict[str, dict[str, int]] = {}
    for turn_id, dims in raw.items():
        if confirmed is not None and str(turn_id) not in confirmed:
            continue
        if any(str(dims.get(d, "")).strip() == "" for d in DIMENSIONS):
            continue
        labels = {d: int(dims[d]) for d in DIMENSIONS}
        if any(v not in (0, 1, 2) for v in labels.values()):
            raise ValueError(f"turn {turn_id}: labels must be 0, 1 or 2, got {labels}")
        out[str(turn_id)] = labels
    return out


def save_labels(path: str | Path, labels: dict[str, dict[str, int]], labeller: str, kind: str = "human",
                extra: dict | None = None) -> None:
    """Write labels in the JSON format above, creating the folder if needed. `extra` adds review fields."""
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"labeller": labeller, "kind": kind, "labels": labels, **(extra or {})}
    tmp = path.with_suffix(".tmp")  # write beside the target, then swap, so a crash never leaves a half-written file
    tmp.write_text(json.dumps(payload, indent=1, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def labeller_kind(path: str | Path) -> str:
    """Who produced a labels file: "human", "llm" or "human_reviewed". A CSV sheet is filled in by hand: "human"."""
    path = Path(path)
    return read_file(path)["kind"] if path.suffix == ".json" else "human"


def start_review(source_path: str | Path, dest_path: str | Path, labeller: str) -> None:
    """Create a `human_reviewed` file for `labeller` holding a copy of a model's labels as an unconfirmed draft."""
    src = read_file(source_path)
    if src["kind"] != "llm":
        raise ValueError("only model-written labels can be used as a draft to review")
    draft = {t: dict(d) for t, d in src["labels"].items()}
    save_labels(dest_path, {t: dict(d) for t, d in draft.items()}, labeller, kind="human_reviewed",
                extra={"source": src.get("labeller", Path(source_path).stem), "draft": draft, "confirmed": {}})


def review_summary(path: str | Path) -> dict:
    """For a `human_reviewed` file: the confirmed turn ids and those where the reviewer changed the draft."""
    data = read_file(path)
    confirmed = sorted((t for t, ok in data.get("confirmed", {}).items() if ok), key=int)
    draft = data.get("draft", {})
    changed = [t for t in confirmed if data["labels"].get(t) != draft.get(t)]
    return {"source": data.get("source"), "confirmed": confirmed, "changed": changed}
