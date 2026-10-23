"""API for the monitor's Label and Tune tabs: collect labels in the browser, and run and watch judge tuning.

    GET  /api/validation                         the blind turns to label, and which label files exist
    GET  /api/validation/labels/{name}           one labeller's labels
    PUT  /api/validation/labels/{name}/{turn}    set the five 0/1/2 labels for a turn (human labellers only)
    POST /api/validation/review/{name}           start reviewing a model's labels as an unconfirmed draft
    POST /api/validation/labels/{name}/{turn}/confirm   mark a draft turn as reviewed
    GET  /api/agreement?labels={name}            judge-vs-labeller agreement per dimension
    GET  /api/tuning                             every tuning run, newest first
    POST /api/tuning/start                       start `validation.tune` in the background (one at a time)
"""
from __future__ import annotations

import csv
import json
import math
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from evals.rubric import RUBRIC
from observability.tracing import DIMENSIONS
from validation.kappa import agreement, load_pairs
from validation.labels import load_labels, read_file, review_summary, save_labels, start_review

_NAME = re.compile(r"^[A-Za-z0-9_-]{1,40}$")


class LabelBody(BaseModel):
    """The five rubric labels for one turn, each 0, 1 or 2."""

    labels: dict[str, int]


class TuneBody(BaseModel):
    """Options for starting a tuning run from the UI."""

    labels: str = "claude"
    rounds: int = 4


class ReviewBody(BaseModel):
    """Which model-written label file to take as the draft to review."""

    source: str = "claude"


def _store(path: Path, data: dict, name: str) -> None:
    """Write a labels file back, keeping its kind and any review fields (`source`, `draft`, `confirmed`)."""
    extra = {k: v for k, v in data.items() if k not in ("labeller", "kind", "labels")}
    save_labels(path, data["labels"], labeller=name, kind=data["kind"], extra=extra)


def _check_name(name: str) -> str:
    if not _NAME.match(name):
        raise HTTPException(400, "labeller name must be 1-40 letters, digits, '_' or '-'")
    return name


def build_router(validation_dir: str | Path) -> APIRouter:
    """Routes bound to a validation directory holding `label_sheet.csv`, `label_key.csv`, `labels/` and `tuning/`."""
    root = Path(validation_dir)
    router = APIRouter()
    state: dict = {"proc": None}
    lock = threading.Lock()

    def labels_path(name: str) -> Path:
        return root / "labels" / f"{_check_name(name)}.json"

    @router.get("/api/validation")
    def overview():
        sheet = root / "label_sheet.csv"
        turns = []
        if sheet.exists():
            with sheet.open(newline="", encoding="utf-8") as f:
                turns = [{"turn_id": r["turn_id"], "persona": r["persona"], "context": r["context"],
                          "user": r["user_message"], "assistant": r["assistant_reply"]} for r in csv.DictReader(f)]
        labellers = {}
        for p in sorted((root / "labels").glob("*.json")):  # `*.log.jsonl` audit files do not match this pattern
            labellers[p.stem] = {"kind": read_file(p)["kind"], "count": len(load_labels(p))}
        return {"turns": turns, "labellers": labellers, "dimensions": list(DIMENSIONS), "rubric": RUBRIC}

    @router.get("/api/validation/labels/{name}")
    def get_labels(name: str):
        """A labeller's labels and kind. For a review file this also lists confirmed and changed turns."""
        p = labels_path(name)
        if not p.exists():
            return {"kind": "human", "labels": {}}
        data = read_file(p)
        out = {"kind": data["kind"], "labels": data["labels"]}
        if data["kind"] == "human_reviewed":
            out.update(review_summary(p))
        return out

    @router.post("/api/validation/review/{name}")
    def begin_review(name: str, body: ReviewBody):
        """Start reviewing a model's labels: copy them into `name` as an unconfirmed draft."""
        dest, src = labels_path(name), labels_path(body.source)
        if dest.exists():
            raise HTTPException(409, "that labeller already has a label file")
        if not src.exists():
            raise HTTPException(404, "no such source labels")
        with lock:
            try:
                start_review(src, dest, name)
            except ValueError as exc:
                raise HTTPException(400, str(exc)) from exc
        return {"ok": True}

    @router.post("/api/validation/labels/{name}/{turn_id}/confirm")
    def confirm_turn(name: str, turn_id: str):
        """Mark one draft turn as reviewed. Needs all five labels set."""
        p = labels_path(name)
        if not p.exists() or turn_id not in sheet_turn_ids():
            raise HTTPException(404, "no such labeller or turn")
        with lock:
            data = read_file(p)
            if data["kind"] != "human_reviewed":
                raise HTTPException(400, "only review files have turns to confirm")
            if set(data["labels"].get(turn_id, {})) != set(DIMENSIONS):
                raise HTTPException(400, "the turn needs all five labels before it can be confirmed")
            data.setdefault("confirmed", {})[turn_id] = True
            _store(p, data, name)
            with p.with_suffix(".log.jsonl").open("a", encoding="utf-8") as log:
                log.write(json.dumps({"ts": time.time(), "turn": turn_id, "confirmed": True}) + "\n")
        return {"ok": True}

    def sheet_turn_ids() -> set[str]:
        sheet = root / "label_sheet.csv"
        if not sheet.exists():
            return set()
        with sheet.open(newline="", encoding="utf-8") as f:
            return {r["turn_id"] for r in csv.DictReader(f)}

    @router.put("/api/validation/labels/{name}/{turn_id}")
    def put_label(name: str, turn_id: str, body: LabelBody):
        """Set some of a turn's labels. A later value for the same dimension replaces the earlier one.

        Every change is checked, applied under a lock (so quick successive clicks cannot overwrite each other),
        written atomically, and appended to `<name>.log.jsonl` as an audit trail.
        """
        p = labels_path(name)
        if turn_id not in sheet_turn_ids():
            raise HTTPException(404, "no such turn in the label sheet")
        if not body.labels or set(body.labels) - set(DIMENSIONS) or any(v not in (0, 1, 2) for v in body.labels.values()):
            raise HTTPException(400, "labels must use the five rubric dimensions with values 0, 1 or 2")
        with lock:
            existing = read_file(p) if p.exists() else {"kind": "human", "labels": {}}
            if existing["kind"] == "llm":
                raise HTTPException(403, "that label file was written by a model and is read-only here")
            existing["labels"].setdefault(turn_id, {}).update(body.labels)
            if existing["kind"] == "human_reviewed":
                existing.setdefault("confirmed", {})[turn_id] = False  # an edited turn must be confirmed again
            _store(p, existing, name)
            with p.with_suffix(".log.jsonl").open("a", encoding="utf-8") as log:
                log.write(json.dumps({"ts": time.time(), "turn": turn_id, "set": body.labels}) + "\n")
        return {"ok": True}

    @router.get("/api/agreement")
    def get_agreement(labels: str):
        key = root / "label_key.csv"
        p = labels_path(labels)
        if not p.exists() or not key.exists():
            raise HTTPException(404, "labels or key file missing")
        rows = agreement(load_pairs(p, key))
        # Kappa is NaN when a rater used one constant label, and NaN is not valid JSON: send null instead.
        return {"rows": [{k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in r.items()} for r in rows]}

    @router.get("/api/tuning")
    def tuning_runs():
        runs = []
        for p in sorted((root / "tuning").glob("*.json"), reverse=True):
            try:
                runs.append(json.loads(p.read_text(encoding="utf-8")))
            except json.JSONDecodeError:
                continue  # mid-write; the next poll will see it
        running = state["proc"] is not None and state["proc"].poll() is None
        return {"runs": runs, "running": running}

    @router.post("/api/tuning/start")
    def start_tuning(body: TuneBody):
        if state["proc"] is not None and state["proc"].poll() is None:
            raise HTTPException(409, "a tuning run is already in progress")
        p = labels_path(body.labels)
        if not p.exists():
            raise HTTPException(404, "no such label file")
        if not 1 <= body.rounds <= 8:
            raise HTTPException(400, "rounds must be between 1 and 8")
        state["proc"] = subprocess.Popen(
            [sys.executable, "-m", "validation.tune", "--labels", str(p), "--sheet", str(root / "label_sheet.csv"),
             "--rounds", str(body.rounds), "--out-dir", str(root / "tuning"),
             "--rubric-out", str(root / "rubrics" / "tuned.json")],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return {"started": True}

    return router
