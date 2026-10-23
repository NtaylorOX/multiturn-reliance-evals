"""API for the monitor's Label and Tune tabs: collect labels in the browser, and run and watch judge tuning.

    GET  /api/validation                         the blind turns to label, and which label files exist
    GET  /api/validation/labels/{name}           one labeller's labels
    PUT  /api/validation/labels/{name}/{turn}    set the five 0/1/2 labels for a turn (human labellers only)
    GET  /api/agreement?labels={name}            judge-vs-labeller agreement per dimension
    GET  /api/tuning                             every tuning run, newest first
    POST /api/tuning/start                       start `validation.tune` in the background (one at a time)
"""
from __future__ import annotations

import csv
import json
import re
import subprocess
import sys
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from evals.rubric import RUBRIC
from observability.tracing import DIMENSIONS
from validation.kappa import agreement, load_pairs
from validation.labels import load_labels, save_labels

_NAME = re.compile(r"^[A-Za-z0-9_-]{1,40}$")


class LabelBody(BaseModel):
    """The five rubric labels for one turn, each 0, 1 or 2."""

    labels: dict[str, int]


class TuneBody(BaseModel):
    """Options for starting a tuning run from the UI."""

    labels: str = "claude"
    rounds: int = 4


def _check_name(name: str) -> str:
    if not _NAME.match(name):
        raise HTTPException(400, "labeller name must be 1-40 letters, digits, '_' or '-'")
    return name


def build_router(validation_dir: str | Path) -> APIRouter:
    """Routes bound to a validation directory holding `label_sheet.csv`, `label_key.csv`, `labels/` and `tuning/`."""
    root = Path(validation_dir)
    router = APIRouter()
    state: dict = {"proc": None}

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
        for p in sorted((root / "labels").glob("*.json")):
            raw = json.loads(p.read_text(encoding="utf-8"))
            labellers[p.stem] = {"kind": raw.get("kind", "human"), "count": len(load_labels(p))}
        return {"turns": turns, "labellers": labellers, "dimensions": list(DIMENSIONS), "rubric": RUBRIC}

    @router.get("/api/validation/labels/{name}")
    def get_labels(name: str):
        p = labels_path(name)
        if not p.exists():
            return {"kind": "human", "labels": {}}
        raw = json.loads(p.read_text(encoding="utf-8"))
        return {"kind": raw.get("kind", "human"), "labels": raw["labels"]}

    @router.put("/api/validation/labels/{name}/{turn_id}")
    def put_label(name: str, turn_id: str, body: LabelBody):
        p = labels_path(name)
        existing = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {"kind": "human", "labels": {}}
        if existing.get("kind") == "llm":
            raise HTTPException(403, "that label file was written by a model and is read-only here")
        if set(body.labels) - set(DIMENSIONS) or any(v not in (0, 1, 2) for v in body.labels.values()):
            raise HTTPException(400, "labels must use the five rubric dimensions with values 0, 1 or 2")
        existing["labels"].setdefault(turn_id, {}).update(body.labels)
        save_labels(p, existing["labels"], labeller=name, kind="human")
        return {"ok": True}

    @router.get("/api/agreement")
    def get_agreement(labels: str):
        key = root / "label_key.csv"
        p = labels_path(labels)
        if not p.exists() or not key.exists():
            raise HTTPException(404, "labels or key file missing")
        return {"rows": agreement(load_pairs(p, key))}

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
