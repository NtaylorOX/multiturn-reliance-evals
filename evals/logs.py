"""Read judged turns back out of Inspect logs.

Every analysis in this repo (plots, validation, the monitor) starts from the same place: the per-turn
records that `evals.conversation` stores in each sample's metadata. This module is the one reader.
"""
from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

from evals.models import MODELS

_SHORT_NAMES = {v: k for k, v in MODELS.items()}


def short_model_name(model: str) -> str:
    """Map an Inspect model string back to the short name in `evals.models.MODELS` when there is one."""
    return _SHORT_NAMES.get(model, model)


def iter_turn_records(log_dir: str | Path) -> Iterator[dict]:
    """Yield one record per assistant turn from every successful `.eval` log under `log_dir`.

    Each record has: `model`, `persona`, `pressure`, `sample` (conversation id), `turn` (0-based),
    `user`, `assistant`, `scores` (rubric dimension -> 0..1, or None if the judge reply could not be
    parsed) and `log` (the log file the turn came from). Logs that did not finish are skipped.
    """
    from inspect_ai.log import read_eval_log

    for path in sorted(Path(log_dir).rglob("*.eval")):
        log = read_eval_log(str(path))
        if log.status != "success" or not log.samples:
            continue
        for sample in log.samples:
            for rec in sample.metadata.get("turns", []):
                yield {
                    "model": short_model_name(log.eval.model),
                    "persona": sample.metadata["persona"],
                    "pressure": sample.metadata["pressure"],
                    "sample": str(sample.id),
                    "turn": rec["turn"],
                    "user": rec["user"],
                    "assistant": rec["assistant"],
                    "scores": rec["scores"],
                    "log": path.name,
                }
