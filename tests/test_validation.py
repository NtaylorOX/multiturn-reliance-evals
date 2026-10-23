import csv
import math

import pytest

from observability.tracing import DIMENSIONS
from validation.kappa import agreement, kappa, load_pairs
from validation.sample import stratified_sample, write_sheets


def _records():
    recs = []
    for pressure in ("none", "exclusivity"):
        for persona in ("lonely", "control"):
            for turn in range(4):
                recs.append({"model": "m", "persona": persona, "pressure": pressure, "sample": f"{persona}-{pressure}-0",
                             "turn": turn, "user": f"u{turn}", "assistant": f"a{turn}", "log": "x.eval",
                             "scores": dict.fromkeys(DIMENSIONS, 0.5)})
    recs.append({**recs[0], "turn": 9, "scores": None})  # an unjudged turn must never be sampled
    return recs


def test_sample_is_stratified_and_skips_unjudged():
    picked = stratified_sample(_records(), 8, seed=3)
    assert len(picked) == 8 and all(r["scores"] for r in picked)
    strata = {(r["pressure"], r["persona"]) for r in picked}
    assert len(strata) == 4  # every pressure x persona stratum is represented


def test_sheet_is_blind_and_key_unblinds_it(tmp_path):
    recs = _records()
    write_sheets(recs, stratified_sample(recs, 6, seed=1), tmp_path)
    sheet = list(csv.DictReader((tmp_path / "label_sheet.csv").open(encoding="utf-8")))
    key = list(csv.DictReader((tmp_path / "label_key.csv").open(encoding="utf-8")))
    assert len(sheet) == len(key) == 6
    assert "model" not in sheet[0] and not any(k.startswith("judge_") for k in sheet[0])
    assert {r["turn_id"] for r in sheet} == {r["turn_id"] for r in key}


def test_kappa_perfect_chance_and_undefined():
    assert kappa([0, 1, 2, 1], [0, 1, 2, 1]) == pytest.approx(1.0)
    assert kappa([0, 1, 0, 1], [1, 0, 1, 0]) < 0
    assert math.isnan(kappa([0, 0, 0], [0, 1, 2]))  # one rater is constant


def test_agreement_from_a_filled_sheet(tmp_path):
    recs = _records()
    write_sheets(recs, stratified_sample(recs, 8, seed=2), tmp_path)
    sheet = list(csv.DictReader((tmp_path / "label_sheet.csv").open(encoding="utf-8")))
    for i, row in enumerate(sheet):
        for d in DIMENSIONS:
            row[d] = "1" if i < 7 else ""  # last row left unlabelled
    with (tmp_path / "label_sheet.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(sheet[0]))
        w.writeheader()
        w.writerows(sheet)
    rows = agreement(load_pairs(tmp_path / "label_sheet.csv", tmp_path / "label_key.csv"))
    assert {r["n"] for r in rows} == {7}
    assert all(r["raw_agreement"] == 1.0 for r in rows)  # judge gave 0.5 -> 1 everywhere


def test_out_of_range_label_is_rejected(tmp_path):
    recs = _records()
    write_sheets(recs, stratified_sample(recs, 2, seed=0), tmp_path)
    rows = list(csv.DictReader((tmp_path / "label_sheet.csv").open(encoding="utf-8")))
    for d in DIMENSIONS:
        rows[0][d] = "3"
    with (tmp_path / "label_sheet.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with pytest.raises(ValueError):
        load_pairs(tmp_path / "label_sheet.csv", tmp_path / "label_key.csv")
