import json

from fastapi import FastAPI
from fastapi.testclient import TestClient

from monitor.validation_api import build_router
from observability.tracing import DIMENSIONS
from validation.labels import save_labels

SHEET = "turn_id,persona,context,user_message,assistant_reply,notes\n1,lonely,,hi,hello,\n2,lonely,,hey,yo,\n"


def _client(tmp_path):
    (tmp_path / "label_sheet.csv").write_text(SHEET, encoding="utf-8")
    app = FastAPI()
    app.include_router(build_router(tmp_path))
    return TestClient(app)


def _put(c, name, turn, labels):
    return c.put(f"/api/validation/labels/{name}/{turn}", json={"labels": labels})


def test_clicking_another_option_replaces_the_value_and_partial_turns_persist(tmp_path):
    c = _client(tmp_path)
    assert _put(c, "ann", "1", {"exclusivity": 2}).status_code == 200
    assert _put(c, "ann", "1", {"exclusivity": 0}).status_code == 200  # a second click replaces, never accumulates
    assert c.get("/api/validation/labels/ann").json()["labels"] == {"1": {"exclusivity": 0}}


def test_invalid_input_is_rejected_and_nothing_is_written(tmp_path):
    c = _client(tmp_path)
    assert _put(c, "ann", "1", {"exclusivity": 3}).status_code == 400
    assert _put(c, "ann", "1", {"mood": 1}).status_code == 400
    assert _put(c, "ann", "1", {}).status_code == 400
    assert _put(c, "ann", "99", {"exclusivity": 1}).status_code == 404   # not a turn in the sheet
    assert _put(c, "../evil", "1", {"exclusivity": 1}).status_code in (400, 404)
    assert not (tmp_path / "labels" / "ann.json").exists()


def test_model_written_labels_are_read_only(tmp_path):
    c = _client(tmp_path)
    save_labels(tmp_path / "labels" / "claude.json", {"1": dict.fromkeys(DIMENSIONS, 0)}, "claude", kind="llm")
    assert _put(c, "claude", "1", {"exclusivity": 2}).status_code == 403


def test_a_refresh_sees_saved_labels_and_every_change_is_logged(tmp_path):
    c = _client(tmp_path)
    for d in DIMENSIONS:
        _put(c, "ann", "2", {d: 1})
    fresh = _client(tmp_path)  # a new client stands in for a page refresh
    assert fresh.get("/api/validation/labels/ann").json()["labels"]["2"] == dict.fromkeys(DIMENSIONS, 1)
    log = (tmp_path / "labels" / "ann.log.jsonl").read_text().splitlines()
    assert len(log) == len(DIMENSIONS) and json.loads(log[0])["turn"] == "2"
    assert fresh.get("/api/validation").json()["labellers"]["ann"]["count"] == 1  # one fully labelled turn
