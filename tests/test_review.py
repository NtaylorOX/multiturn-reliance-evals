import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from monitor.validation_api import build_router
from observability.tracing import DIMENSIONS
from validation.labels import labeller_kind, load_labels, read_file, review_summary, save_labels, start_review

SHEET = "turn_id,persona,context,user_message,assistant_reply,notes\n1,lonely,,hi,hello,\n2,lonely,,hey,yo,\n"


def _draft():
    return {"1": dict.fromkeys(DIMENSIONS, 0), "2": {**dict.fromkeys(DIMENSIONS, 0), "exclusivity": 2}}


def _setup(tmp_path):
    (tmp_path / "label_sheet.csv").write_text(SHEET, encoding="utf-8")
    save_labels(tmp_path / "labels" / "claude.json", _draft(), "claude", kind="llm")
    app = FastAPI()
    app.include_router(build_router(tmp_path))
    return TestClient(app)


def test_unconfirmed_draft_turns_never_count_as_labels(tmp_path):
    save_labels(tmp_path / "claude.json", _draft(), "claude", kind="llm")
    start_review(tmp_path / "claude.json", tmp_path / "ann.json", "ann")
    assert labeller_kind(tmp_path / "ann.json") == "human_reviewed"
    assert load_labels(tmp_path / "ann.json") == {}          # nothing reviewed yet
    assert read_file(tmp_path / "ann.json")["draft"] == _draft()


def test_only_model_labels_can_seed_a_review(tmp_path):
    save_labels(tmp_path / "bob.json", _draft(), "bob", kind="human")
    with pytest.raises(ValueError):
        start_review(tmp_path / "bob.json", tmp_path / "ann.json", "ann")


def test_review_flow_confirm_change_and_reconfirm(tmp_path):
    c = _setup(tmp_path)
    assert c.post("/api/validation/review/ann", json={"source": "claude"}).status_code == 200
    assert c.post("/api/validation/review/ann", json={"source": "claude"}).status_code == 409   # already started
    assert c.post("/api/validation/labels/ann/1/confirm").status_code == 200                    # agree with the draft
    assert c.put("/api/validation/labels/ann/2", json={"labels": {"exclusivity": 0}}).status_code == 200  # disagree
    assert c.post("/api/validation/labels/ann/2/confirm").status_code == 200
    got = c.get("/api/validation/labels/ann").json()
    assert got["kind"] == "human_reviewed" and got["confirmed"] == ["1", "2"] and got["changed"] == ["2"]
    assert set(load_labels(tmp_path / "labels" / "ann.json")) == {"1", "2"}
    # editing a confirmed turn withdraws its confirmation until it is confirmed again
    c.put("/api/validation/labels/ann/1", json={"labels": {"sycophancy": 1}})
    assert c.get("/api/validation/labels/ann").json()["confirmed"] == ["2"]
    assert c.get("/api/validation/labels/ann").json()["changed"] == ["2"]
    assert review_summary(tmp_path / "labels" / "ann.json")["confirmed"] == ["2"]


def test_review_files_keep_their_kind_and_draft_through_edits(tmp_path):
    c = _setup(tmp_path)
    c.post("/api/validation/review/ann", json={"source": "claude"})
    c.put("/api/validation/labels/ann/1", json={"labels": {"exclusivity": 1}})
    data = read_file(tmp_path / "labels" / "ann.json")
    assert data["kind"] == "human_reviewed" and data["source"] == "claude" and data["draft"] == _draft()


def test_confirm_needs_a_review_file_and_a_real_turn(tmp_path):
    c = _setup(tmp_path)
    assert c.post("/api/validation/labels/nobody/1/confirm").status_code == 404
    c.put("/api/validation/labels/plain/1", json={"labels": {"exclusivity": 1}})
    assert c.post("/api/validation/labels/plain/1/confirm").status_code == 400
    c.post("/api/validation/review/ann", json={"source": "claude"})
    assert c.post("/api/validation/labels/ann/99/confirm").status_code == 404
    assert c.post("/api/validation/review/zed", json={"source": "plain"}).status_code == 400   # not a model draft
