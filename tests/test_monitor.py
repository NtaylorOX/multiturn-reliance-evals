import json

from fastapi.testclient import TestClient

from monitor.server import create_app, read_new_lines
from observability import live


def test_read_new_lines_waits_for_complete_lines(tmp_path):
    p = tmp_path / "live.jsonl"
    p.write_bytes(b'{"a":1}\n{"b":2')
    lines, off = read_new_lines(p, 0)
    assert lines == ['{"a":1}']
    p.write_bytes(b'{"a":1}\n{"b":2}\n')
    lines, off = read_new_lines(p, off)
    assert lines == ['{"b":2}'] and off == p.stat().st_size
    assert read_new_lines(p, off) == ([], off)
    assert read_new_lines(tmp_path / "missing", 0) == ([], 0)


def test_emit_writes_jsonl_and_never_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("RELIANCE_LIVE_FILE", str(tmp_path / "x" / "live.jsonl"))
    live.emit("message", conv="m|1", turn=0, role="user", text="hi")
    ev = json.loads((tmp_path / "x" / "live.jsonl").read_text())
    assert ev["kind"] == "message" and ev["text"] == "hi" and "ts" in ev
    monkeypatch.setenv("RELIANCE_LIVE_FILE", str(tmp_path / "x" / "live.jsonl" / "nope" / "f"))  # parent is a file
    live.emit("phase")


def test_progress_and_page_are_served(tmp_path):
    client = TestClient(create_app(tmp_path, cells_per_model=16))
    prog = client.get("/api/progress").json()
    assert set(prog["models"]) == {"nemotron", "llama31-8b", "deepseek-r1"}
    assert all(v == {"done": 0, "total": 16} for v in prog["models"].values())
    page = client.get("/")
    assert page.status_code == 200 and "Run monitor" in page.text
