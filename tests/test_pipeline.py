"""End-to-end tests that need no model server and no network: Inspect's `mockllm` stands in for every model."""
import asyncio
import itertools
import json

import pytest
from fastapi.testclient import TestClient
from inspect_ai import eval as inspect_eval
from inspect_ai.model import ModelOutput, get_model

import observability.export as export
from evals.conversation import reliance_eval
from evals.judge import judge_reply
from evals.logs import conversations_from_log, iter_turn_records
from evals.progress import snapshot
from evals.rejudge import rejudge
from monitor.server import create_app
from observability.export import load_spans

SCORES = {"discourages_outside_support": 0, "exclusivity": 2, "sycophancy": 0, "false_intimacy": 1, "failure_to_redirect": 0}


def _mock(content: str):
    outputs = itertools.cycle([ModelOutput.from_content(model="mockllm/model", content=content)])
    return get_model("mockllm/model", custom_outputs=outputs)


@pytest.fixture
def run(tmp_path, monkeypatch):
    """One 2-turn conversation run offline. Returns (folder, log) with live events and spans written under the folder."""
    monkeypatch.setenv("RELIANCE_LIVE_FILE", str(tmp_path / "live.jsonl"))
    monkeypatch.setenv("RELIANCE_TRACE_FILE", str(tmp_path / "spans.jsonl"))
    monkeypatch.setattr(export, "_telemetry", None)   # the process-wide Telemetry reads the env when first created
    task = reliance_eval(persona="lonely", pressure="availability", n=1, turns=2,
                         simulator="mockllm/model", judge="mockllm/model")
    log = inspect_eval(task, model="mockllm/model", log_dir=str(tmp_path / "logs" / "llama31-8b"), display="none")[0]
    return tmp_path, log


def test_judge_reply_parses_the_models_json_and_gives_up_on_garbage():
    scores = asyncio.run(judge_reply(_mock(json.dumps(SCORES)), "", "hi", "hello"))
    assert scores == {k: v / 2 for k, v in SCORES.items()}
    assert asyncio.run(judge_reply(_mock("not json"), "", "hi", "hello", attempts=2)) is None


def test_task_runs_offline_and_records_turns(run):
    _, log = run
    assert log.status == "success" and log.samples
    turns = log.samples[0].metadata["turns"]
    assert [t["turn"] for t in turns] == [0, 1]
    assert all(t["user"] and t["assistant"] for t in turns)
    assert all(t["scores"] is None for t in turns)           # the default mock reply is not judge JSON
    score = log.samples[0].scores["reliance"]
    assert score.metadata["unjudged_turns"] == 2 and score.value == 0.0


def test_live_events_and_spans_are_written(run):
    folder, _ = run
    events = [json.loads(line) for line in (folder / "live.jsonl").read_text().splitlines()]
    kinds = [e["kind"] for e in events]
    assert kinds[0] == "conversation_start" and kinds[-1] == "conversation_end"
    assert kinds.count("message") == 4 and len({e["conv"] for e in events}) == 1
    assert all(s["attributes"] for s in load_spans(folder / "spans.jsonl"))


def test_log_readers_see_the_run(run):
    folder, _ = run
    records = list(iter_turn_records(folder / "logs"))
    assert [r["turn"] for r in records] == [0, 1] and {r["persona"] for r in records} == {"lonely"}
    assert {r["model"] for r in records} == {"mockllm/model"}
    path = next((folder / "logs" / "llama31-8b").glob("*.eval"))
    summary = conversations_from_log(path)
    assert len(summary) == 1 and summary[0]["turns"] == 2 and summary[0]["mean"] is None and summary[0]["unjudged"] == 2


def test_rejudge_scores_every_turn_once_and_resumes(run):
    folder, _ = run
    out = folder / "rescored.jsonl"
    assert asyncio.run(rejudge(folder / "logs", out, judge="mockllm/model")) == 2
    assert asyncio.run(rejudge(folder / "logs", out, judge="mockllm/model")) == 0     # already done: skipped
    assert len(out.read_text().splitlines()) == 2


def test_progress_view_counts_finished_cells(run):
    folder, _ = run
    text = snapshot(folder / "logs", cells_per_model=16)
    assert "llama31-8b" in text and "1/16" in text


def test_monitor_api_serves_the_finished_conversation(run):
    folder, _ = run
    client = TestClient(create_app(folder / "logs", validation_dir=folder / "validation"))
    assert client.get("/api/progress").json()["models"]["llama31-8b"] == {"done": 1, "total": 16}
    finished = client.get("/api/finished").json()
    assert len(finished) == 1 and finished[0]["persona"] == "lonely" and len(finished[0]["msgs"]) == 4
