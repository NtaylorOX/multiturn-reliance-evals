import asyncio
import json

from evals.rubric import RUBRIC, load_rubric
from observability.tracing import DIMENSIONS
from validation.tune import parse_rubric, reflect_prompt, split_ids, summarise, tune

MARKER = "ONLY WHAT THE ASSISTANT SAYS"


def _data(n=12):
    turns = {str(i): {"user": f"u{i}", "assistant": f"a{i}", "context": ""} for i in range(1, n + 1)}
    labels = {str(i): dict.fromkeys(DIMENSIONS, 0) for i in range(1, n + 1)}
    return turns, labels


async def fake_judge(rubric, turn):
    """Over-scores exclusivity unless the rubric tells it to look only at the assistant."""
    scores = dict.fromkeys(DIMENSIONS, 0)
    scores["exclusivity"] = 0 if MARKER in rubric["exclusivity"] else 2
    return scores


def _proposal():
    return json.dumps({**RUBRIC, "exclusivity": f"Count {MARKER}, never the user's request."})


def test_split_is_disjoint_and_keeps_both_sides():
    train, held = split_ids([str(i) for i in range(1, 11)], 0.3, seed=1)
    assert set(train).isdisjoint(held) and len(train) + len(held) == 10 and held


def test_summarise_reports_mae_and_undefined_kappa():
    _, labels = _data(4)
    judged = {i: dict.fromkeys(DIMENSIONS, 0) for i in labels}
    judged["1"]["exclusivity"] = 2
    s = summarise(labels, judged, list(labels))
    assert s["per_dimension"]["exclusivity"]["mae"] == 0.5
    assert s["kappa"] is None  # constant labels make kappa undefined everywhere here
    assert s["mae"] == 0.1


def test_parse_rubric_requires_exactly_the_five_keys():
    assert parse_rubric("noise " + _proposal(), RUBRIC)["exclusivity"].startswith("Count")
    assert parse_rubric(json.dumps({"exclusivity": "x"}), RUBRIC) is None
    assert parse_rubric("no json", RUBRIC) is None


def test_reflect_prompt_shows_disagreements_and_rules():
    turns, labels = _data(3)
    judged = {i: asyncio.run(fake_judge(RUBRIC, turns[i])) for i in turns}
    p = reflect_prompt(RUBRIC, turns, labels, judged, list(turns))
    assert "exclusivity: label 0 vs judge 2" in p and "never quote" in p


def test_tuning_accepts_an_improving_rubric_and_reports_heldout(tmp_path):
    turns, labels = _data()
    calls = []

    async def optimiser(prompt):
        calls.append(prompt)
        return _proposal()

    out = tmp_path / "run.json"
    run = asyncio.run(tune(turns, labels, fake_judge, optimiser, rounds=2, out_path=out, config={"judge": "fake"}))
    assert run["status"] == "done" and len(calls) == 2
    assert [h["accepted"] for h in run["history"]] == [True, True, False]  # baseline, improved, then no further gain
    assert MARKER in run["best_rubric"]["exclusivity"]
    assert run["heldout"]["baseline"]["mae"] > run["heldout"]["best"]["mae"] == 0
    assert json.loads(out.read_text())["status"] == "done"


def test_invalid_optimiser_output_is_recorded_and_rubric_unchanged():
    turns, labels = _data()

    async def optimiser(prompt):
        return "sorry, I cannot"

    run = asyncio.run(tune(turns, labels, fake_judge, optimiser, rounds=1))
    assert run["history"][1]["accepted"] is False and run["best_rubric"] == RUBRIC


def test_load_rubric_roundtrip_and_validation(tmp_path):
    p = tmp_path / "r.json"
    p.write_text(json.dumps({"rubric": json.loads(_proposal())}))
    assert load_rubric(p)["exclusivity"].startswith("Count")
    p.write_text(json.dumps({"exclusivity": "x"}))
    try:
        load_rubric(p)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError")
    assert load_rubric(None) == RUBRIC
