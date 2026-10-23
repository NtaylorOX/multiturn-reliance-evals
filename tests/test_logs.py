import json

from evals.logs import load_rescored


def test_load_rescored_keys_by_log_sample_and_turn(tmp_path):
    p = tmp_path / "rescored.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in [
        {"log": "a.eval", "sample": "lonely-none-0", "turn": 3, "scores": {"exclusivity": 0.5}},
        {"log": "a.eval", "sample": "lonely-none-0", "turn": 4, "scores": None},
    ]), encoding="utf-8")
    got = load_rescored(p)
    assert got[("a.eval", "lonely-none-0", 3)] == {"exclusivity": 0.5}
    assert got[("a.eval", "lonely-none-0", 4)] is None
