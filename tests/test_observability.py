import json

import pandas as pd

from observability.export import JsonlSpanExporter, load_spans
from observability.plots import plot_by_model, plot_by_pressure, plot_headline, summary_table
from observability.tracing import DIMENSIONS, Telemetry


def _df():
    rows = []
    for model in ("a", "b"):
        for pressure in ("none", "availability"):
            for s in range(3):
                for turn in range(1, 5):
                    rows.append({"model": model, "persona": "lonely", "pressure": pressure,
                                 "sample": f"{model}-{pressure}-{s}", "turn": turn,
                                 "score": 0.1 * turn + (0.2 if pressure != "none" else 0), **{d: 0.0 for d in DIMENSIONS}})
    return pd.DataFrame(rows)


def test_jsonl_exporter_round_trips_turn_spans(tmp_path):
    path = tmp_path / "t" / "spans.jsonl"
    tel = Telemetry(exporter=JsonlSpanExporter(path))
    with tel.conversation("c1", model="m", persona="lonely", pressure="none") as obs:
        obs.turn(0, user_text="hi", assistant_text="hello", scores={d: 0.5 for d in DIMENSIONS})
    spans = load_spans(path)
    assert [s["name"] for s in spans] == ["turn 0", "conversation m lonely"]
    turn, conv = spans
    assert turn["parent_id"] == conv["span_id"] and turn["trace_id"] == conv["trace_id"]
    assert turn["attributes"]["reliance.score"] == 0.5
    assert "hello" not in json.dumps(spans)


def test_plots_build_and_summary_reports_drift():
    df = _df()
    assert plot_by_model(df, "none").axes
    assert plot_by_pressure(df, "a").axes
    assert plot_headline(df).axes
    t = summary_table(df)
    assert (t.late_minus_early > 0).all()
    assert set(t.conversations) == {3}
