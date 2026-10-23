"""Reliance score vs turn, by model x persona, from Inspect logs. Writes PNGs and one static HTML page.

    uv run python -m observability.plots logs/full --out report
"""
from __future__ import annotations

import argparse
import base64
import io
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from evals.logs import iter_turn_records
from observability.tracing import DIMENSIONS

# Okabe-Ito: distinguishable for common colour-vision deficiencies.
PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]
PERSONA_ORDER = ["lonely", "distressed", "dependent", "control"]
PRESSURE_ORDER = ["none", "exclusivity", "availability", "discourage_support"]


def turns_from_logs(log_dir: str | Path, rescored: str | Path | None = None) -> pd.DataFrame:
    """One row per judged turn: model, persona, pressure, sample, 1-based turn, mean `score` and each dimension.

    `rescored` optionally points at an `evals.rejudge` file whose scores replace the logged ones.
    """
    rows = [{
        "model": r["model"], "persona": r["persona"], "pressure": r["pressure"], "sample": r["sample"],
        "turn": r["turn"] + 1, "score": sum(r["scores"].values()) / len(DIMENSIONS), **r["scores"],
    } for r in iter_turn_records(log_dir, rescored) if r["scores"]]
    return pd.DataFrame(rows)


def _band(df: pd.DataFrame, by: str) -> dict[Any, pd.DataFrame]:
    """Mean and a 95% normal-approximation band over conversations, per `by` group and turn."""
    out: dict[Any, pd.DataFrame] = {}
    for key, g in df.groupby(by):
        per_conv = g.groupby(["sample", "turn"]).score.mean().reset_index()
        stats = per_conv.groupby("turn").score.agg(["mean", "sem", "count"]).reset_index()
        stats["sem"] = stats["sem"].fillna(0)
        out[key] = stats
    return out


def _grid(df: pd.DataFrame, by: str, title: str, order: list[str]) -> plt.Figure:
    personas = [p for p in PERSONA_ORDER if p in set(df.persona)]
    keys = sorted(set(df[by]), key=lambda k: order.index(k) if k in order else 99)
    colour = {k: PALETTE[i % len(PALETTE)] for i, k in enumerate(keys)}
    fig, axes = plt.subplots(1, len(personas), figsize=(4.2 * len(personas), 3.8), sharey=True, squeeze=False)
    for ax, persona in zip(axes[0], personas, strict=False):
        for key, st in _band(df[df.persona == persona], by).items():
            ax.plot(st.turn, st["mean"], color=colour[key], lw=2)
            ax.fill_between(st.turn, st["mean"] - 1.96 * st["sem"], st["mean"] + 1.96 * st["sem"],
                            color=colour[key], alpha=0.15, lw=0)
        ax.set_title(persona)
        ax.set_xlabel("turn")
        ax.set_ylim(0, 1)
        ax.set_xticks(sorted(set(df.turn)))
        ax.grid(alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0][0].set_ylabel("reliance score (0 to 1)")
    n = df.groupby(by)["sample"].nunique()
    handles = [plt.Line2D([], [], color=colour[k], lw=2) for k in keys]
    labels = [f"{k} (n={n[k]})" for k in keys]
    fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 4), frameon=False)
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0.1, 1, 0.95))
    return fig


def plot_by_model(df: pd.DataFrame, pressure: str = "none") -> plt.Figure:
    """Headline: reliance vs turn, one panel per persona, one line per model, at one pressure level."""
    sub = df[df.pressure == pressure]
    return _grid(sub, "model", f"Reliance vs turn by model (pressure: {pressure})", sorted(set(sub.model)))


def plot_by_pressure(df: pd.DataFrame, model: str) -> plt.Figure:
    """Does light adversarial pressure move the curve? One panel per persona, one line per condition."""
    return _grid(df[df.model == model], "pressure", f"Effect of pressure on {model}", PRESSURE_ORDER)


def plot_headline(df: pd.DataFrame) -> plt.Figure:
    """Mean reliance per pressure condition, one bar per model, with each conversation as a dot.

    Pools the personas, so it answers "does pressure change what models do?" in one picture. The dots show
    the spread across conversations, which is the honest measure of how much one run can be trusted.
    """
    per_conv = df.groupby(["model", "pressure", "persona", "sample"]).score.mean().reset_index()
    pressures = [p for p in PRESSURE_ORDER if p in set(per_conv.pressure)]
    models = sorted(set(per_conv.model))
    width = 0.8 / len(models)
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    for i, model in enumerate(models):
        xs, means, errs = [], [], []
        for j, pressure in enumerate(pressures):
            vals = per_conv[(per_conv.model == model) & (per_conv.pressure == pressure)].score
            x = j + (i - (len(models) - 1) / 2) * width
            xs.append(x)
            means.append(vals.mean())
            errs.append(1.96 * vals.sem() if len(vals) > 1 else 0)
            ax.scatter([x] * len(vals), vals, s=14, color="black", alpha=0.35, zorder=3)
        ax.bar(xs, means, width * 0.9, yerr=errs, color=PALETTE[i % len(PALETTE)], label=model, capsize=3, zorder=2)
    ax.set_xticks(range(len(pressures)), pressures)
    ax.set_ylabel("mean reliance score (0 to 1)")
    ax.set_ylim(0, 1)
    ax.set_title("Reliance by pressure condition (dots: individual conversations; bars: mean, 95% CI)")
    ax.grid(axis="y", alpha=0.25, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """Mean reliance per model x persona x pressure, with first-half vs second-half turns."""
    half = df.turn.max() // 2
    t = df.assign(half=lambda d: d.turn.map(lambda x: "early" if x <= half else "late"))
    piv = (t.groupby(["model", "persona", "pressure", "half"]).score.mean().unstack("half").round(3))
    piv["late_minus_early"] = (piv["late"] - piv["early"]).round(3)
    piv["conversations"] = t.groupby(["model", "persona", "pressure"])["sample"].nunique()
    return piv.reset_index()


def _png_b64(fig: plt.Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def build_report(log_dir: str | Path, out_dir: str | Path, rescored: str | Path | None = None) -> Path:
    """Write every figure, `turns.csv`, `summary.csv` and a static `report.html` to `out_dir`; return the page path."""
    df = turns_from_logs(log_dir, rescored)
    if df.empty:
        raise SystemExit(f"no judged turns found under {log_dir}")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "turns.csv", index=False)
    table = summary_table(df)
    table.to_csv(out / "summary.csv", index=False)

    figs: list[tuple[str, plt.Figure]] = [("headline_by_pressure", plot_headline(df))]
    for pressure in PRESSURE_ORDER:
        if pressure in set(df.pressure):
            figs.append((f"by_model_{pressure}", plot_by_model(df, pressure)))
    if len(set(df.pressure)) > 1:
        for model in sorted(set(df.model)):
            figs.append((f"pressure_{model}", plot_by_pressure(df, model)))

    sections = []
    for name, fig in figs:
        fig.savefig(out / f"{name}.png", dpi=150)
        sections.append(f'<h2>{name}</h2><img alt="{name}" src="data:image/png;base64,{_png_b64(fig)}">')
    html = (
        "<!doctype html><meta charset=utf-8><title>Reliance vs turn</title>"
        "<style>body{font:15px/1.5 system-ui;max-width:1100px;margin:2rem auto;padding:0 1rem}"
        "img{max-width:100%}table{border-collapse:collapse;font-size:13px}"
        "td,th{border:1px solid #ccc;padding:3px 8px;text-align:right}</style>"
        "<h1>Reliance vs turn</h1>" + "".join(sections) +
        "<h2>Summary</h2>" + table.to_html(index=False, border=0)
    )
    page = out / "report.html"
    page.write_text(html, encoding="utf-8")
    return page


def main() -> None:
    """Command-line entry point."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("log_dir")
    ap.add_argument("--out", default="report")
    ap.add_argument("--rescored", default=None, help="an evals.rejudge output file to use instead of the logged scores")
    args = ap.parse_args()
    print("wrote", build_report(args.log_dir, args.out, args.rescored))


if __name__ == "__main__":
    main()
