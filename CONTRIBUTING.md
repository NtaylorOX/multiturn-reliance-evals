# Contributing

Thanks for looking. The repo is one finished study plus the tooling around it, so changes that keep it small and
legible are the most useful: a new persona, pressure condition, rubric dimension or model, a better analysis, or a fix.

## Setup

```bash
uv sync                 # Python 3.11+
uv run pre-commit install   # optional: lint and type-check on every commit
make check              # ruff, mypy, and the tests with a coverage floor
```

Tests need no model server and no network: Inspect's `mockllm` stands in for every model. A change that needs a real
model to test should add a mock-based test as well.

## Where things go

The README's "Reusing and extending" section says which single file to edit for each kind of change. Notes that apply to
all of them:

- Keep personas and pressure conditions as plain data in `redteam/`, so a new one is a dictionary entry, not new code.
- A new rubric dimension needs the same name in `evals/rubric.py` and `observability/tracing.py::DIMENSIONS`; a test
  checks they match.
- Anything that writes under `validation/labels/` must keep the audit log and the `kind` field honest (`human`, `llm`,
  `human_reviewed`), because every agreement figure in the README depends on who made the labels.

## Style

- Every public function, class and module has a docstring that says what it does and what it returns or raises.
- `ruff` and `mypy` are the style guide; CI runs both on Python 3.11, 3.12 and 3.13.
- Commit messages: a short imperative summary line, then a body only if the reason is not obvious.
- Do not commit anything under `logs/` (full conversations) or `.env`.

## Reporting results

If you run the study with other models, put per-turn scores under `results/<name>/` using
`reliance plots <log-dir> --out results/<name>` and say in the pull request which judge and rubric produced them.
