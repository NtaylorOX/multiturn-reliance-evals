# Changelog

All notable changes. Versions follow [Semantic Versioning](https://semver.org/).

## 0.1.0

First release.

- Inspect AI task running 10-turn conversations between a target model and an LLM-simulated user, with four personas and
  four pressure conditions.
- Per-turn LLM judge on five rubric dimensions; a local Qwen3 judge and simulator by default.
- OpenTelemetry per-turn traces (message text never exported) and a local live event stream.
- Monitor: FastAPI server and one React page with Run, Label and Tune tabs; markdown rendering of transcripts, sanitised.
- Validation: blind label sampling, human, model and human-reviewed label kinds with an audit log, Cohen's kappa.
- Judge tuning by reflective rubric optimisation, and re-scoring of existing logs without rerunning conversations.
- `reliance` command, ruff and mypy configuration, offline end-to-end tests, CI on Python 3.11 to 3.13.
- Pilot results: 48 conversations, one per model x persona x pressure cell, under the original and the tuned rubric.
