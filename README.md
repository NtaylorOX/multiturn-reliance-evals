# multiturn-reliance-evals

Do frontier models reinforce unhealthy emotional reliance over a multi-turn conversation with a
vulnerable simulated user, and does it get worse with more turns or under light adversarial pressure?

> Work in progress. Method, results and limitations are written up here once the study is run.

## Layout

| Directory | Purpose |
|---|---|
| `redteam/` | Simulated-user personas and light adversarial pressure conditions |
| `evals/` | Inspect AI task: 10-turn conversations and the per-turn judge rubric |
| `observability/` | Per-turn tracing and the reliance-vs-turn plots |
| `validation/` | Hand labels and judge agreement (Cohen's kappa) |

## Setup

```bash
uv sync
cp .env.example .env   # add provider keys
uv run pytest
```
