"""Calling the judge model on one assistant reply. Shared by the eval, re-scoring and judge tuning."""
from __future__ import annotations

from typing import Any

from inspect_ai.model import GenerateConfig, Model, get_model

from evals.models import JUDGE
from evals.rubric import RUBRIC, judge_prompt, parse_scores

# Qwen3 thinks by default, and the thinking eats max_tokens and leaves an empty reply. Through Ollama's
# OpenAI-compatible API only reasoning_effort="none" in the raw request body switches it off: the "/no_think"
# prompt switch is ignored, and Inspect's own reasoning_effort setting is not forwarded to Ollama.
NO_THINK = {"reasoning_effort": "none"}
# The simulator and judge run with reasoning off, so short limits are enough. A judge that thinks would need more.
JUDGE_CONFIG: dict[str, Any] = {"temperature": 0.0, "max_tokens": 200, "extra_body": NO_THINK}


async def judge_reply(model: Model | str, transcript: str, user_text: str, assistant_text: str,
                      rubric: dict[str, str] | None = None, attempts: int = 2) -> dict[str, float] | None:
    """Score one assistant reply. Returns dimension -> 0..1, or None if no attempt gave a parsable answer."""
    jdg = get_model(model) if isinstance(model, str) else model
    prompt = judge_prompt(transcript, user_text, assistant_text, rubric or RUBRIC)
    for _ in range(attempts):
        out = await jdg.generate(prompt, config=GenerateConfig(**JUDGE_CONFIG))
        scores = parse_scores(out.completion)
        if scores:
            return scores
    return None


def default_judge() -> str:
    """The Inspect model string of the default judge."""
    return JUDGE
