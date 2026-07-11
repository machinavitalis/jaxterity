# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Tests for the LLM-vs-baseline runner.

Exercises the whole pipeline (prompt → extract → exec → run_eval → score) offline
by injecting a stub ``complete`` — no API needed. The live Anthropic backend is
covered only at the "raises without credentials" boundary.
"""

from __future__ import annotations

import pytest

from jaxterity.policies.eval_harness import CANNED_TASKS
from jaxterity.policies.llm_eval import (
    LLMEvalReport,
    build_prompt,
    extract_policy,
    gather_api_docs,
    score_llm_vs_baseline,
)

# A "competent" agent: emits the right S2 skill for each task, keyed off the
# task name carried in the prompt (mirrors what reference_policy_for does).
_COMPETENT = {
    "cartpole_slide_to_0.05": (
        "def policy(s4, s3):\n"
        "    move_joint(s3, 'slider_to_cart', 0.05, max_steps=300, dt=0.01)\n"
    ),
    "cartpole_pole_upright": (
        "def policy(s4, s3):\n"
        "    move_joint(\n"
        "        s3, 'cart_to_pole', 0.0, max_steps=400, dt=0.01, tolerance=0.02\n"
        "    )\n"
    ),
    "so100_shoulder_to_0.3": (
        "def policy(s4, s3):\n"
        "    move_joint(\n"
        "        s3, 'shoulder_pan', 0.3, max_steps=400, dt=0.01, tolerance=0.02\n"
        "    )\n"
    ),
}


def _competent_complete(prompt: str) -> str:
    for name, code in _COMPETENT.items():
        if name in prompt:
            return f"```python\n{code}```"
    return "I don't know."  # pragma: no cover - all tasks are covered


def _useless_complete(prompt: str) -> str:
    return "Sorry, I can't help with that."  # no code block → no policy


def test_gather_api_docs_surfaces_the_layers() -> None:
    docs = gather_api_docs()
    assert "move_joint" in docs
    assert "S4" in docs and "S3" in docs
    assert len(docs) > 200  # real docstrings, not stubs


def test_build_prompt_includes_task_and_api() -> None:
    prompt = build_prompt(CANNED_TASKS[0], gather_api_docs())
    assert CANNED_TASKS[0].name in prompt
    assert "policy(s4, s3)" in prompt


def test_extract_policy_parses_fenced_code() -> None:
    fn = extract_policy("here:\n```python\ndef policy(s4, s3):\n    pass\n```\ndone")
    assert callable(fn)


def test_extract_policy_rejects_response_without_a_policy() -> None:
    with pytest.raises(ValueError, match="code block"):  # prose, no fenced code
        extract_policy("no code here")
    with pytest.raises(ValueError, match="policy"):  # code block, but no policy fn
        extract_policy("```python\nx = 1\n```")


@pytest.mark.slow
def test_competent_llm_matches_the_baseline() -> None:
    """An agent that emits the right skills scores the baseline rate (ratio ≈ 1)."""
    report = score_llm_vs_baseline(_competent_complete)
    assert isinstance(report, LLMEvalReport)
    assert report.baseline_rate == 1.0
    assert report.llm_rate == 1.0
    assert report.ratio == pytest.approx(1.0)


@pytest.mark.slow
def test_useless_llm_scores_zero_against_a_nonvacuous_harness() -> None:
    """A non-answer scores 0 — the harness is non-vacuous, so the ratio
    actually measures skill."""
    report = score_llm_vs_baseline(_useless_complete)
    assert report.baseline_rate == 1.0
    assert report.llm_rate == 0.0
    assert report.ratio == 0.0
    # every task should record why the LLM failed (no policy supplied / no code)
    assert all(err is not None for _, _, _, err in report.per_task)
