# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""LLM-vs-baseline eval runner.

Gives an LLM agent **only** the public S1–S4 docstrings — the same surface a
human agent would read — asks it to write a ``policy(s4, s3)`` for each canned
task, runs the generated policies through :func:`eval_harness.run_eval`, and
reports the LLM success rate against the deterministic reference baselines
(:func:`eval_harness.reference_policy_for`). The headline number is the *ratio*
``llm_rate / baseline_rate``.

The LLM is injected as a ``complete(prompt) -> str`` callable, so the runner is
**fully testable offline** with a stub and pluggable with a real backend.
:func:`anthropic_complete` is the live backend — it needs the ``anthropic`` SDK
and ``ANTHROPIC_API_KEY``; without them it raises a structured error rather than
guessing.

This runner is meaningful only because the harness is **non-vacuous**: a
do-nothing policy scores 0, so a real ratio measures real skill.

.. warning::
   **Scoring an LLM means executing model-written code.** ``run_eval`` already
   runs arbitrary policy callables — it is a benchmark harness — and
   :func:`extract_policy` ``exec()``s the generated function in a minimal
   namespace. Run this only with a ``complete`` you trust to return benign code
   (the same trust model as any code-writing benchmark). It is intentionally not
   imported by ``jaxterity.policies.__init__`` so merely importing the package
   never pulls in the ``exec`` path.
"""

from __future__ import annotations

import dataclasses
import re
import textwrap
from typing import Callable

import jax.numpy as jnp
import numpy as np

from .eval_harness import CANNED_TASKS, EvalTask, reference_policy_for, run_eval
from .s2 import move_joint, track_trajectory
from .s3 import S3
from .s4 import S4

#: Default Claude model for :func:`anthropic_complete`. Override per your access.
DEFAULT_MODEL = "claude-sonnet-5"


def gather_api_docs() -> str:
    """The public S1–S4 surface shown to the agent (the actionable docstrings)."""
    surfaces = [
        ("S4 (joint-space hardware facade)", S4),
        ("S3 (whole-body control layer)", S3),
        ("S2.move_joint", move_joint),
        ("S2.track_trajectory", track_trajectory),
    ]
    parts = []
    for label, obj in surfaces:
        doc = textwrap.dedent(obj.__doc__ or "(no docstring)").strip()
        parts.append(f"### {label}\n{doc}")
    return "\n\n".join(parts)


_PROMPT_TEMPLATE = """You control a robot through a layered Python API. You may use
ONLY what the docstrings below describe — no other attributes or imports.

{api_docs}

## Task: {task_name}
Robot: {robot}

{prompt}

Write a single Python function ``policy(s4, s3)`` that accomplishes the task.
In scope: ``move_joint(s3, joint, q_des, ...)`` and ``track_trajectory(...)``
(already imported), plus ``jnp`` and ``np``. The function returns nothing.
Respond with ONLY the function inside one ```python code block."""


def build_prompt(task: EvalTask, api_docs: str) -> str:
    """The single-task prompt: the API surface + the task brief."""
    return _PROMPT_TEMPLATE.format(
        api_docs=api_docs,
        task_name=task.name,
        robot=task.robot,
        prompt=task.description,
    )


_CODE_BLOCK = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)


def extract_policy(response: str) -> Callable:
    """Parse an LLM response into a ``policy(s4, s3)`` callable.

    Pulls the first fenced ```python code block and ``exec``s it in a minimal
    namespace exposing the in-scope helpers. Raises ``ValueError`` if there is no
    code block or it defines no callable ``policy``. See the module warning about
    executing model-written code.
    """
    match = _CODE_BLOCK.search(response)
    if match is None:
        raise ValueError("LLM response contained no ```python code block")
    code = match.group(1)
    namespace: dict = {
        "move_joint": move_joint,
        "track_trajectory": track_trajectory,
        "jnp": jnp,
        "np": np,
    }
    exec(code, namespace)  # noqa: S102 - benchmark: runs model-written policy (see module warning)
    policy = namespace.get("policy")
    if not callable(policy):
        raise ValueError("LLM response did not define a callable `policy(s4, s3)`")
    return policy


@dataclasses.dataclass(frozen=True)
class LLMEvalReport:
    """The outcome of one LLM-vs-baseline run."""

    llm_rate: float
    baseline_rate: float
    #: ``(task_name, llm_success, baseline_success, llm_error)`` per task.
    per_task: tuple[tuple[str, bool, bool, str | None], ...]

    @property
    def ratio(self) -> float:
        """``llm_rate / baseline_rate`` — the headline LLM-vs-baseline number."""
        if self.baseline_rate == 0.0:
            return float("nan")
        return self.llm_rate / self.baseline_rate

    def summary(self) -> str:
        lines = [
            f"LLM {self.llm_rate:.0%} vs baseline {self.baseline_rate:.0%} "
            f"(ratio {self.ratio:.2f})"
        ]
        for name, llm_ok, base_ok, err in self.per_task:
            tag = "ok " if llm_ok else "FAIL"
            extra = f"  [{err}]" if err else ""
            lines.append(
                f"  {tag} {name}  (baseline {'ok' if base_ok else 'FAIL'}){extra}"
            )
        return "\n".join(lines)


def score_llm_vs_baseline(
    complete: Callable[[str], str],
    *,
    tasks: tuple[EvalTask, ...] = CANNED_TASKS,
) -> LLMEvalReport:
    """Score LLM-written policies against the reference baselines.

    For each task: prompt ``complete`` with the API docs + brief, extract the
    policy, and run both the LLM policies and the baselines through
    :func:`run_eval`. A failed extraction (no code / not callable) leaves the
    task without a policy, which ``run_eval`` counts as a failure.
    """
    api_docs = gather_api_docs()
    llm_policies: dict[str, Callable] = {}
    llm_errors: dict[str, str] = {}
    baseline_policies: dict[str, Callable] = {}
    for task in tasks:
        baseline_policies[task.name] = reference_policy_for(task)
        try:
            llm_policies[task.name] = extract_policy(
                complete(build_prompt(task, api_docs))
            )
        except Exception as exc:  # generation/parse failure → no policy → failure
            llm_errors[task.name] = repr(exc)

    llm_results, llm_rate = run_eval(llm_policies, tasks=tasks)
    base_results, base_rate = run_eval(baseline_policies, tasks=tasks)
    per_task = tuple(
        (
            t.name,
            lr.success,
            br.success,
            lr.error or llm_errors.get(t.name),
        )
        for t, lr, br in zip(tasks, llm_results, base_results)
    )
    return LLMEvalReport(llm_rate=llm_rate, baseline_rate=base_rate, per_task=per_task)


def anthropic_complete(
    *, model: str = DEFAULT_MODEL, max_tokens: int = 1024
) -> Callable[[str], str]:
    """A live ``complete(prompt) -> str`` backed by the Anthropic API.

    Needs the ``anthropic`` SDK and ``ANTHROPIC_API_KEY`` in the environment;
    raises ``RuntimeError`` with install/setup instructions if either is missing
    (invariant: loud failure, never a silent guess).
    """
    try:
        import anthropic
    except ModuleNotFoundError as exc:  # pragma: no cover - optional live backend
        raise RuntimeError(
            "the `anthropic` SDK is required for the live LLM runner; "
            "`pip install anthropic` and set ANTHROPIC_API_KEY"
        ) from exc

    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY

    def complete(prompt: str) -> str:  # pragma: no cover - needs live credentials
        message = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(block.text for block in message.content if block.type == "text")

    return complete


__all__ = [
    "gather_api_docs",
    "build_prompt",
    "extract_policy",
    "LLMEvalReport",
    "score_llm_vs_baseline",
    "anthropic_complete",
    "DEFAULT_MODEL",
]
