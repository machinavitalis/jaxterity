# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Run the LLM-vs-baseline eval with the live Anthropic backend.

Shows a Claude model only the public S1–S4 docstrings, has it write a policy per
canned task, scores them through the (non-vacuous) eval harness, and prints the
headline LLM-vs-baseline ratio — the number for the README.

Needs ``pip install anthropic`` and ``ANTHROPIC_API_KEY``::

    python examples/llm_eval_run.py --model claude-sonnet-5

Everything except this credentialed run is covered offline by
``test/policies/test_llm_eval.py`` (stub backend).
"""

from __future__ import annotations

import argparse

from jaxterity.policies.llm_eval import (
    DEFAULT_MODEL,
    anthropic_complete,
    score_llm_vs_baseline,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Claude model id")
    parser.add_argument("--max-tokens", type=int, default=1024)
    args = parser.parse_args()

    complete = anthropic_complete(model=args.model, max_tokens=args.max_tokens)
    report = score_llm_vs_baseline(complete)
    print(report.summary())
    print(f"\nLLM-vs-baseline ratio: {report.ratio:.2f}  (model: {args.model})")


if __name__ == "__main__":
    main()
