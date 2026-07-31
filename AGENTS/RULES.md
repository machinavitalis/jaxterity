# RULES.md

Operating principles and discipline for jaxterity. Read once; it governs every session.

## Operating principles

1. **Think before coding.** Three or more steps → write a plan first. Surface assumptions; ask if two interpretations are possible.
2. **Simplest implementation that fits.** If 200 lines could be 50, write 50. No speculative flexibility.
3. **Surgical changes only.** Touch only what the task requires; match existing style. Don't reformat or refactor untouched code.
4. **Define success, then loop.** Write the verification first (test, benchmark, demo). No task is done without proof.

## Shippable surfaces

Public-facing files — `README.md`, `CHANGELOG.md`, `docs/**`, `examples/**`, `CLAIMS.md`, `KNOWN_GAPS.md` — must be real and accurate. In them, never: fabricate numbers or success rates, leave placeholder stubs (`TODO`/`FIXME`/`NotImplementedError`), write examples for functions that don't behave as written, or invent names/URLs/testimonials. If something can't be substantiated, remove the surface that depends on it until it can — removed beats fake. Run a skeptical adversarial read before shipping a shippable-surface change.

## Claims & gaps

New or changed claims in a shippable surface get an evidence row in `CLAIMS.md` pointing at a test/benchmark that passes; a claim with no evidence comes down. `KNOWN_GAPS.md` is the public inverse — what we don't yet do or only partially do. Keep the two symmetric.

## Workflow

One task = one branch (`task/short-title`). Commit at each acceptance criterion, not just at the end. Run the test suite before merging; don't merge red. Merge to `main` autonomously when acceptance passes and nothing below applies.

Escalate — via the commit/PR surface, not a local file — only for: intent or scope ambiguity you can't resolve from context; ADR-worthy design choices (new dependency, public-API change, module restructure); backward-incompatible public API changes; changes to correctness invariants; test failures you can't resolve in-session. For everything else, proceed.

Unrelated bugs found mid-task: don't fix them inline.

## Self-improvement

After a real correction or mistake, propose a tightened rule here and surface it before editing. Code-path-specific dev gotchas go in the affected code's docstring (tagged with the followup task ID), not here. Cross-repo usage lessons go in `LESSONS.md`.
