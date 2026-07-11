# AGENTS/

Orientation for AI coding agents and human contributors working on jaxterity.
Fresh session: read this file, then CONTEXT.md, then start.

## Files here

- **CONTEXT.md** — what jaxterity is, architecture, invariants. Read before writing code.
- **PATTERNS.md** — coding conventions. Consult before any non-trivial code.
- **DECISIONS.md** — ADRs (why it's built this way). Check before re-litigating a design choice.
- **RULES.md** — operating principles + shippable-surface/claims discipline + workflow. Read once.
- **LIBRARIES.md** — third-party dependency choices and licensing.
- **LESSONS.md** — AI-to-AI lessons from using jaxterity. Skim if you'll consume the library.

## Session protocol

1. `git status` / `git log --oneline -20` — what the last session left behind.
2. CONTEXT.md + PATTERNS.md before writing code; DECISIONS.md before design choices.
3. End green: tests pass, work committed. Anything needing human eyes goes in a commit message or PR description — not a local scratchpad file.

See RULES.md for operating principles and escalation.
