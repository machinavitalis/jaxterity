# AGENTS.md — Jaxterity

Agent bootstrap for any coding agent (Claude Code, Codex, Gemini, Cursor, …)
and human contributors. This is the **canonical, tool-neutral entry file**;
`CLAUDE.md`, `GEMINI.md`, `.github/copilot-instructions.md`, and
`CONVENTIONS.md` are symlinks to it, and `.cursor/rules/` points here (see
"Entry points" at the end).

Jaxterity is the robotics layer of a JAX-native sim-to-silicon stack: URDF/MJCF
import, calibrated actuators/sensors, system identification, whole-body
control. For the full architecture and invariants, read `AGENTS/CONTEXT.md`.

Two doors, depending on what you're here to do:

- **Modifying / adding code in Jaxterity** → start at `AGENTS/README.md`, which
  indexes the `AGENTS/` docs and gives the session protocol and read order.
- **Using Jaxterity's public API** in your own code (importing a robot,
  calibrating, writing a controller, deploying) → read `SKILL.md`, the consumer
  operating manual, instead.

## Read first (for code changes)

1. `AGENTS/README.md` — navigation + session protocol.
2. `AGENTS/CONTEXT.md` — what Jaxterity is, architecture, invariants.
3. `AGENTS/PATTERNS.md` — coding conventions; consult before any non-trivial code.
4. `AGENTS/DECISIONS.md` — ADRs; check before re-litigating a design choice.
5. `AGENTS/RULES.md` — operating principles, shippable-surface/claims discipline,
   the one-task-one-branch-one-PR workflow, and the test gate.

Two reminders that bite most often: changes to a shippable surface (`README.md`,
`docs/**`, `examples/**`, `CLAIMS.md`, `KNOWN_GAPS.md`) must be real and
evidence-backed — removed beats fake; and confirm the repo's test gate in
`AGENTS/RULES.md` before proposing a change is done.

## Entry points (why there are several files)

The same content is reachable under every AI tool's expected filename, with no
duplication:

- **`AGENTS.md`** (this file) — the one real, canonical bootstrap. Read
  natively by Codex/ChatGPT and by humans.
- **`CLAUDE.md`, `GEMINI.md`, `.github/copilot-instructions.md`,
  `CONVENTIONS.md`** — symlinks to this file (Claude Code, Gemini CLI, GitHub
  Copilot, Aider). Edit `AGENTS.md`; the rest follow automatically.
- **`.cursor/rules/`** — a one-line rule pointing Cursor here.
- **`SKILL.md`** — the *consumer* manual (using the API), a separate document;
  it is also surfaced as a Claude skill at `.claude/skills/jaxterity/SKILL.md`,
  with root `SKILL.md` a symlink into it.
