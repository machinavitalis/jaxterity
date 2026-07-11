# Contributing to Jaxterity

Thanks for your interest in contributing. Jaxterity is the JAX-native robotics
layer on top of [Jaxonomy](https://github.com/machinavitalis/jaxonomy) — URDF/
MJCF import, MJX-backed dynamics, calibration/system-identification, and attested
digital twins. It is released under the [MIT License](LICENSE.md); by
contributing you agree your contributions are licensed under the same terms.

## Development setup

Jaxterity requires **Python 3.10+** and builds on Jaxonomy (pinned to `2.x`).

```bash
git clone https://github.com/machinavitalis/jaxterity.git
cd jaxterity
python -m pip install --upgrade pip
pip install -e ".[all,dev]"
```

If you're developing Jaxonomy and Jaxterity together, install Jaxonomy editable
first (`pip install -e ../jaxonomy --no-deps`) so cross-repo changes are picked
up live.

## Running the tests

```bash
pytest -m "not nightly" -q     # the default/PR suite
pytest -m nightly -q           # the heavier nightly sweeps
bash scripts/check.sh          # the full local gate (lint + tests), if present
```

## What we value in a change

- **Calibration/sysid changes** come with a recipe + identifiability check and a
  test that recovers known parameters.
- **Hardware claims stay scoped.** Results validated on one platform/unit must
  not be generalized in code, docs, or tests to embodiments that haven't run on
  hardware. Keep `CLAIMS.md` / `KNOWN_GAPS.md` honest.
- **Tests + docstrings** ship with the change; user-visible changes get a
  `CHANGELOG` entry.

Match the surrounding code's conventions. The `AGENTS/` directory holds the
architecture and decision records (`CONTEXT.md`, `PATTERNS.md`, `DECISIONS.md`,
the phase gates) — skim the relevant ones before a non-trivial change.

## Pull request process

1. Branch from `main`; keep the change focused.
2. Ensure `pytest -m "not nightly" -q` passes and CI is green.
3. Describe the change and its motivation; link any related issue.

## Reporting bugs & security issues

Open a GitHub issue for bugs. For anything with security impact, **do not open a
public issue** — follow [SECURITY.md](SECURITY.md).
