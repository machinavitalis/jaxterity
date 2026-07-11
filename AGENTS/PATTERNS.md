# Jaxterity — Coding Patterns

This file documents the coding conventions agents and humans follow when
writing Jaxterity code. It is the reference Claude Code consults on every
diff. The conventions exist because of specific bugs they prevent or
specific properties they preserve (sim-to-real parity, vmap safety,
attestation chain integrity, agent legibility).

Conventions inherited from Jaxonomy still apply. This file documents
additions and Jaxterity-specific specializations only.

If a pattern here disagrees with a pattern in the Jaxonomy PATTERNS.md
that Jaxterity inherits, this file wins for code under `jaxterity/`.

---

## 1. File and module conventions

### 1.1 SPDX header

Every source file begins with:

```python
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors
```

Test files, example scripts, and zoo robot drivers included.

### 1.2 Module docstring

Every module's first non-header line is a one-paragraph docstring stating
what the module owns. The docstring is consumed by mkdocstrings and by
LLMs reading the module fresh, so it must answer "what is this for" in
the first sentence.

```python
"""Calibrated DC motor with Coulomb and viscous friction.

This module provides ``DCMotorWithFriction``, a Jaxonomy ``LeafSystem``
implementing the WPILib SysId motor model. The actuator's parameters
(Ks, Kv, Ka, viscous friction) are identifiable from voltage-ramp
telemetry via :func:`jaxterity.sysid.actuator`.
"""
```

### 1.3 Imports

Imports are grouped in this order, each group separated by a blank line:

1. Standard library
2. Third-party non-JAX
3. JAX and JAX ecosystem (`jax`, `jaxlib`, `equinox`, `optax`, `diffrax`)
4. Jaxonomy
5. Jaxterity (relative imports preferred within the package)

JAX imports follow the canonical aliases:

```python
import jax
import jax.numpy as jnp
from jax import lax
```

Do not alias `jax.numpy` as `np`. `np` is reserved for `numpy`. Mixing
the two is the most common source of "this works under JIT but fails
under vmap" bugs.

The Jaxonomy backend-dispatch alias `numpy_api as npa` is **not** used
in Jaxterity. Jaxterity is JAX-only by policy (see DECISIONS.md ADR-007);
if a function genuinely needs to run on NumPy for debugging, it lives in
`jaxterity.testing` and is explicit about the backend choice.

### 1.4 Type hints

Type hints are mandatory on every public function, method, and module-
level variable. Internal helpers (`_lower_case_with_leading_underscore`)
are strongly encouraged to have them.

For arrays, use `jax.Array` as the runtime type and add a shape comment
or a `jaxtyping` annotation:

```python
from jaxtyping import Array, Float

def compute_inertia(
    mass: Float[Array, ""],
    com: Float[Array, "3"],
    inertia_diag: Float[Array, "3"],
) -> Float[Array, "3 3"]:
    ...
```

Use `jaxtyping` annotations on every public function whose arguments are
arrays. The runtime cost is zero with type-checking disabled, and the
documentation value is high for LLM consumption.

---

## 2. Docstring conventions

Docstrings are not optional. They are the primary surface an LLM uses to
decide whether and how to call a function. A function without a docstring
is broken from the agent's point of view.

### 2.1 Required sections

Every public function gets a docstring with at minimum:

- **Summary line** (one sentence, imperative mood, ends with a period).
- **Extended description** (one paragraph). What the function does, when
  to use it, what it does not do.
- **Args** — every parameter with type, units, frame (if applicable),
  shape (if applicable), and meaning.
- **Returns** — type, units, frame, shape, meaning.
- **Raises** — every exception type the function may raise and the
  condition.
- **Frame** — for any function operating on poses, twists, wrenches, or
  velocities, an explicit frame contract.
- **Units** — for any function with physical inputs, the unit contract.
- **Examples** — at least one runnable example. The example must be
  doctested.

### 2.2 Format

NumPy docstring format, extended with the Frame and Units sections:

```python
def transform_wrench(
    wrench: FramedQuantity[Wrench],
    target_frame: Frame,
) -> FramedQuantity[Wrench]:
    """Express a wrench in a different coordinate frame.

    Converts a wrench (force + torque pair) from its current frame to the
    target frame using the transform tree currently bound on the robot.

    Args
    ----
    wrench : FramedQuantity[Wrench]
        The wrench to transform. Carries its source frame internally.
    target_frame : Frame
        The frame to express the wrench in.

    Returns
    -------
    FramedQuantity[Wrench]
        The wrench, now expressed in ``target_frame``.

    Raises
    ------
    FrameError
        If ``wrench.frame`` and ``target_frame`` cannot be related
        through the currently bound transform tree.

    Frame
    -----
    Input: ``wrench.frame`` (any).
    Output: ``target_frame``.

    Units
    -----
    Forces in newtons, torques in newton-meters. The function is
    unit-agnostic — it preserves whatever units the input carries.

    Examples
    --------
    >>> from jaxterity.robot import Frame, FramedQuantity, Wrench
    >>> w = FramedQuantity(Wrench(f=jnp.zeros(3), tau=jnp.zeros(3)),
    ...                    frame=Frame("foot_left"))
    >>> w_world = transform_wrench(w, Frame("world"))
    >>> w_world.frame
    Frame("world")
    """
```

The Frame and Units sections are required whenever the function touches
physical quantities — a reviewed convention today; the lint that will enforce
it in CI is planned (§7.3).

### 2.3 LLM legibility test

A docstring is sufficient if an LLM, given only the docstring and the
module-level imports, can call the function correctly on the first try.
This is testable — via an agent-driven evaluation
harness — and is the bar every public function must clear.

---

## 3. Naming conventions

### 3.1 Robotics-specific names

Names match the field's standard usage where one exists, even when a
more "Pythonic" name would feel more natural:

| Concept                | Name                            |
|------------------------|---------------------------------|
| Joint angles           | `q` (positions), `qd` (velocities), `qdd` (accelerations) |
| Joint torques          | `tau`                           |
| Generalised forces     | `f` for linear, `m` for moment, `wrench` for the pair |
| Mass matrix            | `M`                             |
| Coriolis term          | `C`                             |
| Gravity term           | `g_vec` (`g` is reserved for gravity scalar) |
| Jacobian               | `J`, `J_lin`, `J_ang`           |
| Pose                   | `T` (4x4 transform), `pose`     |
| Twist (velocity)       | `v` for linear, `w` for angular, `twist` for the pair |
| Center of mass         | `com`                           |
| Inertia tensor         | `I` (deliberate — overrides Python convention of avoiding `I`) |
| Time                   | `t`                             |
| Sample rate            | `fs` (hertz, scalar)            |
| Sample step            | `dt` (seconds, scalar)          |

When `I` is used for inertia in a scope, comment at the top of the scope
to mute any linter complaints; do not rename it.

### 3.2 Frame suffixes

Pose, velocity, and force variables carry a frame suffix when the frame
matters and is not already in the type:

```python
foot_pos_world = ...      # foot position expressed in world frame
foot_pos_body = ...       # foot position expressed in body frame
T_world_body = ...        # transform world ← body
```

The transform naming convention is always `T_target_source`. `T_a_b * p_b
= p_a`.

### 3.3 No abbreviations for domain-foreign concepts

Robotics-foreign concepts spell out:

- `parameter` not `param` (except when matching an upstream API)
- `telemetry` not `tele`
- `attestation_handle` not `att_handle`

Robotics-foreign abbreviations confuse LLMs trained mostly on
non-robotics code. Spell them out.

---

## 4. Error messages

Errors are structured for agent consumption. The model: when something
fails, the next thing that reads the error message is an LLM trying to
fix it. The message must contain enough information for the LLM to
diagnose and patch without re-running the code.

### 4.1 Structure

Every error includes:

1. **What was being attempted** (subject + verb).
2. **Why it failed** (the violated invariant or expected shape/unit).
3. **The offending value(s)** verbatim.
4. **The likely fix** (concrete suggestion).

```python
raise FrameError(
    f"Cannot add wrench in frame {a.frame.name!r} to wrench in frame "
    f"{b.frame.name!r}. Wrenches must be in the same frame before "
    f"addition. Fix: call transform_wrench(b, {a.frame.name!r}) first."
)
```

### 4.2 Exception hierarchy

Jaxterity defines a small exception hierarchy rooted at `JaxterityError`:

- `JaxterityError` — base.
  - `FrameError` — frame mismatch or unknown frame.
  - `UnitError` — unit mismatch.
  - `CalibrationError` — sysid recipe failed or telemetry incompatible.
  - `AttestationError` — handle mismatch or invalidation.
  - `SchemaError` — telemetry or model schema violation.
  - `ZooError` — reference robot zoo violation.

Library code raises one of these, never a bare `ValueError`,
`RuntimeError`, or `AssertionError`. The narrow types let agents and
tests catch the specific failure mode.

### 4.3 ErrorCollector pattern

Following Jaxonomy's `ErrorCollector`, Jaxterity uses the same pattern
for telemetry validation, robot construction, and sysid recipe
preparation. Surface all errors, not the first.

```python
with ErrorCollector() as ec:
    for joint in robot.joints:
        if not joint.has_inertial():
            ec.add(SchemaError(f"Joint {joint.name!r} has no inertial."))
    if ec.has_errors():
        raise ec.aggregate()
```

---

## 5. JAX patterns

### 5.1 Pure functions on the simulation hot path

Any function that ends up inside a Jaxonomy `Diagram` simulation step
must be a pure JAX-traceable function. No `print`, no file I/O, no
mutation, no Python control flow over traced values (use `lax.cond`,
`lax.switch`, `lax.scan`).

This is inherited from Jaxonomy invariant 2. Jaxterity flags it because
the temptation in robotics code is to add ad-hoc logging or
visualization callbacks; those go through Rerun or through
`SimulationResults` post-hoc, never inline.

### 5.2 PyTree registration

Every Jaxterity stateful object is a JAX PyTree. Use `equinox.Module`
where possible — it gives PyTree registration, immutability, and
`__init__`-with-type-hints for free. The `Robot`, `Joint`, `Link`,
`Actuator`, `Sensor`, and all WBC `Task` types are `equinox.Module`s.

Parameters (the identifiable kind) are PyTree leaves; structural
metadata (names, joint types, axis directions) is static.

### 5.3 vmap safety

The vmap-safety rule from Jaxonomy carries: any function exposed to
`simulate_batch` or downstream vmapping must produce identical results
vmapped vs. looped. Jaxterity tests this with a `vmap_consistency`
fixture in `jaxterity.testing` applied to every actuator, sensor, and
sysid recipe.

### 5.4 No mutable default arguments

Standard Python rule, but particularly important under JAX where
mutable defaults can survive across traces and cause silent
correctness bugs. Always `None`-default and construct inside.

### 5.5 PRNG discipline

Every stochastic operation receives an explicit `jax.random.PRNGKey`.
No global RNG. Sensor noise, domain randomization, and sysid recipe
multi-start initialization all receive keys; the key splitting
convention follows the standard JAX pattern (`jax.random.split(key, n)`).

This is the precondition for Jaxterity invariant 7 (zoo regression
discipline) — without explicit keys, the zoo golden trajectories aren't
reproducible.

---

## 6. Test patterns

### 6.1 Test file naming and location

Tests live in `test/`, mirroring the package structure under
`jaxterity/`. Each module under `jaxterity/foo/bar.py` has a matching
`test/foo/test_bar.py`.

### 6.2 Parametrization

Tests parametrize over the dimensions that matter:

- `solver` — every supported ODE solver (`rk4`, `dopri5`, `bdf`).
- `dtype` — `float32` and `float64`.
- `robot` — every zoo robot, for tests that act on the whole pipeline.

Use `pytest.mark.parametrize` with `pytest.param(..., id=...)` so test
IDs are readable in CI logs.

### 6.3 Tolerances come from `jaxterity.testing.tolerances`

No magic numbers in tests. Every numerical comparison passes through
`jaxterity.testing.tolerances.atol(quantity, solver, dtype)` or
`rtol(...)`. The tolerance table lives in
`test/TOLERANCES.md` and is the single source of truth.

```python
from jaxterity.testing import tolerances as tol

def test_pendulum_period(solver, dtype):
    ...
    assert jnp.isclose(period, expected,
                       atol=tol.atol("joint_position", solver, dtype),
                       rtol=tol.rtol("joint_position", solver, dtype))
```

### 6.4 Property tests preferred over example tests

Where applicable, prefer Hypothesis-style property tests over
hand-crafted examples. Reverse-mode autodiff vs. finite-difference is a
property test, not an example test. Conservation laws are property
tests.

### 6.5 Zoo end-to-end tests

`test/zoo/` runs every reference robot through the full pipeline
(import → simulate → calibrate → simulate-calibrated → parity-report).
These tests are the regression suite for invariant 7. They run on every
PR. If they are slow, profile and fix; do not skip them.

### 6.6 Golden trajectory tests

Reference simulation outputs are committed under
`test/golden/<robot>/<scenario>.npz`. Tests compare bit-exact against
the golden. When a deliberate change moves the golden, the PR includes
the new golden artifact and a justification in the PR description.

Golden files are small (compressed `.npz`) and committed to git, not
LFS. The repo stays one-clone-and-go.

---

## 7. Frame and unit discipline

These two invariants (CONTEXT.md #2 and #3) are the most commonly
violated. They get their own section.

### 7.1 Every pose, velocity, force is a `FramedQuantity`

There are no bare `jnp.ndarray`s of shape `(3,)` representing positions
in a public API. Bare arrays mean "the caller forgot which frame this
is in." The type system encodes the frame:

```python
foot_pos_world: FramedQuantity[Position]  # ok
foot_pos: jnp.ndarray                     # not ok in a public API
```

Internal numerics — inside an `Actuator.dynamics()` method, for instance
— can use bare arrays, but the public surface is framed.

### 7.2 SI units, no compromises

Internal representation: radians, meters, seconds, kilograms, newtons,
pascals, kelvins, amperes. Always.

Telemetry loaders convert at the boundary. URDF importers convert at
import time (URDF degrees → radians, etc.). Sensor blocks that output
in non-SI units (e.g., an encoder reading in counts) carry their
conversion factor as a `Parameter`, so the conversion is identifiable
and tracked.

### 7.3 Unit verification in CI

**Planned (not yet implemented).** The intended CI lint pass scans docstrings
for Units sections and flags missing ones on public functions touching physical
quantities (triggering keywords: `force, torque, position, velocity,
acceleration, mass, inertia, voltage, current, power, temperature, pressure`).
Until that lint lands, the Units/Frame docstring sections (§7.1) are a reviewed
**convention**, not an enforced check; `scripts/check.sh` runs ruff + mypy + pytest only.

---

## 8. Attestation chain discipline

The attestation handle (CONTEXT.md invariant 5) is the contract between
Jaxterity and Jaxility. Patterns that protect it:

### 8.1 Handle is computed once, validated everywhere

`CalibratedRobot.attestation_handle` is computed lazily on first access
and cached. Once computed, it is immutable. Mutating any input to the
handle algorithm (URDF bytes, parameter values, sysid recipe version)
returns a new `CalibratedRobot` with a new handle.

### 8.2 Handle algorithm is versioned

The handle computation lives in `jaxterity.diagnostics.attestation`
with an explicit `ALGORITHM_VERSION` constant. Bumping the version is
a major-version release. The version number is itself an input to the
handle so that v0.1 handles do not collide with v0.2 handles.

### 8.3 Sensitivity tests are mandatory

Every PR touching `Robot`, `Joint`, `Link`, an actuator, a sensor, or
a sysid recipe runs the attestation sensitivity suite in CI.
PRs touching handle inputs without an explicit
`# attestation: invariant` or `# attestation: changes-handle` comment
in the diff fail review.

---

## 9. Documentation patterns

### 9.1 README, quickstart, API reference

- `README.md` — 90-second pitch, one screenshot, one `pip install`
  command, link to the quickstart.
- `docs/quickstart.md` — 10-minute guided tour ending with the Cartpole
  simulating against the calibrated zoo entry.
- `docs/api/` — auto-generated by mkdocstrings from the docstrings.
  Quality of this page is a function of docstring quality (Section 2).
- `docs/architecture.md` — the architectural deep-dive. Mirrors
  CONTEXT.md's structure for external readers.
- `docs/zoo/` — one page per reference robot, documenting its URDF
  source, its sysid procedure, its calibrated parameters, and its
  demo script.

### 9.2 Examples are tested

Every example script under `examples/` is exercised by CI. An example
that does not run is removed (or fixed). The bar is "fresh clone +
`pip install -e .[all]` + run the script". No "edit this file before
running" instructions.

### 9.3 Mermaid diagrams in docs

Diagrams in docs use Mermaid (rendered by MkDocs Material). No
externally-rendered images for architecture diagrams. The .md file is
the source of truth and stays diffable.

---

## 10. Commit, PR, and changelog patterns

### 10.1 Commit messages

Every commit on a feature branch has a concise, scoped subject line:

```
Robot.to_diagram() handles fixed joints

Closes #42.
```

Commits not tied to a feature get an inline justification:

```
chore: bump ruff to 0.6.4 (dependency hygiene)
```

### 10.2 PR titles

PR titles are a short description of the change. The PR body
includes:

- Summary of what changed.
- "Attestation impact" line: either "no change" or "handle changes;
  major version bump needed".
- Test plan: what was added, what passes.

### 10.3 Changelog discipline

Every user-visible change adds a line under `[Unreleased]` in
`CHANGELOG.md`, following Keep a Changelog format. Pure internal
refactors stay in commits, not the changelog. This matches Jaxonomy's
discipline.

---

## 11. Anti-patterns

A short list of things to refuse, with the reason:

- **Re-implementing Featherstone, FK, or contact in Jaxterity.** Use
  MJX (FK + dynamics + contact + collision). If MJX is insufficient,
  upstream the fix.
- **Frame-less APIs on public surfaces.** A pose, twist, or wrench
  without a `Frame` is a bug.
- **Bare floats from sysid.** Returns are `SysidResult`, never tuples.
- **Silent contact smoothing.** Any smoothing approximation is named
  and parametrized (invariant 6 from CONTEXT.md).
- **Two sources of truth for a robot.** One `CalibratedRobot`. Period.
- **`print` for debugging in library code.** Use the logging
  framework. `print` calls in library code fail review.
- **TODOs without a tracked owner.** Either the work is tracked, or it
  does not exist.
- **Tests that skip on the default CI matrix.** Skips are red flags;
  document and fix.
