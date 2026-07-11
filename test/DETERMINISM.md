# Jaxterity — Determinism Policy

Jaxterity's attestation chain (CONTEXT.md invariant 5) is a content hash over a
fitted `Robot`; its usefulness depends on computations being **bit-reproducible**.
This document states the reproducibility contract the test suite enforces.

## Same-device contract (enforced)

> Given the **same** Jaxterity version, the **same** Jaxonomy version, the
> **same** device and backend, the **same** inputs, and the **same** PRNG seeds,
> a Jaxterity reference computation produces **byte-for-byte identical** output
> across runs.

This is checked by `jaxterity.testing.determinism.assert_deterministic`, which
runs a computation twice and compares `.tobytes()`. It additionally rejects
*degenerate* outputs (all-NaN/Inf, or a constant array) so that a broken
computation cannot pass the bytes check vacuously — `test/determinism/` includes
that negative control.

Preconditions for the contract to hold (PATTERNS.md §5.5): every stochastic
operation takes an explicit `jax.random.PRNGKey` (no global RNG), and the
simulation hot path is pure (no Python-level nondeterminism).

## Cross-device determinism (OPEN — OQ-4, not assumed here)

Whether a computation is bit-identical, or only ULP-bounded, **across different
devices** (CPU vs GPU vs TPU, or different CPU microarchitectures) is **not**
settled. Jaxonomy documents cross-device determinism as best-effort; Jaxterity
needs a stricter, explicit contract because the attestation handle depends on
hash stability across the machines that compute it.

This is tracked as **OQ-4** in `AGENTS/DECISIONS.md` and is deliberately left
open: this policy does **not** claim any cross-device guarantee, and tests do
**not** assert cross-device byte-equality. Resolving OQ-4 (which operations are
bit-exact across devices, which are ULP-bounded, and how the attestation handle
tolerates the difference) is a prerequisite for any multi-device attestation
workflow and will be specified before that workflow is built.

## Scope today

Until `Robot.to_diagram()` exists, the "reference computations" are
representative JAX kernels exercised in `test/determinism/`. As real robot
simulations land, each becomes a determinism fixture run under this same
same-device contract.
