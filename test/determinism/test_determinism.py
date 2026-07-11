# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Tolerance-table and bit-exact determinism tests.

Covers three things:
- the programmatic tolerance table (`jaxterity.testing.tolerances`) is complete,
  positive, and ordered sensibly (float32 looser than float64);
- the table documented in `test/TOLERANCES.md` matches the code (no drift);
- a reference computation is bit-reproducible across two runs, and the
  determinism guard rejects degenerate (non-finite / constant) outputs — the
  negative control that keeps the bit-exact check from passing vacuously.
"""

from __future__ import annotations

import pathlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxterity.errors import JaxterityError
from jaxterity.testing import tolerances as tol
from jaxterity.testing.determinism import assert_deterministic

_TOLERANCES_MD = pathlib.Path(__file__).resolve().parent.parent / "TOLERANCES.md"


# --- tolerance table -------------------------------------------------------


def test_table_is_complete():
    """Every quantity × solver × dtype combination has an entry."""
    t = tol.table()
    assert len(t) == len(tol.QUANTITIES) * len(tol.SOLVERS) * len(tol.DTYPES)
    for q in tol.QUANTITIES:
        for s in tol.SOLVERS:
            for d in tol.DTYPES:
                assert (q, s, d) in t


def test_tolerances_are_positive():
    """All atol/rtol values are finite and strictly positive."""
    for a, r in tol.table().values():
        assert a > 0.0 and np.isfinite(a)
        assert r > 0.0 and np.isfinite(r)


def test_float32_looser_than_float64():
    """float32 tolerances are never tighter than float64 (lower precision)."""
    for q in tol.QUANTITIES:
        for s in tol.SOLVERS:
            assert tol.atol(q, s, "float32") >= tol.atol(q, s, "float64")
            assert tol.rtol(q, s, "float32") >= tol.rtol(q, s, "float64")


def test_unknown_keys_raise_jaxterity_error():
    """Unregistered quantity/solver/dtype raise a typed JaxterityError."""
    with pytest.raises(JaxterityError, match="quantity"):
        tol.atol("nope", "rk4", "float64")
    with pytest.raises(JaxterityError, match="solver"):
        tol.atol("joint_position", "nope", "float64")
    with pytest.raises(JaxterityError, match="dtype"):
        tol.rtol("joint_position", "rk4", "f64")


def test_doc_table_matches_code():
    """The table in test/TOLERANCES.md matches tolerances.py exactly.

    Enforces PATTERNS.md §6.3 ("single source of truth") by parsing the doc's
    pipe table and comparing every cell to the code, formatted identically.
    """
    doc = {}
    for line in _TOLERANCES_MD.read_text().splitlines():
        if not line.startswith("| "):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) != 5 or cells[0] not in tol.QUANTITIES:
            continue  # header, separator, or prose
        q, s, d, a, r = cells
        doc[(q, s, d)] = (a, r)

    code = {k: (f"{a:.0e}", f"{r:.0e}") for k, (a, r) in tol.table().items()}
    assert doc == code


# --- determinism -----------------------------------------------------------


def test_reference_computation_is_bit_reproducible():
    """A representative jitted kernel is byte-identical across two runs."""

    @jax.jit
    def ref(x):
        return jnp.tanh(x) @ jnp.sin(x).T

    out = assert_deterministic(
        lambda: ref(jnp.arange(12.0).reshape(3, 4)), label="ref-kernel"
    )
    assert out.shape == (3, 3)


def test_negative_control_nonfinite_is_rejected():
    """A non-finite output is rejected (bit-exactness would be vacuous)."""
    with pytest.raises(AssertionError, match="not all-finite"):
        assert_deterministic(lambda: np.array([np.nan, 1.0]), label="nan")


def test_negative_control_constant_is_rejected():
    """A constant output is rejected (carries no information)."""
    with pytest.raises(AssertionError, match="constant"):
        assert_deterministic(lambda: np.zeros(5), label="const")
