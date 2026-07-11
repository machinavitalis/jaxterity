# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Bit-exact determinism checks for the test suite.

Jaxterity's attestation chain (CONTEXT.md invariant 5) depends on hash
stability, which depends on computations being bit-reproducible on a fixed
device. :func:`assert_deterministic` runs a function twice and compares the raw
bytes — with a guard against the failure mode where a *degenerate* output (all
NaN/Inf, or a single repeated value) makes a bytes comparison pass vacuously.

The same-device contract this enforces is specified in ``test/DETERMINISM.md``.
Cross-device determinism is intentionally out of scope here — it is the open
question OQ-4 in ``AGENTS/DECISIONS.md`` and must not be silently assumed.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np


def _as_array(value: Any) -> np.ndarray:
    return np.asarray(value)


def assert_deterministic(
    fn: Callable[[], Any],
    *,
    label: str = "computation",
) -> np.ndarray:
    """Assert that ``fn()`` is bit-for-bit reproducible across two calls.

    Runs ``fn`` twice and compares ``.tobytes()``. Before comparing, it rejects
    *degenerate* outputs that would let a broken computation pass a bytes check
    for free — an all-non-finite result (NaN bit patterns compare equal) or a
    constant array carrying no information.

    Args
    ----
    fn : Callable[[], Any]
        A zero-argument callable returning an array-like. Close over inputs and
        seeds so successive calls are nominally identical.
    label : str
        A name used in failure messages.

    Returns
    -------
    numpy.ndarray
        The (first) computed output, for further assertions by the caller.

    Raises
    ------
    AssertionError
        If the output is degenerate (non-finite or constant), or if the two
        runs differ byte-for-byte.

    Examples
    --------
    >>> import numpy as np
    >>> _ = assert_deterministic(lambda: np.arange(4.0), label="arange")
    """
    a = _as_array(fn())
    b = _as_array(fn())

    if not np.all(np.isfinite(a)):
        raise AssertionError(
            f"{label}: output is not all-finite, so a bit-exact comparison is "
            "vacuous (NaN/Inf bit patterns compare equal). A determinism test "
            "must run a finite, non-degenerate reference computation."
        )
    if a.size > 1 and np.all(a == a.flat[0]):
        raise AssertionError(
            f"{label}: output is a constant array (every element == "
            f"{a.flat[0]!r}); a constant carries no information and would pass "
            "a bytes check trivially. Use a non-degenerate reference."
        )
    if a.tobytes() != b.tobytes():
        raise AssertionError(
            f"{label}: not bit-reproducible — two runs on the same device "
            "produced different bytes. See test/DETERMINISM.md for the "
            "same-device contract this violates."
        )
    return a


__all__ = ["assert_deterministic"]
