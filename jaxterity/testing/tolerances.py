# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Per-(quantity, solver, dtype) numerical tolerances for the test suite.

This module is the single **programmatic** source for ``atol`` / ``rtol`` used
by every numerical test (PATTERNS.md §6.3 — no magic numbers in tests). The
human-readable policy and the full generated table live in
``test/TOLERANCES.md``; ``test/determinism/test_determinism.py`` cross-checks
that the doc's table matches what this module computes, so the two cannot drift.
This module ships in the wheel (``test/`` does not), so tests can read the
tolerances regardless of the working directory.

The values are **initial policy defaults**, grounded in machine epsilon and
solver order, not yet validated against real reference trajectories — those land
once ``Robot.to_diagram()`` exists, at which point the factors here are
tightened to the measured floors.

Tolerances are computed from three factor tables rather than 30 hand-typed
rows, so the policy stays legible and internally consistent:

    atol = base_atol[quantity] * dtype_atol_factor[dtype] * solver_factor[solver]
    rtol = dtype_rtol[dtype] * solver_factor[solver]
"""

from __future__ import annotations

from ..errors import JaxterityError

QUANTITIES: tuple[str, ...] = (
    "joint_position",
    "joint_velocity",
    "contact_force",
    "actuator_torque",
    "actuator_voltage",
    "sensor_reading",
    "motor_temperature",
    "bus_voltage",
)
SOLVERS: tuple[str, ...] = ("rk4", "dopri5", "bdf")
DTYPES: tuple[str, ...] = ("float32", "float64")

# Absolute-tolerance floor per quantity, at float64 under a tight adaptive
# solver. Scales reflect the physical magnitude of each quantity in SI units.
_BASE_ATOL_F64: dict[str, float] = {
    "joint_position": 1e-9,  # rad / m, O(1)
    "joint_velocity": 1e-8,  # rad/s, O(1-10)
    "contact_force": 1e-6,  # N, O(1-100)
    "actuator_torque": 1e-7,  # N·m, O(1-10)
    "actuator_voltage": 1e-7,  # V, O(1-12) — WPILib SysId identifies V from (v, a)
    "sensor_reading": 1e-7,  # SI, quantity-dependent
    "motor_temperature": 1e-5,  # K, O(290-400) abs / O(1-100) rise
    "bus_voltage": 1e-7,  # V, O(1-24) — LeKiwi shared rail
}
# float32 carries ~8 fewer decimal digits than float64 (eps ~1.2e-7 vs ~2.2e-16);
# loosen the absolute floor by 1e5 to stay well clear of round-off.
_DTYPE_ATOL_FACTOR: dict[str, float] = {"float64": 1.0, "float32": 1e5}
# Relative tolerance per dtype, comfortably above each dtype's epsilon to
# absorb accumulation over a trajectory.
_DTYPE_RTOL: dict[str, float] = {"float64": 1e-6, "float32": 1e-4}
# Adaptive dopri5 is the reference; fixed-step rk4 and stiff (dissipative) bdf
# accumulate more error per unit work.
_SOLVER_FACTOR: dict[str, float] = {"dopri5": 1.0, "rk4": 5.0, "bdf": 10.0}

# Gradient-agreement tolerances: how closely an autodiff gradient (forward- or
# reverse-mode) must match a *central finite difference*. These are looser than
# the state tolerances above because the comparison is limited by finite-
# difference truncation (O(h²)), not by the autodiff — the gradient itself is
# exact. They also bound forward-vs-reverse cross-checks (which pass far tighter).
# Solver-independent: FD truncation dominates the per-step solver error.
_GRADIENT_RTOL: dict[str, float] = {"float64": 5e-3, "float32": 5e-2}
_GRADIENT_ATOL: dict[str, float] = {"float64": 1e-4, "float32": 1e-2}


def _check(quantity: str, solver: str, dtype: str) -> None:
    if quantity not in QUANTITIES:
        raise JaxterityError(
            f"Unknown tolerance quantity {quantity!r}. Expected one of "
            f"{QUANTITIES}. Fix: pass a registered quantity, or add it to "
            "jaxterity/testing/tolerances.py and test/TOLERANCES.md."
        )
    if solver not in SOLVERS:
        raise JaxterityError(f"Unknown solver {solver!r}. Expected one of {SOLVERS}.")
    if dtype not in DTYPES:
        raise JaxterityError(
            f"Unknown dtype {dtype!r}. Expected one of {DTYPES} (as a string)."
        )


def atol(quantity: str, solver: str, dtype: str) -> float:
    """Absolute tolerance for comparing a ``quantity`` under ``solver``/``dtype``.

    Args
    ----
    quantity : str
        One of :data:`QUANTITIES`.
    solver : str
        One of :data:`SOLVERS`.
    dtype : str
        One of :data:`DTYPES` (``"float32"`` / ``"float64"``).

    Returns
    -------
    float
        The absolute tolerance.

    Raises
    ------
    JaxterityError
        If ``quantity``, ``solver``, or ``dtype`` is not registered.

    Examples
    --------
    >>> atol("joint_position", "dopri5", "float64")
    1e-09
    """
    _check(quantity, solver, dtype)
    return _BASE_ATOL_F64[quantity] * _DTYPE_ATOL_FACTOR[dtype] * _SOLVER_FACTOR[solver]


def rtol(quantity: str, solver: str, dtype: str) -> float:
    """Relative tolerance for comparing a ``quantity`` under ``solver``/``dtype``.

    Args
    ----
    quantity : str
        One of :data:`QUANTITIES`.
    solver : str
        One of :data:`SOLVERS`.
    dtype : str
        One of :data:`DTYPES`.

    Returns
    -------
    float
        The relative tolerance.

    Raises
    ------
    JaxterityError
        If ``quantity``, ``solver``, or ``dtype`` is not registered.

    Examples
    --------
    >>> rtol("contact_force", "bdf", "float32")
    0.001
    """
    _check(quantity, solver, dtype)
    return _DTYPE_RTOL[dtype] * _SOLVER_FACTOR[solver]


def gradient_rtol(dtype: str) -> float:
    """Relative tolerance for an autodiff gradient vs a central finite difference.

    Looser than :func:`rtol` — the comparison is bounded by finite-difference
    truncation, not the (exact) autodiff. Solver-independent. Also bounds
    forward-mode-vs-reverse-mode gradient cross-checks.

    Args
    ----
    dtype : str
        One of :data:`DTYPES`.

    Returns
    -------
    float
        The relative tolerance.

    Raises
    ------
    JaxterityError
        If ``dtype`` is not registered.
    """
    if dtype not in DTYPES:
        raise JaxterityError(f"Unknown dtype {dtype!r}. Expected one of {DTYPES}.")
    return _GRADIENT_RTOL[dtype]


def gradient_atol(dtype: str) -> float:
    """Absolute tolerance for an autodiff gradient vs a central finite difference.

    Companion to :func:`gradient_rtol` for near-zero gradient components. See it
    for the rationale.

    Args
    ----
    dtype : str
        One of :data:`DTYPES`.

    Returns
    -------
    float
        The absolute tolerance.

    Raises
    ------
    JaxterityError
        If ``dtype`` is not registered.
    """
    if dtype not in DTYPES:
        raise JaxterityError(f"Unknown dtype {dtype!r}. Expected one of {DTYPES}.")
    return _GRADIENT_ATOL[dtype]


def table() -> dict[tuple[str, str, str], tuple[float, float]]:
    """Return the full ``(quantity, solver, dtype) -> (atol, rtol)`` table.

    Returns
    -------
    dict[tuple[str, str, str], tuple[float, float]]
        Every combination of :data:`QUANTITIES` × :data:`SOLVERS` ×
        :data:`DTYPES`, in that nested order.
    """
    return {
        (q, s, d): (atol(q, s, d), rtol(q, s, d))
        for q in QUANTITIES
        for s in SOLVERS
        for d in DTYPES
    }


__all__ = [
    "QUANTITIES",
    "SOLVERS",
    "DTYPES",
    "atol",
    "rtol",
    "gradient_rtol",
    "gradient_atol",
    "table",
]
