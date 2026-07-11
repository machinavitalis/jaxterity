# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Manufacturing-variance fleets: sampled per-unit twins.

Two robots off the same line differ by single-digit percentages in friction,
motor torque constant, and link inertia. This module samples a *fleet* of
such units — each a :class:`~jaxterity.twin.Twin` with its own serial,
parameters, and attestation handle — via **Latin-hypercube sampling** (one
sample per stratum per dimension, so even a 30-unit fleet covers the
variance box evenly; i.i.d. sampling at that size leaves holes).

Uses:

- **Per-unit calibration studies**: each sampled twin is a "true" physical
  unit; a factory-default controller uses the *nominal* parameters on all of
  them, a calibrated one fits each unit (the windowed re-fit /
  hold-excitation path) — the difference is the per-unit gap.
- **Healthy-population manifold**: :func:`healthy_population` extracts each
  unit's declared drift parameters in the
  :meth:`Twin.identifiable_params` convention, ready for
  :meth:`jaxterity.selfcheck.HealthyManifold.from_samples` — the "what does
  a healthy unit look like" region the drift anomaly monitor checks against.
"""

from __future__ import annotations

from typing import Mapping

import jax.numpy as jnp
import numpy as np

from ..errors import JaxterityError
from .twin import Twin


def latin_hypercube(n_samples: int, n_dims: int, rng: np.random.Generator):
    """Latin-hypercube sample in the unit cube.

    One point per stratum per dimension: dimension ``d``'s values are a
    random permutation of the ``n_samples`` strata, jittered within each.

    Args
    ----
    n_samples : int
        Number of points (≥ 1).
    n_dims : int
        Dimensionality.
    rng : numpy.random.Generator
        The (seeded) sampler.

    Returns
    -------
    numpy.ndarray, shape ``(n_samples, n_dims)``
        Points in ``[0, 1)``, stratified per dimension.
    """
    strata = np.stack(
        [rng.permutation(n_samples) for _ in range(n_dims)], axis=1
    ) + rng.uniform(size=(n_samples, n_dims))
    return strata / n_samples


def sample_g1_units(
    n_units: int,
    *,
    seed: int = 0,
    friction_spread: float = 0.15,
    kt_spread: float = 0.10,
    mass_spread: float = 0.05,
    serial_prefix: str = "G1",
) -> tuple[Twin, ...]:
    """Sample a fleet of G1 units around the nominal model.

    Each unit is a :func:`~jaxterity.twin.unitree_g1_twin` whose joint
    Coulomb frictions, per-servo actuation scales (``K_t_scale``), and link
    masses are drawn uniformly within ``±spread`` of nominal, Latin-
    hypercube-stratified across the fleet. Serial numbers run
    ``{prefix}-0001…``; distinct units get distinct attestation handles.

    Args
    ----
    n_units : int
        Fleet size (≥ 1).
    seed : int
        Sampling seed (the fleet is reproducible).
    friction_spread : float
        Joint Coulomb friction multiplicative spread (±, fraction of
        nominal). Default ±15 %.
    kt_spread : float
        Actuation (torque-constant) scale spread around 1. Default ±10 %.
    mass_spread : float
        Link mass multiplicative spread. Default ±5 %.
    serial_prefix : str
        Serial number prefix.

    Returns
    -------
    tuple[Twin, ...]
        The fleet — each unit's twin carries its *true* parameters (these
        are the plants; the factory-default model of every one of them is
        the nominal twin).

    Raises
    ------
    JaxterityError
        If ``n_units`` < 1 or a spread is not in ``[0, 1)``.
    """
    # Deferred: fleet.py is imported by the package __init__, which defines
    # the factory this consumes.
    from . import unitree_g1_twin

    if n_units < 1:
        raise JaxterityError(f"sample_g1_units: n_units must be ≥ 1, got {n_units}.")
    for name, spread in (
        ("friction_spread", friction_spread),
        ("kt_spread", kt_spread),
        ("mass_spread", mass_spread),
    ):
        if not (0.0 <= spread < 1.0):
            raise JaxterityError(
                f"sample_g1_units: {name} must be in [0, 1) (a ±fraction of "
                f"nominal), got {spread}."
            )

    nominal = unitree_g1_twin(serial=f"{serial_prefix}-NOMINAL")
    joint_names = nominal.servo_names
    link_names = tuple(link.name for link in nominal.robot.links)
    n_joints, n_links = len(joint_names), len(link_names)

    rng = np.random.default_rng(seed)
    cube = latin_hypercube(n_units, 2 * n_joints + n_links, rng)
    # Map [0,1) → multiplicative factors in [1−s, 1+s) per block.
    friction_factor = 1.0 + friction_spread * (2.0 * cube[:, :n_joints] - 1.0)
    kt_factor = 1.0 + kt_spread * (2.0 * cube[:, n_joints : 2 * n_joints] - 1.0)
    mass_factor = 1.0 + mass_spread * (2.0 * cube[:, 2 * n_joints :] - 1.0)

    nominal_friction = np.asarray(
        [float(nominal.robot.joint_map[j].friction) for j in joint_names]
    )
    nominal_mass = np.asarray(
        [float(nominal.robot.link_map[name].mass) for name in link_names]
    )

    units = []
    for k in range(n_units):
        updates: dict[str, object] = {
            f"{j}.friction": nominal_friction[i] * friction_factor[k, i]
            for i, j in enumerate(joint_names)
        }
        updates.update(
            {
                f"{name}.mass": nominal_mass[i] * mass_factor[k, i]
                for i, name in enumerate(link_names)
            }
        )
        unit = unitree_g1_twin(serial=f"{serial_prefix}-{k + 1:04d}")
        unit = Twin(
            unit.robot.with_parameters(updates),
            serial=unit.serial,
            C_th=unit.C_th,
            R_th=unit.R_th,
            k_pwr=unit.k_pwr,
            K_t_scale=jnp.asarray(kt_factor[k]),
            T_amb=unit.T_amb,
            servo_names=unit.servo_names,
        )
        units.append(unit)
    return tuple(units)


def healthy_population(
    units: tuple[Twin, ...],
    parameter_paths: tuple[str, ...] | None = None,
) -> list[Mapping[str, object]]:
    """Each unit's declared drift parameters, for the healthy manifold.

    Args
    ----
    units : tuple[Twin, ...]
        The fleet (e.g. :func:`sample_g1_units` output, or per-unit fit
        results from real calibrations).
    parameter_paths : tuple[str, ...] | None
        Which :meth:`Twin.identifiable_params` paths to extract. Default:
        the G1 drift schema's paths
        (:func:`jaxterity.twin.g1_drift_parameters`).

    Returns
    -------
    list[Mapping[str, object]]
        One ``{path: value}`` mapping per unit — the
        :meth:`jaxterity.selfcheck.HealthyManifold.from_samples` input.
    """
    if parameter_paths is None:
        from . import g1_drift_parameters

        parameter_paths = tuple(p.path for p in g1_drift_parameters())
    population: list[Mapping[str, object]] = []
    for unit in units:
        params = unit.identifiable_params()
        missing = sorted(set(parameter_paths) - set(params))
        if missing:
            raise JaxterityError(
                f"healthy_population: unit {unit.serial!r} is missing declared "
                f"drift parameters {missing}. Every unit must carry the full "
                "schema (build them via the unit twin factory)."
            )
        population.append({path: params[path] for path in parameter_paths})
    return population


__all__ = ["latin_hypercube", "sample_g1_units", "healthy_population"]
