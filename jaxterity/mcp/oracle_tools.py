# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""MCP tools for the oracle verbs — the physics oracle as an agent tool.

An LLM planner reasons about a task ("carry this tote up the ramp") in language
and has no way to know whether *this* unit's servos will cook doing it. These
tools give it the missing sense: ``oracle_simulate_plan`` answers what happens
if a command runs, ``oracle_what_if`` ranks alternatives, and
``oracle_identify_payload`` asks the arm what it is holding. Every answer comes
back with per-constraint margins, the dispersion of those margins across the
unit's own parameter posterior, and the twin's attestation handle — so a
planner can tell "physics says no" from "the fit is too loose to tell", and a
later reader can tell which unit version answered.

State follows the ADR-008 stable-handle pattern, like
:mod:`~jaxterity.mcp.layered_tools`: create a twin handle once, then pass its id
to the verbs. Twins here are built from the zoo with **representative**
off-body defaults (they are not a per-unit calibration); ``twin_info`` says so
explicitly, and answers from a twin with no attached fit covariance report a
zero-width band rather than a fabricated confidence.

The tools are pure JSON in / JSON out (lists, dicts, numbers, strings) so any
MCP client can call them; the physics lives in :mod:`jaxterity.oracle`.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Any, Callable, Sequence

from .. import oracle as _oracle
from ..calibrate.active import ReplayLink, design_coast_excitation
from ..checkup.stability import LEKIWI_BASE
from ..errors import JaxterityError
from ..objects import PayloadLoopbackLink, RigidPayload, SemanticPrior
from ..twin import Twin, lekiwi_twin, so101_twin

#: Twin builders exposed to agents, by unit kind.
_TWIN_BUILDERS: dict[str, Callable[[str], Twin]] = {
    "so101": so101_twin,
    "lekiwi": lekiwi_twin,
}

#: Declared mobile bases available to the tip-over screen, by name. Both are
#: representative geometry/mass, not a measured unit (see
#: :data:`jaxterity.checkup.stability.LEKIWI_BASE`).
_BASES = {"lekiwi": LEKIWI_BASE}

#: Live twin handles for this server process.
_TWIN_REGISTRY: dict[str, Twin] = {}


def _twin_handle_id(unit: str, serial: str) -> str:
    payload = json.dumps({"unit": unit, "serial": serial}).encode()
    return "@twin-" + hashlib.sha256(payload).hexdigest()[:16]


def _get_twin(handle: str) -> Twin:
    if handle not in _TWIN_REGISTRY:
        raise JaxterityError(
            f"unknown twin handle {handle!r}. Call create_twin_handle first; "
            f"live handles: {list(_TWIN_REGISTRY)}."
        )
    return _TWIN_REGISTRY[handle]


def _base_from_name(base: str | None):
    if base is None:
        return None
    if base not in _BASES:
        raise JaxterityError(
            f"unknown base profile {base!r}. Known: {sorted(_BASES)}. Omit "
            "`base` to skip the tip-over screen (a bolted-down arm has no "
            "tip-over mode)."
        )
    return _BASES[base]


def _prior_from_mapping(prior: dict[str, Any] | None) -> SemanticPrior | None:
    """Build a :class:`~jaxterity.objects.SemanticPrior` from a JSON mapping.

    Two accepted shapes: an explicit bracket (``mass_kg``) or a material +
    bounding box (``material`` + ``extent_m``), which is what a vision model can
    actually report.
    """
    if prior is None:
        return None
    fields = dict(prior)
    label = fields.pop("label", "unlabelled object")
    source = fields.pop("source", "unattributed")
    confidence = float(fields.pop("confidence", 0.5))
    if "material" in fields:
        material = fields.pop("material")
        extent = fields.pop("extent_m", None)
        if extent is None:
            raise JaxterityError(
                "semantic prior: a material guess needs `extent_m` (the three "
                "bounding-box side lengths in m) to become a mass bracket."
            )
        fill = fields.pop("fill", None)
        if fields:
            raise JaxterityError(
                f"semantic prior: unexpected key(s) {sorted(fields)} alongside "
                "`material`. Use either {label, material, extent_m, fill?, "
                "source, confidence?} or {label, mass_kg, com_halfwidth_m?, "
                "inertia_max?, source, confidence?}."
            )
        kwargs: dict[str, Any] = {}
        if fill is not None:
            kwargs["fill"] = (float(fill[0]), float(fill[1]))
        return SemanticPrior.from_material(
            label,
            str(material),
            (float(extent[0]), float(extent[1]), float(extent[2])),
            source=source,
            confidence=confidence,
            **kwargs,
        )
    mass = fields.pop("mass_kg", None)
    if mass is None:
        raise JaxterityError(
            "semantic prior: needs either `mass_kg` (a [lo, hi] bracket in kg) "
            "or `material` + `extent_m`. A guess with no bracket cannot bound "
            "the search."
        )
    half = fields.pop("com_halfwidth_m", None)
    inertia_max = fields.pop("inertia_max", None)
    if fields:
        raise JaxterityError(
            f"semantic prior: unexpected key(s) {sorted(fields)}. Known: label, "
            "mass_kg, com_halfwidth_m, inertia_max, source, confidence."
        )
    return SemanticPrior(
        label=label,
        mass_kg=(float(mass[0]), float(mass[1])),
        source=source,
        com_halfwidth_m=None if half is None else _triple(half, "com_halfwidth_m"),
        inertia_max=(
            None if inertia_max is None else _triple(inertia_max, "inertia_max")
        ),
        confidence=confidence,
    )


def _triple(values, field: str) -> tuple[float, float, float]:
    """Coerce a JSON list to a three-vector, naming the field if it is not one."""
    if len(values) != 3:
        raise JaxterityError(
            f"semantic prior: {field} needs exactly three per-axis values; got "
            f"{values!r}."
        )
    return (float(values[0]), float(values[1]), float(values[2]))


def _answer_to_dict(answer: _oracle.OracleAnswer) -> dict[str, Any]:
    return {
        "question": answer.question,
        "outcome": answer.outcome,
        "pass_fraction": answer.pass_fraction,
        "binding_constraint": answer.binding_constraint,
        "time_to_breach_s": answer.time_to_breach_s,
        "twin_handle": answer.twin_handle,
        "n_draws": answer.n_draws,
        "coverage": answer.coverage,
        "margins": [dataclasses.asdict(band) for band in answer.margins],
        "detail": answer.detail,
        "summary": answer.summary(),
    }


def _plan(
    twin: Twin,
    name: str,
    tau: Sequence[float],
    n_steps: int,
    dt: float,
    q0: Sequence[float] | None,
    T0: Sequence[float] | None,
) -> _oracle.Plan:
    return _oracle.hold_plan(
        name, twin, list(tau), n_steps=int(n_steps), dt=float(dt), q0=q0, T0=T0
    )


# --- twin handle lifecycle --------------------------------------------------


def create_twin_handle(unit: str, serial: str) -> str:
    """Instantiate a per-unit twin and return its stable handle id.

    Args
    ----
    unit : str
        ``"so101"`` (bench arm) or ``"lekiwi"`` (arm on a shared battery — the
        bus-voltage/brownout path).
    serial : str
        The physical unit's serial number. It is part of the twin's identity,
        so two serials get different attestation handles.

    Returns
    -------
    str
        Handle id of the form ``"@twin-<16-hex>"``, accepted by every oracle
        verb.

    Raises
    ------
    JaxterityError
        On an unknown unit kind.
    """
    if unit not in _TWIN_BUILDERS:
        raise JaxterityError(
            f"unknown twin unit {unit!r}. Known: {sorted(_TWIN_BUILDERS)}."
        )
    twin = _TWIN_BUILDERS[unit](serial)
    handle = _twin_handle_id(unit, serial)
    _TWIN_REGISTRY[handle] = twin
    return handle


def list_twin_handles() -> list[str]:
    """Return all live twin handle ids."""
    return list(_TWIN_REGISTRY)


def release_twin_handle(handle: str) -> None:
    """Drop a twin handle from the registry. No-op if absent."""
    _TWIN_REGISTRY.pop(handle, None)


def clear_twin_registry() -> None:
    """Drop every live twin handle (test fixtures, server shutdown)."""
    _TWIN_REGISTRY.clear()


def twin_info(handle: str) -> dict[str, Any]:
    """Describe a twin: identity, size, battery, and what its posterior covers.

    Args
    ----
    handle : str
        A handle from :func:`create_twin_handle`.

    Returns
    -------
    dict
        ``serial``, ``n_servos``, ``has_battery``, ``attestation_handle``,
        ``servo_names``, ``uncertain_leaves`` (the fitted leaves an oracle
        answer's band is drawn over — empty means answers are point
        predictions), and a ``provenance_note`` stating that these zoo-built
        twins carry representative, not per-unit-fitted, off-body defaults.
    """
    twin = _get_twin(handle)
    return {
        "serial": twin.serial,
        "n_servos": int(twin.n_servos),
        "has_battery": bool(twin.has_battery),
        "attestation_handle": twin.attestation_handle,
        "servo_names": list(twin.servo_names),
        "uncertain_leaves": sorted(twin.uncertainty()),
        "provenance_note": (
            "Built from the zoo with representative off-body defaults (thermal, "
            "battery); this is not a per-unit calibration. Answers from it are "
            "as good as those defaults, and its band is zero-width until a fit's "
            "covariance is attached."
        ),
    }


# --- the oracle verbs -------------------------------------------------------


def oracle_simulate_plan(
    twin_handle: str,
    tau: list[float],
    n_steps: int,
    dt: float,
    envelope: dict[str, float],
    *,
    name: str = "plan",
    q0: list[float] | None = None,
    T0: list[float] | None = None,
    base: str | None = None,
    n_draws: int = 8,
    seed: int = 0,
    coverage: float = 0.9,
) -> dict[str, Any]:
    """Ask what happens if this command runs on this unit.

    Simulates a held commanded torque on the unit's twin across a posterior of
    its fitted parameters and reports the outcome (pass / derate / veto), the
    per-constraint margins with their spread, the binding constraint, and — for
    a thermal breach — the time until the ceiling is crossed.

    Args
    ----
    twin_handle : str
        Handle from :func:`create_twin_handle`.
    tau : list[float]
        Commanded generalized force per DOF, N·m, held over the plan.
    n_steps : int
        Number of steps to simulate.
    dt : float
        Step size, s. The twin integrates with a fixed-step RK4 — a free-moving
        arm typically needs ``dt <= 0.01``; too coarse a step is reported as a
        divergence error, not as a verdict.
    envelope : dict[str, float]
        Safety limits: ``temp_ceiling_k`` (required), and optionally
        ``temp_band_k``, ``joint_margin_rad``, ``bus_floor_v``, ``bus_band_v``,
        ``torque_margin_frac``, ``torque_band_frac``. An unknown key is
        refused rather than ignored.
    name : str
        Label for the question, echoed in the answer.
    q0, T0 : list[float] | None
        Initial joint positions (rad) and per-servo temperature (K); default
        zeros and ambient.
    base : str | None
        Declared mobile base for the coupled tip-over screen (``"lekiwi"``).
        Omit for a bolted-down arm.
    n_draws : int
        Posterior draws (collapses to 1 when the twin carries no fit
        covariance).
    seed : int
        PRNG seed for the draws.
    coverage : float
        Quantile coverage of the reported bands.

    Returns
    -------
    dict
        The answer: outcome, pass_fraction, binding_constraint, margins
        (each with worst / median / lo / hi / units), time_to_breach_s, the
        twin's attestation handle, and a one-line summary.

    Units
    -----
    N·m, s, K, V, rad, m (tip-over margin).
    """
    twin = _get_twin(twin_handle)
    plan = _plan(twin, name, tau, n_steps, dt, q0, T0)
    answer = _oracle.simulate_plan(
        twin,
        plan,
        _oracle.envelope_from_mapping(envelope),
        base=_base_from_name(base),
        n_draws=int(n_draws),
        seed=int(seed),
        coverage=float(coverage),
    )
    return _answer_to_dict(answer)


def oracle_what_if(
    twin_handle: str,
    candidates: list[dict[str, Any]],
    envelope: dict[str, float],
    *,
    base: str | None = None,
    n_draws: int = 8,
    seed: int = 0,
    coverage: float = 0.9,
) -> dict[str, Any]:
    """Compare candidate commands and rank them by the headroom each keeps.

    The choosing verb: "which of these keeps the most tip-over margin?", "is
    the slower version worth it?". Every candidate is evaluated on the same
    posterior draws, so the ranking reflects the plans and not the sampling.

    Args
    ----
    twin_handle : str
        Handle from :func:`create_twin_handle`.
    candidates : list[dict]
        One dict per candidate: ``{"name": str, "tau": [float, ...],
        "n_steps": int, "dt": float}`` plus optional ``"q0"`` / ``"T0"``.
        Names must be unique.
    envelope : dict[str, float]
        As in :func:`oracle_simulate_plan`.
    base : str | None
        Declared mobile base for the tip-over screen.
    n_draws, seed, coverage
        As in :func:`oracle_simulate_plan`.

    Returns
    -------
    dict
        ``best`` (winning candidate name), ``ranking`` (best first),
        ``answers`` (the full answer per candidate), and ``detail``.

    Raises
    ------
    JaxterityError
        On an empty candidate list, duplicate names, or a candidate missing
        ``tau`` / ``n_steps`` / ``dt``.
    """
    twin = _get_twin(twin_handle)
    plans = []
    for i, candidate in enumerate(candidates):
        missing = [k for k in ("tau", "n_steps", "dt") if k not in candidate]
        if missing:
            raise JaxterityError(
                f"oracle_what_if: candidate {i} is missing {missing}. Each "
                "candidate needs name, tau (N·m per DOF), n_steps and dt (s)."
            )
        plans.append(
            _plan(
                twin,
                str(candidate.get("name", f"candidate_{i}")),
                candidate["tau"],
                candidate["n_steps"],
                candidate["dt"],
                candidate.get("q0"),
                candidate.get("T0"),
            )
        )
    comparison = _oracle.what_if(
        twin,
        plans,
        _oracle.envelope_from_mapping(envelope),
        base=_base_from_name(base),
        n_draws=int(n_draws),
        seed=int(seed),
        coverage=float(coverage),
    )
    return {
        "best": comparison.best,
        "ranking": list(comparison.ranking),
        "detail": comparison.detail,
        "answers": [_answer_to_dict(a) for a in comparison.answers],
    }


def oracle_identify_payload(
    robot_name: str,
    attach_link: str,
    *,
    payload_name: str = "payload",
    offset: list[float] | None = None,
    prior: dict[str, Any] | None = None,
    telemetry_path: str | None = None,
    planted_mass: float | None = None,
    planted_com: list[float] | None = None,
    fit: list[str] | None = None,
    probe_speed: float = 0.1,
    iterations: int = 8,
    uncertainty: str | None = "laplace",
    coverage: float = 0.95,
) -> dict[str, Any]:
    """Ask the arm what it is holding: mass / CoM with intervals and provenance.

    Two transports, and the answer always says which one it used. With
    ``telemetry_path`` the fit consumes a **real capture** (the on-device script
    records the probe; the fit runs here). Without it the probe is replayed
    through a seeded **simulation** of the arm holding a planted object — useful
    for probe design and for checking what a probe could recover, and labelled
    ``"loopback-simulated"`` so it is never mistaken for a measurement.

    A ``prior`` (a vision model's guess) seeds and bounds the search; the
    returned ``recipe`` records that it did.

    Args
    ----
    robot_name : str
        A zoo robot name (see ``jaxterity.zoo.available``).
    attach_link : str
        Link the object is held by.
    payload_name : str
        Name of the payload body in the lowered model.
    offset : list[float] | None
        Grasp point in the attach link's frame, m. Default the link origin.
    prior : dict | None
        A semantic prior, either ``{"label", "mass_kg": [lo, hi], "source", …}``
        or ``{"label", "material", "extent_m": [x, y, z], "source", …}``.
    telemetry_path : str | None
        Path to a recorded capture (CSV / Parquet / HDF5 in the telemetry
        schema). When given, the fit runs against real data.
    planted_mass, planted_com : float | list[float] | None
        Ground truth for the simulated transport only (ignored with
        ``telemetry_path``).
    fit : list[str] | None
        Parameter groups to fit — a subset of ``["mass", "com",
        "inertia_diag"]``. Default ``["mass", "com"]``.
    probe_speed : float
        Coast-probe initial joint velocity, rad/s.
    iterations : int
        Levenberg–Marquardt iterations.
    uncertainty : str | None
        ``"laplace"`` (default), ``"bootstrap"``, or ``None``.
    coverage : float
        Interval coverage.

    Returns
    -------
    dict
        ``params`` (SI scalars), ``intervals``, ``recipe``, ``telemetry_hash``,
        ``warnings`` (verbatim: identifiability, pinned prior bounds,
        unresolved parameters) and ``transport``.

    Raises
    ------
    JaxterityError
        On an unknown robot/link, or a malformed prior.

    Units
    -----
    kg, m, kg·m²; probe speed rad/s.
    """
    from .. import zoo

    robot = zoo.load(robot_name)
    payload = RigidPayload(
        attach_link,
        mass=0.1,
        name=payload_name,
        offset=(0.0, 0.0, 0.0) if offset is None else _triple(offset, "offset"),
    )
    excitation = design_coast_excitation(robot, speed=float(probe_speed))
    if telemetry_path is not None:
        from ..data.loaders import load_telemetry

        link: Any = ReplayLink(load_telemetry(telemetry_path))
        transport = f"replay:{telemetry_path}"
    else:
        link = PayloadLoopbackLink(
            robot,
            payload,
            true_mass=planted_mass,
            true_com=None if planted_com is None else tuple(planted_com),
        )
        transport = "loopback-simulated"
    answer = _oracle.identify_object(
        robot,
        link,
        payload,
        prior=_prior_from_mapping(prior),
        coverage=float(coverage),
        excitation=excitation,
        fit=tuple(fit) if fit else ("mass", "com"),
        iterations=int(iterations),
        uncertainty=uncertainty,
    )
    intervals = (
        None
        if answer.intervals is None
        else {k: [float(v[0]), float(v[1])] for k, v in answer.intervals.items()}
    )
    return {
        "params": {k: float(v) for k, v in answer.params.items()},
        "intervals": intervals,
        "coverage": answer.coverage,
        "recipe": answer.recipe,
        "telemetry_hash": answer.telemetry_hash,
        "warnings": list(answer.warnings),
        "transport": transport,
        "summary": answer.summary(),
    }


#: Source-of-truth registry the MCP server adds to its tool table.
ORACLE_TOOLS = (
    create_twin_handle,
    list_twin_handles,
    release_twin_handle,
    clear_twin_registry,
    twin_info,
    oracle_simulate_plan,
    oracle_what_if,
    oracle_identify_payload,
)


__all__ = [
    "ORACLE_TOOLS",
    "create_twin_handle",
    "list_twin_handles",
    "release_twin_handle",
    "clear_twin_registry",
    "twin_info",
    "oracle_simulate_plan",
    "oracle_what_if",
    "oracle_identify_payload",
]
