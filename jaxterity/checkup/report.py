# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Per-unit health checkup: compose capture → fit → gates → report card.

This module owns the ``jaxterity checkup`` pipeline: it takes one unit's
commanded-excitation captures (``scripts/lekiwi_commanded_capture.py``), runs
the closed-loop friction fit per joint
(:func:`~jaxterity.calibrate.identify_closed_loop`), optionally fits the
battery Thévenin sag (:func:`~jaxterity.calibrate.fit_battery_sag`), evaluates
a fixed set of pass/warn/fail **gates**, and emits a :class:`CheckupReport` —
a JSON-serializable report card carrying the fitted parameters with
confidence intervals, the held-out fit quality versus the frictionless URDF
baseline, coarse actuation-health checks, and the unit's attestation handle.

The gates are deliberately conservative and each carries a human-readable
``detail`` and ``remedy``. The fidelity score is a documented, versioned
formula over measurable quantities (see :func:`_score`) — a triage number,
not a certification. What this checkup does **not** measure (v0): backlash,
thermal parameters (the short captures carry no usable thermal signal), and
anything requiring a second reference unit.
"""

from __future__ import annotations

import dataclasses
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from .. import zoo
from ..calibrate import (
    fit_battery_sag,
    identify_closed_loop,
    load_bus_load_trace,
    load_commanded_capture,
)
from ..errors import CalibrationError
from .protocol import CaptureProtocol, capture_protocol
from .stability import MULTISINE_KIND_V1, multisine_peak_to_peak

#: v2 adds the pinned capture protocol (:mod:`jaxterity.checkup.protocol`). A v1
#: report is read back with ``protocol=None`` — honest: those runs predate the
#: pin, so nothing can claim their method matched anything.
#: v3 adds the two things a bare Wald interval and a bare scalar could not say:
#: the *total* uncertainty (statistical + capture-length, see
#: :mod:`jaxterity.checkup.uncertainty`) and the pose the parameter was measured
#: at (:mod:`jaxterity.calibrate.pose_conditioned`). Both were measured to matter
#: on this hardware — the Wald interval understated by ~4x and shoulder_pan's
#: damping moved 3.29x with pose alone — so a report that omits them is a
#: quantified claim that is wrong, not merely an incomplete one.
SCHEMA = "jaxterity-checkup-v3"
SCHEMA_V2 = "jaxterity-checkup-v2"
SCHEMA_V1 = "jaxterity-checkup-v1"
SCORE_FORMULA_VERSION = "1"

#: Gate thresholds (documented in each gate's detail string).
_HELD_OUT_PASS = 0.10
_CONFIDENCE_REL_SE = 0.5
_RESPONSE_RATIO_BAND = (0.3, 2.5)
_REFERENCE_RATIO_BAND = (1.0 / 3.0, 3.0)
_BATTERY_R2_MIN = 0.5
_BATTERY_MARGIN_PASS_V = 0.3


@dataclasses.dataclass(frozen=True)
class Gate:
    """One pass/warn/fail verdict with its reason and remedy.

    ``level`` is ``"pass"`` / ``"warn"`` / ``"fail"``; only ``"fail"`` makes
    the overall checkup fail. ``scope`` names what was judged (a joint name,
    ``"battery"``, or ``"unit"``).
    """

    gate: str
    scope: str
    level: str
    detail: str
    remedy: str = ""

    @property
    def ok(self) -> bool:
        """Whether this gate is non-fatal (pass or warn)."""
        return self.level != "fail"


@dataclasses.dataclass(frozen=True)
class JointCheckup:
    """One joint's fitted parameters, confidence, and health metrics.

    ``damping`` / ``friction`` are the identified viscous (N·m·s/rad) and
    Coulomb (N·m) channels — ``None`` when the raw fit came out non-positive
    and the channel was *held* at the (frictionless) prior. The ``*_ci95``
    intervals are Wald intervals from the least-squares covariance.
    ``held_out`` carries the torque-prediction RMS of the frictionless
    baseline vs the fit on the capture's held-out tail, and the reduction
    fraction. ``response_ratio`` is measured peak-to-peak motion over the
    excitation's commanded peak-to-peak — a coarse dead-joint check.

    ``damping_total_ci95`` is the interval to **quote**: the Wald interval
    combined in quadrature with the capture-length bias measured by re-fitting on
    sub-windows (:func:`jaxterity.checkup.total_uncertainty`). On the reference
    capture the Wald interval did not contain its own sub-window re-fits and the
    systematic term was 3.7x the statistical one, so ``damping_ci95`` alone
    understates by roughly 4x. It is kept alongside rather than replaced, because
    the split is the information: ``damping_bias_fraction`` is the share of the
    total *variance* that is capture-length dependence. Both are ``None`` when
    the run did not measure the ladder.

    ``pose_feature`` is the value of the pose variable this joint's friction
    actually depends on, at the pose it was measured at — axis inertia for a yaw
    axis, gravity torque for a pitch axis, named by ``pose_feature_name``. Two
    fits of the same joint at different values of it are not comparable: on
    LK-0001 shoulder_pan's damping moved **3.29x** between identical excitation
    recipes purely because the arm was extended differently, and the ratio
    damping/inertia was consistent to 4 % across the two. Recording it is what
    makes a later comparison checkable rather than a guess.
    """

    joint: str
    servo_id: int
    n_samples: int
    damping: float | None
    friction: float | None
    damping_ci95: tuple[float, float] | None
    friction_ci95: tuple[float, float] | None
    held: tuple[str, ...]
    held_out: dict[str, float]
    K_t: float
    load_offset: float
    response_ratio: float | None
    telemetry_hash: str
    damping_total_ci95: tuple[float, float] | None = None
    damping_bias_fraction: float | None = None
    pose_feature: float | None = None
    pose_feature_name: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Plain-JSON form."""
        d = dataclasses.asdict(self)
        d["held"] = list(self.held)
        for k in ("damping_ci95", "friction_ci95", "damping_total_ci95"):
            if d[k] is not None:
                d[k] = list(d[k])
        return d


@dataclasses.dataclass(frozen=True)
class BatteryCheckup:
    """The battery Thévenin fit and the brownout margin it implies.

    ``predicted_min_v`` is ``V_oc - R_int * peak_current`` at the trace's
    observed peak aggregate current; ``margin_v`` is that minus the declared
    ``bus_floor_v``. The current scale is the duty-proxy frame of
    :func:`~jaxterity.calibrate.fit_battery_sag` (see its caveat), so the
    margin is a scenario estimate, not a calibrated absolute prediction.
    """

    V_oc: float
    R_int: float
    r_squared: float
    sag_v: float
    peak_current_a: float
    bus_floor_v: float
    predicted_min_v: float
    margin_v: float
    telemetry_hash: str

    def to_dict(self) -> dict[str, Any]:
        """Plain-JSON form."""
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class CheckupReport:
    """The full per-unit report card (JSON round-trippable).

    ``score`` maps component names to 0–100 integers plus ``"total"``;
    the formula is versioned (``score_formula``) and documented in
    :func:`_score`. ``attestation_handle`` is the calibrated robot's handle
    after all fitted channels are bound — two units with different fits get
    different handles, and a recalibration changes it.
    """

    schema: str
    serial: str
    robot: str
    created_utc: str
    engine_version: str
    baseline: str
    joints: tuple[JointCheckup, ...]
    battery: BatteryCheckup | None
    gates: tuple[Gate, ...]
    score: dict[str, int]
    score_formula: str
    attestation_handle: str
    protocol: CaptureProtocol | None = None

    @property
    def protocol_hash(self) -> str | None:
        """The pinned capture protocol's hash, or ``None`` for an unpinned run.

        Two runs of the same unit are comparable as a *drift* series only when
        this matches — see :mod:`jaxterity.checkup.protocol` and
        :meth:`jaxterity.checkup.CheckupStore.drift_warnings`.
        """
        return self.protocol.hash if self.protocol is not None else None

    @property
    def passed(self) -> bool:
        """True when no gate failed (warns allowed)."""
        return all(g.ok for g in self.gates)

    def to_dict(self) -> dict[str, Any]:
        """Plain-JSON form (inverse of :meth:`from_dict`)."""
        return {
            "schema": self.schema,
            "serial": self.serial,
            "robot": self.robot,
            "created_utc": self.created_utc,
            "engine_version": self.engine_version,
            "baseline": self.baseline,
            "joints": [j.to_dict() for j in self.joints],
            "battery": self.battery.to_dict() if self.battery else None,
            "gates": [dataclasses.asdict(g) for g in self.gates],
            "score": dict(self.score),
            "score_formula": self.score_formula,
            "attestation_handle": self.attestation_handle,
            "protocol": self.protocol.to_dict() if self.protocol else None,
            "protocol_hash": self.protocol_hash,
            "passed": self.passed,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "CheckupReport":
        """Rebuild a report from its :meth:`to_dict` form.

        Raises
        ------
        CalibrationError
            If the payload's schema field is not :data:`SCHEMA`.
        """
        if d.get("schema") not in (SCHEMA, SCHEMA_V2, SCHEMA_V1):
            raise CalibrationError(
                f"CheckupReport.from_dict: unrecognized schema {d.get('schema')!r}; "
                f"expected {SCHEMA!r} (or the legacy {SCHEMA_V2!r} / {SCHEMA_V1!r})."
            )
        joints = tuple(
            JointCheckup(
                joint=j["joint"],
                servo_id=int(j["servo_id"]),
                n_samples=int(j["n_samples"]),
                damping=j["damping"],
                friction=j["friction"],
                damping_ci95=tuple(j["damping_ci95"]) if j["damping_ci95"] else None,
                damping_total_ci95=(
                    tuple(j["damping_total_ci95"])
                    if j.get("damping_total_ci95")
                    else None
                ),
                damping_bias_fraction=j.get("damping_bias_fraction"),
                pose_feature=j.get("pose_feature"),
                pose_feature_name=j.get("pose_feature_name", ""),
                friction_ci95=tuple(j["friction_ci95"]) if j["friction_ci95"] else None,
                held=tuple(j["held"]),
                held_out=dict(j["held_out"]),
                K_t=float(j["K_t"]),
                load_offset=float(j["load_offset"]),
                response_ratio=j["response_ratio"],
                telemetry_hash=j["telemetry_hash"],
            )
            for j in d["joints"]
        )
        battery = BatteryCheckup(**d["battery"]) if d.get("battery") else None
        gates = tuple(Gate(**g) for g in d["gates"])
        return cls(
            schema=d["schema"],
            serial=d["serial"],
            robot=d["robot"],
            created_utc=d["created_utc"],
            engine_version=d["engine_version"],
            baseline=d["baseline"],
            joints=joints,
            battery=battery,
            gates=gates,
            score={k: int(v) for k, v in d["score"].items()},
            score_formula=d["score_formula"],
            attestation_handle=d["attestation_handle"],
            protocol=(
                CaptureProtocol.from_dict(d["protocol"]) if d.get("protocol") else None
            ),
        )

    def summary(self) -> str:
        """Terminal-friendly verdict table."""
        lines = [
            f"checkup {self.serial} ({self.robot}) — score "
            f"{self.score['total']}/100 — {'PASS' if self.passed else 'FAIL'}",
            f"  baseline: {self.baseline}",
        ]
        for j in self.joints:
            red = j.held_out.get("reduction", 0.0)
            b = "held" if j.damping is None else f"{j.damping:.4f}"
            tc = "held" if j.friction is None else f"{j.friction:.4f}"
            lines.append(
                f"  {j.joint:16s} b={b:>8s}  tau_c={tc:>8s}  "
                f"held-out {red:+.1%} vs frictionless"
            )
        if self.battery:
            lines.append(
                f"  battery          V_oc={self.battery.V_oc:.3f} V  "
                f"R_int={self.battery.R_int * 1e3:.1f} mΩ  "
                f"margin {self.battery.margin_v:+.2f} V vs "
                f"{self.battery.bus_floor_v:.2f} V floor"
            )
        worst = [g for g in self.gates if g.level != "pass"]
        for g in worst:
            mark = "FAIL" if g.level == "fail" else "warn"
            lines.append(f"  [{mark}] {g.scope}: {g.detail}")
        lines.append(f"  attestation: {self.attestation_handle}")
        return "\n".join(lines)


def _interval(res: Any, key: str) -> tuple[float, float] | None:
    """95% Wald interval for one fitted key, or None when not fit/estimated."""
    iv = res.interval(0.95)
    if iv is None or key not in iv:
        return None
    lo, hi = iv[key]
    return (float(lo), float(hi))


def _commanded_peak_to_peak(cap_doc: dict) -> float | None:
    """The commanded peak-to-peak motion of a capture, in radians.

    Prefers the **recorded command** (``goal_ticks``), which is what the servo
    was actually told to do — no waveform bookkeeping, correct for every
    capture generation, and immune to the amplitude having been derated on
    device. Falls back to the waveform model when the channel is absent, and
    that fallback is version-aware: the same ``amp_ticks`` commands a 2.69×
    swing under the legacy ``√(2n)`` normalization and 1.77× under the
    true-peak one (see
    :func:`jaxterity.checkup.stability.multisine_peak_to_peak`). Assuming a 2×
    swing — as this did before — inflated every response ratio by ~34% on
    legacy captures and would have deflated it on new ones.
    """
    cal = cap_doc.get("calibration") or {}
    tpr = cal.get("ticks_per_rad")
    if not tpr:
        return None
    goal = cap_doc.get("goal_ticks")
    if goal:
        commanded = float(np.ptp(np.asarray(goal, dtype=float))) / float(tpr)
        if commanded > 0.0:
            return commanded
    exc = cap_doc.get("excitation") or {}
    amp = exc.get("amp_ticks")
    if not amp:
        return None
    kind = str(exc.get("kind") or MULTISINE_KIND_V1)
    duration = float(exc.get("duration_s") or 0.0)
    if duration <= 0.0:
        return None
    try:
        scale = multisine_peak_to_peak(duration, kind)
    except CalibrationError:
        return None  # an excitation this module does not model; no gate rather
        # than a wrong one
    expected = scale * float(amp) / float(tpr)
    return expected if expected > 0.0 else None


def _response_ratio(cap_doc: dict, q: np.ndarray) -> float | None:
    """Measured over commanded peak-to-peak motion; None when meta is missing."""
    commanded = _commanded_peak_to_peak(cap_doc)
    if commanded is None:
        return None
    return float(np.ptp(q) / commanded)


def _joint_gates(j: JointCheckup, reference_damping: float | None) -> list[Gate]:
    """The per-joint gates: fit / held-out / confidence / response / reference."""
    gates: list[Gate] = []
    if j.held:
        gates.append(
            Gate(
                gate="fit",
                scope=j.joint,
                level="warn",
                detail=f"channel(s) {', '.join(j.held)} came out non-positive and "
                "were held at the frictionless prior (not identifiable from this "
                "capture).",
                remedy="Re-capture with a larger amplitude or a gravity-loaded "
                "pose; see the identifiability report.",
            )
        )
    red = j.held_out.get("reduction", 0.0)
    if red < 0.0:
        level, detail = (
            "fail",
            f"fit predicts held-out torque WORSE than the frictionless model "
            f"({red:+.1%}).",
        )
    elif red < _HELD_OUT_PASS:
        level, detail = (
            "warn",
            f"held-out torque-error reduction {red:+.1%} is below "
            f"{_HELD_OUT_PASS:.0%} — the per-unit fit adds little on this joint.",
        )
    else:
        level, detail = (
            "pass",
            f"held-out torque-error reduction {red:+.1%} vs frictionless baseline.",
        )
    gates.append(
        Gate(
            gate="held-out",
            scope=j.joint,
            level=level,
            detail=detail,
            remedy=""
            if level == "pass"
            else "Re-run the capture; check the joint "
            "moves freely and the calibration offsets are current.",
        )
    )
    # Judge the *total* interval when the run measured one. Gating on the Wald
    # interval alone passed channels whose real uncertainty was ~4x larger,
    # because least squares does not model capture-length dependence.
    interval = j.damping_total_ci95 or j.damping_ci95
    if j.damping is not None and interval is not None:
        half = (interval[1] - interval[0]) / 2.0
        rel = half / j.damping if j.damping > 0 else float("inf")
        if rel > _CONFIDENCE_REL_SE * 1.96 * 2:  # ~rel SE > threshold
            share = (
                ""
                if j.damping_bias_fraction is None
                else f", of which {j.damping_bias_fraction:.0%} is capture-length "
                "dependence rather than fit noise"
            )
            gates.append(
                Gate(
                    gate="confidence",
                    scope=j.joint,
                    level="warn",
                    detail=f"damping CI half-width is {rel:.0%} of the value{share}"
                    " — weakly identified; treat as indicative.",
                    remedy="Longer capture or richer excitation band."
                    if not j.damping_bias_fraction
                    else "Capture longer — a tighter least-squares interval alone "
                    "will not close a length-dependent one.",
                )
            )

    if j.response_ratio is not None:
        lo, hi = _RESPONSE_RATIO_BAND
        if not (lo <= j.response_ratio <= hi):
            gates.append(
                Gate(
                    gate="response",
                    scope=j.joint,
                    level="warn",
                    detail=f"measured motion is {j.response_ratio:.2f}× the "
                    f"commanded excitation (outside [{lo}, {hi}]) — possible "
                    "dead/stalled joint, slipping horn, or stale calibration.",
                    remedy="Check the servo horn and wiring; re-run `jaxterity setup`.",
                )
            )
    if j.damping is not None and reference_damping and reference_damping > 0:
        ratio = j.damping / reference_damping
        lo, hi = _REFERENCE_RATIO_BAND
        if not (lo <= ratio <= hi):
            gates.append(
                Gate(
                    gate="reference",
                    scope=j.joint,
                    level="warn",
                    detail=f"fitted damping is {ratio:.2f}× the committed "
                    "reference unit's fit (a single reference unit, not a fleet "
                    "distribution) — your unit may genuinely differ.",
                    remedy="Nothing to fix if the held-out gate passes; the fit "
                    "IS your unit.",
                )
            )
    return gates


def _composition_gate(joints: list[JointCheckup]) -> list[Gate]:
    """One unit-scoped warning when damping intervals are length-dominated.

    Deliberately *not* per joint. On the reference unit every one of the six
    channels came out 79-97 % capture-length dependence, so a per-joint gate said
    the same thing six times — and it was not six joint faults, it was one
    property of the capture protocol: 16 s at these tones does not pin damping
    well enough for the least-squares interval to be the whole story. A gate that
    fires on every channel of a healthy robot stops carrying information, so this
    one fires once and says how many channels it covers.
    """
    measured = [
        (j, j.damping_bias_fraction)
        for j in joints
        if j.damping_bias_fraction is not None
    ]
    dominated = [(j, share) for j, share in measured if share >= 0.5]
    if not measured or len(dominated) * 2 < len(measured):
        return [
            Gate(
                gate="uncertainty_composition",
                scope=j.joint,
                level="warn",
                detail=(
                    f"{share:.0%} of damping's reported interval "
                    "is capture-length dependence, not fit noise."
                ),
                remedy="Quote the total interval, and do not compare this "
                "channel against a fit from a different capture length.",
            )
            for j, share in dominated
        ]
    worst, worst_share = max(dominated, key=lambda pair: pair[1])
    return [
        Gate(
            gate="uncertainty_composition",
            scope="unit",
            level="warn",
            detail=(
                f"{len(dominated)} of {len(measured)} damping channels have "
                "intervals dominated by capture-length dependence rather than fit "
                f"noise (worst: {worst.joint} at "
                f"{worst_share:.0%}). This is a property of the "
                "capture protocol, not of the joints."
            ),
            remedy="Capture longer before comparing these parameters across "
            "runs; a tighter least-squares interval will not close it.",
        )
    ]


def _battery_gates(b: BatteryCheckup) -> list[Gate]:
    """Battery fit-quality and brownout-margin gates."""
    gates: list[Gate] = []
    if b.r_squared < _BATTERY_R2_MIN:
        gates.append(
            Gate(
                gate="battery-fit",
                scope="battery",
                level="warn",
                detail=f"Thévenin fit r²={b.r_squared:.2f} is below "
                f"{_BATTERY_R2_MIN} — sag estimate is noisy.",
                remedy="Re-run the aggregate load capture with more servos moving.",
            )
        )
    if b.margin_v < 0.0:
        level = "fail"
        detail = (
            f"predicted bus minimum {b.predicted_min_v:.2f} V is BELOW the "
            f"{b.bus_floor_v:.2f} V floor at the observed peak load "
            f"({b.margin_v:+.2f} V margin)."
        )
        remedy = "Charge or replace the pack; avoid sustained multi-joint load."
    elif b.margin_v < _BATTERY_MARGIN_PASS_V:
        level = "warn"
        detail = (
            f"brownout margin is only {b.margin_v:+.2f} V at observed peak load "
            f"(floor {b.bus_floor_v:.2f} V)."
        )
        remedy = "Charge before long sessions; watch aggregate load."
    else:
        level = "pass"
        detail = (
            f"predicted bus minimum {b.predicted_min_v:.2f} V clears the "
            f"{b.bus_floor_v:.2f} V floor by {b.margin_v:+.2f} V."
        )
        remedy = ""
    gates.append(
        Gate(
            gate="battery-margin",
            scope="battery",
            level=level,
            detail=detail,
            remedy=remedy,
        )
    )
    return gates


def _score(
    joints: tuple[JointCheckup, ...], battery: BatteryCheckup | None
) -> dict[str, int]:
    """Fidelity score v1 — documented, bounded components, equal-weight mean.

    - ``dynamics``: mean held-out reduction, clipped to [0, 0.5] and scaled so
      50% reduction = 100.
    - ``identification``: fraction of the 2·n_joints friction channels that
      were actually bound (not held).
    - ``actuation``: fraction of joints whose response ratio is inside the
      dead-joint band (joints without the metadata count as passing).
    - ``power`` (only when a battery trace was fit): brownout margin clipped
      to [0, 0.5 V] scaled so 0.5 V = 100.

    ``total`` is the rounded mean of the available components. This is a
    triage number for comparing runs of the *same* unit over time; it is not
    calibrated across robots.
    """
    n = len(joints)
    dyn = (
        float(
            np.mean(
                [np.clip(j.held_out.get("reduction", 0.0), 0.0, 0.5) for j in joints]
            )
        )
        / 0.5
    )
    bound = sum((j.damping is not None) + (j.friction is not None) for j in joints)
    ident = bound / (2.0 * n)
    ok_resp = sum(
        1
        for j in joints
        if j.response_ratio is None
        or _RESPONSE_RATIO_BAND[0] <= j.response_ratio <= _RESPONSE_RATIO_BAND[1]
    )
    act = ok_resp / n
    parts = {
        "dynamics": round(100 * dyn),
        "identification": round(100 * ident),
        "actuation": round(100 * act),
    }
    if battery is not None:
        parts["power"] = round(100 * float(np.clip(battery.margin_v / 0.5, 0.0, 1.0)))
    parts["total"] = round(float(np.mean(list(parts.values()))))
    return parts


#: Joints whose friction load is set by *inertia* rather than gravity torque —
#: the yaw axes, where gravity torque about a vertical axis is identically zero.
#: Everything else is scored against the gravity torque it holds. This is the
#: per-joint physics from :mod:`jaxterity.calibrate.pose_conditioned`, applied.
_YAW_JOINTS = ("shoulder_pan",)


def _pose_feature(robot, calibration, joint: str, pose_ticks):
    """``(value, name)`` of the pose variable this joint's friction depends on."""
    from ..calibrate.pose_conditioned import (
        axis_inertia_feature,
        gravity_torque_feature,
    )
    from ..calibrate.types import JointCalibration

    if calibration is None or not pose_ticks:
        return None, ""
    doc = (
        calibration
        if isinstance(calibration, dict)
        else json.loads(Path(calibration).read_text())
    )
    fields = JointCalibration.__dataclass_fields__
    rows = [
        JointCalibration(**{k: v for k, v in c.items() if k in fields})
        for c in doc.get("calibrations", [])
    ]
    if not rows:
        return None, ""
    try:
        if joint in _YAW_JOINTS:
            return float(axis_inertia_feature(robot, rows, joint)(pose_ticks)), (
                "axis_inertia_kg_m2"
            )
        return float(gravity_torque_feature(robot, rows, joint)(pose_ticks)), (
            "gravity_torque_Nm"
        )
    except Exception:
        # A model/name mismatch must not fail a checkup: the feature is context
        # for a later comparison, not a verdict on this run.
        return None, ""


def _uncertainty_fields(robot, capture, result, enabled: bool) -> dict:
    """Total-uncertainty fields for one joint, or empty when not measured.

    Failure to measure the ladder is *not* fatal: a short capture that cannot be
    sub-windowed still deserves a report, it just does not get the extra term.
    The fields stay ``None`` so the omission is visible rather than being papered
    over with the Wald interval under a name that promises more.
    """
    if not enabled:
        return {}
    from .probe import fit_stability
    from .uncertainty import total_uncertainty as _total

    channel = f"{capture.joint}.damping"
    if channel not in result.fitted:
        return {}
    try:
        entries = fit_stability(robot, capture)
        entry = next(e for e in entries if e.channel == channel)
        combined = _total(entry)
    except (CalibrationError, StopIteration):
        return {}
    return {
        "damping_total_ci95": combined.ci95,
        "damping_bias_fraction": combined.bias_fraction,
    }


def run_checkup(
    capture_paths: list[str | Path],
    *,
    serial: str,
    robot_name: str = "so100",
    calibration: str | Path | None = None,
    battery_trace: str | Path | None = None,
    bus_floor_v: float = 11.0,
    now: str | None = None,
    total_uncertainty: bool = True,
) -> CheckupReport:
    """Run the full per-unit checkup from on-device capture files.

    Args
    ----
    capture_paths : list[str | Path]
        Per-joint commanded-excitation capture JSONs
        (schema ``lekiwi-commanded-capture-v1``), one per excited joint.
    serial : str
        The unit's serial — keys the run store and the report.
    robot_name : str
        Zoo robot supplying the baseline model (default ``"so100"``).
    calibration : str | Path | None
        Full per-joint calibration JSON for pose conversion (recommended).
    battery_trace : str | Path | None
        Optional aggregate-load capture (schema ``lekiwi-load-capture-v1``)
        for the Thévenin sag fit and brownout-margin gate.
    bus_floor_v : float
        Operator-declared brownout floor, V (servo reset threshold).
    total_uncertainty : bool
        Measure each damping channel's capture-length dependence by re-fitting on
        sub-windows, and report the combined interval
        (:func:`jaxterity.checkup.total_uncertainty`). On by default because the
        Wald interval alone was measured to understate by ~4x, and an interval a
        trivial re-fit escapes is worse than none. Costs a handful of extra fits
        per joint; turn it off only to reproduce a v2 report.
    now : str | None
        ISO-8601 UTC timestamp override (for reproducible tests); defaults to
        the current time.

    Returns
    -------
    CheckupReport
        The report card. ``report.passed`` is False iff any gate failed.

    Raises
    ------
    CalibrationError
        If no captures are given or a capture cannot be loaded/fit.
    """
    if not capture_paths:
        raise CalibrationError("run_checkup: no capture files given.")
    # Pin the method before fitting anything: a run whose protocol cannot be
    # read is a run that must not be compared to others (see
    # jaxterity.checkup.protocol).
    protocol = capture_protocol(list(capture_paths), battery_trace=battery_trace)
    robot = zoo.load(robot_name)
    try:
        reference = zoo.calibrated(robot_name)
    except Exception:
        reference = None

    joints: list[JointCheckup] = []
    gates: list[Gate] = []
    calibrated_robot = robot
    cal_arg = None if calibration is None else str(calibration)
    for path in capture_paths:
        cap_doc = json.loads(Path(path).read_text())
        cap = load_commanded_capture(cap_doc, robot, calibration=cal_arg)
        res = identify_closed_loop(robot, cap)
        r = res.residuals or {}
        held = tuple(
            name
            for name, key in (
                ("damping", f"{cap.joint}.damping"),
                ("friction", f"{cap.joint}.friction"),
            )
            if key not in res.fitted
        )
        j = JointCheckup(
            joint=cap.joint,
            servo_id=cap.servo_id,
            n_samples=cap.n_samples,
            damping=(
                float(res.fitted[f"{cap.joint}.damping"])
                if f"{cap.joint}.damping" in res.fitted
                else None
            ),
            friction=(
                float(res.fitted[f"{cap.joint}.friction"])
                if f"{cap.joint}.friction" in res.fitted
                else None
            ),
            damping_ci95=_interval(res, f"{cap.joint}.damping"),
            friction_ci95=_interval(res, f"{cap.joint}.friction"),
            held=held,
            held_out={
                "rms_baseline": float(r.get("held_out_rms_baseline", 0.0)),
                "rms_fitted": float(r.get("held_out_rms_fitted", 0.0)),
                "reduction": float(r.get("held_out_reduction", 0.0)),
            },
            K_t=float(r.get("K_t", 0.0)),
            load_offset=float(r.get("load_offset", 0.0)),
            response_ratio=_response_ratio(cap_doc, cap.q),
            telemetry_hash=res.telemetry_hash,
            **_uncertainty_fields(robot, cap, res, total_uncertainty),
            **dict(
                zip(
                    ("pose_feature", "pose_feature_name"),
                    _pose_feature(
                        robot,
                        calibration,
                        cap.joint,
                        tuple(
                            (int(sid), int(ticks))
                            for sid, ticks in sorted(
                                (cap_doc.get("pose_ticks") or {}).items(),
                                key=lambda kv: int(kv[0]),
                            )
                        ),
                    ),
                    strict=True,
                )
            ),
        )
        joints.append(j)
        ref_damping = None
        if reference is not None:
            ref_joint = reference.joint_map.get(cap.joint)
            if ref_joint is not None:
                ref_damping = float(ref_joint.damping)
        gates.extend(_joint_gates(j, ref_damping))
        calibrated_robot = res.apply_to(calibrated_robot)
    gates.extend(_composition_gate(joints))

    battery: BatteryCheckup | None = None
    if battery_trace is not None:
        trace = load_bus_load_trace(str(battery_trace))
        bres = fit_battery_sag(trace)
        br = bres.residuals or {}
        v_oc = float(bres.fitted["V_oc"])
        r_int = float(bres.fitted["R_int"])
        peak = float(br.get("peak_current_a", 0.0))
        predicted_min = v_oc - r_int * peak
        battery = BatteryCheckup(
            V_oc=v_oc,
            R_int=r_int,
            r_squared=float(br.get("r_squared", 0.0)),
            sag_v=float(br.get("sag_v", 0.0)),
            peak_current_a=peak,
            bus_floor_v=float(bus_floor_v),
            predicted_min_v=predicted_min,
            margin_v=predicted_min - float(bus_floor_v),
            telemetry_hash=bres.telemetry_hash,
        )
        gates.extend(_battery_gates(battery))

    from .. import __version__

    return CheckupReport(
        schema=SCHEMA,
        serial=serial,
        robot=robot_name,
        created_utc=now or datetime.now(timezone.utc).isoformat(timespec="seconds"),
        engine_version=__version__,
        baseline=f"{robot_name} canonical URDF (frictionless)",
        joints=tuple(joints),
        battery=battery,
        gates=tuple(gates),
        score=_score(tuple(joints), battery),
        score_formula=SCORE_FORMULA_VERSION,
        attestation_handle=calibrated_robot.attestation_handle,
        protocol=protocol,
    )


__all__ = [
    "SCHEMA",
    "Gate",
    "JointCheckup",
    "BatteryCheckup",
    "CheckupReport",
    "run_checkup",
]
