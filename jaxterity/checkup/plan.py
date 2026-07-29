# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Build a stability-vetted checkup capture plan (the pre-checkup setup).

The T-228 v0 checkup let the on-device capture script pick its own excitation
centers (each joint's band center) from whatever pose the arm was in — which
extended a LeKiwi's arm and tipped the base (see
:mod:`jaxterity.checkup.stability`). This module is the fix: **plan first,
then move.** :func:`build_capture_plan` takes the arm model, the unit's
per-joint calibration, a declared :class:`~jaxterity.checkup.stability.BaseProfile`,
and a start pose (e.g. the bring-up choreography's home pose), and emits a
:class:`CapturePlan` in which every motion the capture scripts will command —
the bounded-speed transit into the pose, each single-joint multisine *around
the pose* (not the band center), and the synchronized battery load trace —
has been simulated and carries a ZMP margin above the profile's floor.
Excitations that fail are amplitude-derated until they pass; entries that
cannot pass even at the minimum useful amplitude are marked infeasible with
the reason, and a pose whose *static* margin is already below the floor is
rejected outright (no excitation from it can be safe).

The plan serializes to JSON in **servo ticks** (schema
``jaxterity-checkup-plan-v1``) so the dependency-free on-device scripts
(``scripts/lekiwi_commanded_capture.py --plan``) consume it without any model
or unit conversion. Honesty note: margins inherit the base profile's
fidelity — :data:`~jaxterity.checkup.stability.LEKIWI_BASE` is representative,
not measured; the pass/fail *ordering* of poses is robust, the absolute
margin is only as good as the declared geometry.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import numpy as np

from ..calibrate.active import dof_joint_names
from ..calibrate.closed_loop import LEKIWI_ARM_JOINT_BY_ID
from ..errors import CalibrationError
from ..robot import Robot
from .stability import (
    BaseProfile,
    StabilityReport,
    load_trace_trajectory,
    multisine_trajectory,
    transit_trajectory,
    zmp_margins,
)

PLAN_SCHEMA = "jaxterity-checkup-plan-v1"

#: Amplitude derate ladder: multiply by this factor until the margin passes.
_DERATE_FACTOR = 0.75
#: Smallest excitation amplitude (ticks) still worth fitting; below this the
#: entry is infeasible from the chosen pose.
_MIN_AMP_TICKS = 60
#: Default bounded transit speed (ticks/s); ~0.23 rad/s at 652 ticks/rad —
#: deliberately slower than the v0 script's 600.
_DEFAULT_TRANSIT_TICKS_PER_S = 150


@dataclasses.dataclass(frozen=True)
class PlanEntry:
    """One vetted capture: a single-joint multisine or the load trace.

    ``amp_ticks`` may be smaller than requested (derated to pass the margin);
    ``feasible`` is False when even the minimum amplitude fails, with the
    reason in ``note``. ``min_margin_m`` is the simulated worst ZMP margin of
    this entry's motion (excitation only; the shared transit is reported on
    the plan).
    """

    kind: str  # "joint" | "load"
    joints: tuple[str, ...]
    servo_ids: tuple[int, ...]
    center_ticks: dict[str, int]
    amp_ticks: int
    duration_s: float
    rate_hz: float
    min_margin_m: float
    feasible: bool
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        d["joints"] = list(self.joints)
        d["servo_ids"] = list(self.servo_ids)
        return d


@dataclasses.dataclass(frozen=True)
class CapturePlan:
    """A stability-vetted capture plan for one unit (JSON round-trippable)."""

    schema: str
    robot: str
    serial: str
    base_profile: str
    margin_floor_m: float
    pose_ticks: dict[str, int]  # servo-id (str) -> ticks, the vetted start pose
    pose_static_margin_m: float
    transit_margin_m: float
    transit_ok: bool
    transit_ticks_per_s: int
    entries: tuple[PlanEntry, ...]

    @property
    def feasible_entries(self) -> tuple[PlanEntry, ...]:
        """The entries that passed the margin floor."""
        return tuple(e for e in self.entries if e.feasible)

    def to_dict(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        d["entries"] = [e.to_dict() for e in self.entries]
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "CapturePlan":
        if d.get("schema") != PLAN_SCHEMA:
            raise CalibrationError(
                f"CapturePlan.from_dict: unrecognized schema {d.get('schema')!r}; "
                f"expected {PLAN_SCHEMA!r}."
            )
        entries = tuple(
            PlanEntry(
                kind=e["kind"],
                joints=tuple(e["joints"]),
                servo_ids=tuple(int(s) for s in e["servo_ids"]),
                center_ticks={k: int(v) for k, v in e["center_ticks"].items()},
                amp_ticks=int(e["amp_ticks"]),
                duration_s=float(e["duration_s"]),
                rate_hz=float(e["rate_hz"]),
                min_margin_m=float(e["min_margin_m"]),
                feasible=bool(e["feasible"]),
                note=e.get("note", ""),
            )
            for e in d["entries"]
        )
        return cls(
            schema=d["schema"],
            robot=d["robot"],
            serial=d["serial"],
            base_profile=d["base_profile"],
            margin_floor_m=float(d["margin_floor_m"]),
            pose_ticks={k: int(v) for k, v in d["pose_ticks"].items()},
            pose_static_margin_m=float(d["pose_static_margin_m"]),
            transit_margin_m=float(d["transit_margin_m"]),
            transit_ok=bool(d["transit_ok"]),
            transit_ticks_per_s=int(d["transit_ticks_per_s"]),
            entries=entries,
        )

    def summary(self) -> str:
        """Terminal-friendly plan summary."""
        lines = [
            f"capture plan {self.serial} ({self.robot}) on base "
            f"'{self.base_profile}' — floor {self.margin_floor_m * 1000:.0f} mm",
            f"  pose static margin {self.pose_static_margin_m * 1000:.0f} mm · "
            f"transit margin {self.transit_margin_m * 1000:.0f} mm @ "
            f"{self.transit_ticks_per_s} ticks/s"
            + (
                ""
                if self.transit_ok
                else "  [WARN: support the arm by hand "
                "while it ramps to the pose — worst-case transit is marginal]"
            ),
        ]
        for e in self.entries:
            tag = "ok  " if e.feasible else "SKIP"
            lines.append(
                f"  [{tag}] {e.kind:5s} {'+'.join(e.joints):24s} "
                f"amp {e.amp_ticks:3d} ticks  margin "
                f"{e.min_margin_m * 1000:6.1f} mm{('  — ' + e.note) if e.note else ''}"
            )
        return "\n".join(lines)


def _cal_rows(calibration: str | Path | dict) -> dict[int, dict]:
    """Per-servo calibration rows keyed by servo id."""
    doc = (
        calibration
        if isinstance(calibration, dict)
        else json.loads(Path(calibration).read_text())
    )
    return {int(c["servo_id"]): c for c in doc["calibrations"]}


def _rad_to_ticks(rad: float, row: dict) -> int:
    return int(round(row["offset_ticks"] + row["sign"] * rad * row["ticks_per_rad"]))


def _ticks_to_rad(ticks: float, row: dict) -> float:
    return row["sign"] * (ticks - row["offset_ticks"]) / row["ticks_per_rad"]


def build_capture_plan(
    robot: Robot,
    calibration: str | Path | dict,
    *,
    serial: str,
    robot_name: str = "so100",
    base: BaseProfile,
    pose_rad: dict[str, float],
    amp_ticks: int = 200,
    duration_s: float = 16.0,
    rate_hz: float = 80.0,
    load_servo_ids: tuple[int, ...] = (1, 2, 3),
    load_amp_ticks: int = 180,
    transit_ticks_per_s: int = _DEFAULT_TRANSIT_TICKS_PER_S,
    guard_margin_ticks: int = 80,
) -> CapturePlan:
    """Build and vet the full capture plan from a declared start pose.

    Every excitation is centered on the pose (clamped into the joint's guard
    band), simulated with the exact commanded waveform, and amplitude-derated
    by :data:`_DERATE_FACTOR` steps until its worst ZMP margin clears
    ``base.margin_min_m`` (or marked infeasible below :data:`_MIN_AMP_TICKS`).
    The transit *into* the pose is validated at ``transit_ticks_per_s`` from a
    worst-case start (each joint at its band edge farthest from the pose).

    Args
    ----
    robot : Robot
        The arm model (e.g. ``zoo.load("so100")``).
    calibration : str | Path | dict
        The unit's per-joint calibration (bring-up output) — supplies the
        tick/radian mapping and guard bands.
    serial : str
        Unit serial recorded in the plan.
    robot_name : str
        Zoo name recorded in the plan.
    base : BaseProfile
        The declared base under the arm (e.g. ``LEKIWI_BASE``).
    pose_rad : dict[str, float]
        The start pose, joint name → radians (joints omitted here hold their
        calibration mid-range). Vet it before trusting it — the function does.
    amp_ticks, duration_s, rate_hz : int, float, float
        Requested per-joint excitation parameters (defaults mirror the
        committed bring-up captures).
    load_servo_ids, load_amp_ticks : tuple[int, ...], int
        The battery load-trace servos and requested amplitude.
    transit_ticks_per_s : int
        Bounded transit speed the scripts must use.
    guard_margin_ticks : int
        Stay this far inside each joint's calibrated range (mirrors the
        capture script's guard).

    Returns
    -------
    CapturePlan
        The vetted plan (entries may be derated or marked infeasible).

    Raises
    ------
    CalibrationError
        If the pose itself is statically unstable on the declared base, or a
        pose joint is unknown to the robot/calibration.
    """
    dof_names = dof_joint_names(robot)
    rows = _cal_rows(calibration)
    joint_by_id = {
        sid: name for sid, name in LEKIWI_ARM_JOINT_BY_ID.items() if sid in rows
    }
    id_by_joint = {name: sid for sid, name in joint_by_id.items()}
    unknown = set(pose_rad) - set(dof_names)
    if unknown:
        raise CalibrationError(
            f"build_capture_plan: pose names unknown joints {sorted(unknown)}; "
            f"robot DOFs are {list(dof_names)}."
        )

    # Full pose vector: declared pose where given, calibration mid-range else.
    pose_vec = np.zeros(len(dof_names))
    pose_ticks: dict[str, int] = {}
    bands: dict[str, tuple[int, int]] = {}
    for j, name in enumerate(dof_names):
        sid = id_by_joint.get(name)
        if sid is None:
            continue
        row = rows[sid]
        lo = int(row["range_min_ticks"]) + guard_margin_ticks
        hi = int(row["range_max_ticks"]) - guard_margin_ticks
        bands[name] = (lo, hi)
        if name in pose_rad:
            ticks = _rad_to_ticks(float(pose_rad[name]), row)
        else:
            ticks = (lo + hi) // 2
        ticks = max(lo, min(hi, ticks))
        pose_ticks[str(sid)] = ticks
        pose_vec[j] = _ticks_to_rad(ticks, row)

    # Gate 0 — the pose must be *statically* stable with margin to spare.
    static_q = np.tile(pose_vec, (3, 1))
    static_margin = float(zmp_margins(robot, static_q, 0.1, dof_names, base).min())
    if static_margin < base.margin_min_m:
        raise CalibrationError(
            f"build_capture_plan: the start pose is statically unstable on base "
            f"'{base.name}' (ZMP margin {static_margin * 1000:.1f} mm < floor "
            f"{base.margin_min_m * 1000:.0f} mm). Choose a tucked pose; no "
            "excitation from this pose can be safe."
        )

    # Gate 1 — the bounded-speed transit into the pose, from a worst-case
    # start (every joint at the band edge farthest from the pose).
    worst = pose_vec.copy()
    for j, name in enumerate(dof_names):
        if name not in bands:
            continue
        sid = id_by_joint[name]
        row = rows[sid]
        lo, hi = bands[name]
        p0 = pose_ticks[str(sid)]
        far = lo if abs(p0 - lo) > abs(p0 - hi) else hi
        worst[j] = _ticks_to_rad(far, row)
    ticks_per_rad = float(np.median([rows[s]["ticks_per_rad"] for s in rows]))
    speed_rad_s = transit_ticks_per_s / ticks_per_rad
    q_t, dt = transit_trajectory(worst, pose_vec, speed_rad_s=speed_rad_s)
    transit_margin = float(zmp_margins(robot, q_t, dt, dof_names, base).min())
    # A transit that *starts* in a bad pose passes through low-margin
    # territory no matter how slowly it moves — retracting must stay allowed,
    # so the transit floor is half the excitation floor, and a shortfall is a
    # loud warning (support the arm by hand), not a refusal.
    transit_ok = transit_margin >= base.margin_min_m / 2.0

    # Gate 2 — each single-joint multisine around the pose, derated to pass.
    entries: list[PlanEntry] = []
    for sid in sorted(joint_by_id):
        name = joint_by_id[sid]
        row = rows[sid]
        j = dof_names.index(name)
        lo, hi = bands[name]
        # Excite as close to the vetted pose as the guard band allows: a pose
        # may legitimately park a joint at (or just outside) its band edge —
        # nudge the center inward just enough for a useful amplitude.
        center = max(
            lo + _MIN_AMP_TICKS, min(hi - _MIN_AMP_TICKS, pose_ticks[str(sid)])
        )
        amp = min(int(amp_ticks), center - lo, hi - center)
        margin = -np.inf
        while amp >= _MIN_AMP_TICKS:
            amp_rad = amp / float(row["ticks_per_rad"])
            center_rad = _ticks_to_rad(center, row)
            q_t, dt = multisine_trajectory(
                pose_vec, j, center_rad, amp_rad, duration_s=duration_s
            )
            margin = float(zmp_margins(robot, q_t, dt, dof_names, base).min())
            if margin >= base.margin_min_m:
                break
            amp = int(amp * _DERATE_FACTOR)
        feasible = margin >= base.margin_min_m and amp >= _MIN_AMP_TICKS
        entries.append(
            PlanEntry(
                kind="joint",
                joints=(name,),
                servo_ids=(sid,),
                center_ticks={str(sid): center},
                amp_ticks=int(amp) if feasible else 0,
                duration_s=duration_s,
                rate_hz=rate_hz,
                min_margin_m=margin,
                feasible=feasible,
                note=""
                if feasible
                else "infeasible from this pose — even minimum amplitude tips",
            )
        )

    # Gate 3 — the synchronized load trace (worst-case in-phase rocking).
    load_joints = tuple(joint_by_id[s] for s in load_servo_ids if s in joint_by_id)
    load_idx = tuple(dof_names.index(n) for n in load_joints)
    amp = int(load_amp_ticks)
    margin = -np.inf
    while amp >= _MIN_AMP_TICKS:
        amps_rad = tuple(
            min(
                amp,
                pose_ticks[str(id_by_joint[n])] - bands[n][0],
                bands[n][1] - pose_ticks[str(id_by_joint[n])],
            )
            / float(rows[id_by_joint[n]]["ticks_per_rad"])
            for n in load_joints
        )
        q_t, dt = load_trace_trajectory(
            pose_vec, load_idx, amps_rad, duration_s=duration_s
        )
        margin = float(zmp_margins(robot, q_t, dt, dof_names, base).min())
        if margin >= base.margin_min_m:
            break
        amp = int(amp * _DERATE_FACTOR)
    load_feasible = margin >= base.margin_min_m and amp >= _MIN_AMP_TICKS
    entries.append(
        PlanEntry(
            kind="load",
            joints=load_joints,
            servo_ids=tuple(load_servo_ids),
            center_ticks={
                str(s): pose_ticks[str(s)]
                for s in load_servo_ids
                if str(s) in pose_ticks
            },
            amp_ticks=int(amp) if load_feasible else 0,
            duration_s=duration_s,
            rate_hz=60.0,
            min_margin_m=margin,
            feasible=load_feasible,
            note=""
            if load_feasible
            else "infeasible from this pose — in-phase trace tips even derated",
        )
    )

    return CapturePlan(
        schema=PLAN_SCHEMA,
        robot=robot_name,
        serial=serial,
        base_profile=base.name,
        margin_floor_m=base.margin_min_m,
        pose_ticks=pose_ticks,
        pose_static_margin_m=static_margin,
        transit_margin_m=transit_margin,
        transit_ok=transit_ok,
        transit_ticks_per_s=int(transit_ticks_per_s),
        entries=tuple(entries),
    )


def validate_capture_motion(
    robot: Robot,
    calibration: str | Path | dict,
    *,
    base: BaseProfile,
    pose_ticks: dict[str, int],
    excited_servo_id: int,
    center_ticks: int,
    amp_ticks: int,
    duration_s: float = 16.0,
) -> StabilityReport:
    """Vet one already-specified capture motion (the T-228 postmortem tool).

    Reconstructs exactly what ``lekiwi_commanded_capture.py`` would command —
    the arm holding ``pose_ticks`` while the excited joint rides the Schroeder
    multisine about ``center_ticks`` — and returns its ZMP report. Use it to
    check a historical or hand-written capture before (or after) it runs.
    """
    dof_names = dof_joint_names(robot)
    rows = _cal_rows(calibration)
    joint = LEKIWI_ARM_JOINT_BY_ID.get(excited_servo_id)
    if joint is None or joint not in dof_names:
        raise CalibrationError(
            f"validate_capture_motion: servo id {excited_servo_id} is not a "
            "known arm joint."
        )
    pose_vec = np.zeros(len(dof_names))
    for sid_str, ticks in pose_ticks.items():
        name = LEKIWI_ARM_JOINT_BY_ID.get(int(sid_str))
        if name in dof_names and int(sid_str) in rows:
            pose_vec[dof_names.index(name)] = _ticks_to_rad(
                float(ticks), rows[int(sid_str)]
            )
    row = rows[excited_servo_id]
    j = dof_names.index(joint)
    q_t, dt = multisine_trajectory(
        pose_vec,
        j,
        _ticks_to_rad(float(center_ticks), row),
        amp_ticks / float(row["ticks_per_rad"]),
        duration_s=duration_s,
    )
    margins = zmp_margins(robot, q_t, dt, dof_names, base)
    static_q = np.tile(pose_vec, (3, 1))
    return StabilityReport(
        name=f"{joint} multisine amp={amp_ticks}",
        min_margin_m=float(margins.min()),
        static_margin_m=float(zmp_margins(robot, static_q, 0.1, dof_names, base).min()),
        margin_floor_m=base.margin_min_m,
    )


__all__ = [
    "PLAN_SCHEMA",
    "PlanEntry",
    "CapturePlan",
    "build_capture_plan",
    "validate_capture_motion",
]
