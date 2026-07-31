# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Lost motion from a slow reversal sweep, on a single-encoder servo.

Backlash is a headline wear mechanism for geared hobby servos and the checkup
measured none of it. :mod:`jaxterity.actuators.backlash` has the model
(:class:`~jaxterity.actuators.backlash.PositionalBacklash`,
:func:`~jaxterity.actuators.backlash.lost_motion`) but its fit wants
**dual-encoder** telemetry — motor-side *and* joint-side angle — so that
``Δθ = θ_motor − θ_joint`` is observable. A Feetech STS/SMS servo has one
encoder, on the output. There is no second angle to difference, so that fit
cannot run on this robot, and pretending otherwise would produce a confident
number from a quantity the hardware never measured.

What a single encoder *can* see is the **command-to-measurement hysteresis
loop**: sweep the goal position slowly up through a range and slowly back down,
and the measured position does not retrace its path. The vertical width of that
loop is this module's measurement.

What that width is, precisely
-----------------------------
It is **total lost motion referred to the output**, and it is an *upper bound* on
gear play, because three things contribute and one encoder cannot separate them:

1. **Gear backlash** — the thing of interest.
2. **The servo's own position-loop deadband** — the controller stops correcting
   once the error is small enough. This is firmware, not wear.
3. **Compliance** — shaft, gear-tooth and horn deflection under the load needed
   to move at all.

So a growing width over a unit's life is real evidence of wear (the other two
terms are constant), while the absolute value is not a gear-play number. This
module therefore reports ``lost_motion_rad`` and refuses to call it backlash;
:attr:`ReversalFit.play_rad` is the compliant-mesh decomposition, and its
docstring says what it still cannot separate.

Separating (1) from (2) needs a second angle reference — an external encoder on
the output, or a dial indicator and a human. That is bench work, and it is the
honest prerequisite for a gear-play claim.

Protocol requirements, both enforced
------------------------------------
* **Both directions.** A single-direction sweep has no loop and therefore no
  width; :func:`reversal_lost_motion` refuses it by joint name rather than
  returning a small number from one branch's noise.
* **Slowly.** The sweep must be quasi-static, or the loop width picks up velocity
  lag from the position loop and inertia — which is a *dynamic* effect that would
  inflate the number without any wear. The capture script bounds the rate; this
  module records what rate it ran at.

Units
-----
Lost motion in **rad** at the joint (converted from ticks via the calibration),
with the raw tick width kept alongside because that is what was measured. The
load channel stays in raw duty units — see :attr:`ReversalFit.k_mesh_proxy`.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import numpy as np

from ..errors import CalibrationError

BACKLASH_CAPTURE_SCHEMA = "lekiwi-backlash-capture-v1"

#: Fraction of each sweep trimmed from both ends before the loop is measured.
#: The turnarounds are where the direction reverses, so the branches meet there
#: and the width collapses toward zero — including them would dilute the median
#: with samples that are not on either branch.
_TURNAROUND_TRIM = 0.15

#: Minimum samples per branch after trimming. Below this the median gap is not a
#: measurement, it is a couple of readings.
_MIN_BRANCH_SAMPLES = 20

#: Load gate for the engaged-sample fit, as a fraction of the sweep's peak |load|.
#: Mirrors the actuator module's ``_ENGAGED_FRACTION`` so the two agree on what
#: "engaged" means.
_ENGAGED_FRACTION = 0.25

#: Bootstrap resamples for the lost-motion interval. 2000 is enough for a stable
#: 95 % percentile interval and cheap on a few hundred gaps.
_BOOTSTRAP_DRAWS = 2000

#: Percentile bounds of the reported interval.
_CI_PERCENTILES = (2.5, 97.5)


@dataclasses.dataclass(frozen=True)
class LostMotion:
    """One joint's command-to-measurement hysteresis width.

    Attributes
    ----------
    joint : str
        Joint name.
    servo_id : int
        Bus id.
    lost_motion_ticks : float
        Median vertical width of the loop, raw encoder ticks — the measurement.
    lost_motion_rad : float
        The same width in radians at the joint.
    ci95_rad : tuple[float, float]
        Percentile bootstrap interval on the median width, rad. Resampled over
        the per-goal gaps, so it reports how consistent the loop width is along
        the sweep — not the encoder's noise, which is far smaller.
    n_gaps : int
        Number of goal positions at which the two branches were compared.
    quantization_limited : bool
        Whether the bootstrap interval is no wider than one encoder tick. On
        LK-0001 every joint's lost motion came out 8-12 ticks, and the per-goal
        gaps are integers, so resampling them returns the same median almost
        every time and three of five joints reported a **zero-width** interval.
        That is not certainty, it is the encoder: the honest resolution is
        ±half a tick. When this is true, read ``ci95_rad`` as "no finer than the
        quantum" rather than as a measurement of spread.
    sweep_rate_ticks_per_s : float
        Rate the sweep ran at, carried through so a reader can see whether the
        quasi-static assumption held.
    """

    joint: str
    servo_id: int
    lost_motion_ticks: float
    lost_motion_rad: float
    ci95_rad: tuple[float, float]
    n_gaps: int
    sweep_rate_ticks_per_s: float
    quantization_limited: bool = False

    @property
    def relative_ci(self) -> float:
        """Interval half-width relative to the value (NaN at zero width)."""
        if self.lost_motion_rad == 0:
            return float("nan")
        half = (self.ci95_rad[1] - self.ci95_rad[0]) / 2.0
        return half / abs(self.lost_motion_rad)

    def to_dict(self) -> dict[str, Any]:
        """Plain-JSON form."""
        d = dataclasses.asdict(self)
        d["ci95_rad"] = list(self.ci95_rad)
        d["relative_ci"] = self.relative_ci
        return d


@dataclasses.dataclass(frozen=True)
class ReversalFit:
    """The compliant-mesh decomposition of one joint's reversal loop.

    Fits ``load = k·Δ − k·play·sign(Δ)`` over the engaged samples, where ``Δ`` is
    the goal-minus-measured error — the same piecewise-linear form as
    :func:`jaxterity.actuators.backlash._fit_backlash`, but with the servo's load
    register standing in for joint torque and one encoder standing in for two.

    Attributes
    ----------
    joint : str
        Joint name.
    play_rad : float
        Half-width of the fitted dead zone, rad. **Still not gear play**: it is
        the dead zone of the whole command-to-output path, so the servo's
        position-loop deadband is inside it. It is a decomposition of the same
        measurement, not a new measurement.
    k_mesh_proxy : float
        Fitted stiffness in **load-register units per radian**, not N·m/rad. The
        load register is a duty proxy whose torque scale is not calibrated here,
        so converting would invent a constant. Comparable across captures of the
        same unit; not comparable to a datasheet.
    r_squared : float
        Fit quality over the engaged samples. A poor value means the loop is not
        the piecewise-linear shape this model assumes — read
        :class:`LostMotion` instead and distrust ``play_rad``.
    n_engaged : int
        Samples above the load gate that the fit used.
    """

    joint: str
    play_rad: float
    k_mesh_proxy: float
    r_squared: float
    n_engaged: int

    def to_dict(self) -> dict[str, Any]:
        """Plain-JSON form."""
        return dataclasses.asdict(self)


def load_backlash_capture(source: str | Path | dict) -> dict:
    """Parse a reversal capture, validating its schema.

    Args
    ----
    source : str | Path | dict
        Capture path or parsed document.

    Returns
    -------
    dict
        The document unchanged.

    Raises
    ------
    CalibrationError
        If the schema is unrecognized or the capture has no samples.
    """
    doc = source if isinstance(source, dict) else json.loads(Path(source).read_text())
    if doc.get("schema") != BACKLASH_CAPTURE_SCHEMA:
        raise CalibrationError(
            f"load_backlash_capture: unrecognized schema {doc.get('schema')!r}; "
            f"expected {BACKLASH_CAPTURE_SCHEMA!r}."
        )
    if not doc.get("t"):
        raise CalibrationError("load_backlash_capture: the capture has no samples.")
    return doc


def _ticks_per_rad(calibration: dict | None, servo_id: int) -> float:
    if not calibration:
        return float("nan")
    for row in calibration.get("calibrations", []):
        if int(row.get("servo_id", -1)) == int(servo_id):
            return float(row["ticks_per_rad"])
    return float("nan")


def _branch(goal: np.ndarray, pos: np.ndarray, trim: float):
    """Trim a branch's turnarounds and sort it by goal."""
    n = goal.size
    cut = int(trim * n)
    if n - 2 * cut < _MIN_BRANCH_SAMPLES:
        return None
    sliced_goal = goal[cut : n - cut]
    sliced_pos = pos[cut : n - cut]
    order = np.argsort(sliced_goal)
    return sliced_goal[order], sliced_pos[order]


def reversal_lost_motion(
    source: str | Path | dict,
    *,
    calibration: str | Path | dict | None = None,
    trim: float = _TURNAROUND_TRIM,
) -> tuple[LostMotion, ...]:
    """Measure each joint's command-to-measurement hysteresis width.

    For each joint, the forward and reverse branches are trimmed of their
    turnarounds, the reverse branch is interpolated onto the forward branch's
    goal positions, and the reported width is the **median** signed gap between
    them. Median rather than mean because a single stiction event on one branch
    would otherwise set the number, and median rather than peak-to-peak because
    the width is a property of the whole loop, not of its widest point.

    Args
    ----
    source : str | Path | dict
        A ``lekiwi-backlash-capture-v1`` capture.
    calibration : str | Path | dict | None
        The unit's calibration, for ticks → rad. Without it the radian fields
        come back NaN and the tick width is still reported — a width in ticks is
        a real measurement, just not in physical units.
    trim : float
        Fraction of each branch dropped from both ends.

    Returns
    -------
    tuple[LostMotion, ...]
        One per joint, in servo-id order.

    Raises
    ------
    CalibrationError
        If any joint's capture covers only one sweep direction (named), or a
        branch is too short after trimming to support a median.
    """
    doc = load_backlash_capture(source)
    cal = (
        calibration
        if isinstance(calibration, dict) or calibration is None
        else json.loads(Path(calibration).read_text())
    )
    sid_all = np.asarray(doc["servo_id"], dtype=int)
    phase_all = np.asarray(doc["phase"], dtype=object)
    goal_all = np.asarray(doc["goal_ticks"], dtype=float)
    pos_all = np.asarray(
        [np.nan if v is None else v for v in doc["pos_ticks"]], dtype=float
    )
    time_all = np.asarray(doc["t"], dtype=float)
    joints = doc.get("joints", {})

    out: list[LostMotion] = []
    rng = np.random.default_rng(0)
    for sid in doc["servo_ids"]:
        sid = int(sid)
        name = str(joints.get(str(sid), "")) or f"servo{sid}"
        mine = sid_all == sid
        directions = {str(p) for p in np.unique(phase_all[mine])}
        missing = {"forward", "reverse"} - directions
        if missing:
            raise CalibrationError(
                f"reversal_lost_motion: {name} (servo {sid}) has no "
                f"{'/'.join(sorted(missing))} sweep — only {sorted(directions)}. "
                "A single-direction sweep has no hysteresis loop, so it has no "
                "width to measure: the number it would produce is one branch's "
                "noise. Fix: capture the full up-and-back sweep."
            )

        branches = {}
        for direction in ("forward", "reverse"):
            selected = mine & (phase_all == direction)
            valid = selected & np.isfinite(pos_all)
            branch = _branch(goal_all[valid], pos_all[valid], trim)
            if branch is None:
                raise CalibrationError(
                    f"reversal_lost_motion: {name}'s {direction} sweep has fewer "
                    f"than {_MIN_BRANCH_SAMPLES} samples after trimming "
                    f"{trim:.0%} off each end ({int(valid.sum())} valid samples "
                    "in total). Fix: sweep longer or sample faster."
                )
            branches[direction] = branch

        fwd_goal, fwd_pos = branches["forward"]
        rev_goal, rev_pos = branches["reverse"]
        # Compare only where the branches overlap in goal: outside that, one
        # branch would be extrapolated and the "gap" would be an artifact.
        lo, hi = (
            max(fwd_goal.min(), rev_goal.min()),
            min(fwd_goal.max(), rev_goal.max()),
        )
        inside = (fwd_goal >= lo) & (fwd_goal <= hi)
        if int(inside.sum()) < _MIN_BRANCH_SAMPLES:
            raise CalibrationError(
                f"reversal_lost_motion: {name}'s two sweeps overlap over only "
                f"{int(inside.sum())} goal positions, fewer than the "
                f"{_MIN_BRANCH_SAMPLES} needed. Fix: sweep both directions over "
                "the same range."
            )
        gaps = np.abs(fwd_pos[inside] - np.interp(fwd_goal[inside], rev_goal, rev_pos))
        width_ticks = float(np.median(gaps))

        draws = rng.choice(gaps, size=(_BOOTSTRAP_DRAWS, gaps.size), replace=True)
        medians = np.median(draws, axis=1)
        lo_t, hi_t = (float(v) for v in np.percentile(medians, _CI_PERCENTILES))
        # Integer gaps make the bootstrap blind below one tick; widen to the
        # quantum rather than report a zero-width interval as if it were
        # certainty. Half a tick either way is the encoder's actual resolution.
        limited = (hi_t - lo_t) <= 1.0
        if limited:
            lo_t, hi_t = width_ticks - 0.5, width_ticks + 0.5

        scale = _ticks_per_rad(cal, sid)
        mine_t = time_all[mine]
        span = float(mine_t[-1] - mine_t[0]) if mine_t.size > 1 else 0.0
        rate = float(np.ptp(goal_all[mine]) * 2.0 / span) if span > 0 else float("nan")
        out.append(
            LostMotion(
                joint=name,
                servo_id=sid,
                lost_motion_ticks=width_ticks,
                lost_motion_rad=width_ticks / scale,
                ci95_rad=(lo_t / scale, hi_t / scale),
                n_gaps=int(inside.sum()),
                sweep_rate_ticks_per_s=rate,
                quantization_limited=bool(limited),
            )
        )
    return tuple(out)


def fit_reversal(
    source: str | Path | dict,
    *,
    calibration: str | Path | dict | None = None,
    engaged_fraction: float = _ENGAGED_FRACTION,
) -> tuple[ReversalFit, ...]:
    """Decompose each joint's loop into a dead zone and a stiffness.

    Run :func:`reversal_lost_motion` first and compare: this fit is the
    model-dependent number, and the stiffness-free width is what tells you
    whether to believe it. A ``play_rad`` that disagrees with half the measured
    width by more than a factor of about two means the loop is not the shape this
    model assumes.

    Args
    ----
    source : str | Path | dict
        A reversal capture.
    calibration : str | Path | dict | None
        For ticks → rad. Without it ``play_rad`` is NaN.
    engaged_fraction : float
        Load gate as a fraction of the joint's peak |load|. Samples below it are
        inside the dead zone and are excluded: including them drags the slope
        toward zero and shrinks the apparent play.

    Returns
    -------
    tuple[ReversalFit, ...]
        One per joint, in servo-id order.

    Raises
    ------
    CalibrationError
        If a joint has too few engaged samples, or the capture is missing the
        load channel the fit needs.
    """
    doc = load_backlash_capture(source)
    if not doc.get("load_raw"):
        raise CalibrationError(
            "fit_reversal: the capture has no load channel, so there is no "
            "torque proxy to fit a stiffness against. Use reversal_lost_motion, "
            "which needs only position."
        )
    cal = (
        calibration
        if isinstance(calibration, dict) or calibration is None
        else json.loads(Path(calibration).read_text())
    )
    sid_all = np.asarray(doc["servo_id"], dtype=int)
    goal = np.asarray(doc["goal_ticks"], dtype=float)
    pos = np.asarray([np.nan if v is None else v for v in doc["pos_ticks"]], float)
    load = np.asarray([np.nan if v is None else v for v in doc["load_raw"]], float)
    joints = doc.get("joints", {})

    out: list[ReversalFit] = []
    for sid in doc["servo_ids"]:
        sid = int(sid)
        name = str(joints.get(str(sid), "")) or f"servo{sid}"
        mine = (sid_all == sid) & np.isfinite(pos) & np.isfinite(load)
        error = goal[mine] - pos[mine]
        tau = load[mine]
        peak = float(np.max(np.abs(tau))) if tau.size else 0.0
        engaged = (
            np.abs(tau) > engaged_fraction * peak if peak > 0 else np.zeros(0, bool)
        )
        if int(np.sum(engaged)) < _MIN_BRANCH_SAMPLES:
            raise CalibrationError(
                f"fit_reversal: {name} has {int(np.sum(engaged))} samples above "
                f"the {engaged_fraction:.0%} load gate, fewer than the "
                f"{_MIN_BRANCH_SAMPLES} needed. The sweep never loaded the mesh "
                "enough to see its stiffness — sweep against gravity, or over a "
                "range where the joint carries load."
            )
        x = error[engaged]
        y = tau[engaged]
        design = np.column_stack([x, -np.sign(x)])
        coef, *_ = np.linalg.lstsq(design, y, rcond=None)
        k_ticks, k_play = float(coef[0]), float(coef[1])
        residual = y - design @ coef
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r_squared = 1.0 - float(np.sum(residual**2)) / ss_tot if ss_tot > 0 else 0.0

        scale = _ticks_per_rad(cal, sid)
        play_ticks = k_play / k_ticks if k_ticks != 0 else float("nan")
        out.append(
            ReversalFit(
                joint=name,
                play_rad=play_ticks / scale,
                k_mesh_proxy=k_ticks * scale,
                r_squared=r_squared,
                n_engaged=int(np.sum(engaged)),
            )
        )
    return tuple(out)


def backlash_summary(
    widths: tuple[LostMotion, ...] | list[LostMotion],
) -> str:
    """Terminal-friendly table of measured lost motion."""
    lines = [
        f"  {'joint':16s} {'ticks':>8} {'mrad':>8} {'ci95 mrad':>18} "
        f"{'quantized':>10} {'gaps':>6} {'ticks/s':>8}"
    ]
    for item in widths:
        lo, hi = item.ci95_rad
        lines.append(
            f"  {item.joint:16s} {item.lost_motion_ticks:8.1f} "
            f"{item.lost_motion_rad * 1e3:8.2f} "
            f"{f'[{lo * 1e3:.2f}, {hi * 1e3:.2f}]':>18} "
            f"{'yes' if item.quantization_limited else 'no':>10} "
            f"{item.n_gaps:6d} {item.sweep_rate_ticks_per_s:8.1f}"
        )
    return "\n".join(lines)


__all__ = [
    "BACKLASH_CAPTURE_SCHEMA",
    "LostMotion",
    "ReversalFit",
    "backlash_summary",
    "fit_reversal",
    "load_backlash_capture",
    "reversal_lost_motion",
]
