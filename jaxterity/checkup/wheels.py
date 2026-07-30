# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Drive-wheel health from a free-spin capture — the half the checkup ignored.

The per-unit checkup fits six arm joints and says nothing about the three drive
wheels, which is half a LeKiwi's mechanism. This module reads
``scripts/lekiwi_wheel_capture.py``'s speed-staircase capture and reports what a
*free-spinning* wheel can honestly support.

Two metrics carry the verdict, and they were chosen from what the first real
capture (LK-0001, 2026-07-29, wheels on blocks) showed to be trustworthy:

* **Speed tracking** — commanded versus measured speed per step. On a healthy
  unloaded wheel this is exact: all three wheels hit ±200/400/600 dead on, in
  both directions. A wheel that cannot reach its commanded speed with nothing
  attached to it has something wrong.
* **No-load duty symmetry** — the torque a servo spends to spin nothing, per
  step, compared *across* the wheels. This needs no reference unit and no
  absolute calibration, which is why it works on day one. LK-0001's three wheels
  matched within 2 % at every step (78/140/200 duty at 200/400/600).

And one metric is deliberately **reported but not gated**: the coast-down. With
the wheels off the ground it is not rolling resistance — there is no ground
contact — it is the wheel's own internal drag. On LK-0001 all three wheels
stopped in 0.10-0.12 s, which at 50 Hz is *five samples*: the quantity is real
but its resolution is ~20 ms, far too coarse to gate a fault on. Capture at a
higher rate (or with a heavier wheel) before trusting a difference between two
of these numbers.

That reading also depends entirely on the :data:`_SPEED_QUANTUM` threshold
below. Counting a lone one-count reading as motion put two of the three wheels
at 2.98 s against the same 3.0 s window — a 25× spread that looked like one bad
bearing, was really a stopped wheel's register dithering between 0 and 50, and
contradicted the duty numbers, which had all three wheels matching within 2 %.
Extra drag raises duty. When a "fault" appears in one channel and nothing
corroborates it in another, suspect the channel.

The on-floor comparison is the useful future use: the difference between an
on-blocks coast and an on-floor coast *is* the rolling resistance
(:func:`jaxterity.physics.compliance.fit_coast_down`), with this capture as the
no-contact baseline.

Units
-----
Speeds and duties are the servo's **raw register units** throughout (signed
magnitude speed; duty in 0.1 % of full scale). No conversion to rad/s is
attempted: it would need a wheel-radius and gear-ratio calibration this capture
does not measure, and inventing one would dress a guess as a unit.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any

import numpy as np

from ..errors import CalibrationError
from .report import Gate

WHEEL_CAPTURE_SCHEMA = "lekiwi-wheel-capture-v1"

#: Fraction of each step's samples dropped from the front, so the spin-up
#: transient does not contaminate the steady-state medians.
_SETTLE_FRACTION = 0.25

#: Speed-tracking tolerance: measured must be within this fraction of commanded.
#: Loose enough for register granularity, tight enough that a wheel failing to
#: reach speed is caught — LK-0001 tracked exactly, so any real miss is large.
_TRACKING_TOLERANCE = 0.10

#: No-load duty symmetry: a wheel whose duty departs from the wheel-set median
#: by more than this fraction is flagged. 25 % against the 2 % spread the healthy
#: unit showed, so this fires on a real outlier, not on unit variation.
_DUTY_SYMMETRY_TOLERANCE = 0.25

#: Distance from the median, °C, past which a temperature sample is counted as a
#: garbled read rather than a real excursion. Wide, because it only has to
#: separate the register's wild values (up to 78 °C on a 32 °C wheel) from the
#: real few-degree rise across a 15 s staircase.
_TEMP_OUTLIER_C = 3.0

#: Quantum of the present-speed register, raw units — every reading LK-0001
#: produced was a multiple of 50. A wheel at rest dithers between 0 and one
#: quantum, so "still moving" has to mean *strictly more* than one count:
#: reading a lone 50 as motion put a stopped wheel's coast-down at 2.98 s
#: instead of a small fraction of a second.
_SPEED_QUANTUM = 50.0


@dataclasses.dataclass(frozen=True)
class WheelHealth:
    """One wheel's free-spin health summary.

    Attributes
    ----------
    servo_id : int
        Bus id.
    joint : str
        Wheel name from the calibration (cosmetic).
    tracking : dict[int, float]
        Commanded step → ``measured/commanded`` ratio (1.0 = exact). Keyed by the
        *signed* commanded speed, so forward and reverse are separate.
    duty : dict[int, float]
        Commanded step → median |duty| while holding it, raw register units.
    coast_s : float
        How long the wheel kept reporting non-zero speed after the goal was
        zeroed and torque released, s. **Reported, not gated** — see the module
        docstring.
    median_temp_c : float
        Median temperature during this wheel's segment, °C — deliberately the
        median and not a maximum. See :attr:`garbled_temp_fraction`.
    garbled_temp_fraction : float
        Fraction of this wheel's temperature samples more than 3 °C from the
        median. On LK-0001 this was 1.5-1.7 %, and **every** bad sample fell in
        the driven phase with none at all during coast: the register garbles
        while the servo is commutating. A ``max`` over that channel reported
        78 °C on a wheel the bus showed at 32 °C, and a rolling-median despike
        cannot fix it either, because the bad samples sometimes land adjacent.
        Report the median, and read this fraction to know how much of the
        channel you had to throw away.
    """

    servo_id: int
    joint: str
    tracking: dict[int, float]
    duty: dict[int, float]
    coast_s: float
    coast_censored: bool
    median_temp_c: float
    garbled_temp_fraction: float

    @property
    def worst_tracking(self) -> float:
        """The step whose speed tracking was furthest from exact (as a ratio)."""
        if not self.tracking:
            return float("nan")
        return min(self.tracking.values(), key=lambda r: abs(1.0 - r) * -1)

    @property
    def tracks_every_step(self) -> bool:
        """Whether every commanded step was reached within tolerance."""
        return all(
            abs(1.0 - ratio) <= _TRACKING_TOLERANCE for ratio in self.tracking.values()
        )

    def to_dict(self) -> dict[str, Any]:
        """Plain-JSON form (integer keys become strings)."""
        d = dataclasses.asdict(self)
        d["tracking"] = {str(k): v for k, v in self.tracking.items()}
        d["duty"] = {str(k): v for k, v in self.duty.items()}
        return d


def load_wheel_capture(source: str | Path | dict) -> dict:
    """Parse a wheel capture, validating its schema.

    Args
    ----
    source : str | Path | dict
        Capture path or parsed document.

    Returns
    -------
    dict
        The document, unchanged (kept as a plain dict — the per-wheel structure
        is what :func:`wheel_health` produces).

    Raises
    ------
    CalibrationError
        If the schema is unrecognized or the capture carries no samples.
    """
    doc = source if isinstance(source, dict) else json.loads(Path(source).read_text())
    if doc.get("schema") != WHEEL_CAPTURE_SCHEMA:
        raise CalibrationError(
            f"load_wheel_capture: unrecognized schema {doc.get('schema')!r}; "
            f"expected {WHEEL_CAPTURE_SCHEMA!r}."
        )
    if not doc.get("t"):
        raise CalibrationError("load_wheel_capture: the capture has no samples.")
    return doc


def wheel_health(source: str | Path | dict) -> tuple[WheelHealth, ...]:
    """Summarize each wheel's free-spin behaviour.

    Per wheel and per commanded step, takes the median measured speed and |duty|
    over the step's settled portion (the first :data:`_SETTLE_FRACTION` of each
    step is dropped as spin-up), then measures how long the wheel kept reporting
    motion after release.

    Args
    ----
    source : str | Path | dict
        A wheel capture (see :func:`load_wheel_capture`).

    Returns
    -------
    tuple[WheelHealth, ...]
        One entry per wheel, in capture order.

    Raises
    ------
    CalibrationError
        If the capture is unreadable, or a wheel has no staircase samples (a
        capture that never commanded it says nothing about it).

    Units
    -----
    Raw register units (see the module docstring).
    """
    doc = load_wheel_capture(source)
    time = np.asarray(doc["t"], dtype=float)
    servo = np.asarray(doc["servo_id"], dtype=int)
    phase = np.asarray(doc["phase"], dtype=object)
    commanded = np.asarray(doc["commanded_speed"], dtype=float)

    def channel(name: str) -> np.ndarray:
        return np.asarray(
            [np.nan if v is None else float(v) for v in doc[name]], dtype=float
        )

    speed, duty, temp = channel("speed_raw"), channel("load_raw"), channel("temp_C")

    out: list[WheelHealth] = []
    for sid in doc["servo_ids"]:
        sid = int(sid)
        mine = servo == sid
        stair = mine & (phase == "staircase")
        if not stair.any():
            raise CalibrationError(
                f"wheel_health: wheel {sid} has no staircase samples, so nothing "
                "about it is measured. Fix: include it in --servo-ids."
            )
        tracking: dict[int, float] = {}
        duties: dict[int, float] = {}
        for step in sorted(set(commanded[stair])):
            at_step = np.flatnonzero(stair & (commanded == step))
            settled = at_step[int(len(at_step) * _SETTLE_FRACTION) :]
            measured = float(np.nanmedian(speed[settled]))
            tracking[int(step)] = measured / step if step else float("nan")
            duties[int(step)] = float(np.nanmedian(np.abs(duty[settled])))

        coast = mine & (phase == "coast")
        coast_s, coast_censored = 0.0, False
        if coast.any():
            coast_speed = np.abs(speed[coast])
            coast_time = time[coast] - time[coast][0]
            moving = np.flatnonzero(coast_speed > _SPEED_QUANTUM)
            coast_s = float(coast_time[moving[-1]]) if moving.size else 0.0
            coast_censored = bool(moving.size and moving[-1] == coast_speed.size - 1)
        mine_temp = temp[mine]
        has_temp = bool(mine.any()) and bool(np.isfinite(mine_temp).any())
        median_temp = float(np.nanmedian(mine_temp)) if has_temp else float("nan")
        garbled = (
            float(np.mean(np.abs(temp[mine] - median_temp) > _TEMP_OUTLIER_C))
            if has_temp
            else float("nan")
        )
        out.append(
            WheelHealth(
                servo_id=sid,
                joint=str(doc.get("joints", {}).get(str(sid), "")),
                tracking=tracking,
                duty=duties,
                coast_s=coast_s,
                coast_censored=coast_censored,
                median_temp_c=median_temp,
                garbled_temp_fraction=garbled,
            )
        )
    return tuple(out)


def wheel_gates(healths: tuple[WheelHealth, ...]) -> list[Gate]:
    """Pass/warn/fail gates over the wheel summaries.

    Two gates, both chosen because the first real capture showed them to be
    trustworthy (see the module docstring):

    * ``wheel_tracking`` — every commanded step reached within
      :data:`_TRACKING_TOLERANCE`. A miss is a **fail**: an unloaded wheel that
      cannot reach speed is broken, not merely odd.
    * ``wheel_symmetry`` — each wheel's no-load duty against the median across
      **all** the wheels, per step. A departure beyond
      :data:`_DUTY_SYMMETRY_TOLERANCE` is a **warn**, because a genuinely
      different-but-fine wheel is possible (a different tyre, a fresh bearing)
      and this check has no reference unit behind it.

      Comparing each wheel against the median of the *others* is the obvious
      formulation and it is wrong at this fleet size: with three wheels the
      "median of the others" is a two-sample average, so one bad wheel drags the
      reference for both healthy ones and a single outlier warns three times.
      Against the median of all three, one outlier moves the median not at all —
      which is the whole reason a median is used. It needs three wheels to work,
      so symmetry is skipped below that rather than degraded silently.

    The coast-down is deliberately not gated. It is reported in the summary for
    a human to read, because on the reference unit it disagreed with the duty
    evidence and gating it would have failed a healthy robot.

    Args
    ----
    healths : tuple[WheelHealth, ...]
        Output of :func:`wheel_health`. Symmetry needs at least three wheels
        (see above); with fewer, only tracking is gated.

    Returns
    -------
    list[Gate]
        Gates in the checkup's own :class:`~jaxterity.checkup.report.Gate` form,
        so they compose with the rest of the report.
    """
    gates: list[Gate] = []
    for health in healths:
        scope = health.joint or f"servo{health.servo_id}"
        misses = {
            step: ratio
            for step, ratio in health.tracking.items()
            if abs(1.0 - ratio) > _TRACKING_TOLERANCE
        }
        if misses:
            worst = min(misses.items(), key=lambda kv: kv[1])
            gates.append(
                Gate(
                    gate="wheel_tracking",
                    scope=scope,
                    level="fail",
                    detail=(
                        f"commanded {worst[0]} but reached {worst[1]:.0%} of it "
                        f"({len(misses)} step(s) outside "
                        f"{_TRACKING_TOLERANCE:.0%}) with the wheel unloaded."
                    ),
                    remedy="Check the wheel for mechanical drag and the servo "
                    "for a derating driver; a free wheel should track exactly.",
                )
            )

    if len(healths) >= 3:
        for health in healths:
            scope = health.joint or f"servo{health.servo_id}"
            for step, mine in health.duty.items():
                peers = [h.duty[step] for h in healths if step in h.duty]
                if len(peers) < 3 or mine <= 0:
                    continue
                median = float(np.median(peers))
                if median <= 0:
                    continue
                departure = abs(mine - median) / median
                if departure > _DUTY_SYMMETRY_TOLERANCE:
                    gates.append(
                        Gate(
                            gate="wheel_symmetry",
                            scope=scope,
                            level="warn",
                            detail=(
                                f"no-load duty {mine:.0f} at commanded {step} is "
                                f"{departure:.0%} from the wheel-set median "
                                f"({median:.0f})."
                            ),
                            remedy="Compare the wheels mechanically; a duty "
                            "outlier with matching speed is usually a bearing "
                            "or a tyre, not the servo.",
                        )
                    )
                    break  # one symmetry warning per wheel is enough
    return gates


def wheel_summary(healths: tuple[WheelHealth, ...]) -> str:
    """Terminal-friendly table of the wheel summaries."""
    lines = [
        f"  {'wheel':14s} {'tracking':>9} {'duty @ max':>11} {'coast_s':>8} "
        f"{'med_C':>6} {'garbled':>8}"
    ]
    for health in healths:
        top = max(health.duty, key=lambda s: abs(s)) if health.duty else 0
        tracking = "exact" if health.tracks_every_step else "MISSED"
        coast = f"{health.coast_s:.2f}" + (">" if health.coast_censored else "")
        lines.append(
            f"  {(health.joint or health.servo_id):14} {tracking:>9} "
            f"{health.duty.get(top, float('nan')):11.0f} {coast:>8} "
            f"{health.median_temp_c:6.0f} {health.garbled_temp_fraction:7.1%}"
        )
    return "\n".join(lines)


__all__ = [
    "WHEEL_CAPTURE_SCHEMA",
    "WheelHealth",
    "load_wheel_capture",
    "wheel_gates",
    "wheel_health",
    "wheel_summary",
]
