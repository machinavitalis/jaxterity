# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""The capture protocol, pinned: what was commanded, hashed, and comparable.

A drift series is only a drift series if every run measured the same thing the
same way. The first LeKiwi unit's stored runs did not: run-0001's damping
estimates sat at 0.12× and 0.17× of a later session's for two joints, and the
cause was not the hardware aging between sessions — it was that the sessions
excited different poses about different centers. Cross-session numbers compared
that way are *method sensitivity dressed as degradation*, and every downstream
claim built on them (per-unit trends, remaining-life projections, any fleet
percentile) inherits the error.

This module makes the method an artifact you can compare. :func:`capture_protocol`
reads the same capture files the checkup fits and canonicalizes what was
*commanded* — per joint: the waveform kind and its tone set, the amplitude, the
duration, the command rate, the excitation center, and the held pose of every
other joint; plus the battery trace's own parameters. :attr:`CaptureProtocol.hash`
is a stable digest of that canonical form, and :meth:`CaptureProtocol.differences`
says in words what changed when two runs disagree.

**Two kinds of quantity, compared two ways.** The *recipe* — waveform kind, tone
set, amplitude, duration, command rate — is set by the plan and is exactly
reproducible, so it goes into the hash and an equal hash means an identical
recipe. Positions are not: a pose commanded identically lands a few ticks apart
every session, because servos do not settle on the same tick twice. Hashing them
(even quantized) makes the pin flip on settling noise whenever a joint sits near
a bucket boundary — a pin that fires at random is worse than none. So poses and
excitation centers are kept as raw ticks and compared by *distance*, against
:data:`POSE_TOLERANCE_TICKS` (25 ticks ≈ 2.2° on a 652 ticks/rad servo): far
above settling, far below the pose differences that moved run-0001's fits by
5–8×. :meth:`CaptureProtocol.matches` is therefore the comparison to use —
``hash`` alone answers "same recipe?", not "same experiment?".

What a matching hash does **not** claim: that the robot was in the same thermal
state, that the battery was equally charged, or that nothing was serviced in
between. It claims the commanded motion was the same — which is the part a
capture protocol can control, and the part run-0001 got wrong.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Any

from ..errors import CalibrationError

PROTOCOL_SCHEMA = "jaxterity-checkup-protocol-v1"

#: How far a commanded position may differ between sessions and still count as
#: the same experiment, in encoder ticks. 25 ticks is ~2.2° on the LeKiwi's
#: 652 ticks/rad servos — an order of magnitude above servo settling, an order
#: of magnitude below the pose differences that corrupted run-0001's drift
#: series. Compared as a distance, never hashed: see the module docstring.
POSE_TOLERANCE_TICKS = 25

#: Digest length (hex chars) of :attr:`CaptureProtocol.hash`. Long enough that a
#: collision is not a practical concern, short enough to read in a report.
_HASH_CHARS = 16


@dataclasses.dataclass(frozen=True)
class JointProtocol:
    """How one joint was excited (the canonical, hashable form).

    Attributes
    ----------
    joint : str
        Joint name.
    servo_id : int
        Bus id.
    kind : str
        Waveform version (see
        :func:`jaxterity.checkup.stability.schroeder_multisine`).
    tones_hz : tuple[float, ...]
        Tone set, Hz.
    amp_ticks : int
        Commanded amplitude, ticks.
    duration_s : float
        Capture duration, s.
    rate_hz : float
        Commanded update rate, Hz.
    center_ticks : int
        Excitation center, raw ticks (compared by distance, not hashed).
    pose_ticks : tuple[tuple[int, int], ...]
        The held pose as sorted ``(servo_id, ticks)`` pairs — the *other*
        joints' positions, which set the gravity/inertia operating point this
        joint was identified at. Also compared by distance.
    """

    joint: str
    servo_id: int
    kind: str
    tones_hz: tuple[float, ...]
    amp_ticks: int
    duration_s: float
    rate_hz: float
    center_ticks: int
    pose_ticks: tuple[tuple[int, int], ...]

    def to_dict(self) -> dict[str, Any]:
        """Plain-JSON form (canonical: keys sorted, tuples as lists)."""
        d = dataclasses.asdict(self)
        d["tones_hz"] = [float(f) for f in self.tones_hz]
        d["pose_ticks"] = [[int(s), int(t)] for s, t in self.pose_ticks]
        return d

    def recipe(self) -> dict[str, Any]:
        """The exactly-reproducible half — what the hash is taken over.

        Excludes positions by design (see the module docstring): they are
        continuous, so they are compared by distance instead.
        """
        return {
            "joint": self.joint,
            "servo_id": self.servo_id,
            "kind": self.kind,
            "tones_hz": [float(f) for f in self.tones_hz],
            "amp_ticks": self.amp_ticks,
            "duration_s": self.duration_s,
            "rate_hz": self.rate_hz,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "JointProtocol":
        """Inverse of :meth:`to_dict`."""
        return cls(
            joint=str(d["joint"]),
            servo_id=int(d["servo_id"]),
            kind=str(d["kind"]),
            tones_hz=tuple(float(f) for f in d["tones_hz"]),
            amp_ticks=int(d["amp_ticks"]),
            duration_s=float(d["duration_s"]),
            rate_hz=float(d["rate_hz"]),
            center_ticks=int(d["center_ticks"]),
            pose_ticks=tuple((int(s), int(t)) for s, t in d["pose_ticks"]),
        )


@dataclasses.dataclass(frozen=True)
class BatteryProtocol:
    """How the battery load trace was commanded (canonical form).

    Attributes
    ----------
    servo_ids : tuple[int, ...]
        Loaded servos, sorted.
    kind : str
        Waveform version.
    tones_hz : tuple[float, ...]
        Tone set, Hz.
    amp_ticks : int
        Commanded amplitude, ticks.
    duration_s : float
        Trace duration, s.
    """

    servo_ids: tuple[int, ...]
    kind: str
    tones_hz: tuple[float, ...]
    amp_ticks: int
    duration_s: float

    def to_dict(self) -> dict[str, Any]:
        """Plain-JSON form."""
        d = dataclasses.asdict(self)
        d["servo_ids"] = [int(s) for s in self.servo_ids]
        d["tones_hz"] = [float(f) for f in self.tones_hz]
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "BatteryProtocol":
        """Inverse of :meth:`to_dict`."""
        return cls(
            servo_ids=tuple(int(s) for s in d["servo_ids"]),
            kind=str(d["kind"]),
            tones_hz=tuple(float(f) for f in d["tones_hz"]),
            amp_ticks=int(d["amp_ticks"]),
            duration_s=float(d["duration_s"]),
        )


@dataclasses.dataclass(frozen=True)
class CaptureProtocol:
    """The whole session's commanded method, hashable and diffable.

    Attributes
    ----------
    joints : tuple[JointProtocol, ...]
        Per-joint excitation, sorted by joint name.
    battery : BatteryProtocol | None
        The load trace, when one was captured.
    """

    joints: tuple[JointProtocol, ...]
    battery: BatteryProtocol | None = None

    @property
    def hash(self) -> str:
        """Stable digest of the **recipe** (:data:`_HASH_CHARS` hex chars).

        BLAKE2b over the sorted-key JSON of every joint's
        :meth:`JointProtocol.recipe` plus the battery trace's parameters.
        Identical on every machine for an identical recipe; different the moment
        a waveform, tone set, amplitude, duration or rate changes. Positions are
        deliberately **not** in it — use :meth:`matches` for the "same
        experiment?" question, which is the one a drift series needs.
        """
        payload = json.dumps(
            {
                "schema": PROTOCOL_SCHEMA,
                "joints": [j.recipe() for j in self.joints],
                "battery": self.battery.to_dict() if self.battery else None,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.blake2b(payload.encode(), digest_size=32).hexdigest()[
            :_HASH_CHARS
        ]

    def matches(
        self,
        other: "CaptureProtocol",
        *,
        pose_tolerance_ticks: int = POSE_TOLERANCE_TICKS,
    ) -> bool:
        """Whether two sessions ran the same experiment.

        Recipe hashes equal **and** every excitation center and held-pose
        position within ``pose_tolerance_ticks``. This is the predicate a drift
        comparison should gate on; :attr:`hash` alone is only half of it.

        Args
        ----
        other : CaptureProtocol
            The protocol to compare against.
        pose_tolerance_ticks : int
            Position tolerance, encoder ticks (default
            :data:`POSE_TOLERANCE_TICKS`).

        Returns
        -------
        bool
            True when the two are the same experiment within tolerance.
        """
        return not self.differences(other, pose_tolerance_ticks=pose_tolerance_ticks)

    def to_dict(self) -> dict[str, Any]:
        """Plain-JSON form (inverse of :meth:`from_dict`)."""
        return {
            "schema": PROTOCOL_SCHEMA,
            "joints": [j.to_dict() for j in self.joints],
            "battery": self.battery.to_dict() if self.battery else None,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "CaptureProtocol":
        """Rebuild from :meth:`to_dict`.

        Raises
        ------
        CalibrationError
            If the payload's schema field is not :data:`PROTOCOL_SCHEMA`.
        """
        if d.get("schema") != PROTOCOL_SCHEMA:
            raise CalibrationError(
                f"CaptureProtocol.from_dict: unrecognized schema "
                f"{d.get('schema')!r}; expected {PROTOCOL_SCHEMA!r}."
            )
        return cls(
            joints=tuple(JointProtocol.from_dict(j) for j in d["joints"]),
            battery=(
                BatteryProtocol.from_dict(d["battery"]) if d.get("battery") else None
            ),
        )

    def differences(
        self,
        other: "CaptureProtocol",
        *,
        pose_tolerance_ticks: int = POSE_TOLERANCE_TICKS,
    ) -> tuple[str, ...]:
        """Human-readable differences from ``other`` (empty when equivalent).

        The message a drift warning needs: not "the hash changed" but *what*
        changed, so a reader can tell a re-planned amplitude from a moved pose
        from a waveform upgrade. Recipe fields are compared exactly; positions
        only register beyond ``pose_tolerance_ticks``, so ordinary servo
        settling produces no lines at all.

        Args
        ----
        other : CaptureProtocol
            The protocol to compare against (typically the previous run's).
        pose_tolerance_ticks : int
            Position tolerance, encoder ticks.

        Returns
        -------
        tuple[str, ...]
            One line per difference, ordered joints-then-battery.
        """
        out: list[str] = []
        mine = {j.joint: j for j in self.joints}
        theirs = {j.joint: j for j in other.joints}
        for joint in sorted(set(mine) - set(theirs)):
            out.append(f"{joint}: captured now, absent before")
        for joint in sorted(set(theirs) - set(mine)):
            out.append(f"{joint}: captured before, absent now")
        for joint in sorted(set(mine) & set(theirs)):
            a, b = mine[joint], theirs[joint]
            if a.kind != b.kind:
                out.append(f"{joint}: waveform {b.kind} → {a.kind}")
            if a.tones_hz != b.tones_hz:
                out.append(f"{joint}: tones {list(b.tones_hz)} → {list(a.tones_hz)}")
            if a.amp_ticks != b.amp_ticks:
                out.append(f"{joint}: amplitude {b.amp_ticks} → {a.amp_ticks} ticks")
            if a.duration_s != b.duration_s:
                out.append(f"{joint}: duration {b.duration_s:g} → {a.duration_s:g} s")
            if a.rate_hz != b.rate_hz:
                out.append(f"{joint}: rate {b.rate_hz:g} → {a.rate_hz:g} Hz")
            center_delta = abs(a.center_ticks - b.center_ticks)
            if center_delta > pose_tolerance_ticks:
                out.append(f"{joint}: center moved {center_delta} ticks")
            moved = _pose_deltas(a.pose_ticks, b.pose_ticks, pose_tolerance_ticks)
            if moved:
                out.append(f"{joint}: held pose differs ({moved})")
        if (self.battery is None) != (other.battery is None):
            out.append(
                "battery load trace present now, absent before"
                if self.battery is not None
                else "battery load trace absent now, present before"
            )
        elif self.battery is not None and other.battery is not None:
            if self.battery != other.battery:
                out.append(
                    f"battery trace: {other.battery.amp_ticks} ticks / "
                    f"{other.battery.duration_s:g} s → {self.battery.amp_ticks} "
                    f"ticks / {self.battery.duration_s:g} s"
                )
        return tuple(out)


def _pose_deltas(
    mine: tuple[tuple[int, int], ...],
    theirs: tuple[tuple[int, int], ...],
    tolerance_ticks: int,
) -> str:
    """Compact 'servo N: M ticks' summary of out-of-tolerance pose differences.

    Empty string when every posed servo agrees within tolerance (the settling
    case). Reports the three largest offenders — enough to identify what moved
    without turning a warning into a table.
    """
    a, b = dict(mine), dict(theirs)
    if set(a) != set(b):
        return "different servos posed"
    deltas = sorted(
        ((abs(a[s] - b[s]), s) for s in a if abs(a[s] - b[s]) > tolerance_ticks),
        reverse=True,
    )
    return ", ".join(f"servo {s}: {d} ticks" for d, s in deltas[:3])


def capture_protocol(
    capture_paths: list[str | Path] | tuple[str | Path, ...],
    *,
    battery_trace: str | Path | None = None,
) -> CaptureProtocol:
    """Read the commanded protocol out of a session's capture files.

    Args
    ----
    capture_paths : list[str | Path] | tuple[str | Path, ...]
        The per-joint commanded-excitation captures (schema
        ``lekiwi-commanded-capture-v1``) — the same files the checkup fits.
    battery_trace : str | Path | None
        The aggregate load capture, when one was recorded.

    Returns
    -------
    CaptureProtocol
        The canonical, hashable protocol.

    Raises
    ------
    CalibrationError
        If a capture is missing the ``excitation`` metadata the protocol is
        made of — a capture that does not say how it was commanded cannot be
        pinned, and silently hashing an incomplete protocol would produce a
        pin that compares unequal things.

    Examples
    --------
    >>> protocol = capture_protocol(["commanded_shoulder_pan.json"])  # doctest: +SKIP
    >>> protocol.hash  # doctest: +SKIP
    'b0d4c1f7a9e2...'
    """
    joints: list[JointProtocol] = []
    for path in capture_paths:
        doc = json.loads(Path(path).read_text())
        exc = doc.get("excitation") or {}
        if not exc or "amp_ticks" not in exc:
            raise CalibrationError(
                f"capture_protocol: {path} carries no excitation metadata, so "
                "the commanded protocol is unknown. Fix: re-capture with a "
                "current script (every capture records its excitation), or "
                "leave this session unpinned rather than comparing it to "
                "others as if the method matched."
            )
        pose = doc.get("pose_ticks") or {}
        joints.append(
            JointProtocol(
                joint=str(doc["joint"]),
                servo_id=int(doc["servo_id"]),
                kind=str(exc.get("kind", "unknown")),
                tones_hz=tuple(float(f) for f in exc.get("multisine_hz", ())),
                amp_ticks=int(exc["amp_ticks"]),
                duration_s=float(exc.get("duration_s", 0.0)),
                rate_hz=float(exc.get("target_rate_hz", 0.0)),
                center_ticks=int(exc.get("center_ticks", 0)),
                pose_ticks=tuple(
                    sorted((int(sid), int(ticks)) for sid, ticks in pose.items())
                ),
            )
        )

    battery: BatteryProtocol | None = None
    if battery_trace is not None:
        doc = json.loads(Path(battery_trace).read_text())
        exc = doc.get("excitation") or {}
        battery = BatteryProtocol(
            servo_ids=tuple(sorted(int(s) for s in doc.get("servo_ids", ()))),
            kind=str(exc.get("kind", "unknown")),
            tones_hz=tuple(float(f) for f in exc.get("multisine_hz", ())),
            amp_ticks=int(exc.get("amp_ticks", 0)),
            duration_s=float(exc.get("duration_s", 0.0)),
        )
    return CaptureProtocol(
        joints=tuple(sorted(joints, key=lambda j: j.joint)), battery=battery
    )


__all__ = [
    "POSE_TOLERANCE_TICKS",
    "PROTOCOL_SCHEMA",
    "BatteryProtocol",
    "CaptureProtocol",
    "JointProtocol",
    "capture_protocol",
]
