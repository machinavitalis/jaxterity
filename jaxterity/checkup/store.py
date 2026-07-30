# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Serial-keyed on-disk store of checkup runs, and drift across them.

Each saved :class:`~jaxterity.checkup.CheckupReport` becomes
``<root>/<serial>/run-NNNN.json``; the index is monotonically increasing so
the run order is the file order. :meth:`CheckupStore.drift` turns the stored
runs into per-parameter series (damping/friction per joint, score, battery
R_int) — the raw material for the report's drift sparklines and for "what
changed since the dataset was recorded" questions.

A series is only a *drift* series if every run commanded the same motion, so
each run also carries its pinned capture protocol
(:mod:`jaxterity.checkup.protocol`) and :meth:`CheckupStore.drift_warnings`
names every boundary where the method changed. Read those before reading a
trend: the first LeKiwi unit's runs differ by 5–8× on two joints for no reason
other than a different pose.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from .report import CheckupReport

_RUN_RE = re.compile(r"run-(\d{4})\.json$")


class CheckupStore:
    """Filesystem store of checkup runs, keyed by unit serial.

    Args
    ----
    root : str | Path
        Directory that holds one subdirectory per serial. Created lazily.
    """

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def _unit_dir(self, serial: str) -> Path:
        if not serial or "/" in serial or serial.startswith("."):
            raise ValueError(f"CheckupStore: invalid serial {serial!r}.")
        return self.root / serial

    def save(self, report: CheckupReport) -> Path:
        """Persist a report as the next run for its serial; returns the path."""
        d = self._unit_dir(report.serial)
        d.mkdir(parents=True, exist_ok=True)
        n = 1 + max(
            (int(m.group(1)) for p in d.iterdir() if (m := _RUN_RE.search(p.name))),
            default=0,
        )
        path = d / f"run-{n:04d}.json"
        path.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n")
        return path

    def runs(self, serial: str) -> list[CheckupReport]:
        """All stored runs for a serial, oldest first (empty when none)."""
        d = self._unit_dir(serial)
        if not d.is_dir():
            return []
        paths = sorted(p for p in d.iterdir() if _RUN_RE.search(p.name))
        return [CheckupReport.from_dict(json.loads(p.read_text())) for p in paths]

    def drift(self, serial: str) -> dict[str, list[float | None]]:
        """Per-parameter series across the stored runs (oldest first).

        Returns
        -------
        dict[str, list[float | None]]
            Keys ``"<joint>.damping"`` / ``"<joint>.friction"`` (``None`` at
            runs where the channel was held), ``"score.total"``, and — when
            any run fit a battery — ``"battery.R_int"``.
        """
        runs = self.runs(serial)
        series: dict[str, list[float | None]] = {}
        joints = sorted({j.joint for r in runs for j in r.joints})
        for joint in joints:
            for field in ("damping", "friction"):
                key = f"{joint}.{field}"
                series[key] = [
                    next(
                        (getattr(j, field) for j in r.joints if j.joint == joint),
                        None,
                    )
                    for r in runs
                ]
        series["score.total"] = [float(r.score["total"]) for r in runs]
        if any(r.battery is not None for r in runs):
            series["battery.R_int"] = [
                r.battery.R_int if r.battery else None for r in runs
            ]
        return series

    def protocol_hashes(self, serial: str) -> list[str | None]:
        """Each stored run's pinned protocol hash, oldest first.

        ``None`` for runs recorded before protocol pinning existed — those
        cannot be shown to share a method with anything, which is a fact about
        them, not a gap to paper over.
        """
        return [r.protocol_hash for r in self.runs(serial)]

    def pose_attribution(self, serial: str) -> dict[str, dict[str, float]]:
        """How much of each joint's apparent drift is geometry, not wear.

        The quantitative half of the canonical-pose contract. ``drift_warnings``
        says *that* the pose moved; this says what it cost. For each joint with a
        recorded pose feature in both the first and last run, it reports the raw
        damping ratio, the pose-feature ratio, and the ratio of the two —
        ``damping/feature``, the quantity that stayed put on LK-0001.

        The measurement behind it: two sessions under *identical* excitation
        recipes moved shoulder_pan's damping 3.29x while its axis inertia moved
        3.42x, so ``damping/inertia`` agreed to 4 %. A raw drift series would have
        called that a joint falling apart. ``normalized_ratio`` near 1.0 means the
        change was geometry; far from 1.0 is what actually deserves attention.

        Args
        ----
        serial : str
            The unit.

        Returns
        -------
        dict[str, dict[str, float]]
            Joint → ``{"damping_ratio", "feature_ratio", "normalized_ratio"}``.
            Joints without a feature in both runs, or with a zero denominator,
            are omitted rather than reported as infinite: an unattributable
            change is not a change of zero.
        """
        runs = self.runs(serial)
        if len(runs) < 2:
            return {}
        first = {j.joint: j for j in runs[0].joints}
        last = {j.joint: j for j in runs[-1].joints}
        out: dict[str, dict[str, float]] = {}
        for joint, end in last.items():
            start = first.get(joint)
            if start is None:
                continue
            values = (start.damping, end.damping, start.pose_feature, end.pose_feature)
            if any(v is None for v in values):
                continue
            d0, d1, f0, f1 = (float(v) for v in values)  # type: ignore[arg-type]
            if d0 == 0 or f0 == 0 or f1 == 0:
                continue
            damping_ratio = d1 / d0
            feature_ratio = f1 / f0
            if feature_ratio == 0:
                continue
            out[joint] = {
                "damping_ratio": damping_ratio,
                "feature_ratio": feature_ratio,
                "normalized_ratio": damping_ratio / feature_ratio,
            }
        return out

    def drift_warnings(self, serial: str) -> tuple[str, ...]:
        """What makes this unit's drift series untrustworthy, in words.

        A drift series only measures drift if every run commanded the same
        motion. This walks the stored runs and reports each boundary where the
        pinned capture protocol changed — with the specific differences, so a
        reader can tell a re-planned amplitude from a moved pose — plus any runs
        that carry no pin at all.

        The first LeKiwi unit is the cautionary case this exists for: run-0001's
        damping estimates were 0.12× and 0.17× of a later session's on two
        joints, entirely because the sessions excited different poses. Without
        these warnings that reads as a robot falling apart.

        Args
        ----
        serial : str
            The unit.

        Returns
        -------
        tuple[str, ...]
            Empty when every run shares one protocol (the only case in which
            the series is a clean drift measurement).

        Examples
        --------
        >>> store = CheckupStore("/tmp/checkup")           # doctest: +SKIP
        >>> for line in store.drift_warnings("LK-0001"):   # doctest: +SKIP
        ...     print(line)
        runs 1-2 → run 3: capture protocol changed (shoulder_lift: amplitude
        200 → 150 ticks); the series across that boundary mixes drift with
        method change.
        """
        runs = self.runs(serial)
        if len(runs) < 2:
            return ()
        warnings: list[str] = []
        unpinned = [i + 1 for i, r in enumerate(runs) if r.protocol is None]
        if unpinned:
            warnings.append(
                f"run(s) {', '.join(str(i) for i in unpinned)} carry no capture "
                "protocol (recorded before pinning); they cannot be shown to "
                "share a method with the rest of the series."
            )
        start = 1
        for i in range(1, len(runs)):
            previous, current = runs[i - 1], runs[i]
            if previous.protocol is None or current.protocol is None:
                start = i + 1
                continue
            if current.protocol.matches(previous.protocol):
                continue
            differences = current.protocol.differences(previous.protocol)
            detail = "; ".join(differences) if differences else "hash differs"
            span = f"run {start}" if start == i else f"runs {start}-{i}"
            warnings.append(
                f"{span} → run {i + 1}: capture protocol changed ({detail}); the "
                "series across that boundary mixes drift with method change."
            )
            start = i + 1
        return tuple(warnings)


__all__ = ["CheckupStore"]
