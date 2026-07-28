# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Serial-keyed on-disk store of checkup runs, and drift across them.

Each saved :class:`~jaxterity.checkup.CheckupReport` becomes
``<root>/<serial>/run-NNNN.json``; the index is monotonically increasing so
the run order is the file order. :meth:`CheckupStore.drift` turns the stored
runs into per-parameter series (damping/friction per joint, score, battery
R_int) — the raw material for the report's drift sparklines and for "what
changed since the dataset was recorded" questions.
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


__all__ = ["CheckupStore"]
