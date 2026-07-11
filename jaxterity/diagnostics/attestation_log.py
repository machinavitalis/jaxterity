# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""The rolling attestation log (re-fit companion).

Every windowed re-fit commit produces a new per-unit attestation handle
(ADR-008). This module keeps the *record* of that rolling chain: an
append-only, timestamped log of ``(handle, recipe, telemetry hash, residual)``
entries — the proof that a unit's self-model is being kept current, and the
ticker a drift dashboard renders. A unit whose log stops growing is a unit
whose model is going stale; a unit without a log has nothing to show at all.

This is bookkeeping, not cryptography: handles come from
:func:`jaxterity.diagnostics.compute_handle` (BLAKE3 content hashes);
signing and tamper evidence live in the deployment layer (Jaxility manifests
and the commercial signers). The log's integrity claim is structural —
monotonic time, append-only, each entry's handle recomputable from the twin
it recorded.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

from ..errors import JaxterityError

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..calibrate.windowed import WindowedFitResult
    from ..twin import Twin


class AttestationRecord(NamedTuple):
    """One commit on a unit's rolling attestation chain.

    Fields
    ------
    t : float
        Commit time, s (simulated or wall — the caller's clock).
    handle : str
        The unit's full attestation handle after the commit.
    recipe : str
        What produced it (e.g. ``"sysid.windowed_drift"``; ``"baseline"``
        for the initial entry).
    telemetry_hash : str
        Content hash of the telemetry behind the commit (``""`` for the
        baseline entry).
    residual_rms : float | None
        The window's torque prediction-residual RMS, N·m (``None`` when not
        applicable).
    """

    t: float
    handle: str
    recipe: str
    telemetry_hash: str
    residual_rms: float | None

    @property
    def short_handle(self) -> str:
        """The dashboard-friendly ``7f3a…91c2`` rendering."""
        return f"{self.handle[:4]}…{self.handle[-4:]}"


class AttestationLog:
    """Append-only attestation history for one unit.

    Open the log with the unit's starting twin (the Day-0 baseline entry),
    then :meth:`commit` each re-fit. ``entries`` is the full chain;
    :meth:`render` produces the rolling-ticker lines.

    Args
    ----
    twin : Twin
        The unit at log-open time.
    t : float
        The baseline timestamp, s.
    """

    def __init__(self, twin: "Twin", *, t: float = 0.0):
        self._serial = twin.serial
        self._entries: list[AttestationRecord] = [
            AttestationRecord(
                t=float(t),
                handle=twin.attestation_handle,
                recipe="baseline",
                telemetry_hash="",
                residual_rms=None,
            )
        ]

    @property
    def serial(self) -> str:
        """The unit this log belongs to."""
        return self._serial

    @property
    def entries(self) -> tuple[AttestationRecord, ...]:
        """The chain, oldest first (the baseline entry is index 0)."""
        return tuple(self._entries)

    @property
    def last_verified(self) -> AttestationRecord:
        """The most recent commit — 'last verified' on a dashboard."""
        return self._entries[-1]

    def commit(self, t: float, fit: "WindowedFitResult") -> AttestationRecord:
        """Append a re-fit commit to the chain.

        Args
        ----
        t : float
            Commit time, s. Must be after the previous entry.
        fit : WindowedFitResult
            The re-fit (:func:`jaxterity.calibrate.refit_drift_parameters`
            output). Its twin must be this unit (same serial).

        Returns
        -------
        AttestationRecord
            The appended entry.

        Raises
        ------
        JaxterityError
            On a serial mismatch, non-increasing time, or an unchanged
            handle (a commit that didn't change the model is not a commit).
        """
        twin = fit.twin
        if twin.serial != self._serial:
            raise JaxterityError(
                f"AttestationLog.commit: this log belongs to unit "
                f"{self._serial!r} but the fit is for {twin.serial!r}."
            )
        if t <= self._entries[-1].t:
            raise JaxterityError(
                f"AttestationLog.commit: time went backwards ({t} after "
                f"{self._entries[-1].t}); the chain is append-only."
            )
        handle = twin.attestation_handle
        if handle == self._entries[-1].handle:
            raise JaxterityError(
                "AttestationLog.commit: the fit's handle equals the previous "
                "entry's — a commit must change the model (did you pass the "
                "pre-fit twin?)."
            )
        record = AttestationRecord(
            t=float(t),
            handle=handle,
            recipe=fit.sysid_result.recipe_name,
            telemetry_hash=fit.sysid_result.telemetry_hash,
            residual_rms=fit.residual_rms_after,
        )
        self._entries.append(record)
        return record

    def render(self, last: int = 5, *, time_unit: str = "s") -> list[str]:
        """The rolling-ticker lines, newest first.

        Args
        ----
        last : int
            How many entries to render.
        time_unit : str
            Label appended to the timestamp.

        Returns
        -------
        list[str]
            e.g. ``"t=3600 s · 7f3a…91c2 · sysid.windowed_drift · rms 0.42"``.
        """
        lines = []
        for record in reversed(self._entries[-last:]):
            rms = (
                f" · rms {record.residual_rms:.3g}"
                if record.residual_rms is not None
                else ""
            )
            lines.append(
                f"t={record.t:g} {time_unit} · {record.short_handle} · "
                f"{record.recipe}{rms}"
            )
        return lines


__all__ = ["AttestationLog", "AttestationRecord"]
