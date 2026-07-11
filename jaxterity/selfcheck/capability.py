# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Live capability estimation: shrink the operating envelope as a unit degrades.

The safety monitor checks commands against a *fixed* operator-declared
:class:`~jaxterity.selfcheck.monitor.SafetyEnvelope`. But a unit's real
capability is not fixed — as it heats (or ages, running hotter for the same
torque), the torque it can sustain without crossing its thermal ceiling falls.
This module publishes a **degraded** ``SafetyEnvelope`` from the unit's *current*
thermal state, so the same `SafetyMonitor` that consumed the nominal envelope now
enforces the shrunk one — the controller automatically respects a smaller torque
budget as the unit gets hot.

The capability is read off the lumped thermal model the twin already carries:
the remaining headroom to the ceiling, ``(T_ceiling − T_now) / (T_ceiling −
T_amb)``, scales the allowed torque margin from nominal (cool unit, full margin)
down to zero (at the ceiling). Because a measured/predicted ``T_now`` already
reflects both transient heating *and* aging (an aged unit with higher ``R_th``
runs hotter at the same torque), the single headroom factor captures both.

Honest scope: this is envelope estimation from the twin + a **declared limit
model** (the nominal limits + the headroom derate), not a learned capability
model. The output is a standard `SafetyEnvelope` — nothing downstream changes.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from ..twin import Twin
from .monitor import SafetyEnvelope


@dataclasses.dataclass(frozen=True)
class LimitModel:
    """The declared nominal limits a unit derates *from*.

    Fields
    ------
    temp_ceiling_k : float
        The servo temperature ceiling, K — the envelope's hard limit and the
        reference the headroom derate measures against.
    temp_band_k : float
        Derate band below the ceiling, K (passed through to the envelope).
    nominal_torque_margin_frac : float
        The torque margin (fraction of effort limit) a fully-cool unit is
        allowed. Scaled down by the thermal headroom as the unit heats.
    torque_band_frac : float
        Derate band below the torque margin (passed through to the envelope).
    joint_margin_rad : float | None
        Joint-limit margin to publish (rad), or ``None`` to leave it unchecked.
    bus_floor_v : float | None
        Bus-voltage floor to publish (V), or ``None``.
    min_torque_margin_frac : float
        Floor on the derated torque margin so a hot unit keeps a usable (if
        small) budget rather than collapsing to zero. Default 0.05.
    """

    temp_ceiling_k: float
    temp_band_k: float = 10.0
    nominal_torque_margin_frac: float = 0.9
    torque_band_frac: float = 0.1
    joint_margin_rad: float | None = None
    bus_floor_v: float | None = None
    min_torque_margin_frac: float = 0.05


class CapabilityEstimator:
    """Publish a degraded :class:`SafetyEnvelope` from a unit's thermal state.

    Args
    ----
    twin : Twin
        The unit's twin (its ``T_amb`` anchors the headroom scale).
    limit_model : LimitModel
        The nominal limits and the derate policy.
    """

    def __init__(self, twin: Twin, limit_model: LimitModel):
        self.twin = twin
        self.limits = limit_model

    def capability_factor(self, temperature) -> float:
        """Thermal capability in ``[0, 1]`` from the current per-servo temperature.

        The minimum over servos of the remaining headroom fraction
        ``(T_ceiling − T_now) / (T_ceiling − T_amb)`` — 1 when cool, 0 at the
        ceiling. The hottest servo binds (it sets the unit's capability).

        Args
        ----
        temperature : array-like, shape ``(n_servos,)``
            Current/predicted per-servo temperature, K.

        Returns
        -------
        float
            Capability factor in ``[0, 1]``, monotone decreasing in temperature.
        """
        T_now = np.asarray(temperature, dtype=float)
        ceiling = self.limits.temp_ceiling_k
        span = ceiling - self.twin.T_amb
        if span <= 0:
            return 0.0
        headroom = (ceiling - T_now) / span
        return float(np.clip(np.min(headroom), 0.0, 1.0))

    def estimate(self, temperature) -> SafetyEnvelope:
        """Publish the degraded envelope for the current thermal state.

        Scales the nominal torque margin by :meth:`capability_factor` (floored at
        ``min_torque_margin_frac``) and returns a `SafetyEnvelope` carrying the
        declared ceiling / joint / bus limits — ready to hand straight to a
        :class:`~jaxterity.selfcheck.monitor.SafetyMonitor`.

        Args
        ----
        temperature : array-like, shape ``(n_servos,)``
            Current/predicted per-servo temperature, K.

        Returns
        -------
        SafetyEnvelope
            The shrunk envelope (smaller ``torque_margin_frac`` when hot).
        """
        factor = self.capability_factor(temperature)
        margin = max(
            self.limits.nominal_torque_margin_frac * factor,
            self.limits.min_torque_margin_frac,
        )
        return SafetyEnvelope(
            temp_ceiling_k=self.limits.temp_ceiling_k,
            temp_band_k=self.limits.temp_band_k,
            joint_margin_rad=self.limits.joint_margin_rad,
            bus_floor_v=self.limits.bus_floor_v,
            torque_margin_frac=margin,
            torque_band_frac=self.limits.torque_band_frac,
        )


__all__ = ["LimitModel", "CapabilityEstimator"]
