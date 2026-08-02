# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Predictive safety vetoing.

A body-only model answers "is there torque available?" — and says *proceed*. The
twin answers "where is this unit *headed*?" The :class:`SafetyMonitor` turns the
self-check co-processor's forecast plus the unit's thermal model into a
decision, against an operator-declared :class:`SafetyEnvelope`:

- :data:`SafetyAction.OK` — the predicted horizon (and the sustained-hold
  steady state) stay clear,
- :data:`SafetyAction.DERATE` — a constraint enters its warning band; scale the
  command by ``factor ∈ (0, 1)``,
- :data:`SafetyAction.VETO` — a constraint is breached; ``factor = 0``.

The binding constraint is named, and ``factor`` is a smooth (clipped-linear),
monotone function of predicted severity bounded to ``[0, 1]``.

The thermal check is the catch. A servo's heat is driven by the held torque
(∝ ``tau²``), not the body trajectory, so the monitor checks two things a
body-only model cannot: the near-term predicted temperature over the forecast,
**and** the sustained-hold steady state ``T_ss = T_amb + k_pwr·R_th·tau²`` with
an **analytic time-to-ceiling** ``t = −τ·ln((ceiling − T_ss)/(T0 − T_ss))`` —
so "hold this and you cook in N seconds" is caught from a cold start, long before
a short real-time horizon would show it.

Honest scope: predict-and-veto only — no recovery controller (runtime
assurance / simplex is later work). The envelope is operator-declared.
"""

from __future__ import annotations

import dataclasses
import enum

import numpy as np

from ..calibrate.active import dof_joint_names
from ..twin import Twin, TwinRollout


class SafetyAction(enum.Enum):
    """The monitor's verdict on a predicted horizon."""

    OK = "ok"
    DERATE = "derate"
    VETO = "veto"


@dataclasses.dataclass(frozen=True)
class SafetyEnvelope:
    """Operator-declared safety limits the predicted horizon must respect.

    Each constraint has a *limit* and a *warning band*: severity ramps linearly
    from safe (``factor = 1``) at the band edge to a breach (``factor = 0``) at
    the limit. A constraint with a ``None`` limit is not checked.

    Args
    ----
    temp_ceiling_k : float
        Maximum allowed servo temperature, K (checked against both the near-term
        forecast and the sustained-hold steady state).
    temp_band_k : float
        Width of the derate band below ``temp_ceiling_k``, K.
    joint_margin_rad : float | None
        Minimum allowed distance from any joint position limit, rad (also the
        derate band). ``None`` (default) skips the check — only meaningful on a
        control-fine-horizon prediction, since a long open-loop forecast under a
        held torque coasts past the stops (a position controller would not).
    bus_floor_v : float | None
        Minimum allowed bus voltage, V (LeKiwi). ``None`` to skip.
    bus_band_v : float
        Width of the derate band above ``bus_floor_v``, V.
    torque_margin_frac : float | None
        Maximum allowed commanded torque as a fraction of each joint's effort
        limit (e.g. ``0.9``). ``None`` to skip. Checked against the action.
    torque_band_frac : float
        Width of the derate band below ``torque_margin_frac`` (fraction).
    """

    temp_ceiling_k: float
    temp_band_k: float = 10.0
    joint_margin_rad: float | None = None
    bus_floor_v: float | None = None
    bus_band_v: float = 1.0
    torque_margin_frac: float | None = None
    torque_band_frac: float = 0.1


@dataclasses.dataclass(frozen=True)
class SafetyDecision:
    """The monitor's decision for one predicted horizon.

    Args
    ----
    action : SafetyAction
        OK / DERATE / VETO.
    factor : float
        Command scale in ``[0, 1]`` — ``1.0`` for OK, ``0.0`` for VETO, the
        binding constraint's smooth severity for DERATE.
    binding_constraint : str
        Name of the constraint with the least margin (e.g. ``"servo_temperature"``).
    worst_value : float
        The binding constraint's worst predicted value over the horizon (SI).
    time_to_breach_s : float | None
        For a binding *thermal* constraint, the predicted time until the ceiling
        is crossed at the held torque (s), if a breach is coming.
    detail : str
        Human/agent-readable one-line explanation.
    factors : dict[str, float]
        Per-constraint factor (for diagnostics).
    worst_values : dict[str, float]
        Per-constraint worst predicted value over the horizon, in that
        constraint's own SI unit (K, rad, V, dimensionless fraction) — the
        un-normalized companion to ``factors``, which the oracle bands across
        a parameter posterior.
    """

    action: SafetyAction
    factor: float
    binding_constraint: str
    worst_value: float
    time_to_breach_s: float | None
    detail: str
    factors: dict[str, float]
    worst_values: dict[str, float] = dataclasses.field(default_factory=dict)

    @property
    def ok(self) -> bool:
        """True iff the verdict is OK (no scaling needed)."""
        return self.action is SafetyAction.OK

    @property
    def vetoed(self) -> bool:
        """True iff the command is vetoed (``factor == 0``)."""
        return self.action is SafetyAction.VETO


def _ceiling_factor(worst_max: float, limit: float, band: float) -> float:
    """Smooth derate for a quantity that must stay *below* ``limit``."""
    if band <= 0:
        return 1.0 if worst_max < limit else 0.0
    return float(np.clip((limit - worst_max) / band, 0.0, 1.0))


def _floor_factor(worst_min: float, floor: float, band: float) -> float:
    """Smooth derate for a quantity that must stay *above* ``floor``."""
    if band <= 0:
        return 1.0 if worst_min > floor else 0.0
    return float(np.clip((worst_min - floor) / band, 0.0, 1.0))


class SafetyMonitor:
    """Turns a co-processor forecast + the unit's thermal model into a decision.

    See the module docstring for the contract.

    Args
    ----
    twin : Twin
        The unit's twin — its robot supplies per-joint position/effort limits and
        its thermal leaves (``C_th``/``R_th``/``k_pwr``/``T_amb``) drive the
        sustained-hold steady-state and time-to-breach checks.
    envelope : SafetyEnvelope
        The operator-declared limits.
    """

    def __init__(self, twin: Twin, envelope: SafetyEnvelope):
        self.twin = twin
        self.envelope = envelope
        self._names = dof_joint_names(twin.robot)
        jmap = twin.robot.joint_map
        self._lower = np.array([jmap[n].limits.lower for n in self._names])
        self._upper = np.array([jmap[n].limits.upper for n in self._names])
        self._effort = np.array([jmap[n].limits.effort for n in self._names])
        self._C_th = np.asarray(twin.C_th, dtype=float)
        self._R_th = np.asarray(twin.R_th, dtype=float)
        self._k_pwr = np.asarray(twin.k_pwr, dtype=float)
        self._T_amb = float(twin.T_amb)

    def evaluate(self, prediction: TwinRollout, *, action=None) -> SafetyDecision:
        """Decide OK/Derate/Veto for a predicted horizon.

        Args
        ----
        prediction : TwinRollout
            A co-processor forecast (temperature, q, optional bus voltage). Its
            first temperature row is the measured "now" used for time-to-breach.
        action : array-like | None
            The commanded generalized force (N·m). Enables the sustained-hold
            thermal check and the torque-margin check.

        Returns
        -------
        SafetyDecision
            The verdict, the binding constraint, and per-constraint factors.
        """
        env = self.envelope
        T_pred = np.asarray(prediction.temperature)
        q = np.asarray(prediction.q)
        factors: dict[str, float] = {}
        worst: dict[str, float] = {}

        # --- servo temperature: near-term forecast AND sustained-hold steady state ---
        t_max = float(T_pred.max())
        time_to_breach: float | None = None
        if action is not None:
            a = np.asarray(action, dtype=float)
            T_ss = self._T_amb + self._k_pwr * self._R_th * a * a  # per servo
            t_ss = float(T_ss.max())
            worst_temp = max(t_max, t_ss)
            time_to_breach = self._time_to_breach(T_pred[0], T_ss, env.temp_ceiling_k)
        else:
            worst_temp = t_max
        factors["servo_temperature"] = _ceiling_factor(
            worst_temp, env.temp_ceiling_k, env.temp_band_k
        )
        worst["servo_temperature"] = worst_temp

        # --- joint position limits (opt-in; control-fine horizon only) ---
        if env.joint_margin_rad is not None and env.joint_margin_rad > 0:
            dist = np.minimum(q - self._lower[None, :], self._upper[None, :] - q)
            min_dist = float(dist.min())
            factors["joint_limit"] = float(
                np.clip(min_dist / env.joint_margin_rad, 0.0, 1.0)
            )
            worst["joint_limit"] = min_dist

        # --- bus voltage floor (LeKiwi) ---
        if env.bus_floor_v is not None and prediction.bus_voltage is not None:
            v_min = float(np.asarray(prediction.bus_voltage).min())
            factors["bus_voltage"] = _floor_factor(
                v_min, env.bus_floor_v, env.bus_band_v
            )
            worst["bus_voltage"] = v_min

        # --- torque / saturation margin (on the commanded action) ---
        if env.torque_margin_frac is not None and action is not None:
            a = np.abs(np.asarray(action, dtype=float))
            with np.errstate(divide="ignore", invalid="ignore"):
                frac = np.where(self._effort > 0, a / self._effort, 0.0)
            worst_frac = float(frac.max())
            factors["torque"] = _ceiling_factor(
                worst_frac, env.torque_margin_frac, env.torque_band_frac
            )
            worst["torque"] = worst_frac

        binding = min(factors, key=lambda k: factors[k])
        factor = factors[binding]
        if factor <= 0.0:
            act, factor = SafetyAction.VETO, 0.0
        elif factor < 1.0:
            act = SafetyAction.DERATE
        else:
            act = SafetyAction.OK

        ttb = time_to_breach if binding == "servo_temperature" else None
        return SafetyDecision(
            action=act,
            factor=float(factor),
            binding_constraint=binding,
            worst_value=worst[binding],
            time_to_breach_s=ttb if act is not SafetyAction.OK else None,
            detail=self._describe(act, binding, worst[binding], ttb),
            factors=factors,
            worst_values=worst,
        )

    def _time_to_breach(self, T0, T_ss, ceiling: float) -> float | None:
        """Earliest analytic time any servo's first-order rise crosses ``ceiling``.

        ``T(t) = T_ss + (T0 − T_ss)·exp(−t/τ)``, ``τ = R_th·C_th``. Returns the
        minimum positive crossing time over servos whose ``T_ss`` exceeds the
        ceiling, ``0.0`` if already above, or ``None`` if none will breach.
        """
        T0 = np.asarray(T0, dtype=float)
        tau = self._R_th * self._C_th
        times: list[float] = []
        for i in range(T_ss.shape[0]):
            if T0[i] >= ceiling:
                return 0.0
            if T_ss[i] <= ceiling:
                continue  # this servo asymptotes below the ceiling — never breaches
            ratio = (ceiling - T_ss[i]) / (T0[i] - T_ss[i])  # in (0, 1)
            times.append(float(-tau[i] * np.log(ratio)))
        return min(times) if times else None

    @staticmethod
    def _describe(action, binding, worst, ttb):
        if action is SafetyAction.OK:
            return "ok: predicted horizon stays within the envelope"
        when = f" in {ttb:.1f} s" if ttb is not None else ""
        verb = "veto" if action is SafetyAction.VETO else "derate"
        return f"{verb}: {binding} binding (worst {worst:.3g}){when}"


__all__ = ["SafetyMonitor", "SafetyEnvelope", "SafetyDecision", "SafetyAction"]
