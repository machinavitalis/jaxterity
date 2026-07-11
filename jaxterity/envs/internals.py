# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Internals-in-the-loop training plants (the Category-A enabler).

A body-only simulator delivers any torque a policy asks for, forever. Real
actuation runs through electro-thermal internals: windings heat under load
and the torque constant falls with temperature; protection firmware
throttles a hot motor; every motor draws from one battery whose bus sags
under aggregate current, and a deep-enough sag browns out the controller.
This module makes those internals part of the *training plant*:

- :class:`InternalsState` — the per-motor temperature, the bus voltage, and
  the brownout dwell timer, carried across env steps (and, if the caller
  wants long-horizon soak, across episodes).
- :func:`actuation_scale` — the actuation feedback: delivered torque =
  commanded × ``K_t(T)/K_t(T_ref)`` × ``throttle(T)``.
- :func:`step_internals` — one step of the coupled state: thermal
  integration (the single-node model), bus voltage under the
  delivered torques (the Thévenin model), brownout dwell.
- :func:`sample_conditions` — per-episode operating conditions
  (ambient temperature → battery internal resistance *and* joint friction,
  jointly). The ``correlated`` flag is the entire A/B difference of a
  correlated-degradation study: ``True`` draws every channel from one
  latent temperature (the physical truth); ``False`` draws each channel
  from its *own* independent latent temperature — so the **marginal**
  distribution of every channel is identical by construction and only the
  joint distribution differs. That is the deck-stacking-proof contrast.

**Neutral mode is an exact no-op.** ``InternalsParams.neutral(n)`` sets
``alpha_kt = 0``, disables the throttle, and zeroes the bus resistance: the
actuation scale is exactly 1.0, the brownout never fires, and a wrapped
env's trajectory is bit-identical to the unwrapped one (the state is still
integrated, so both arms of a paired study run the same code path). This is
asserted in ``test/envs/test_internals.py`` and is the contract a paired
training study leans on.

Units: SI throughout (K, V, Ω, N·m, s). All functions are pure and
``jit``/``vmap``/``scan``-safe.
"""

from __future__ import annotations

import dataclasses
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp

from ..errors import JaxterityError
from ..physics import bus_voltage

_T_AMB_DEFAULT = 298.15


@dataclasses.dataclass(frozen=True)
class InternalsParams:
    """The internals model coefficients (static per training arm).

    Args
    ----
    C_th, R_th, k_pwr : jax.Array, shape ``(n,)``
        Per-motor thermal capacitance (J/K), thermal resistance (K/W), and
        torque²→power gain (W/(N·m)²) — the single-node model.
    alpha_kt : jax.Array, shape ``(n,)``
        Torque-constant temperature coefficient, 1/K:
        ``K_t(T) = K_t(T_ref) · (1 − alpha_kt·(T − T_ref))``. Datasheet-class
        values are ~0.001–0.003 /K for copper-magnet motors. 0 disables.
    T_ref : float
        Reference temperature for the torque constant, K.
    T_throttle, T_max : jax.Array, shape ``(n,)``
        Protection-firmware derating band, K: full torque below
        ``T_throttle``, linear derate to ``throttle_floor`` at ``T_max``.
        Set ``T_throttle = T_max = +inf`` to disable.
    throttle_floor : float
        The derate floor (the firmware never fully kills a motor mid-motion).
    V_oc : float
        Battery open-circuit voltage, V.
    R_int : float
        Battery internal resistance, Ω. 0 disables the bus (V_bus ≡ V_oc).
    k_i : jax.Array, shape ``(n,)``
        Per-motor torque→current gain, A/(N·m).
    V_min : float
        Brownout threshold, V.
    brownout_dwell_s : float
        How long ``V_bus < V_min`` must persist before the controller
        browns out (real supervisors ride through sub-ms dips).
    T_amb : float
        Ambient temperature, K (the thermal sink; overridden per episode by
        :func:`sample_conditions` in conditioned studies).
    """

    C_th: jax.Array
    R_th: jax.Array
    k_pwr: jax.Array
    alpha_kt: jax.Array
    T_ref: float
    T_throttle: jax.Array
    T_max: jax.Array
    throttle_floor: float
    V_oc: float
    R_int: float
    k_i: jax.Array
    V_min: float
    brownout_dwell_s: float
    T_amb: float

    def __post_init__(self):
        n = jnp.shape(self.C_th)[0]
        for name in ("R_th", "k_pwr", "alpha_kt", "T_throttle", "T_max", "k_i"):
            if jnp.shape(getattr(self, name)) != (n,):
                raise JaxterityError(
                    f"InternalsParams.{name} must be shape ({n},) — one entry "
                    f"per motor — got {jnp.shape(getattr(self, name))}."
                )
        if self.throttle_floor <= 0.0 or self.throttle_floor > 1.0:
            raise JaxterityError(
                f"InternalsParams.throttle_floor must be in (0, 1], got "
                f"{self.throttle_floor} (a zero floor stalls the motor and a "
                ">1 floor would amplify)."
            )

    @property
    def n_motors(self) -> int:
        return int(jnp.shape(self.C_th)[0])

    @classmethod
    def neutral(cls, n_motors: int) -> "InternalsParams":
        """The exact-no-op coefficients (the paired study's A arm).

        The thermal state is still integrated (same code path, same
        compile), but nothing feeds back: ``alpha_kt = 0``, throttle
        disabled, ``R_int = 0`` (no sag, no brownout).
        """
        return cls(
            C_th=jnp.full(n_motors, 100.0),
            R_th=jnp.full(n_motors, 2.0),
            k_pwr=jnp.full(n_motors, 0.1),
            alpha_kt=jnp.zeros(n_motors),
            T_ref=_T_AMB_DEFAULT,
            T_throttle=jnp.full(n_motors, jnp.inf),
            T_max=jnp.full(n_motors, jnp.inf),
            throttle_floor=1.0,
            V_oc=48.0,
            R_int=0.0,
            k_i=jnp.full(n_motors, 1.0),
            V_min=-jnp.inf,
            brownout_dwell_s=jnp.inf,
            T_amb=_T_AMB_DEFAULT,
        )


class InternalsState(NamedTuple):
    """The episode-carried internals state.

    Fields
    ------
    T : jax.Array, shape ``(n,)``
        Per-motor winding temperature, K.
    V_bus : jax.Array, scalar
        Bus voltage at the last step, V.
    brownout_timer : jax.Array, scalar
        Continuous time spent below ``V_min``, s (resets on recovery).
    browned_out : jax.Array, scalar bool
        Latched once the dwell is exceeded (the episode-termination flag).
    """

    T: jax.Array
    V_bus: jax.Array
    brownout_timer: jax.Array
    browned_out: jax.Array


def init_state(params: InternalsParams, T0=None) -> InternalsState:
    """The start-of-episode state (motors at ambient unless carried over).

    Args
    ----
    params : InternalsParams
        The arm's coefficients.
    T0 : array-like | None
        Initial winding temperatures, K — pass the previous episode's final
        ``T`` for soak studies where heat carries across episodes.
    """
    T = (
        jnp.full(params.n_motors, params.T_amb)
        if T0 is None
        else jnp.asarray(T0, dtype=float)
    )
    return InternalsState(
        T=T,
        V_bus=jnp.asarray(params.V_oc, dtype=float),
        brownout_timer=jnp.asarray(0.0),
        browned_out=jnp.asarray(False),
    )


def actuation_scale(params: InternalsParams, state: InternalsState) -> jax.Array:
    """Delivered-per-commanded torque, per motor: ``K_t(T)/K_t(T_ref) · throttle(T)``.

    With neutral params this is exactly 1.0 (no floating-point residue):
    both factors reduce to literal ones.
    """
    kt = 1.0 - params.alpha_kt * (state.T - params.T_ref)
    # Linear derate between T_throttle and T_max, clipped to the floor.
    # Below the threshold the scale is *exactly* 1.0 — selected via where so
    # the disabled case (T_throttle = T_max = inf, whose band is inf − inf =
    # NaN) never contaminates the result.
    band = jnp.maximum(params.T_max - params.T_throttle, 1e-9)
    derate = jnp.clip((params.T_max - state.T) / band, params.throttle_floor, 1.0)
    throttle = jnp.where(state.T <= params.T_throttle, 1.0, derate)
    return jnp.maximum(kt, 0.0) * throttle


def step_internals(
    params: InternalsParams,
    state: InternalsState,
    tau_delivered,
    dt: float,
    T_amb=None,
) -> InternalsState:
    """Advance the internals one control step.

    Args
    ----
    params : InternalsParams
        Coefficients.
    state : InternalsState
        Current state.
    tau_delivered : jax.Array, shape ``(n,)``
        The torque each motor actually produced this step, N·m (i.e. the
        commanded torque × :func:`actuation_scale` — heating and current
        follow delivered torque).
    dt : float
        Step duration, s.
    T_amb : float | jax.Array | None
        Episode ambient override (from :func:`sample_conditions`); defaults
        to ``params.T_amb``.

    Returns
    -------
    InternalsState
    """
    tau = jnp.asarray(tau_delivered, dtype=float)
    ambient = params.T_amb if T_amb is None else T_amb
    # Single-node thermal model, explicit Euler over the control
    # step — control steps (≈ 2–20 ms) are far below the thermal constants.
    dT = (params.k_pwr * tau * tau - (state.T - ambient) / params.R_th) / params.C_th
    T = state.T + dt * dT
    V_bus = bus_voltage(tau, V_oc=params.V_oc, R_int=params.R_int, k_i=params.k_i)
    timer = jnp.where(V_bus < params.V_min, state.brownout_timer + dt, 0.0)
    browned_out = jnp.logical_or(state.browned_out, timer >= params.brownout_dwell_s)
    return InternalsState(
        T=T, V_bus=V_bus, brownout_timer=timer, browned_out=browned_out
    )


def apply_actuation_scale(mjx_model: Any, actuator_ids, scale) -> Any:
    """An MJX model whose selected actuators deliver ``scale ×`` torque.

    Scales the affine actuator rows (``gainprm[:, 0]`` and
    ``biasprm[:, 1:3]`` in lockstep — the torque-constant model for MuJoCo
    ``<position>``/``<motor>`` actuators; the same mechanism as
    :class:`jaxterity.drift.DriftSchedule`'s ``actuator_gain``). Pure and
    jit-safe; call it per control step with :func:`actuation_scale`'s
    output mapped to the actuator ordering.

    Args
    ----
    mjx_model : mjx.Model
        The *nominal* model (scale multiplies its values — don't compound).
    actuator_ids : array-like of int, shape ``(m,)``
        Which actuators.
    scale : jax.Array, shape ``(m,)``
        Per-actuator delivered/commanded factor.
    """
    ids = jnp.asarray(actuator_ids)
    scale = jnp.asarray(scale)
    gain = mjx_model.actuator_gainprm
    bias = mjx_model.actuator_biasprm
    return mjx_model.replace(
        actuator_gainprm=gain.at[ids, 0].set(gain[ids, 0] * scale),
        actuator_biasprm=bias.at[ids, 1:3].set(bias[ids, 1:3] * scale[:, None]),
    )


class ConditionDraw(NamedTuple):
    """One episode's operating conditions (see :func:`sample_conditions`).

    Fields
    ------
    T_amb : jax.Array, scalar
        Ambient temperature, K.
    friction_scale : jax.Array, scalar
        Multiplier on joint Coulomb friction (cold grease is stiffer).
    damping_scale : jax.Array, scalar
        Multiplier on joint viscous damping.
    R_int_scale : jax.Array, scalar
        Multiplier on battery internal resistance (cold cells sag harder).
    """

    T_amb: jax.Array
    friction_scale: jax.Array
    damping_scale: jax.Array
    R_int_scale: jax.Array


def sample_conditions(
    rng: jax.Array,
    *,
    t_amb_range: tuple[float, float] = (258.15, 303.15),
    kappa_friction: float = 0.02,
    kappa_damping: float = 0.02,
    kappa_r_int: float = 0.02,
    T_ref: float = _T_AMB_DEFAULT,
    correlated: bool = True,
) -> ConditionDraw:
    """Draw one episode's temperature-coupled operating conditions.

    Each derived channel is linear in the cold excursion:
    ``scale = 1 + κ·(T_ref − T_amb)`` (clipped at 1 from below — heat does
    not *help* these channels in the modeled range).

    The ``correlated`` flag is the A/B contrast of a correlated-degradation
    study and is designed to be **marginal-preserving**:

    - ``correlated=True``: one latent temperature drives every channel —
      cold episodes have high friction *and* high battery resistance
      together (the physical truth).
    - ``correlated=False``: each channel is driven by its *own* independent
      latent temperature drawn from the same range — every channel's
      marginal distribution is identical to the correlated case by
      construction; only the joint distribution differs. (This is standard
      independent domain randomization with honestly matched magnitudes.)

    Args
    ----
    rng : jax.Array
        PRNG key (per episode).
    t_amb_range : tuple[float, float]
        Ambient range, K. Default −15 °C … +30 °C.
    kappa_friction, kappa_damping, kappa_r_int : float
        Cold-coupling coefficients, 1/K.
    T_ref : float
        The neutral temperature (scales are 1 there and above).
    correlated : bool
        See above.

    Returns
    -------
    ConditionDraw
        ``T_amb`` is the *friction channel's* latent in the independent
        case (each channel still has its own latent internally).
    """
    keys = jax.random.split(rng, 3)
    lo, hi = t_amb_range

    def latent(key):
        return jax.random.uniform(key, minval=lo, maxval=hi)

    t_friction = latent(keys[0])
    t_damping = jnp.where(correlated, t_friction, latent(keys[1]))
    t_battery = jnp.where(correlated, t_friction, latent(keys[2]))

    def cold_scale(t, kappa):
        return 1.0 + kappa * jnp.maximum(T_ref - t, 0.0)

    return ConditionDraw(
        T_amb=t_friction,
        friction_scale=cold_scale(t_friction, kappa_friction),
        damping_scale=cold_scale(t_damping, kappa_damping),
        R_int_scale=cold_scale(t_battery, kappa_r_int),
    )


__all__ = [
    "InternalsParams",
    "InternalsState",
    "ConditionDraw",
    "init_state",
    "actuation_scale",
    "step_internals",
    "apply_actuation_scale",
    "sample_conditions",
]
