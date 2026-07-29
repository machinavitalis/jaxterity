# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Double-acting pneumatic cylinder: the actuator whose state is a pressure.

Jaxonomy's acausal layer already carries a *hydraulic* domain, and hydraulics
gets away with treating the working fluid as incompressible. Air does not.
Compressibility is the whole character of a pneumatic actuator: the chambers
are gas springs, force is only indirectly commandable, and the bandwidth is set
by how fast the valve can move mass through an orifice rather than by any
electrical time constant. So the model's state is not a current or a
temperature — it is the two chamber pressures.

Model — stateful block, state ``[p_a, p_b]`` (Pa absolute)::

    V_a  = V_dead_a + A_a·x                V_b = V_dead_b + A_b·(stroke − x)
    V̇_a  =  A_a·ẋ                          V̇_b = −A_b·ẋ
    ṁ_a  = orifice(p_supply, p_a)·u₊ − orifice(p_a, p_exhaust)·u₋
    ṁ_b  = orifice(p_supply, p_b)·u₋ − orifice(p_b, p_exhaust)·u₊
    ṗ    = (R_air·T·ṁ − p·V̇) / V                         (isothermal chamber)
    F    = p_a·A_a − p_b·A_b − F_friction·tanh(ẋ/v_eps)

with ``u₊ = max(u, 0)``, ``u₋ = max(−u, 0)`` the two halves of the valve
command ``u ∈ [−1, 1]`` — positive fills chamber A and vents B.

Mass flow is the ISO 6358 orifice law (:func:`orifice_flow`), the standard
pneumatic-component characterization: a *sonic conductance* ``C`` and a
*critical pressure ratio* ``b_crit``, both of which every valve datasheet
publishes, with choked flow below ``b_crit`` and an elliptic subsonic branch
above it. The branch switch is smoothed and the smoothing is exposed.

Two named approximations. **Isothermal chambers**: real fills are closer to
adiabatic (the gas heats on compression), which makes this model slightly
optimistic about how fast a chamber pressurizes; the isothermal form keeps the
state at two variables instead of four and is the standard control-oriented
choice. **No valve dynamics**: ``u`` acts instantly, so the spool's own
bandwidth (typically 50–200 Hz) must be modeled upstream if it matters.

Identifiability. The areas and the stroke are geometry — measure them. The
supply pressure is instrumented. What a trace identifies is ``C`` and
``b_crit`` per flow path (from the pressure rise during a commanded fill into a
*known* volume) and the dead volumes (from the rise rate at each end of
travel). Fitting areas from force data instead of measuring them trades a
number you can read off a caliper for one that is confounded with friction.

Units
-----
Pa (absolute), m, m/s, m², kg/s, N, K.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from ..data.schema import PneumaticTelemetry, ValidationReport
from ..parameter import ParameterSpec
from ..sysid.protocol import SysidProtocol
from ._base import Actuator

#: Specific gas constant of dry air, J/(kg·K).
R_AIR = 287.05
#: Reference density of air at ISO 6358 standard conditions, kg/m³.
RHO_REF = 1.185
#: Standard reference temperature for the ISO 6358 flow law, K.
T_REF = 293.15
#: Default smoothing width for the choked/subsonic branch switch (pressure
#: ratio, dimensionless) and for the friction sign (m/s).
_RATIO_EPS_DEFAULT = 1e-3
_V_EPS_DEFAULT = 1e-4
#: Floor on chamber volume so the pressure ODE cannot divide by zero (m³).
_VOLUME_FLOOR = 1e-12

_SYSID_PROTOCOL = SysidProtocol(
    name="pneumatic-fill-and-stroke",
    excitation="step",
    required_columns=("valve_command", "pressure_a", "pressure_b", "position"),
    duration_s=20.0,
    sample_rate_hz=1000.0,
    setup_notes=(
        "Measure the bore, rod diameter and stroke first — they are geometry. "
        "Then: (1) BLOCKED FILL, piston clamped at each end of travel in turn, "
        "step the valve fully open and record the chamber pressure rise. Into "
        "a known, fixed volume the rise rate gives the sonic conductance C "
        "directly, and its curvature as the chamber approaches supply "
        "pressure gives the critical ratio b_crit; the residual volume at the "
        "end stop is the dead volume. (2) FREE STROKE with a known payload, "
        "for the friction force. Log the supply pressure throughout: a "
        "compressor cycling between cut-in and cut-out moves every fitted "
        "number, exactly as a sagging battery does for an electric drive."
    ),
)


def orifice_flow(
    p_up,
    p_down,
    *,
    conductance,
    b_crit=0.3,
    temperature=T_REF,
    ratio_eps: float = _RATIO_EPS_DEFAULT,
):
    """ISO 6358 mass flow through a pneumatic restriction, signed.

    The standard valve characterization::

        ratio = p_down / p_up
        ṁ = C·ρ_ref·p_up·√(T_ref/T)                        if ratio ≤ b_crit
        ṁ = C·ρ_ref·p_up·√(T_ref/T)·√(1 − ((ratio−b)/(1−b))²)   otherwise

    with the choked/subsonic switch smoothed over ``ratio_eps``. The function
    is antisymmetric: if ``p_down > p_up`` the roles swap and the returned flow
    is negative, so a single call handles both fill and vent without a
    caller-side branch.

    Args
    ----
    p_up, p_down : float or jax.Array
        Absolute pressures on each side, Pa. Must be > 0.
    conductance : float or jax.Array
        Sonic conductance ``C``, m³/(s·Pa) — the datasheet number.
    b_crit : float or jax.Array
        Critical pressure ratio in [0, 1) (≈0.3 for a spool valve, ≈0.5 for a
        smooth nozzle).
    temperature : float or jax.Array
        Upstream absolute temperature, K.
    ratio_eps : float
        Smoothing width for the branch switch (dimensionless pressure ratio).

    Returns
    -------
    jax.Array
        Mass flow from the ``p_up`` side to the ``p_down`` side, kg/s
        (negative when the flow reverses).

    Units
    -----
    Pa, m³/(s·Pa), K → kg/s.

    Examples
    --------
    >>> # Deeply choked: flow is proportional to upstream pressure alone.
    >>> m1 = orifice_flow(6e5, 1e5, conductance=1e-8)
    >>> m2 = orifice_flow(12e5, 1e5, conductance=1e-8)
    >>> bool(abs(float(m2 / m1) - 2.0) < 1e-6)
    True
    """
    high = jnp.maximum(p_up, p_down)
    low = jnp.minimum(p_up, p_down)
    direction = jnp.sign(p_up - p_down)
    ratio = low / high
    subsonic_term = 1.0 - ((ratio - b_crit) / (1.0 - b_crit)) ** 2
    subsonic = jnp.sqrt(jnp.clip(subsonic_term, 0.0, 1.0))
    # Smooth blend between the choked branch (factor 1) and the subsonic one.
    blend = 0.5 * (1.0 + jnp.tanh((ratio - b_crit) / ratio_eps))
    factor = (1.0 - blend) + blend * subsonic
    magnitude = conductance * RHO_REF * high * jnp.sqrt(T_REF / temperature) * factor
    return direction * magnitude


def chamber_pressure_rate(
    pressure, volume, volume_rate, mass_flow, *, temperature=T_REF
):
    """Isothermal chamber pressure derivative.

    ``ṗ = (R_air·T·ṁ − p·V̇)/V`` — mass coming in raises pressure, the piston
    opening the chamber lowers it. The volume is floored so a fully collapsed
    chamber (zero dead volume at an end stop) cannot divide by zero.

    Args
    ----
    pressure : float or jax.Array
        Chamber absolute pressure, Pa.
    volume : float or jax.Array
        Chamber volume, m³.
    volume_rate : float or jax.Array
        Rate of change of chamber volume, m³/s.
    mass_flow : float or jax.Array
        Net mass flow into the chamber, kg/s.
    temperature : float or jax.Array
        Gas temperature, K.

    Returns
    -------
    jax.Array
        Pressure derivative, Pa/s.

    Units
    -----
    Pa, m³, m³/s, kg/s, K → Pa/s.
    """
    safe_volume = jnp.maximum(volume, _VOLUME_FLOOR)
    return (R_AIR * temperature * mass_flow - pressure * volume_rate) / safe_volume


class PneumaticCylinder(Actuator):
    """Double-acting cylinder with two chamber-pressure states.

    See the module docstring for the model, the ISO 6358 flow law, and the two
    named approximations (isothermal chambers, instantaneous valve). Input
    ports are ``valve`` (dimensionless in [-1, 1]), ``x`` (m, piston position
    from the chamber-A end cap) and ``v`` (m/s); output ports are ``force``
    (N, on the rod, positive extending), ``p_a`` and ``p_b`` (Pa).

    Args
    ----
    area_a, area_b : float or jax.Array
        Effective piston areas, m². ``area_b`` is the annular (rod) side and
        is the smaller of the two on a single-rod cylinder — the asymmetry
        that makes retract force differ from extend force. Must be > 0.
    stroke : float or jax.Array
        Full stroke length, m. Must be > 0.
    dead_volume_a, dead_volume_b : float or jax.Array
        End-of-travel residual volumes (port, fitting and cushion), m³.
    conductance : float or jax.Array
        Valve sonic conductance ``C`` at full command, m³/(s·Pa).
    b_crit : float or jax.Array
        Valve critical pressure ratio in [0, 1).
    p_supply, p_exhaust : float or jax.Array
        Supply and exhaust absolute pressures, Pa (exhaust is atmospheric,
        ~101325 Pa, *not* zero — gauge pressure has no place in the model).
    f_friction : float or jax.Array
        Seal Coulomb friction force, N. Must be >= 0.
    temperature : float or jax.Array
        Gas temperature, K.
    v_eps : float
        Friction-sign smoothing width, m/s.
    p0_a, p0_b : float
        Initial chamber pressures, Pa (default: exhaust pressure).
    name : str
        Block name.

    Examples
    --------
    >>> cyl = PneumaticCylinder(area_a=1.13e-3, area_b=9.0e-4)
    >>> tuple(cyl.parameter_spec())[:3]
    ('conductance', 'b_crit', 'f_friction')
    """

    sysid_protocol = _SYSID_PROTOCOL

    def __init__(
        self,
        *,
        area_a=1.13e-3,
        area_b=9.0e-4,
        stroke=0.1,
        dead_volume_a=1.0e-6,
        dead_volume_b=1.0e-6,
        conductance=1.0e-8,
        b_crit=0.3,
        p_supply=6.0e5,
        p_exhaust=1.01325e5,
        f_friction=0.0,
        temperature=T_REF,
        v_eps: float = _V_EPS_DEFAULT,
        p0_a: float | None = None,
        p0_b: float | None = None,
        name: str = "pneumatic_cylinder",
    ):
        super().__init__(name=name)
        self._v_eps = float(v_eps)
        for n, x in (
            ("area_a", area_a),
            ("area_b", area_b),
            ("stroke", stroke),
            ("dead_volume_a", dead_volume_a),
            ("dead_volume_b", dead_volume_b),
            ("conductance", conductance),
            ("b_crit", b_crit),
            ("p_supply", p_supply),
            ("p_exhaust", p_exhaust),
            ("f_friction", f_friction),
            ("temperature", temperature),
        ):
            self.declare_dynamic_parameter(n, jnp.asarray(x, dtype=float))
        for port in ("valve", "x", "v"):
            self.declare_input_port(name=port)
        initial = jnp.asarray(
            [
                float(p_exhaust) if p0_a is None else float(p0_a),
                float(p_exhaust) if p0_b is None else float(p0_b),
            ],
            dtype=float,
        )
        self.declare_continuous_state(default_value=initial, ode=self._ode)
        self.declare_output_port(self._force_out, name="force")
        self.declare_output_port(self._p_a_out, name="p_a", requires_inputs=False)
        self.declare_output_port(self._p_b_out, name="p_b", requires_inputs=False)

    @property
    def area_a(self):
        return self.dynamic_parameters["area_a"].get()

    @property
    def area_b(self):
        return self.dynamic_parameters["area_b"].get()

    @property
    def stroke(self):
        return self.dynamic_parameters["stroke"].get()

    @property
    def conductance(self):
        return self.dynamic_parameters["conductance"].get()

    @property
    def b_crit(self):
        return self.dynamic_parameters["b_crit"].get()

    @property
    def p_supply(self):
        return self.dynamic_parameters["p_supply"].get()

    @property
    def p_exhaust(self):
        return self.dynamic_parameters["p_exhaust"].get()

    @property
    def f_friction(self):
        return self.dynamic_parameters["f_friction"].get()

    @property
    def dead_volume_a(self):
        return self.dynamic_parameters["dead_volume_a"].get()

    @property
    def dead_volume_b(self):
        return self.dynamic_parameters["dead_volume_b"].get()

    @property
    def temperature(self):
        return self.dynamic_parameters["temperature"].get()

    @classmethod
    def uncalibrated(cls, *, name: str = "pneumatic_cylinder") -> "PneumaticCylinder":
        """Return an instance with identifiability-neutral defaults.

        A 38 mm-bore-class cylinder with datasheet-order flow parameters and
        **zero** seal friction, so a fit can only add the loss it measures.
        """
        return cls(f_friction=0.0, name=name)

    @staticmethod
    def _volumes(x, params):
        """Chamber volumes (m³) at piston position ``x``."""
        v_a = params["dead_volume_a"] + params["area_a"] * x
        v_b = params["dead_volume_b"] + params["area_b"] * (params["stroke"] - x)
        return v_a, v_b

    def _flows(self, p_a, p_b, valve, params):
        """Net mass flow into each chamber (kg/s) at a valve command."""
        fill = jnp.maximum(valve, 0.0)
        vent = jnp.maximum(-valve, 0.0)
        kwargs = dict(
            b_crit=params["b_crit"],
            temperature=params["temperature"],
        )
        conductance = params["conductance"]
        exhaust = params["p_exhaust"]

        def through(p_up, p_down):
            return orifice_flow(p_up, p_down, conductance=conductance, **kwargs)

        m_a = fill * through(params["p_supply"], p_a) - vent * through(p_a, exhaust)
        m_b = vent * through(params["p_supply"], p_b) - fill * through(p_b, exhaust)
        return m_a, m_b

    def _ode(self, time, state, *inputs, **params):
        p_a, p_b = state.continuous_state
        valve, x, v = inputs
        v_a, v_b = self._volumes(x, params)
        m_a, m_b = self._flows(p_a, p_b, valve, params)
        return jnp.array(
            [
                chamber_pressure_rate(
                    p_a,
                    v_a,
                    params["area_a"] * v,
                    m_a,
                    temperature=params["temperature"],
                ),
                chamber_pressure_rate(
                    p_b,
                    v_b,
                    -params["area_b"] * v,
                    m_b,
                    temperature=params["temperature"],
                ),
            ]
        )

    def _force_out(self, time, state, *inputs, **params):
        p_a, p_b = state.continuous_state
        v = inputs[2]
        return (
            p_a * params["area_a"]
            - p_b * params["area_b"]
            - params["f_friction"] * jnp.tanh(v / self._v_eps)
        )

    def _p_a_out(self, time, state, *inputs, **params):
        return state.continuous_state[0]

    def _p_b_out(self, time, state, *inputs, **params):
        return state.continuous_state[1]

    def parameter_spec(self) -> ParameterSpec:
        """The leaves a trace can identify — geometry is measured, not fitted."""
        return ParameterSpec(
            ("conductance", "b_crit", "f_friction", "dead_volume_a", "dead_volume_b")
        )

    def validate(self, telemetry: PneumaticTelemetry) -> ValidationReport:
        """Forward-simulate the chamber pressures against a recorded trace.

        Integrates the two-state pressure ODE with a fixed-step NumPy RK4
        (zero-order hold on the valve command and the measured piston motion —
        the piston trajectory is *taken from the data*, so this scores the
        pneumatics alone and not the load dynamics), seeded from the trace's
        first pressure samples.

        Args
        ----
        telemetry : PneumaticTelemetry
            Valve command, chamber pressures, and piston position.

        Returns
        -------
        ValidationReport
            ``ok`` iff both chamber-pressure RMS residuals are within 1 % of
            the supply pressure — the scale a pneumatic model is honest at,
            and stated here rather than borrowed from a tolerance policy that
            has no pressure entry.

        Units
        -----
        Pa, m.
        """
        t = np.asarray(telemetry.time, dtype=np.float64)
        valve = np.asarray(telemetry.valve_command, dtype=np.float64)
        x = np.asarray(telemetry.position, dtype=np.float64)
        p_a_meas = np.asarray(telemetry.pressure_a, dtype=np.float64)
        p_b_meas = np.asarray(telemetry.pressure_b, dtype=np.float64)
        v = np.gradient(x, t)

        params = {
            name: float(getattr(self, name))
            for name in (
                "area_a",
                "area_b",
                "stroke",
                "dead_volume_a",
                "dead_volume_b",
                "conductance",
                "b_crit",
                "p_supply",
                "p_exhaust",
                "temperature",
            )
        }

        def rhs(p, xi, vi, u):
            v_a, v_b = self._volumes(xi, params)
            m_a, m_b = self._flows(p[0], p[1], u, params)
            return np.array(
                [
                    float(
                        chamber_pressure_rate(
                            p[0],
                            v_a,
                            params["area_a"] * vi,
                            m_a,
                            temperature=params["temperature"],
                        )
                    ),
                    float(
                        chamber_pressure_rate(
                            p[1],
                            v_b,
                            -params["area_b"] * vi,
                            m_b,
                            temperature=params["temperature"],
                        )
                    ),
                ]
            )

        simulated = np.empty((t.size, 2))
        simulated[0] = [p_a_meas[0], p_b_meas[0]]
        state = simulated[0].copy()
        for i in range(1, t.size):
            dt = float(t[i] - t[i - 1])
            u, xi, vi = float(valve[i - 1]), float(x[i - 1]), float(v[i - 1])
            k1 = rhs(state, xi, vi, u)
            k2 = rhs(state + 0.5 * dt * k1, xi, vi, u)
            k3 = rhs(state + 0.5 * dt * k2, xi, vi, u)
            k4 = rhs(state + dt * k3, xi, vi, u)
            state = state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            simulated[i] = state

        report = ValidationReport()
        bound = 0.01 * float(self.p_supply)
        for label, sim, meas in (
            ("A", simulated[:, 0], p_a_meas),
            ("B", simulated[:, 1], p_b_meas),
        ):
            residual = sim - meas
            rms = float(np.sqrt(np.mean(residual * residual)))
            if rms > bound:
                report.warnings.append(
                    f"chamber {label} pressure residual RMS={rms:.4g} Pa "
                    f"exceeds 1% of supply ({bound:.4g} Pa) — the valve "
                    "conductance, the critical ratio, or the dead volume may "
                    "not match this trace."
                )
        return report


__all__ = [
    "PneumaticCylinder",
    "R_AIR",
    "RHO_REF",
    "T_REF",
    "chamber_pressure_rate",
    "orifice_flow",
]
