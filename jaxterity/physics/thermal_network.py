# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Thermal, beyond one node: an acausal winding/housing/driver network.

:class:`~jaxterity.physics.thermal.ServoThermal` is one lumped node: a single
temperature that heats under ``k_pwr·τ²`` and cools toward ambient. That model
answers "does this servo cook on this duty cycle" and nothing else, and it
answers it with the wrong sensor: the thermistor a bus servo reports is in the
*case*, while the thing that fails is the *winding*, tens of kelvin hotter and
seconds — not minutes — faster. It also cannot see a neighbor: on a dense arm
or a humanoid hip, servos share a bracket, and a saturated joint warms the one
next to it.

This module builds the network that can, on Jaxonomy's acausal thermal domain:

    ambient ──R_ha── housing ──R_wh── winding ◄── P_winding (causal input)
        │               │
        │               ├──R_ab── neighbouring housing …
        └──R_da── driver ◄── P_driver (causal input), optionally ──R_dh── housing

Per servo, two nodes (winding ``C_w``, housing ``C_h``) plus an optional
driver/amplifier node (``C_d``) which may be thermally tied to its servo's
housing or isolated on its own heatsink. Inter-servo conduction is declared
edge by edge. Ambient is an ideal reservoir.

Inputs are **dissipated power in watts**, not torque: the torque²-proxy
(:func:`~jaxterity.physics.thermal.dissipated_power`) stays a causal block
upstream, so a user who has measured ``I²R`` uses it directly and never has to
launder real watts through a torque gain.

Identifiability — read before fitting. With the winding *and* the housing both
instrumented and the injected power known in watts, all four per-servo
parameters (``C_w``, ``R_wh``, ``C_h``, ``R_ha``) are identifiable:
:func:`fit_two_node` regresses each node's energy balance separately, and the
two balances see four independent coefficients. This is the payoff for the
second sensor. Every weakening of that setup collapses it:

* **Housing only** (the usual bus servo): the winding node is unobserved, the
  response is dominated by the slow pole, and you are back to the single-node
  contract — fit :class:`~jaxterity.physics.thermal.ServoThermal` instead.
* **Torque instead of watts**: the unknown ``k_pwr`` multiplies into ``1/C_w``
  exactly as in the lumped model, so ``C_w`` must be supplied.
* **Coarse temperature quantization**: a servo that reports 1 °C steps puts a
  hard floor under every one of these estimates. That is a *sensor* fact, so
  model it — see :class:`~jaxterity.sensors.temperature_sensor.TemperatureSensor`
  and :func:`~jaxterity.sensors.temperature_sensor.quantization_floor`, which
  turns the quantization step into the standard error the fit cannot beat.

Units
-----
K, W, J/K, K/W, s.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any

import jax.numpy as jnp
import numpy as np
from jaxonomy.acausal import AcausalCompiler, AcausalDiagram, EqnEnv
from jaxonomy.acausal.component_library import thermal as th

from ..data.schema import ThermalTelemetry, ValidationReport
from ..errors import CalibrationError, JaxterityError
from .power_tree import _CONDITION_NUMBER_THRESHOLD, _sympy

#: Default ambient temperature (K) — 25 °C, matching ``ServoThermal``.
_T_AMB_DEFAULT = 298.15

#: Guard on the repeated-eigenvalue branch of the 2×2 matrix exponential.
_EIG_EPS = 1e-12


# --------------------------------------------------------------------------
# Analytic two-node model (the reference the network is checked against)
# --------------------------------------------------------------------------


def two_node_steady_state(power, *, T_amb, R_wh, R_ha):
    """Steady-state winding and housing temperature under a constant power.

    Setting both derivatives to zero, all the injected power must leave through
    the housing-to-ambient path, and the winding sits one more drop above it::

        T_h = T_amb + P·R_ha
        T_w = T_h   + P·R_wh = T_amb + P·(R_wh + R_ha)

    Neither depends on the heat capacities — those set only *how fast* the
    node gets there, which is exactly why a steady-state-only trace cannot
    identify them.

    Args
    ----
    power : float or jax.Array
        Dissipated power in the winding, W.
    T_amb : float or jax.Array
        Ambient temperature, K.
    R_wh : float or jax.Array
        Winding→housing thermal resistance, K/W.
    R_ha : float or jax.Array
        Housing→ambient thermal resistance, K/W.

    Returns
    -------
    tuple[jax.Array, jax.Array]
        ``(T_winding, T_housing)``, K.

    Units
    -----
    power: W; temperatures: K; resistances: K/W.

    Examples
    --------
    >>> tw, th = two_node_steady_state(4.0, T_amb=300.0, R_wh=0.8, R_ha=3.0)
    >>> float(th), float(tw)
    (312.0, 315.2)
    """
    T_h = T_amb + power * R_ha
    return T_h + power * R_wh, T_h


def _two_node_matrix(C_w, C_h, R_wh, R_ha):
    """The 2×2 state matrix of the two-node model (winding, housing)."""
    g_wh = 1.0 / R_wh
    g_ha = 1.0 / R_ha
    return jnp.array(
        [
            [-g_wh / C_w, g_wh / C_w],
            [g_wh / C_h, -(g_wh + g_ha) / C_h],
        ]
    )


def two_node_response(t, power, *, T0_w, T0_h, T_amb, C_w, C_h, R_wh, R_ha):
    """Analytic two-node temperature response to a constant power step.

    Closed-form solution of the coupled pair::

        C_w·Ṫ_w = P − (T_w − T_h)/R_wh
        C_h·Ṫ_h =     (T_w − T_h)/R_wh − (T_h − T_amb)/R_ha

    as ``x(t) = x_ss + exp(A·t)·(x₀ − x_ss)``, with the 2×2 matrix exponential
    evaluated in closed form (Sylvester's formula). The state matrix is
    similar to a symmetric matrix under ``diag(√C_w, √C_h)``, so both
    eigenvalues are real and negative — the response is two decaying
    exponentials, a fast winding pole and a slow housing pole, never an
    oscillation. The repeated-eigenvalue case falls back to the
    ``e^{λt}(I + (A − λI)t)`` limit.

    Args
    ----
    t : float or jax.Array
        Time since the power step, s.
    power : float or jax.Array
        Dissipated power in the winding, W (constant over the step).
    T0_w, T0_h : float or jax.Array
        Initial winding / housing temperature, K.
    T_amb : float or jax.Array
        Ambient temperature, K.
    C_w, C_h : float or jax.Array
        Winding / housing heat capacity, J/K.
    R_wh, R_ha : float or jax.Array
        Winding→housing and housing→ambient resistance, K/W.

    Returns
    -------
    tuple[jax.Array, jax.Array]
        ``(T_winding(t), T_housing(t))``, K.

    Units
    -----
    t: s; power: W; temperatures: K; C: J/K; R: K/W.
    """
    A = _two_node_matrix(C_w, C_h, R_wh, R_ha)
    ss_w, ss_h = two_node_steady_state(power, T_amb=T_amb, R_wh=R_wh, R_ha=R_ha)
    y0 = jnp.stack([jnp.asarray(T0_w) - ss_w, jnp.asarray(T0_h) - ss_h])

    trace = A[0, 0] + A[1, 1]
    det = A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]
    disc = jnp.sqrt(jnp.maximum(trace * trace - 4.0 * det, 0.0))
    lam_p = 0.5 * (trace + disc)
    lam_m = 0.5 * (trace - disc)
    eye = jnp.eye(2)
    gap = lam_p - lam_m
    distinct = (
        jnp.exp(lam_p * t) * (A - lam_m * eye) - jnp.exp(lam_m * t) * (A - lam_p * eye)
    ) / jnp.where(jnp.abs(gap) < _EIG_EPS, 1.0, gap)
    repeated = jnp.exp(lam_p * t) * (eye + (A - lam_p * eye) * t)
    expm = jnp.where(jnp.abs(gap) < _EIG_EPS, repeated, distinct)
    y = expm @ y0
    return y[0] + ss_w, y[1] + ss_h


# --------------------------------------------------------------------------
# Declarative network
# --------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class DriverSpec:
    """A servo's driver/amplifier as its own thermal node.

    The half of the electrical loss that never enters the motor: MOSFET
    conduction and switching loss in the driver stage. Whether it matters
    depends on packaging — a bus servo integrates the driver into the case
    (``R_dh`` small: the driver warms the housing), while a separate amplifier
    on its own heatsink is effectively isolated (``R_dh=None``).

    Args
    ----
    C_th : float
        Driver heat capacity, J/K. Must be > 0.
    R_da : float
        Driver→ambient thermal resistance, K/W. Must be > 0.
    R_dh : float | None
        Driver→(its servo's) housing resistance, K/W. ``None`` = thermally
        isolated from the servo.

    Units
    -----
    J/K, K/W.
    """

    C_th: float
    R_da: float
    R_dh: float | None = None

    def __post_init__(self) -> None:
        if self.C_th <= 0 or self.R_da <= 0:
            raise JaxterityError(
                f"DriverSpec needs C_th > 0 and R_da > 0, got {self.C_th} and "
                f"{self.R_da}."
            )
        if self.R_dh is not None and self.R_dh <= 0:
            raise JaxterityError(
                f"DriverSpec.R_dh must be > 0 when set (use None for a driver "
                f"thermally isolated from its servo), got {self.R_dh}."
            )


@dataclasses.dataclass(frozen=True)
class ServoThermalSpec:
    """One servo as a winding node, a housing node, and an optional driver.

    Args
    ----
    name : str
        Servo name; must be a valid identifier (it names the network's ports).
    C_w : float
        Winding heat capacity, J/K — small, so the winding is the fast pole.
    C_h : float
        Housing/case heat capacity, J/K — the slow pole.
    R_wh : float
        Winding→housing thermal resistance, K/W.
    R_ha : float
        Housing→ambient thermal resistance, K/W (convection + radiation,
        lumped).
    driver : DriverSpec | None
        Optional driver/amplifier node.

    Units
    -----
    J/K, K/W.
    """

    name: str
    C_w: float
    C_h: float
    R_wh: float
    R_ha: float
    driver: DriverSpec | None = None

    def __post_init__(self) -> None:
        if not self.name.isidentifier():
            raise JaxterityError(
                f"ServoThermalSpec.name {self.name!r} must be a valid Python "
                "identifier — it becomes an acausal component and port name."
            )
        if min(self.C_w, self.C_h, self.R_wh, self.R_ha) <= 0:
            raise JaxterityError(
                f"ServoThermalSpec {self.name!r} needs strictly positive "
                f"C_w/C_h/R_wh/R_ha, got {self.C_w}, {self.C_h}, {self.R_wh}, "
                f"{self.R_ha}."
            )


@dataclasses.dataclass(frozen=True)
class ConductionSpec:
    """A conduction path between two servo housings (a shared bracket).

    Args
    ----
    a, b : str
        The two servo names. Must differ and both be declared.
    R_ab : float
        Housing→housing thermal resistance, K/W. Must be > 0.

    Units
    -----
    K/W.
    """

    a: str
    b: str
    R_ab: float

    def __post_init__(self) -> None:
        if self.a == self.b:
            raise JaxterityError(
                f"ConductionSpec connects a servo to itself ({self.a!r}); a "
                "self-edge carries no heat."
            )
        if self.R_ab <= 0:
            raise JaxterityError(
                f"ConductionSpec({self.a!r}, {self.b!r}) needs R_ab > 0, got "
                f"{self.R_ab}. Use a large resistance for a weak path."
            )


@dataclasses.dataclass(frozen=True)
class AmbientSpec:
    """The ambient reservoir every housing and driver ultimately cools into.

    Args
    ----
    T_amb : float
        Ambient temperature, K. Must be > 0.

    Units
    -----
    K.
    """

    T_amb: float = _T_AMB_DEFAULT

    def __post_init__(self) -> None:
        if self.T_amb <= 0:
            raise JaxterityError(
                f"AmbientSpec.T_amb is an absolute temperature in kelvin and "
                f"must be > 0, got {self.T_amb}. (25 °C is 298.15 K — a "
                "celsius value here is the usual cause.)"
            )


@dataclasses.dataclass(frozen=True)
class ThermalNetworkSpec:
    """A robot's thermal network: servos, conduction edges, ambient.

    Args
    ----
    servos : tuple[ServoThermalSpec, ...]
        Every modeled servo. Names must be unique.
    conduction : tuple[ConductionSpec, ...]
        Housing↔housing conduction edges (a shared bracket, a common plate).
    ambient : AmbientSpec
        The reservoir.

    Raises
    ------
    JaxterityError
        On duplicate servo names or a conduction edge naming an undeclared
        servo.
    """

    servos: tuple[ServoThermalSpec, ...]
    conduction: tuple[ConductionSpec, ...] = ()
    ambient: AmbientSpec = dataclasses.field(default_factory=AmbientSpec)

    def __post_init__(self) -> None:
        if not self.servos:
            raise JaxterityError(
                "ThermalNetworkSpec needs at least one servo; an empty network "
                "has no nodes to integrate."
            )
        names = [s.name for s in self.servos]
        duplicates = sorted({n for n in names if names.count(n) > 1})
        if duplicates:
            raise JaxterityError(
                f"ThermalNetworkSpec has duplicate servo name(s) {duplicates}."
            )
        for edge in self.conduction:
            for side in (edge.a, edge.b):
                if side not in names:
                    raise JaxterityError(
                        f"Conduction edge ({edge.a!r}, {edge.b!r}) names servo "
                        f"{side!r}, which is not declared. Known servos: {names}."
                    )

    @property
    def nodes(self) -> tuple[str, ...]:
        """Every node's canonical name, e.g. ``"elbow.winding"``.

        Order: for each servo in declaration order, ``winding``, ``housing``,
        then ``driver`` when present. This is the row/column order of
        :func:`steady_state_temperatures`' linear system.
        """
        out: list[str] = []
        for servo in self.servos:
            out.append(f"{servo.name}.winding")
            out.append(f"{servo.name}.housing")
            if servo.driver is not None:
                out.append(f"{servo.name}.driver")
        return tuple(out)

    def servo(self, name: str) -> ServoThermalSpec:
        """The :class:`ServoThermalSpec` named ``name``.

        Raises
        ------
        JaxterityError
            If no servo has that name.
        """
        for servo in self.servos:
            if servo.name == name:
                return servo
        raise JaxterityError(
            f"No servo named {name!r} in this thermal network. Known servos: "
            f"{[s.name for s in self.servos]}."
        )


def _resolve_node(spec: ThermalNetworkSpec, key: str) -> str:
    """Canonical node name for a user-supplied key (bare name → winding)."""
    if key in spec.nodes:
        return key
    if "." not in key:
        candidate = f"{key}.winding"
        if candidate in spec.nodes:
            return candidate
    raise CalibrationError(
        f"Unknown thermal node {key!r}. Known nodes: {list(spec.nodes)} (a bare "
        "servo name is shorthand for its winding node)."
    )


def steady_state_temperatures(
    spec: ThermalNetworkSpec, powers: Mapping[str, float]
) -> dict[str, float]:
    """Solve the whole network's steady state (a linear conductance system).

    At steady state every capacity drops out and the network is
    ``G·(T − T_amb) = P``, with ``G`` the node conductance matrix assembled
    from every declared resistance. Solving it directly is both the fast path
    for a duty-cycle study and the closed-form reference the compiled acausal
    network is checked against — including the effects a per-servo lumped model
    structurally cannot show: a saturated joint raising its neighbour's floor
    through the bracket, and a driver warming its own servo's housing.

    Args
    ----
    spec : ThermalNetworkSpec
        The network.
    powers : Mapping[str, float]
        Dissipated power per node, W. Keys are node names
        (``"elbow.winding"``, ``"elbow.driver"``); a bare servo name is
        shorthand for its winding. Omitted nodes dissipate nothing.

    Returns
    -------
    dict[str, float]
        Node name → steady-state temperature, K.

    Raises
    ------
    CalibrationError
        If a key names no node in this network.

    Units
    -----
    W in, K out.

    Examples
    --------
    >>> spec = ThermalNetworkSpec(
    ...     servos=(ServoThermalSpec("j0", C_w=20, C_h=200, R_wh=0.8, R_ha=3.0),),
    ...     ambient=AmbientSpec(T_amb=300.0),
    ... )
    >>> temps = steady_state_temperatures(spec, {"j0": 4.0})
    >>> round(temps["j0.housing"], 6), round(temps["j0.winding"], 6)
    (312.0, 315.2)
    """
    nodes = spec.nodes
    index = {name: i for i, name in enumerate(nodes)}
    n = len(nodes)
    conductance = np.zeros((n, n), dtype=float)
    load = np.zeros(n, dtype=float)

    def couple(node_a: str, node_b: str, resistance: float) -> None:
        i, j = index[node_a], index[node_b]
        g = 1.0 / resistance
        conductance[i, i] += g
        conductance[j, j] += g
        conductance[i, j] -= g
        conductance[j, i] -= g

    def to_ambient(node: str, resistance: float) -> None:
        i = index[node]
        conductance[i, i] += 1.0 / resistance

    for servo in spec.servos:
        couple(f"{servo.name}.winding", f"{servo.name}.housing", servo.R_wh)
        to_ambient(f"{servo.name}.housing", servo.R_ha)
        if servo.driver is not None:
            to_ambient(f"{servo.name}.driver", servo.driver.R_da)
            if servo.driver.R_dh is not None:
                couple(
                    f"{servo.name}.driver", f"{servo.name}.housing", servo.driver.R_dh
                )
    for edge in spec.conduction:
        couple(f"{edge.a}.housing", f"{edge.b}.housing", edge.R_ab)

    for key, value in powers.items():
        load[index[_resolve_node(spec, key)]] += float(value)

    rise = np.linalg.solve(conductance, load)
    return {name: spec.ambient.T_amb + float(rise[i]) for i, name in enumerate(nodes)}


@dataclasses.dataclass(frozen=True)
class ThermalNetwork:
    """A compiled thermal network: the acausal system plus its port index maps.

    Build one with :meth:`build`. Inputs are per-node dissipated power (W);
    outputs are per-node absolute temperature (K). Use the index maps rather
    than reconstructing port names.

    **Every** node carries a power input — including housings and drivers that
    nothing drives in your setup — and an acausal input port has no default
    source, so wire the unexcited ones to a zero constant or the diagram will
    refuse to evaluate. Iterating :attr:`power_inputs` and defaulting to 0.0 is
    the idiom.

    Attributes
    ----------
    spec : ThermalNetworkSpec
        The network this was compiled from.
    system : Any
        The compiled Jaxonomy ``AcausalSystem``.
    power_inputs : dict[str, int]
        Node name → input-port index (W, positive = heating).
    temperature_outputs : dict[str, int]
        Node name → output-port index (K).
    """

    spec: ThermalNetworkSpec
    system: Any
    power_inputs: dict[str, int]
    temperature_outputs: dict[str, int]

    @classmethod
    def build(
        cls,
        spec: ThermalNetworkSpec,
        *,
        initial_temperature: float | None = None,
        verbose: bool = False,
    ) -> "ThermalNetwork":
        """Compile a :class:`ThermalNetworkSpec` into an acausal Jaxonomy system.

        Args
        ----
        spec : ThermalNetworkSpec
            The declared network.
        initial_temperature : float | None
            Starting temperature of every capacity, K (default: ambient — the
            robot has been sitting idle).
        verbose : bool
            Pass through to ``AcausalCompiler``.

        Returns
        -------
        ThermalNetwork
            The compiled system and its port index maps.

        Raises
        ------
        JaxterityError
            If ``sympy`` (the ``[acausal]`` extra) is not installed.
        """
        _sympy()  # fail early, with the actionable message
        T0 = spec.ambient.T_amb if initial_temperature is None else initial_temperature
        ev = EqnEnv()
        diagram = AcausalDiagram()
        ambient = th.TemperatureSource(
            ev, name="ambient", temperature=spec.ambient.T_amb
        )

        capacities: dict[str, Any] = {}
        power_names: dict[str, str] = {}
        for servo in spec.servos:
            nodes = [
                (f"{servo.name}.winding", f"{servo.name}_winding", servo.C_w),
                (f"{servo.name}.housing", f"{servo.name}_housing", servo.C_h),
            ]
            if servo.driver is not None:
                nodes.append(
                    (f"{servo.name}.driver", f"{servo.name}_driver", servo.driver.C_th)
                )
            for node, component_name, capacity in nodes:
                mass = th.HeatCapacitor(
                    ev,
                    name=component_name,
                    C=capacity,
                    initial_temperature=T0,
                    initial_temperature_fixed=True,
                )
                capacities[node] = mass
                # Heat is injected between ambient (port_a) and the node
                # (port_b): with this wiring a positive input is positive
                # heating, and the reservoir absorbs the bookkeeping.
                source = th.HeatflowSource(
                    ev,
                    name=f"{component_name}_heat",
                    enable_heat_port=True,
                    enable_port_b=True,
                )
                diagram.connect(source, "port_a", ambient, "port")
                diagram.connect(source, "port_b", mass, "port")
                power_names[node] = f"{component_name}_heat_Q_flow"
                sensor = th.TemperatureSensor(
                    ev, name=f"{component_name}_temp", enable_port_b=False
                )
                diagram.connect(sensor, "port_a", mass, "port")

            winding = capacities[f"{servo.name}.winding"]
            housing = capacities[f"{servo.name}.housing"]
            r_wh = th.Insulator(ev, name=f"{servo.name}_r_wh", R=servo.R_wh)
            diagram.connect(r_wh, "port_a", winding, "port")
            diagram.connect(r_wh, "port_b", housing, "port")
            r_ha = th.Insulator(ev, name=f"{servo.name}_r_ha", R=servo.R_ha)
            diagram.connect(r_ha, "port_a", housing, "port")
            diagram.connect(r_ha, "port_b", ambient, "port")
            if servo.driver is not None:
                driver = capacities[f"{servo.name}.driver"]
                r_da = th.Insulator(ev, name=f"{servo.name}_r_da", R=servo.driver.R_da)
                diagram.connect(r_da, "port_a", driver, "port")
                diagram.connect(r_da, "port_b", ambient, "port")
                if servo.driver.R_dh is not None:
                    r_dh = th.Insulator(
                        ev, name=f"{servo.name}_r_dh", R=servo.driver.R_dh
                    )
                    diagram.connect(r_dh, "port_a", driver, "port")
                    diagram.connect(r_dh, "port_b", housing, "port")

        for k, edge in enumerate(spec.conduction):
            link = th.Insulator(ev, name=f"conduction_{k}", R=edge.R_ab)
            diagram.connect(link, "port_a", capacities[f"{edge.a}.housing"], "port")
            diagram.connect(link, "port_b", capacities[f"{edge.b}.housing"], "port")

        system = AcausalCompiler(
            ev,
            diagram,
            verbose=verbose,
            condition_number_threshold=_CONDITION_NUMBER_THRESHOLD,
        )()

        port_in = {p.name: i for i, p in enumerate(system.input_ports)}
        port_out = {p.name: i for i, p in enumerate(system.output_ports)}
        power_inputs = {node: port_in[name] for node, name in power_names.items()}
        temperature_outputs = {
            node: port_out[f"{node.replace('.', '_')}_temp_T_rel"]
            for node in spec.nodes
        }
        return cls(
            spec=spec,
            system=system,
            power_inputs=power_inputs,
            temperature_outputs=temperature_outputs,
        )


# --------------------------------------------------------------------------
# Fitting and validation
# --------------------------------------------------------------------------


def fit_two_node(
    telemetry_winding: ThermalTelemetry,
    telemetry_housing: ThermalTelemetry,
    power,
    *,
    T_amb: float,
    servo_index: int = 0,
) -> dict[str, float]:
    """Identify one servo's four two-node parameters from both temperatures.

    Regresses each node's energy balance in its identifiable form, using
    central differences for the derivatives::

        Ṫ_w = a₁·P − a₂·(T_w − T_h)      a₁ = 1/C_w,  a₂ = 1/(C_w·R_wh)
        Ṫ_h = b₁·(T_w − T_h) − b₂·(T_h − T_amb)
                                         b₁ = 1/(C_h·R_wh), b₂ = 1/(C_h·R_ha)

    Four coefficients, four parameters, no free direction — *provided* the
    power is in watts and the winding is genuinely instrumented. Read the
    module docstring before assuming it is.

    Args
    ----
    telemetry_winding, telemetry_housing : ThermalTelemetry
        Winding and housing temperature traces on the same clock, K.
    power : numpy.ndarray, shape ``(n,)``
        Dissipated power aligned with the timestamps, W.
    T_amb : float
        Ambient temperature, K.
    servo_index : int
        Which column of each trace to fit.

    Returns
    -------
    dict[str, float]
        ``{"C_w", "R_wh", "C_h", "R_ha"}`` in J/K and K/W.

    Raises
    ------
    CalibrationError
        If the traces disagree in length or clock, are too short to
        differentiate, or the regression is degenerate (a trace with no power
        variation and no winding/housing gradient identifies nothing).

    Units
    -----
    K, W in; J/K and K/W out.
    """
    t = np.asarray(telemetry_winding.time, dtype=np.float64)
    t_h = np.asarray(telemetry_housing.time, dtype=np.float64)
    if t.shape != t_h.shape or not np.allclose(t, t_h):
        raise CalibrationError(
            "fit_two_node needs the winding and housing traces on the same "
            f"clock; got {t.size} and {t_h.size} sample(s). Fix: resample both "
            "onto one time base at the loader boundary."
        )
    if t.size < 5:
        raise CalibrationError(
            f"fit_two_node needs at least 5 samples to central-difference both "
            f"nodes, got {t.size}."
        )
    T_w = np.asarray(telemetry_winding.temperature, dtype=np.float64)[:, servo_index]
    T_h = np.asarray(telemetry_housing.temperature, dtype=np.float64)[:, servo_index]
    P = np.asarray(power, dtype=np.float64)
    if P.shape[0] != t.size:
        raise CalibrationError(
            f"fit_two_node got {P.shape[0]} power sample(s) for {t.size} "
            "temperature sample(s). Fix: align power with the telemetry clock."
        )

    dTw = (T_w[2:] - T_w[:-2]) / (t[2:] - t[:-2])
    dTh = (T_h[2:] - T_h[:-2]) / (t[2:] - t[:-2])
    gradient = (T_w - T_h)[1:-1]
    over_ambient = (T_h - T_amb)[1:-1]
    P_c = P[1:-1]

    design_w = np.stack([P_c, -gradient], axis=1)
    design_h = np.stack([gradient, -over_ambient], axis=1)
    for design, label in ((design_w, "winding"), (design_h, "housing")):
        if np.linalg.matrix_rank(design, tol=1e-12) < 2:
            raise CalibrationError(
                f"fit_two_node: the {label} regression is rank-deficient — this "
                "trace does not separate its two coefficients. Fix: excite with "
                "a power step large enough to open a winding/housing gradient, "
                "and record through both the rise and the decay."
            )
    (a1, a2), *_ = np.linalg.lstsq(design_w, dTw, rcond=None)
    (b1, b2), *_ = np.linalg.lstsq(design_h, dTh, rcond=None)
    if a1 <= 0 or a2 <= 0 or b1 <= 0 or b2 <= 0:
        raise CalibrationError(
            f"fit_two_node produced a non-physical coefficient (a1={a1:.3g}, "
            f"a2={a2:.3g}, b1={b1:.3g}, b2={b2:.3g}); every one must be > 0. "
            "This is the signature of a sign error in the power channel, a "
            "swapped winding/housing pair, or a trace dominated by sensor "
            "quantization."
        )
    C_w = 1.0 / a1
    R_wh = a1 / a2
    C_h = 1.0 / (b1 * R_wh)
    R_ha = 1.0 / (C_h * b2)
    return {"C_w": C_w, "R_wh": R_wh, "C_h": C_h, "R_ha": R_ha}


def validate_two_node(
    telemetry_winding: ThermalTelemetry,
    telemetry_housing: ThermalTelemetry,
    power,
    servo: ServoThermalSpec,
    *,
    T_amb: float,
    servo_index: int = 0,
) -> ValidationReport:
    """Forward-simulate one servo's two-node model against measured traces.

    Integrates the coupled pair with a fixed-step NumPy RK4 between samples
    (zero-order hold on power, matching a real recording), seeded from the
    first sample of each trace, and reports the RMS residual of both nodes.

    Args
    ----
    telemetry_winding, telemetry_housing : ThermalTelemetry
        Measured traces, K.
    power : numpy.ndarray, shape ``(n,)``
        Dissipated power, W.
    servo : ServoThermalSpec
        The parameters under test.
    T_amb : float
        Ambient temperature, K.
    servo_index : int
        Which column of each trace to score.

    Returns
    -------
    ValidationReport
        ``ok`` iff both nodes' RMS residuals are within the
        ``motor_temperature`` policy bound.

    Units
    -----
    K, W.
    """
    from ..testing import tolerances as tol

    t = np.asarray(telemetry_winding.time, dtype=np.float64)
    T_w_meas = np.asarray(telemetry_winding.temperature, dtype=np.float64)[
        :, servo_index
    ]
    T_h_meas = np.asarray(telemetry_housing.temperature, dtype=np.float64)[
        :, servo_index
    ]
    P = np.asarray(power, dtype=np.float64)

    def rhs(x: np.ndarray, p: float) -> np.ndarray:
        """Derivative of ``[T_w, T_h]`` under a dissipated power ``p``."""
        through = (x[0] - x[1]) / servo.R_wh
        return np.array(
            [
                (p - through) / servo.C_w,
                (through - (x[1] - T_amb) / servo.R_ha) / servo.C_h,
            ]
        )

    T_w_sim = np.empty_like(T_w_meas)
    T_h_sim = np.empty_like(T_h_meas)
    T_w_sim[0], T_h_sim[0] = T_w_meas[0], T_h_meas[0]
    state = np.array([T_w_meas[0], T_h_meas[0]])
    for i in range(1, t.size):
        dt = float(t[i] - t[i - 1])
        u = float(P[i - 1])
        k1 = rhs(state, u)
        k2 = rhs(state + 0.5 * dt * k1, u)
        k3 = rhs(state + 0.5 * dt * k2, u)
        k4 = rhs(state + dt * k3, u)
        state = state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        T_w_sim[i], T_h_sim[i] = state

    report = ValidationReport()
    bound = tol.atol("motor_temperature", "rk4", "float64") * 100.0
    for label, sim, meas in (
        ("winding", T_w_sim, T_w_meas),
        ("housing", T_h_sim, T_h_meas),
    ):
        residual = sim - meas
        rms = float(np.sqrt(np.mean(residual * residual)))
        peak = float(np.max(np.abs(residual)))
        if rms > bound:
            report.warnings.append(
                f"{servo.name} {label} residual RMS={rms:.3g} K "
                f"(peak={peak:.3g} K) exceeds policy bound {bound:.3g} K — the "
                "two-node parameters may not match this trace."
            )
    return report


__all__ = [
    "AmbientSpec",
    "ConductionSpec",
    "DriverSpec",
    "ServoThermalSpec",
    "ThermalNetwork",
    "ThermalNetworkSpec",
    "fit_two_node",
    "steady_state_temperatures",
    "two_node_response",
    "two_node_steady_state",
    "validate_two_node",
]
