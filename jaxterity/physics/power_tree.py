# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""The power tree as an acausal electrical network.

:class:`~jaxterity.physics.battery.BatterySag` lumps a robot's whole power
system into ``V_bus = V_oc − R_int·Σ k_i|τ_i|``: one rail, one resistance, one
open-circuit voltage, no state. That is a *causal shortcut* — it answers "how
far does the shared rail droop under aggregate load" and nothing else. A real
robot's power tree is a network: a pack whose open-circuit voltage depends on
state of charge (and ages), a harness whose per-branch resistance means the
wrist servo sees a lower voltage than the shoulder servo, DC-DC regulators
feeding the compute rails at an efficiency that *raises* pack current as the
pack sags, and (sometimes) a charger pushing current back in.

This module builds that network with Jaxonomy's acausal (Modelica-style)
electrical domain, so the topology is declared and the equations are derived:

    pack (ECM cell)  ──R_main──  main rail ──┬── R_branch ── load current source
       │                            │        └── DC-DC ── regulated rail ── …
    charger ────────────────────────┘        (bus capacitance to ground)

Division of labor (the stack's rule): **devices are causal blocks, physical
networks are acausal**. Each servo remains a causal
:class:`~jaxterity.actuators.servo.Servo`; what it *draws* is a causal input to
this network (:func:`servo_currents` maps torque → current), and what it *sees*
is a network output. The body stays an implicit MJX monolith.

What you get
------------
* :class:`PowerTreeSpec` — the declarative tree (pack, harness, loads,
  regulators, charger).
* :class:`PowerTree` — the compiled Jaxonomy ``AcausalSystem`` plus the index
  maps that name its ports, so callers never string-build port names.
* :func:`dc_solve` — the same network solved at DC in closed form (NumPy). This
  is the reference the simulated network is checked against, the model
  :meth:`PowerTree.validate` scores telemetry with, and the fast path for a
  quasi-static duty study.
* :class:`CellAging` — capacity-fade / resistance-growth *priors* (see
  :mod:`jaxterity.aging` for the fitted-from-history counterpart).
* :func:`fit_power_tree` — the honest fitter, plus what it cannot identify.

Identifiability — read before fitting. From a **rail-voltage** trace and known
load currents, the pack's ohmic resistance ``R0``, its polarization resistance
``R1``, and the main harness resistance ``R_main`` are *not* separable: the
trace sees only their sum ``R_series = R0 + R1 + R_main`` (at DC), because they
carry the same current. :func:`fit_power_tree` therefore reports ``R_series``
and refuses to invent a split; a second voltage tap at the pack terminal (or a
branch) is what separates them. Per-*branch* resistances **are** identifiable
— but only from a trace that measures that branch's terminal voltage while its
own current varies.

Units
-----
V, A, Ω, F, A·h, s. SoC is dimensionless in [0, 1].
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from jaxonomy.acausal import AcausalCompiler, AcausalDiagram, EqnEnv
from jaxonomy.acausal.component_library import battery as _acausal_battery
from jaxonomy.acausal.component_library import electrical as elec
from jaxonomy.acausal.component_library.base import SymKind
from jaxonomy.acausal.component_library.component_base import ComponentBase

from ..data.schema import BusVoltageTelemetry, ValidationReport
from ..errors import CalibrationError, JaxterityError

#: Smoothing width (A) for the charger's CC/CV corner and its no-discharge
#: clamp. Small enough to be invisible next to any real charger's regulation
#: band; it exists so the network stays C∞ (differentiable) at the corner.
_CHARGER_EPS_A = 1e-3

#: Conditioning threshold handed to the acausal compiler. A power tree mixes
#: milliohms with millifarads, so its Jacobian is legitimately ill-scaled and
#: the default 1e4 warning fires on healthy models.
_CONDITION_NUMBER_THRESHOLD = 1e9

#: DC fixed-point solve: iteration cap and convergence tolerance (V).
_DC_MAX_ITERS = 200
_DC_TOL_V = 1e-12


def _sympy():
    """Import ``sympy`` with a Jaxterity-shaped error if the extra is missing."""
    try:
        import sympy as sp
    except ImportError as e:  # pragma: no cover - exercised only without sympy
        raise JaxterityError(
            "The acausal power tree needs sympy (Jaxonomy's acausal layer "
            "builds its equations symbolically), which is an optional "
            "dependency. Fix: pip install 'jaxterity[acausal]'."
        ) from e
    return sp


# --------------------------------------------------------------------------
# Acausal components Jaxonomy's electrical library does not ship
# --------------------------------------------------------------------------


class DCDCConverter(ComponentBase):
    """Regulated DC-DC converter: a constant output rail at a fixed efficiency.

    Four electrical pins (``p1``/``n1`` input, ``p2``/``n2`` output). The
    output voltage is held at ``v_out`` regardless of input voltage, and the
    input current follows from power balance::

        V_out          = v_out
        η · V_in · I_in = V_out · I_out

    with the Modelica passive sign convention (``I_in`` flows *into* ``p1``,
    so a converter drawing power has ``I_in > 0`` and ``I_p2 < 0``). The
    coupling is the point: as the pack sags, a regulated rail draws *more*
    pack current for the same load — the destabilizing feedback a lumped
    constant-current model cannot show.

    Scope — deliberately not modeled: dropout (this converter regulates even
    when ``V_in < v_out``, which is unphysical; keep the pack above the
    regulator's dropout or check the DC solve), load-dependent efficiency,
    switching ripple, soft-start inrush, and current limiting. A regulated
    output pins its rail voltage, so do **not** hang a capacitor on it — the
    capacitor's state would be algebraically determined and the acausal
    compiler rejects the model.

    Args
    ----
    ev : EqnEnv
        The diagram's equation environment.
    name : str
        Component name. Its parameters appear on the compiled system as
        ``f"{name}_v_out"`` / ``f"{name}_efficiency"``.
    v_out : float
        Regulated output voltage, V. Must be > 0.
    efficiency : float
        Conversion efficiency in (0, 1].

    Units
    -----
    v_out: V; efficiency: dimensionless.
    """

    def __init__(
        self,
        ev: Any,
        name: str = "dcdc",
        v_out: float = 5.0,
        efficiency: float = 0.9,
    ):
        sp = _sympy()
        self.name = name
        super().__init__()
        Vp1, Ip1 = self.declare_electrical_port(ev, "p1")
        Vn1, In1 = self.declare_electrical_port(ev, "n1")
        Vp2, Ip2 = self.declare_electrical_port(ev, "p2")
        Vn2, In2 = self.declare_electrical_port(ev, "n2")
        self.port_idx_to_name = {-1: "p1", -2: "n1", 1: "p2", 2: "n2"}
        v_set = self.declare_symbol(
            ev,
            "v_out",
            self.name,
            kind=SymKind.param,
            val=v_out,
            validator=lambda x: x > 0.0,
            invalid_msg=f"DCDCConverter {name} must have v_out > 0",
        )
        eta = self.declare_symbol(
            ev,
            "efficiency",
            self.name,
            kind=SymKind.param,
            val=efficiency,
            validator=lambda x: 0.0 < x <= 1.0,
            invalid_msg=f"DCDCConverter {name} must have 0 < efficiency <= 1",
        )
        self.add_eqs(
            [
                sp.Eq(Vp2.s - Vn2.s, v_set.s),
                sp.Eq(0, Ip1.s + In1.s),
                sp.Eq(0, Ip2.s + In2.s),
                sp.Eq(eta.s * (Vp1.s - Vn1.s) * Ip1.s, -(Vp2.s - Vn2.s) * Ip2.s),
            ]
        )


class Charger(elec.ElecTwoPin):
    """CC/CV charger: constant current up to a constant-voltage set point.

    A two-pin source that *supplies* current into the node at ``p``::

        I_supply = max(0, min(i_cc, (v_cv − V) / r_out))

    — constant-current while the pack is low, tapering as the terminal
    voltage approaches ``v_cv``, and never sinking current (a charger does not
    discharge the pack). Both corners are smoothed with the
    ``½(a+b±√((a−b)²+ε²))`` soft min/max so the network stays differentiable;
    ``eps`` is the (named, exposed) smoothing width — no hidden smoothing.

    Scope: no charge-termination logic, no temperature derating, no
    balancing. Those are BMS behaviors, not tree topology.

    Args
    ----
    ev : EqnEnv
        The diagram's equation environment.
    name : str
        Component name.
    v_cv : float
        Constant-voltage set point, V.
    i_cc : float
        Constant-current limit, A. Must be >= 0.
    r_out : float
        Output impedance, Ω. Must be > 0 (it sets the CC→CV taper slope).
    eps : float
        Smoothing width for the CC/CV corner, A.

    Units
    -----
    v_cv: V; i_cc, eps: A; r_out: Ω.
    """

    def __init__(
        self,
        ev: Any,
        name: str = "charger",
        v_cv: float = 12.6,
        i_cc: float = 2.0,
        r_out: float = 0.2,
        eps: float = _CHARGER_EPS_A,
    ):
        sp = _sympy()
        self.name = name
        super().__init__(ev, self.name)
        v_cv_s = self.declare_symbol(
            ev, "v_cv", self.name, kind=SymKind.param, val=v_cv
        )
        i_cc_s = self.declare_symbol(
            ev,
            "i_cc",
            self.name,
            kind=SymKind.param,
            val=i_cc,
            validator=lambda x: x >= 0.0,
            invalid_msg=f"Charger {name} must have i_cc >= 0",
        )
        r_out_s = self.declare_symbol(
            ev,
            "r_out",
            self.name,
            kind=SymKind.param,
            val=r_out,
            validator=lambda x: x > 0.0,
            invalid_msg=f"Charger {name} must have r_out > 0",
        )
        i_cv = (v_cv_s.s - self.V.s) / r_out_s.s

        def soft_min(a, b):
            return 0.5 * (a + b - sp.sqrt((a - b) ** 2 + eps**2))

        def soft_max(a, b):
            return 0.5 * (a + b + sp.sqrt((a - b) ** 2 + eps**2))

        # Passive convention: Ip is current *into* p, so supplying is negative.
        self.add_eqs([sp.Eq(self.Ip.s, -soft_max(soft_min(i_cc_s.s, i_cv), 0.0))])


# --------------------------------------------------------------------------
# The declarative tree
# --------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class CellSpec:
    """The pack: a series/parallel-scaled equivalent-circuit (ECM) cell.

    Wraps Jaxonomy's ``BatteryCellECM`` with the series/parallel arithmetic a
    robot pack needs: ``series`` cells multiply the OCV table and the
    resistances, ``parallel`` strings divide the resistances and multiply the
    capacity.

    Args
    ----
    capacity_Ah : float
        Per-cell capacity, A·h. Must be > 0.
    R0 : float
        Per-cell ohmic (instantaneous) resistance, Ω. Must be > 0.
    R1, C1 : float
        Per-cell polarization RC pair: resistance Ω and capacitance F. The
        pair sets the sag *transient* (τ = R1·C1); at DC it adds ``R1`` to
        the series resistance.
    ocv_soc : tuple[float, ...]
        Increasing SoC breakpoints in [0, 1] for the OCV table.
    ocv_volts : tuple[float, ...]
        Per-cell open-circuit voltage at each breakpoint, V.
    series, parallel : int
        Pack geometry (``3s2p`` → ``series=3, parallel=2``). Must be >= 1.
    initial_soc : float
        Starting state of charge in [0, 1].

    Units
    -----
    A·h, Ω, F, V.
    """

    capacity_Ah: float
    R0: float
    R1: float
    C1: float
    ocv_soc: tuple[float, ...]
    ocv_volts: tuple[float, ...]
    series: int = 1
    parallel: int = 1
    initial_soc: float = 1.0

    def __post_init__(self) -> None:
        if self.capacity_Ah <= 0 or self.R0 <= 0 or self.R1 <= 0 or self.C1 <= 0:
            raise JaxterityError(
                f"CellSpec needs strictly positive capacity_Ah/R0/R1/C1, got "
                f"{self.capacity_Ah}, {self.R0}, {self.R1}, {self.C1}. "
                "Jaxonomy's ECM cell validates these; a zero R1/C1 has no "
                "polarization transient — use a tiny value instead."
            )
        if len(self.ocv_soc) != len(self.ocv_volts) or len(self.ocv_soc) < 2:
            raise JaxterityError(
                f"CellSpec needs matching ocv_soc/ocv_volts tables with >= 2 "
                f"breakpoints, got {len(self.ocv_soc)} and "
                f"{len(self.ocv_volts)}."
            )
        if any(b <= a for a, b in zip(self.ocv_soc, self.ocv_soc[1:])):
            raise JaxterityError(
                f"CellSpec.ocv_soc must be strictly increasing, got {self.ocv_soc}."
            )
        if self.series < 1 or self.parallel < 1:
            raise JaxterityError(
                f"CellSpec needs series >= 1 and parallel >= 1, got "
                f"{self.series} and {self.parallel}."
            )
        if not 0.0 <= self.initial_soc <= 1.0:
            raise JaxterityError(
                f"CellSpec.initial_soc must lie in [0, 1], got {self.initial_soc}."
            )

    @property
    def pack_capacity_Ah(self) -> float:
        """Pack capacity, A·h (per-cell capacity × parallel strings)."""
        return self.capacity_Ah * self.parallel

    @property
    def pack_R0(self) -> float:
        """Pack ohmic resistance, Ω (× series, ÷ parallel)."""
        return self.R0 * self.series / self.parallel

    @property
    def pack_R1(self) -> float:
        """Pack polarization resistance, Ω (× series, ÷ parallel)."""
        return self.R1 * self.series / self.parallel

    @property
    def pack_C1(self) -> float:
        """Pack polarization capacitance, F (× parallel, ÷ series — τ is fixed)."""
        return self.C1 * self.parallel / self.series

    @property
    def pack_ocv_volts(self) -> tuple[float, ...]:
        """Pack open-circuit voltage table, V (per-cell × series)."""
        return tuple(v * self.series for v in self.ocv_volts)

    def open_circuit_voltage(self, soc: float) -> float:
        """Pack OCV at a state of charge, V (linear interpolation, clamped)."""
        return float(
            np.interp(
                float(soc),
                np.asarray(self.ocv_soc, dtype=float),
                np.asarray(self.pack_ocv_volts, dtype=float),
            )
        )


@dataclasses.dataclass(frozen=True)
class LoadSpec:
    """One current-drawing load hanging off a rail through its own harness.

    Args
    ----
    name : str
        Load name (a servo, the compute board, a camera). Must be unique and
        a valid identifier — it names the network's ports.
    rail : str
        The rail it hangs off (``"main"`` or a regulator's ``output_rail``).
    branch_ohm : float
        Wiring + connector resistance from the rail to this load's terminal,
        Ω. Zero means "tapped directly at the rail" (no resistor is built).
    quiescent_a : float
        Always-on current draw, A — added to the commanded input current.

    Units
    -----
    Ω, A.
    """

    name: str
    rail: str = "main"
    branch_ohm: float = 0.0
    quiescent_a: float = 0.0

    def __post_init__(self) -> None:
        if not self.name.isidentifier():
            raise JaxterityError(
                f"LoadSpec.name {self.name!r} must be a valid Python identifier "
                "— it becomes an acausal component name and a port name."
            )
        if self.branch_ohm < 0 or self.quiescent_a < 0:
            raise JaxterityError(
                f"LoadSpec {self.name!r} needs branch_ohm >= 0 and "
                f"quiescent_a >= 0, got {self.branch_ohm} and {self.quiescent_a}."
            )


@dataclasses.dataclass(frozen=True)
class RegulatorSpec:
    """A DC-DC regulator between two rails (see :class:`DCDCConverter`).

    Args
    ----
    name : str
        Regulator name; must be a valid identifier and unique.
    output_rail : str
        Name of the rail it creates.
    v_out : float
        Regulated output voltage, V.
    efficiency : float
        Conversion efficiency in (0, 1].
    input_rail : str
        Rail it draws from (default the pack's ``"main"``).

    Units
    -----
    V, dimensionless.
    """

    name: str
    output_rail: str
    v_out: float
    efficiency: float = 0.9
    input_rail: str = "main"

    def __post_init__(self) -> None:
        if not self.name.isidentifier() or not self.output_rail.isidentifier():
            raise JaxterityError(
                f"RegulatorSpec name/output_rail must be valid identifiers, got "
                f"{self.name!r} and {self.output_rail!r}."
            )
        if self.v_out <= 0 or not 0.0 < self.efficiency <= 1.0:
            raise JaxterityError(
                f"RegulatorSpec {self.name!r} needs v_out > 0 and 0 < "
                f"efficiency <= 1, got {self.v_out} and {self.efficiency}."
            )


@dataclasses.dataclass(frozen=True)
class ChargerSpec:
    """A CC/CV charger on the main rail (see :class:`Charger`).

    Args
    ----
    v_cv : float
        Constant-voltage set point, V (pack terminal, not per cell).
    i_cc : float
        Constant-current limit, A.
    r_out : float
        Output impedance, Ω.
    name : str
        Component name.

    Units
    -----
    V, A, Ω.
    """

    v_cv: float
    i_cc: float
    r_out: float = 0.2
    name: str = "charger"


@dataclasses.dataclass(frozen=True)
class PowerTreeSpec:
    """A robot's whole power tree, declared.

    Args
    ----
    pack : CellSpec
        The battery.
    loads : tuple[LoadSpec, ...]
        Every current-drawing load. Names must be unique.
    regulators : tuple[RegulatorSpec, ...]
        DC-DC stages. Each ``input_rail`` must already exist (``"main"`` or an
        earlier regulator's ``output_rail``) — the tuple order *is* the
        topological order, and a cycle raises.
    charger : ChargerSpec | None
        Optional charger on the main rail.
    main_ohm : float
        Pack → main-rail harness + connector resistance, Ω. Must be > 0: the
        main rail carries a bulk capacitance, and a capacitor tied straight
        across an ideal source has no dynamics of its own (the acausal
        compiler rejects that model).
    bus_capacitance : float
        Main-rail bulk capacitance, F. Must be > 0 — it is the network's
        energy-storage element, and without one the tree is a purely algebraic
        system the compiler refuses.

    Raises
    ------
    JaxterityError
        On duplicate names, an unknown or forward-referenced rail, or a
        non-positive ``main_ohm`` / ``bus_capacitance``.
    """

    pack: CellSpec
    loads: tuple[LoadSpec, ...]
    regulators: tuple[RegulatorSpec, ...] = ()
    charger: ChargerSpec | None = None
    main_ohm: float = 0.02
    bus_capacitance: float = 1e-3

    def __post_init__(self) -> None:
        if self.main_ohm <= 0 or self.bus_capacitance <= 0:
            raise JaxterityError(
                f"PowerTreeSpec needs main_ohm > 0 and bus_capacitance > 0, got "
                f"{self.main_ohm} and {self.bus_capacitance}. See the class "
                "docstring for why the acausal compiler requires both."
            )
        if not self.loads:
            raise JaxterityError(
                "PowerTreeSpec needs at least one load; a tree with nothing to "
                "power has no current-drawing input port."
            )
        names = [load.name for load in self.loads] + [r.name for r in self.regulators]
        duplicates = sorted({n for n in names if names.count(n) > 1})
        if duplicates:
            raise JaxterityError(
                f"PowerTreeSpec has duplicate component name(s) {duplicates}; "
                "load and regulator names share one namespace (they become "
                "acausal component names)."
            )
        rails = {"main"}
        for reg in self.regulators:
            if reg.input_rail not in rails:
                raise JaxterityError(
                    f"Regulator {reg.name!r} draws from rail "
                    f"{reg.input_rail!r}, which is not defined yet. Known rails "
                    f"at this point: {sorted(rails)}. Fix: order `regulators` so "
                    "each stage follows the one feeding it (cycles are not "
                    "physical)."
                )
            if reg.output_rail in rails:
                raise JaxterityError(
                    f"Regulator {reg.name!r} defines rail {reg.output_rail!r}, "
                    "which already exists. Each rail has exactly one source."
                )
            rails.add(reg.output_rail)
        for load in self.loads:
            if load.rail not in rails:
                raise JaxterityError(
                    f"Load {load.name!r} hangs off rail {load.rail!r}, which no "
                    f"regulator defines. Known rails: {sorted(rails)}."
                )

    @property
    def rails(self) -> tuple[str, ...]:
        """Rail names in topological order, ``"main"`` first."""
        return ("main",) + tuple(r.output_rail for r in self.regulators)

    def load(self, name: str) -> LoadSpec:
        """The :class:`LoadSpec` named ``name``.

        Raises
        ------
        JaxterityError
            If no load has that name.
        """
        for load in self.loads:
            if load.name == name:
                return load
        raise JaxterityError(
            f"No load named {name!r} in this power tree. Known loads: "
            f"{[x.name for x in self.loads]}."
        )


#: REPRESENTATIVE LeKiwi power tree — a 3S li-ion pack, six bus servos on the
#: main rail through a daisy-chained harness (the wrist is electrically
#: farthest, so it sags most), and a 5 V regulator for the Raspberry Pi. The
#: numbers are plausible, **not measured from a unit** (same honesty contract
#: as :data:`jaxterity.checkup.stability.LEKIWI_BASE`): measure your harness
#: and re-declare before trusting an absolute voltage.
LEKIWI_POWER_TREE = PowerTreeSpec(
    pack=CellSpec(
        capacity_Ah=2.6,
        R0=0.025,
        R1=0.012,
        C1=900.0,
        ocv_soc=(0.0, 0.1, 0.5, 0.9, 1.0),
        ocv_volts=(3.20, 3.50, 3.70, 4.05, 4.20),
        series=3,
        initial_soc=0.9,
    ),
    loads=(
        LoadSpec("shoulder_pan", branch_ohm=0.010),
        LoadSpec("shoulder_lift", branch_ohm=0.014),
        LoadSpec("elbow_flex", branch_ohm=0.018),
        LoadSpec("wrist_flex", branch_ohm=0.022),
        LoadSpec("wrist_roll", branch_ohm=0.026),
        LoadSpec("gripper", branch_ohm=0.030),
        LoadSpec("compute", rail="rail5", branch_ohm=0.005, quiescent_a=1.2),
    ),
    regulators=(
        RegulatorSpec("reg5", output_rail="rail5", v_out=5.0, efficiency=0.88),
    ),
    main_ohm=0.030,
    bus_capacitance=2.2e-3,
)


# --------------------------------------------------------------------------
# Compilation
# --------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class PowerTree:
    """A compiled power tree: the acausal system plus its port index maps.

    Build one with :meth:`build`. The compiled ``system`` is an ordinary
    Jaxonomy ``LeafSystem`` — add it to a ``DiagramBuilder``, wire each load's
    current input, and read the rail/terminal voltages and SoC from its
    outputs. Use the index accessors rather than reconstructing port names.

    Attributes
    ----------
    spec : PowerTreeSpec
        The tree this was compiled from.
    system : Any
        The compiled Jaxonomy ``AcausalSystem``.
    load_inputs : dict[str, int]
        Load name → input-port index (units: A, positive = drawn).
    rail_outputs : dict[str, int]
        Rail name → output-port index (units: V).
    load_outputs : dict[str, int]
        Load name → output-port index of that load's *terminal* voltage (V),
        i.e. after its own branch resistance.
    soc_output : int
        Output-port index of the pack state of charge (dimensionless).
    pack_current_output : int
        Output-port index of the pack current (A, positive = charging, per
        the Modelica passive convention).

    Examples
    --------
    >>> import jaxonomy
    >>> from jaxterity.physics import PowerTree, LEKIWI_POWER_TREE
    >>> tree = PowerTree.build(LEKIWI_POWER_TREE)  # doctest: +SKIP
    >>> builder = jaxonomy.DiagramBuilder()        # doctest: +SKIP
    >>> sys_ = builder.add(tree.system)            # doctest: +SKIP
    >>> sys_.input_ports[tree.load_inputs["elbow_flex"]]  # doctest: +SKIP
    """

    spec: PowerTreeSpec
    system: Any
    load_inputs: dict[str, int]
    rail_outputs: dict[str, int]
    load_outputs: dict[str, int]
    soc_output: int
    pack_current_output: int

    @classmethod
    def build(
        cls,
        spec: PowerTreeSpec,
        *,
        verbose: bool = False,
    ) -> "PowerTree":
        """Compile a :class:`PowerTreeSpec` into an acausal Jaxonomy system.

        Args
        ----
        spec : PowerTreeSpec
            The declared tree.
        verbose : bool
            Pass through to ``AcausalCompiler`` (prints the equation pipeline;
            the switch to reach for when a topology change stops compiling).

        Returns
        -------
        PowerTree
            The compiled system and its port index maps.

        Raises
        ------
        JaxterityError
            If ``sympy`` (the ``[acausal]`` extra) is not installed.
        """
        _sympy()  # fail early, with the actionable message
        ev = EqnEnv()
        diagram = AcausalDiagram()
        ground = elec.Ground(ev, name="gnd")

        pack = _acausal_battery.BatteryCellECM(
            ev,
            name="pack",
            R0=spec.pack.pack_R0,
            R1=spec.pack.pack_R1,
            C1=spec.pack.pack_C1,
            capacity_Ah=spec.pack.pack_capacity_Ah,
            ocv_soc=list(spec.pack.ocv_soc),
            ocv_volts=list(spec.pack.pack_ocv_volts),
            initial_soc=spec.pack.initial_soc,
            initial_soc_fixed=True,
            enable_soc_port=True,
        )
        pack_current = elec.CurrentSensor(ev, name="pack_current")
        main_wire = elec.Resistor(ev, name="main_wire", R=spec.main_ohm)
        bus_cap = elec.Capacitor(
            ev,
            name="bus_cap",
            C=spec.bus_capacitance,
            initial_voltage=spec.pack.open_circuit_voltage(spec.pack.initial_soc),
            initial_voltage_fixed=True,
        )
        # pack(+) -> current sensor -> main harness -> main rail node (= bus cap +)
        diagram.connect(pack, "p", pack_current, "p")
        diagram.connect(pack_current, "n", main_wire, "p")
        diagram.connect(main_wire, "n", bus_cap, "p")
        diagram.connect(pack, "n", ground, "p")
        diagram.connect(bus_cap, "n", ground, "p")

        # Rail "anchor" components: the component + pin every consumer of that
        # rail connects to. "main" is the bus capacitor's + node; a regulated
        # rail is its converter's p2 pin.
        rail_anchor: dict[str, tuple[Any, str]] = {"main": (bus_cap, "p")}

        if spec.charger is not None:
            charger = Charger(
                ev,
                name=spec.charger.name,
                v_cv=spec.charger.v_cv,
                i_cc=spec.charger.i_cc,
                r_out=spec.charger.r_out,
            )
            diagram.connect(charger, "p", bus_cap, "p")
            diagram.connect(charger, "n", ground, "p")

        for reg in spec.regulators:
            converter = DCDCConverter(
                ev, name=reg.name, v_out=reg.v_out, efficiency=reg.efficiency
            )
            src_component, src_pin = rail_anchor[reg.input_rail]
            diagram.connect(converter, "p1", src_component, src_pin)
            diagram.connect(converter, "n1", ground, "p")
            diagram.connect(converter, "n2", ground, "p")
            rail_anchor[reg.output_rail] = (converter, "p2")

        load_inputs: dict[str, int] = {}
        load_outputs: dict[str, int] = {}
        input_names: list[str] = []
        output_names: list[str] = []
        for load in spec.loads:
            draw = elec.CurrentSource(
                ev, name=load.name, i=0.0, enable_current_port=True
            )
            terminal = elec.VoltageSensor(ev, name=f"{load.name}_term")
            src_component, src_pin = rail_anchor[load.rail]
            if load.branch_ohm > 0.0:
                branch = elec.Resistor(ev, name=f"{load.name}_wire", R=load.branch_ohm)
                diagram.connect(branch, "p", src_component, src_pin)
                diagram.connect(branch, "n", draw, "p")
                diagram.connect(terminal, "p", draw, "p")
            else:
                diagram.connect(draw, "p", src_component, src_pin)
                diagram.connect(terminal, "p", src_component, src_pin)
            diagram.connect(draw, "n", ground, "p")
            diagram.connect(terminal, "n", ground, "p")
            input_names.append(f"{load.name}_i")
            output_names.append(f"{load.name}_term_v")

        rail_outputs: dict[str, int] = {}
        for rail in spec.rails:
            sensor = elec.VoltageSensor(ev, name=f"{rail}_rail")
            src_component, src_pin = rail_anchor[rail]
            diagram.connect(sensor, "p", src_component, src_pin)
            diagram.connect(sensor, "n", ground, "p")
            output_names.append(f"{rail}_rail_v")

        compiler = AcausalCompiler(
            ev,
            diagram,
            verbose=verbose,
            condition_number_threshold=_CONDITION_NUMBER_THRESHOLD,
        )
        system = compiler()

        port_in = {p.name: i for i, p in enumerate(system.input_ports)}
        port_out = {p.name: i for i, p in enumerate(system.output_ports)}
        for load, name in zip(spec.loads, input_names):
            load_inputs[load.name] = port_in[name]
        for load in spec.loads:
            load_outputs[load.name] = port_out[f"{load.name}_term_v"]
        for rail in spec.rails:
            rail_outputs[rail] = port_out[f"{rail}_rail_v"]
        return cls(
            spec=spec,
            system=system,
            load_inputs=load_inputs,
            rail_outputs=rail_outputs,
            load_outputs=load_outputs,
            soc_output=port_out["pack_soc"],
            pack_current_output=port_out["pack_current_i"],
        )

    def validate(
        self,
        telemetry: BusVoltageTelemetry,
        load_currents: Mapping[str, Sequence[float]] | np.ndarray,
        *,
        rail: str = "main",
        soc: float | None = None,
    ) -> ValidationReport:
        """Score a measured rail-voltage trace against the tree's DC solution.

        Evaluates :func:`dc_solve` at each sample's load currents and compares
        the predicted rail voltage to the recorded ``voltage`` channel. DC (not
        the transient network) is the right reference here: the polarization
        RC settles in seconds while a capture is minutes, and the residual this
        reports is a *parameter* mismatch, not a transient one.

        Args
        ----
        telemetry : BusVoltageTelemetry
            Measured rail voltage.
        load_currents : Mapping[str, Sequence[float]] | numpy.ndarray
            Per-load current draw, A. Either a mapping load-name → per-sample
            current, or an ``(n, n_loads)`` array in ``spec.loads`` order.
        rail : str
            Which rail the telemetry measures.
        soc : float | None
            State of charge to evaluate the OCV at (default the pack's
            ``initial_soc``).

        Returns
        -------
        ValidationReport
            ``ok`` iff the RMS residual is within the ``bus_voltage`` policy
            bound (scaled, as elsewhere, to allow for the modeling gap).

        Units
        -----
        telemetry voltage: V; load_currents: A.
        """
        from ..testing import tolerances as tol

        currents = _current_matrix(self.spec, load_currents)
        measured = np.asarray(telemetry.voltage, dtype=np.float64)
        if currents.shape[0] != measured.shape[0]:
            raise CalibrationError(
                f"PowerTree.validate: {currents.shape[0]} current sample(s) but "
                f"{measured.shape[0]} voltage sample(s). Fix: align the load "
                "currents with the telemetry timestamps."
            )
        predicted = np.array(
            [
                dc_solve(
                    self.spec,
                    dict(zip([lo.name for lo in self.spec.loads], row)),
                    soc=soc,
                ).rail_voltage[rail]
                for row in currents
            ]
        )
        residual = predicted - measured
        rms = float(np.sqrt(np.mean(residual * residual)))
        peak = float(np.max(np.abs(residual)))
        report = ValidationReport()
        bound = tol.atol("bus_voltage", "rk4", "float64") * 100.0
        if rms > bound:
            report.warnings.append(
                f"rail {rail!r} voltage residual RMS={rms:.3g} V "
                f"(peak={peak:.3g} V) exceeds policy bound {bound:.3g} V — the "
                "pack/harness parameters may not match this trace."
            )
        return report


# --------------------------------------------------------------------------
# DC solve — the closed-form reference
# --------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class PowerTreeSolution:
    """The tree's DC operating point (see :func:`dc_solve`).

    Attributes
    ----------
    rail_voltage : dict[str, float]
        Rail name → voltage, V.
    load_voltage : dict[str, float]
        Load name → terminal voltage after its branch resistance, V.
    rail_current : dict[str, float]
        Rail name → total current leaving that rail, A (loads + downstream
        regulator inputs).
    pack_current : float
        Net current *out of* the pack, A (positive = discharging).
    charger_current : float
        Current the charger supplies into the main rail, A (0 if no charger).
    """

    rail_voltage: dict[str, float]
    load_voltage: dict[str, float]
    rail_current: dict[str, float]
    pack_current: float
    charger_current: float


def _current_matrix(
    spec: PowerTreeSpec,
    load_currents: Mapping[str, Sequence[float]] | np.ndarray,
) -> np.ndarray:
    """Per-load currents as an ``(n, n_loads)`` array in ``spec.loads`` order."""
    names = [load.name for load in spec.loads]
    if isinstance(load_currents, Mapping):
        missing = [n for n in names if n not in load_currents]
        if missing:
            raise CalibrationError(
                f"Power-tree currents are missing load(s) {missing}. Every "
                f"declared load needs a current channel; known loads: {names}."
            )
        columns = [
            np.atleast_1d(np.asarray(load_currents[n], dtype=float)) for n in names
        ]
        return np.stack(columns, axis=-1)
    array = np.atleast_2d(np.asarray(load_currents, dtype=float))
    if array.shape[-1] != len(names):
        raise CalibrationError(
            f"Power-tree currents have {array.shape[-1]} column(s) but the tree "
            f"declares {len(names)} load(s) {names}. Fix: pass an "
            "(n, n_loads) array in spec.loads order, or a name→current mapping."
        )
    return array


def dc_solve(
    spec: PowerTreeSpec,
    load_currents: Mapping[str, float],
    *,
    soc: float | None = None,
) -> PowerTreeSolution:
    """Solve the power tree at DC (steady state) in closed form.

    The same network the acausal system integrates, with the capacitors open
    and the polarization RC settled (so the pack's series resistance is
    ``R0 + R1``). Regulated rails hold their set point, so only the main rail
    is unknown; its voltage and the regulators' input currents are coupled
    (a sagging pack makes a regulated rail draw *more* current), so the solve
    is a damped fixed-point iteration — a handful of passes to machine
    precision.

    Args
    ----
    spec : PowerTreeSpec
        The declared tree.
    load_currents : Mapping[str, float]
        Per-load current draw, A (positive = drawn). Loads omitted here draw
        only their ``quiescent_a``.
    soc : float | None
        State of charge to evaluate the pack OCV at; defaults to
        ``spec.pack.initial_soc``.

    Returns
    -------
    PowerTreeSolution
        Rail and load-terminal voltages, per-rail currents, pack and charger
        currents.

    Raises
    ------
    CalibrationError
        If a name in ``load_currents`` is not a declared load, or the
        fixed point does not converge (a physically impossible operating
        point — e.g. a regulator asked to deliver more power than the pack can
        source at any voltage).

    Units
    -----
    A in, V/A out.

    Examples
    --------
    >>> from jaxterity.physics import LEKIWI_POWER_TREE, dc_solve
    >>> sol = dc_solve(LEKIWI_POWER_TREE, {"elbow_flex": 2.0})
    >>> sol.load_voltage["elbow_flex"] < sol.rail_voltage["main"]
    True
    """
    known = {load.name for load in spec.loads}
    unknown = sorted(set(load_currents) - known)
    if unknown:
        raise CalibrationError(
            f"dc_solve got current(s) for unknown load(s) {unknown}. Declared "
            f"loads: {sorted(known)}."
        )
    soc_value = spec.pack.initial_soc if soc is None else float(soc)
    ocv = spec.pack.open_circuit_voltage(soc_value)
    r_series = spec.pack.pack_R0 + spec.pack.pack_R1 + spec.main_ohm

    draw = {
        load.name: float(load_currents.get(load.name, 0.0)) + load.quiescent_a
        for load in spec.loads
    }
    # Current each rail must source from its loads (regulator inputs come next).
    load_current_by_rail = {rail: 0.0 for rail in spec.rails}
    for load in spec.loads:
        load_current_by_rail[load.rail] += draw[load.name]

    rail_voltage = {rail: 0.0 for rail in spec.rails}
    for reg in spec.regulators:
        rail_voltage[reg.output_rail] = reg.v_out
    v_main = ocv

    for _ in range(_DC_MAX_ITERS):
        # Downstream first: a regulator's input current depends on everything
        # its output rail (and any rail below that) draws.
        rail_current = dict(load_current_by_rail)
        for reg in reversed(spec.regulators):
            out_power = reg.v_out * rail_current[reg.output_rail]
            v_in = v_main if reg.input_rail == "main" else rail_voltage[reg.input_rail]
            if v_in <= 0.0:
                raise CalibrationError(
                    f"dc_solve: rail {reg.input_rail!r} collapsed to {v_in:.3g} V "
                    f"while solving regulator {reg.name!r}. The tree cannot "
                    "supply this load at any operating point — reduce the load, "
                    "lower the branch resistance, or raise the pack SoC."
                )
            rail_current[reg.input_rail] += out_power / (reg.efficiency * v_in)

        charger_current = 0.0
        if spec.charger is not None:
            charger = spec.charger
            charger_current = max(
                0.0, min(charger.i_cc, (charger.v_cv - v_main) / charger.r_out)
            )
        pack_current = rail_current["main"] - charger_current
        v_new = ocv - r_series * pack_current
        if abs(v_new - v_main) < _DC_TOL_V:
            v_main = v_new
            break
        # Damped update: the regulator feedback is 1/V, so an undamped step can
        # oscillate when the pack is heavily loaded.
        v_main = 0.5 * (v_main + v_new)
    else:
        raise CalibrationError(
            f"dc_solve did not converge in {_DC_MAX_ITERS} iterations (last "
            f"main-rail estimate {v_main:.4g} V). This is the signature of an "
            "operating point the tree cannot reach — check the regulator loads "
            "against the pack's series resistance."
        )

    rail_voltage["main"] = v_main
    load_voltage = {
        load.name: rail_voltage[load.rail] - draw[load.name] * load.branch_ohm
        for load in spec.loads
    }
    return PowerTreeSolution(
        rail_voltage=rail_voltage,
        load_voltage=load_voltage,
        rail_current=rail_current,
        pack_current=pack_current,
        charger_current=charger_current,
    )


def servo_currents(tau, k_i, *, quiescent=0.0, i_eps: float = 1e-3):
    """Per-servo current draw from per-servo torque (the causal bridge).

    ``I_i = k_i · |τ_i| + quiescent_i`` with the same smoothed ``|·|`` the
    lumped :class:`~jaxterity.physics.battery.BatterySag` uses (``τ·tanh(τ/ε)``
    — exactly zero at zero torque, no kink), so the map composes through
    forward-mode autodiff. This is the *only* place torque meets the power
    tree: everything downstream is electrical.

    Args
    ----
    tau : jax.Array, shape ``(n,)``
        Per-servo torque, N·m.
    k_i : jax.Array, shape ``(n,)``
        Per-servo torque→current gains, A per N·m.
    quiescent : float or jax.Array
        Always-on per-servo current, A.
    i_eps : float
        Smoothing width for ``|τ|``, N·m.

    Returns
    -------
    jax.Array, shape ``(n,)``
        Per-servo current draw, A.

    Units
    -----
    tau: N·m; k_i: A/(N·m); returns A.
    """
    import jax.numpy as jnp

    from .battery import _smooth_abs

    return jnp.asarray(k_i) * _smooth_abs(jnp.asarray(tau), i_eps) + quiescent


# --------------------------------------------------------------------------
# Aging and fitting
# --------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class CellAging:
    """Capacity-fade and resistance-growth *priors* for a pack.

    Two mechanisms, each split into a cycle term and a calendar term, applied
    multiplicatively::

        capacity(n, d) = capacity₀ · (1 − fade_cycle·n − fade_day·d)
        R(n, d)        = R₀        · (1 + growth_cycle·n + growth_day·d)

    where ``n`` is equivalent full cycles and ``d`` is calendar days. Linear
    is deliberately the *simplest defensible* form: real fade is closer to
    ``√n`` early and accelerates late, so treat this as a prior over the
    middle of a pack's life, not a life model. Capacity is floored at 1 % so
    an over-long extrapolation cannot produce a non-physical spec.

    The defaults are representative Li-ion numbers (roughly 20 % capacity loss
    and 60 % resistance growth at 1000 cycles), **not measured**. Fit them
    from a unit's own history with :mod:`jaxterity.aging`.

    Args
    ----
    capacity_fade_per_cycle, capacity_fade_per_day : float
        Fractional capacity lost per equivalent full cycle / per day.
    resistance_growth_per_cycle, resistance_growth_per_day : float
        Fractional resistance gained per equivalent full cycle / per day.

    Units
    -----
    Dimensionless fractions per cycle / per day.

    Examples
    --------
    >>> from jaxterity.physics import CellAging, LEKIWI_POWER_TREE
    >>> aged = CellAging().apply(LEKIWI_POWER_TREE.pack, cycles=500, days=365)
    >>> aged.capacity_Ah < LEKIWI_POWER_TREE.pack.capacity_Ah
    True
    >>> aged.R0 > LEKIWI_POWER_TREE.pack.R0
    True
    """

    capacity_fade_per_cycle: float = 2.0e-4
    capacity_fade_per_day: float = 5.0e-5
    resistance_growth_per_cycle: float = 6.0e-4
    resistance_growth_per_day: float = 1.5e-4

    def apply(
        self, cell: CellSpec, *, cycles: float = 0.0, days: float = 0.0
    ) -> CellSpec:
        """The pack after ``cycles`` equivalent full cycles and ``days`` days.

        Args
        ----
        cell : CellSpec
            The as-new pack.
        cycles : float
            Equivalent full cycles (charge throughput / capacity). Must be >= 0.
        days : float
            Calendar age, days. Must be >= 0.

        Returns
        -------
        CellSpec
            A new spec with faded capacity and grown resistances (the RC time
            constant ``R1·C1`` is held fixed, so ``C1`` shrinks as ``R1``
            grows — the polarization *transient* is far more stable with age
            than its magnitude).

        Raises
        ------
        JaxterityError
            If ``cycles`` or ``days`` is negative.
        """
        if cycles < 0 or days < 0:
            raise JaxterityError(
                f"CellAging.apply needs cycles >= 0 and days >= 0, got "
                f"{cycles} and {days}."
            )
        fade = self.capacity_fade_per_cycle * cycles + self.capacity_fade_per_day * days
        growth = (
            1.0
            + self.resistance_growth_per_cycle * cycles
            + self.resistance_growth_per_day * days
        )
        capacity = cell.capacity_Ah * max(1.0 - fade, 0.01)
        return dataclasses.replace(
            cell,
            capacity_Ah=capacity,
            R0=cell.R0 * growth,
            R1=cell.R1 * growth,
            C1=cell.C1 / growth,
        )


def fit_power_tree(
    telemetry: BusVoltageTelemetry,
    load_currents: Mapping[str, Sequence[float]] | np.ndarray,
    spec: PowerTreeSpec,
) -> dict[str, float]:
    """Identify the tree's series resistance and OCV from a rail-voltage trace.

    Ordinary least squares on ``V_rail = V_oc − R_series · I_pack``, the DC
    relation with the *sum* ``R_series = pack_R0 + pack_R1 + main_ohm``.
    Regulator input currents are reconstructed from the declared efficiencies
    at the measured rail voltage, so a tree with a 5 V rail is fitted correctly
    even though only the main rail is instrumented.

    What this does **not** identify (see the module docstring): the split of
    ``R_series`` into pack vs harness, per-branch resistances, the OCV *curve*
    (only its value over the trace's SoC span, reported as ``V_oc``), and
    anything about the polarization time constant — that needs a load step and
    a transient fit, not a DC regression.

    Args
    ----
    telemetry : BusVoltageTelemetry
        Main-rail voltage trace under a known load.
    load_currents : Mapping[str, Sequence[float]] | numpy.ndarray
        Per-load current, A: a name→series mapping or an ``(n, n_loads)``
        array in ``spec.loads`` order.
    spec : PowerTreeSpec
        The declared tree (topology + regulator efficiencies).

    Returns
    -------
    dict[str, float]
        ``{"V_oc": ..., "R_series": ...}`` in V and Ω.

    Raises
    ------
    CalibrationError
        If the currents and telemetry lengths disagree, or the trace has no
        current variation (``R_series`` is then unidentifiable — a constant
        load cannot separate the drop from the open-circuit voltage).

    Units
    -----
    V, A, Ω.
    """
    currents = _current_matrix(spec, load_currents)
    voltage = np.asarray(telemetry.voltage, dtype=np.float64)
    if currents.shape[0] != voltage.shape[0]:
        raise CalibrationError(
            f"fit_power_tree: {currents.shape[0]} current sample(s) but "
            f"{voltage.shape[0]} voltage sample(s). Fix: align the load "
            "currents with the telemetry timestamps."
        )
    names = [load.name for load in spec.loads]
    quiescent = np.asarray([load.quiescent_a for load in spec.loads])
    draw = currents + quiescent[None, :]

    # Pack current per sample: main-rail loads directly, regulated-rail loads
    # through their converter at the *measured* rail voltage.
    rail_of = {load.name: load.rail for load in spec.loads}
    reg_by_rail = {reg.output_rail: reg for reg in spec.regulators}
    pack_current = np.zeros_like(voltage)
    for j, name in enumerate(names):
        rail = rail_of[name]
        if rail == "main":
            pack_current += draw[:, j]
            continue
        reg = reg_by_rail[rail]
        if reg.input_rail != "main":
            raise CalibrationError(
                f"fit_power_tree only reconstructs single-stage regulators from "
                f"the main rail; {reg.name!r} draws from {reg.input_rail!r}. "
                "Fix: instrument that rail directly, or fit the stages "
                "separately."
            )
        with np.errstate(divide="ignore", invalid="ignore"):
            pack_current += reg.v_out * draw[:, j] / (reg.efficiency * voltage)

    if float(np.ptp(pack_current)) <= 0.0:
        raise CalibrationError(
            "fit_power_tree needs current variation to separate R_series from "
            "V_oc; this trace draws a constant "
            f"{float(pack_current[0]):.3g} A. Fix: capture a load sweep (idle "
            "→ loaded → idle), not a single operating point."
        )
    design = np.stack([np.ones_like(pack_current), -pack_current], axis=1)
    coefficients, *_ = np.linalg.lstsq(design, voltage, rcond=None)
    return {"V_oc": float(coefficients[0]), "R_series": float(coefficients[1])}


__all__ = [
    "CellAging",
    "CellSpec",
    "Charger",
    "ChargerSpec",
    "DCDCConverter",
    "LEKIWI_POWER_TREE",
    "LoadSpec",
    "PowerTree",
    "PowerTreeSolution",
    "PowerTreeSpec",
    "RegulatorSpec",
    "dc_solve",
    "fit_power_tree",
    "servo_currents",
]
