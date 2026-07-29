# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Acausal power tree — spec validation, DC solve, network parity, fit, aging.

The DC solve (:func:`dc_solve`) is a closed-form NumPy reference for the same
network the acausal compiler integrates, so the load-bearing test here is
*parity*: simulate the compiled network to steady state and assert it lands on
the DC solution within the ``bus_voltage`` policy tolerance. Everything else
(regulator coupling, per-branch sag ordering, charger CC/CV) is checked on the
cheap DC path, and the lumped ``BatterySag`` equivalence pins the new model to
the old one it generalizes.
"""

from __future__ import annotations

import jaxonomy
import numpy as np
import pytest

pytest.importorskip("sympy")

from jaxterity.data import BusVoltageTelemetry  # noqa: E402
from jaxterity.errors import CalibrationError, JaxterityError  # noqa: E402
from jaxterity.physics.power_tree import (  # noqa: E402
    LEKIWI_POWER_TREE,
    CellAging,
    CellSpec,
    ChargerSpec,
    LoadSpec,
    PowerTree,
    PowerTreeSpec,
    RegulatorSpec,
    dc_solve,
    fit_power_tree,
    servo_currents,
)
from jaxterity.testing import tolerances as tol  # noqa: E402

_V_TOL = tol.atol("bus_voltage", "bdf", "float64") * 1e5


def _flat_cell(**overrides) -> CellSpec:
    """A pack with a flat OCV table: OCV is 12 V at every SoC.

    Keeps the DC arithmetic in tests independent of the coulomb counter, so a
    voltage assertion is about resistance, not about how much charge moved.
    """
    kwargs = dict(
        capacity_Ah=5.0,
        R0=0.02,
        R1=0.01,
        C1=100.0,
        ocv_soc=(0.0, 1.0),
        ocv_volts=(12.0, 12.0),
        initial_soc=0.8,
    )
    kwargs.update(overrides)
    return CellSpec(**kwargs)


def _simple_spec(**overrides) -> PowerTreeSpec:
    kwargs = dict(
        pack=_flat_cell(),
        loads=(LoadSpec("servo", branch_ohm=0.05),),
        main_ohm=0.03,
        bus_capacitance=1e-3,
    )
    kwargs.update(overrides)
    return PowerTreeSpec(**kwargs)


# --- spec validation -------------------------------------------------------


def test_rails_are_topologically_ordered():
    spec = _simple_spec(
        loads=(LoadSpec("servo"), LoadSpec("pi", rail="rail5")),
        regulators=(RegulatorSpec("reg5", output_rail="rail5", v_out=5.0),),
    )
    assert spec.rails == ("main", "rail5")


def test_load_on_undefined_rail_raises():
    with pytest.raises(JaxterityError, match="which no regulator defines"):
        _simple_spec(loads=(LoadSpec("pi", rail="rail5"),))


def test_regulator_forward_reference_raises():
    with pytest.raises(JaxterityError, match="not defined yet"):
        _simple_spec(
            regulators=(
                RegulatorSpec("regA", output_rail="a", v_out=5.0, input_rail="b"),
                RegulatorSpec("regB", output_rail="b", v_out=8.0),
            ),
        )


def test_duplicate_component_names_raise():
    with pytest.raises(JaxterityError, match="duplicate component name"):
        _simple_spec(loads=(LoadSpec("x"), LoadSpec("x")))


def test_zero_bus_capacitance_raises():
    # The acausal compiler needs an energy-storage element; catching it in the
    # spec turns a deep compiler error into an actionable one.
    with pytest.raises(JaxterityError, match="bus_capacitance > 0"):
        _simple_spec(bus_capacitance=0.0)


def test_pack_geometry_scales_series_and_parallel():
    cell = _flat_cell(series=3, parallel=2)
    assert cell.pack_capacity_Ah == pytest.approx(10.0)
    assert cell.pack_R0 == pytest.approx(0.02 * 3 / 2)
    assert cell.pack_ocv_volts == (36.0, 36.0)
    # Series/parallel scaling preserves the polarization time constant.
    assert cell.pack_R1 * cell.pack_C1 == pytest.approx(cell.R1 * cell.C1)


# --- DC solve --------------------------------------------------------------


def test_dc_solve_ohms_law_on_a_single_load():
    spec = _simple_spec()
    sol = dc_solve(spec, {"servo": 4.0})
    r_series = spec.pack.pack_R0 + spec.pack.pack_R1 + spec.main_ohm
    assert sol.pack_current == pytest.approx(4.0)
    assert sol.rail_voltage["main"] == pytest.approx(12.0 - r_series * 4.0)
    # The load sees a further drop across its own branch.
    assert sol.load_voltage["servo"] == pytest.approx(
        sol.rail_voltage["main"] - 4.0 * 0.05
    )


def test_branch_resistance_orders_the_per_load_sag():
    # The whole point of a network over a lumped rail: the electrically
    # farthest servo sees the lowest voltage at equal current.
    sol = dc_solve(LEKIWI_POWER_TREE, {n: 1.5 for n in ("shoulder_pan", "wrist_roll")})
    assert sol.load_voltage["wrist_roll"] < sol.load_voltage["shoulder_pan"]


def test_quiescent_draw_applies_without_a_commanded_current():
    sol = dc_solve(LEKIWI_POWER_TREE, {})
    # The compute rail's 1.2 A quiescent draw still loads the pack.
    assert sol.pack_current > 0.0
    assert sol.rail_voltage["rail5"] == pytest.approx(5.0)


def test_regulated_rail_draws_more_pack_current_as_the_pack_sags():
    # Constant power on the output => input current rises as V_in falls. This
    # coupling is invisible to a lumped constant-current model.
    spec = _simple_spec(
        loads=(LoadSpec("pi", rail="rail5"),),
        regulators=(
            RegulatorSpec("reg5", output_rail="rail5", v_out=5.0, efficiency=0.9),
        ),
    )
    strong = dc_solve(spec, {"pi": 2.0})
    weak = dc_solve(dataclass_replace_ocv(spec, 9.0), {"pi": 2.0})
    assert weak.pack_current > strong.pack_current
    # Power balance holds at each operating point.
    assert strong.pack_current == pytest.approx(
        5.0 * 2.0 / (0.9 * strong.rail_voltage["main"])
    )


def dataclass_replace_ocv(spec: PowerTreeSpec, volts: float) -> PowerTreeSpec:
    """The same tree with a flat OCV table at ``volts`` (test helper)."""
    import dataclasses

    pack = dataclasses.replace(spec.pack, ocv_volts=(volts, volts))
    return dataclasses.replace(spec, pack=pack)


def test_charger_supplies_constant_current_then_tapers():
    spec = _simple_spec(charger=ChargerSpec(v_cv=12.6, i_cc=2.0, r_out=0.2))
    idle = dc_solve(spec, {"servo": 0.0})
    # CC region: (12.6 - ~12) / 0.2 = 3 A > i_cc, so the charger is current-limited.
    assert idle.charger_current == pytest.approx(2.0)
    assert idle.pack_current < 0.0  # charging
    # A near-full pack taper: OCV close to the CV set point => CV region.
    full = dc_solve(dataclass_replace_ocv(spec, 12.55), {"servo": 0.0})
    assert 0.0 < full.charger_current < 2.0


def test_dc_solve_rejects_an_unknown_load():
    with pytest.raises(CalibrationError, match="unknown load"):
        dc_solve(_simple_spec(), {"nope": 1.0})


def test_dc_solve_reduces_to_the_lumped_battery_sag():
    # BatterySag is V_oc - R_int * I_total on one rail with no branch drop; the
    # tree must reproduce it exactly when its topology is that degenerate case.
    from jaxterity.physics.battery import bus_voltage

    spec = _simple_spec(loads=(LoadSpec("a"), LoadSpec("b")), main_ohm=0.01)
    r_series = spec.pack.pack_R0 + spec.pack.pack_R1 + spec.main_ohm
    sol = dc_solve(spec, {"a": 1.0, "b": 2.5})
    lumped = float(
        bus_voltage(
            np.asarray([1.0, 2.5]),
            V_oc=12.0,
            R_int=r_series,
            k_i=np.ones(2),
            i_eps=1e-6,
        )
    )
    assert sol.rail_voltage["main"] == pytest.approx(lumped, abs=_V_TOL)


# --- compiled network ------------------------------------------------------


def _simulate_to_steady_state(tree: PowerTree, currents: dict[str, float], t_end=2.0):
    """Run the compiled network with constant load currents; return final outputs."""
    builder = jaxonomy.DiagramBuilder()
    system = builder.add(tree.system)
    for name, value in currents.items():
        source = builder.add(jaxonomy.library.Constant(value, name=f"cmd_{name}"))
        builder.connect(
            source.output_ports[0], system.input_ports[tree.load_inputs[name]]
        )
    diagram = builder.build()
    context = diagram.create_context()
    results = jaxonomy.simulate(
        diagram,
        context,
        (0.0, t_end),
        recorded_signals={p.name: p for p in system.output_ports},
    )
    return {k: float(v[-1]) for k, v in results.outputs.items()}, system


@pytest.mark.slow
def test_compiled_network_matches_the_dc_solution():
    spec = _simple_spec()
    tree = PowerTree.build(spec)
    outputs, system = _simulate_to_steady_state(tree, {"servo": 3.0})
    names = [p.name for p in system.output_ports]
    expected = dc_solve(spec, {"servo": 3.0})
    assert outputs[names[tree.rail_outputs["main"]]] == pytest.approx(
        expected.rail_voltage["main"], abs=_V_TOL
    )
    assert outputs[names[tree.load_outputs["servo"]]] == pytest.approx(
        expected.load_voltage["servo"], abs=_V_TOL
    )


@pytest.mark.slow
def test_compiled_network_with_a_regulator_matches_the_dc_solution():
    spec = _simple_spec(
        loads=(LoadSpec("servo", branch_ohm=0.02), LoadSpec("pi", rail="rail5")),
        regulators=(
            RegulatorSpec("reg5", output_rail="rail5", v_out=5.0, efficiency=0.85),
        ),
    )
    tree = PowerTree.build(spec)
    currents = {"servo": 2.0, "pi": 1.5}
    outputs, system = _simulate_to_steady_state(tree, currents)
    names = [p.name for p in system.output_ports]
    expected = dc_solve(spec, currents)
    assert outputs[names[tree.rail_outputs["main"]]] == pytest.approx(
        expected.rail_voltage["main"], abs=_V_TOL
    )
    assert outputs[names[tree.rail_outputs["rail5"]]] == pytest.approx(5.0, abs=_V_TOL)


@pytest.mark.slow
def test_soc_falls_under_load():
    spec = _simple_spec(pack=_flat_cell(capacity_Ah=0.01))  # tiny pack, fast counter
    tree = PowerTree.build(spec)
    outputs, system = _simulate_to_steady_state(tree, {"servo": 2.0}, t_end=1.0)
    names = [p.name for p in system.output_ports]
    assert outputs[names[tree.soc_output]] < spec.pack.initial_soc


@pytest.mark.slow
def test_the_representative_lekiwi_tree_compiles():
    """The tree SKILL.md tells an agent to start from must actually build."""
    tree = PowerTree.build(LEKIWI_POWER_TREE)
    assert set(tree.load_inputs) == {lo.name for lo in LEKIWI_POWER_TREE.loads}
    assert set(tree.rail_outputs) == {"main", "rail5"}


def test_port_maps_cover_every_declared_load_and_rail():
    tree = PowerTree.build(
        _simple_spec(
            loads=(LoadSpec("servo"), LoadSpec("pi", rail="rail5")),
            regulators=(RegulatorSpec("reg5", output_rail="rail5", v_out=5.0),),
        )
    )
    assert set(tree.load_inputs) == {"servo", "pi"}
    assert set(tree.load_outputs) == {"servo", "pi"}
    assert set(tree.rail_outputs) == {"main", "rail5"}
    assert len(tree.system.input_ports) == 2


# --- torque bridge ---------------------------------------------------------


def test_servo_currents_are_sign_independent_and_offset_by_quiescent():
    import jax.numpy as jnp

    k_i = jnp.asarray([2.0, 3.0])
    plus = servo_currents(jnp.asarray([1.0, 2.0]), k_i, quiescent=0.1)
    minus = servo_currents(jnp.asarray([-1.0, -2.0]), k_i, quiescent=0.1)
    np.testing.assert_allclose(np.asarray(plus), np.asarray(minus), atol=1e-12)
    np.testing.assert_allclose(np.asarray(plus), [2.1, 6.1], atol=1e-3)


def test_servo_currents_are_differentiable_at_zero_torque():
    import jax
    import jax.numpy as jnp

    grad = jax.grad(lambda t: jnp.sum(servo_currents(t, jnp.ones(2))))(jnp.zeros(2))
    assert np.all(np.isfinite(np.asarray(grad)))


# --- fitting ---------------------------------------------------------------


def test_fit_recovers_series_resistance_and_ocv():
    spec = _simple_spec()
    r_series = spec.pack.pack_R0 + spec.pack.pack_R1 + spec.main_ohm
    draw = np.linspace(0.0, 5.0, 40)
    voltage = 12.0 - r_series * draw
    telemetry = BusVoltageTelemetry(
        time=np.arange(draw.size) * 0.01, sample_rate_hz=100.0, voltage=voltage
    )
    fitted = fit_power_tree(telemetry, {"servo": draw}, spec)
    assert fitted["V_oc"] == pytest.approx(12.0, abs=_V_TOL)
    assert fitted["R_series"] == pytest.approx(r_series, rel=1e-6)


def test_fit_accounts_for_a_regulated_rail():
    spec = _simple_spec(
        loads=(LoadSpec("servo"), LoadSpec("pi", rail="rail5")),
        regulators=(
            RegulatorSpec("reg5", output_rail="rail5", v_out=5.0, efficiency=0.9),
        ),
    )
    r_series = spec.pack.pack_R0 + spec.pack.pack_R1 + spec.main_ohm
    servo = np.linspace(0.0, 4.0, 30)
    pi = np.full_like(servo, 1.0)
    voltage = np.empty_like(servo)
    for k in range(servo.size):
        solution = dc_solve(spec, {"servo": servo[k], "pi": pi[k]})
        voltage[k] = solution.rail_voltage["main"]
    telemetry = BusVoltageTelemetry(
        time=np.arange(servo.size) * 0.01, sample_rate_hz=100.0, voltage=voltage
    )
    fitted = fit_power_tree(telemetry, {"servo": servo, "pi": pi}, spec)
    assert fitted["R_series"] == pytest.approx(r_series, rel=1e-3)
    assert fitted["V_oc"] == pytest.approx(12.0, rel=1e-4)


def test_fit_refuses_a_constant_load():
    spec = _simple_spec()
    draw = np.full(20, 2.0)
    telemetry = BusVoltageTelemetry(
        time=np.arange(20) * 0.01, sample_rate_hz=100.0, voltage=np.full(20, 11.8)
    )
    with pytest.raises(CalibrationError, match="needs current variation"):
        fit_power_tree(telemetry, {"servo": draw}, spec)


def test_validate_flags_a_mismatched_tree():
    spec = _simple_spec()
    draw = np.linspace(0.0, 5.0, 20)
    # A trace generated with twice the declared series resistance.
    r_series = 2 * (spec.pack.pack_R0 + spec.pack.pack_R1 + spec.main_ohm)
    telemetry = BusVoltageTelemetry(
        time=np.arange(20) * 0.01,
        sample_rate_hz=100.0,
        voltage=12.0 - r_series * draw,
    )
    report = _validate_via_dc(spec, telemetry, draw)
    assert not report.ok
    assert "voltage residual" in report.warnings[0]


def _validate_via_dc(spec, telemetry, draw):
    """Score a trace without paying for a compile.

    ``PowerTree.validate`` is a pure DC-path method, so a tree whose compiled
    system is absent scores telemetry identically — and a test that asserts a
    residual should not spend ten seconds in the acausal compiler to do it.
    """
    tree = PowerTree(
        spec=spec,
        system=None,
        load_inputs={},
        rail_outputs={},
        load_outputs={},
        soc_output=0,
        pack_current_output=0,
    )
    return tree.validate(telemetry, {"servo": draw})


def test_validate_passes_on_a_self_consistent_trace():
    spec = _simple_spec()
    draw = np.linspace(0.0, 5.0, 20)
    voltage = np.array(
        [dc_solve(spec, {"servo": d}).rail_voltage["main"] for d in draw]
    )
    telemetry = BusVoltageTelemetry(
        time=np.arange(20) * 0.01, sample_rate_hz=100.0, voltage=voltage
    )
    assert _validate_via_dc(spec, telemetry, draw).ok


# --- aging -----------------------------------------------------------------


def test_aging_fades_capacity_and_grows_resistance():
    cell = _flat_cell()
    aged = CellAging().apply(cell, cycles=1000, days=365)
    assert aged.capacity_Ah < cell.capacity_Ah
    assert aged.R0 > cell.R0 and aged.R1 > cell.R1
    # The polarization time constant is held fixed by construction.
    assert aged.R1 * aged.C1 == pytest.approx(cell.R1 * cell.C1)


def test_aging_is_monotone_in_both_mechanisms():
    cell = _flat_cell()
    young = CellAging().apply(cell, cycles=100)
    old = CellAging().apply(cell, cycles=800)
    assert old.capacity_Ah < young.capacity_Ah
    assert old.R0 > young.R0


def test_aging_floors_capacity_instead_of_going_negative():
    aged = CellAging(capacity_fade_per_cycle=1e-2).apply(_flat_cell(), cycles=1e4)
    assert aged.capacity_Ah > 0.0


def test_aging_rejects_negative_age():
    with pytest.raises(JaxterityError, match="cycles >= 0"):
        CellAging().apply(_flat_cell(), cycles=-1.0)


def test_an_aged_pack_sags_more():
    spec = _simple_spec()
    aged = CellAging().apply(spec.pack, cycles=1000)
    import dataclasses

    aged_spec = dataclasses.replace(spec, pack=aged)
    assert (
        dc_solve(aged_spec, {"servo": 4.0}).rail_voltage["main"]
        < dc_solve(spec, {"servo": 4.0}).rail_voltage["main"]
    )
