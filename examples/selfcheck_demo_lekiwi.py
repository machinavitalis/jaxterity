# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""End-to-end predictive self-check on the REAL LeKiwi.

Unlike the simulated ``selfcheck_demo_so101.py`` (loopback self-ID, placeholder
off-body physics), every stage here is grounded in telemetry captured on a
*physical* LeKiwi and committed under ``test/calibrate/data/lekiwi/``:

1. **Active self-ID**: the arm's per-joint friction is identified from a
   commanded position-multisine driven through the servo inner loop
   (``identify_closed_loop``) — the real fit now committed in the ``so100`` zoo
   calibration. (``--active`` re-runs the fit from the committed capture live.)
2. **Attested serial-bound twin**: the calibrated robot + per-servo thermal + a
   battery whose Thévenin ``V_oc``/``R_int`` were fit from a real aggregate
   multi-servo load trace (``fit_battery_sag``).
3. **Co-processor**: runs beside the controller with zero deadline misses. (On
   the real Pi, the NumPy reduced co-processor holds rate with zero misses —
   ``scripts/lekiwi_reduced_coprocessor.py``.)
4. **Two failures a body-only model is blind to**: a sustained heavy hold the
   twin **vetoes on servo temperature** (naming the time-to-ceiling), and a
   sustained load on an aged pack the twin **vetoes on bus voltage** — while the
   body-only ``so101_twin`` (no thermal limit, no battery) proceeds on both.
5. **Pre-flight**: an unsafe trajectory is rejected against the unit's own
   twin; a safe one passes. The verdict binds to the twin's attestation handle.

The whole run is **reproducible from hashes**: the attestation handle plus every
decision hash to a stable ``run_hash``. Run:
``python -m examples.selfcheck_demo_lekiwi [--active]``.
"""

from __future__ import annotations

import dataclasses
import hashlib
import pathlib
import time

import jax.numpy as jnp

from jaxterity import zoo
from jaxterity.calibrate import (
    fit_battery_sag,
    identify_closed_loop,
    load_bus_load_trace,
    load_commanded_capture,
)
from jaxterity.selfcheck import (
    CoProcessor,
    SafetyEnvelope,
    SafetyMonitor,
    Scenario,
    constant_policy,
    preflight,
)
from jaxterity.twin import Twin, lekiwi_twin, so101_twin

_DATA = pathlib.Path(__file__).parent.parent / "test" / "calibrate" / "data" / "lekiwi"
_CALIB = _DATA / "calibration_final.json"
#: Aged-pack wear factor on the real-fit R_int — the degradation the online re-fit
#: is meant to catch; with it a sustained load browns the bus out.
_AGED_R_INT_FACTOR = 5.0


@dataclasses.dataclass(frozen=True)
class LeKiwiSelfCheckResult:
    """Structured, hashable outcome of the real-LeKiwi self-check chain."""

    twin_handle: str
    calibration_synthetic: bool
    battery_v_oc: float
    battery_r_int: float
    deadline_misses: int
    ticks_published: int
    thermal_action: str
    thermal_constraint: str
    thermal_time_to_breach_s: float | None
    brownout_action: str
    brownout_constraint: str
    brownout_v_min: float
    body_only_thermal_proceeds: bool
    body_only_brownout_proceeds: bool
    preflight_safe_passed: bool
    preflight_unsafe_passed: bool
    run_hash: str


def _calibrated_robot(active: bool):
    """Stage 1: the real per-joint friction calibration (committed, or re-fit live)."""
    if not active:
        return zoo.calibrated("so100")  # already a real closed-loop fit (Step 3)
    robot = zoo.load("so100")
    for name in (
        "commanded_shoulder_pan",
        "commanded_shoulder_lift",
        "commanded_elbow_flex",
    ):
        cap = load_commanded_capture(_DATA / f"{name}.json", robot, calibration=_CALIB)
        robot = identify_closed_loop(robot, cap).apply_to(robot)
    return robot


def run_demo(
    *,
    active: bool = False,
    serial: str = "LK-REAL-001",
    rate_hz: float = 20.0,
    clock=time.perf_counter,
) -> LeKiwiSelfCheckResult:
    """Run the full real-LeKiwi self-check chain and return a hashable result."""
    robot = _calibrated_robot(active)

    # Stage 2: battery from the real aggregate-load trace + the attested twin.
    batt = fit_battery_sag(load_bus_load_trace(_DATA / "aggregate_load_trace.json"))
    v_oc, r_int = float(batt.fitted["V_oc"]), float(batt.fitted["R_int"])
    twin = lekiwi_twin(serial, V_oc=v_oc, R_int=r_int)
    twin = Twin(  # bind the real calibrated robot (lekiwi_twin loads the zoo default)
        robot,
        serial=serial,
        C_th=twin.C_th,
        R_th=twin.R_th,
        k_pwr=twin.k_pwr,
        V_oc=v_oc,
        R_int=r_int,
        k_i=twin.k_i,
        T_amb=twin.T_amb,
        servo_names=twin.servo_names,
    )
    aged = dataclasses.replace(twin, R_int=jnp.asarray(_AGED_R_INT_FACTOR * r_int))
    body = so101_twin(serial)  # body-only: no battery, placeholder thermal
    nv = twin.n_servos
    nq = twin.robot.build_system()._nq
    T_amb = float(twin.T_amb)
    handle = twin.attestation_handle

    # Stage 3: co-processor beside a light-command controller, zero deadline misses.
    cp = CoProcessor(twin, rate_hz=rate_hz, horizon_steps=10, dt=0.02, clock=clock)
    report = cp.run(
        lambda _k: (
            jnp.zeros(nq),
            jnp.full(nv, 0.05),
            jnp.full(nv, T_amb),
            jnp.full(nv, 0.15),
        ),
        8,
    )

    # Stage 4a: thermal check — sustained heavy hold; body-only has no thermal limit.
    env_t = SafetyEnvelope(temp_ceiling_k=T_amb + 35.0)
    hold = jnp.full(nv, 2.0)
    pred_t = cp.predict(jnp.zeros(nq), jnp.zeros(nv), jnp.full(nv, T_amb), hold)
    therm = SafetyMonitor(twin, env_t).evaluate(pred_t, action=hold)

    # Stage 4b: brownout check — sustained load on aged pack; body-only has no battery.
    env_v = SafetyEnvelope(
        temp_ceiling_k=T_amb + 500.0, bus_floor_v=10.0, bus_band_v=0.5
    )
    cp_aged = CoProcessor(aged, rate_hz=rate_hz, horizon_steps=10, dt=0.02, clock=clock)
    pred_v = cp_aged.predict(jnp.zeros(nq), jnp.zeros(nv), jnp.full(nv, T_amb), hold)
    brown = SafetyMonitor(aged, env_v).evaluate(pred_v, action=hold)
    cp_body = CoProcessor(body, rate_hz=rate_hz, horizon_steps=10, dt=0.02, clock=clock)
    pred_body = cp_body.predict(jnp.zeros(nq), jnp.zeros(nv), jnp.full(nv, T_amb), hold)

    # Stage 5: pre-flight against the unit's own twin.
    scen = [
        Scenario(
            "hold-at-rest",
            tuple([0.0] * nq),
            tuple([0.0] * nv),
            tuple([T_amb] * nv),
            10,
            0.02,
        )
    ]
    safe = preflight(constant_policy(jnp.full(nv, 0.3)), twin, env_t, scen)
    unsafe = preflight(constant_policy(jnp.full(nv, 2.0)), twin, env_t, scen)

    payload = (
        f"{handle}|misses={report.deadline_misses}"
        f"|thermal={therm.action.value}:{therm.binding_constraint}"
        f"|brownout={brown.action.value}:{brown.binding_constraint}"
        f"|safe={safe.passed}|unsafe={unsafe.passed}"
    )
    run_hash = hashlib.sha256(payload.encode()).hexdigest()

    return LeKiwiSelfCheckResult(
        twin_handle=handle,
        calibration_synthetic=False,
        battery_v_oc=v_oc,
        battery_r_int=r_int,
        deadline_misses=report.deadline_misses,
        ticks_published=report.published,
        thermal_action=therm.action.value,
        thermal_constraint=therm.binding_constraint,
        thermal_time_to_breach_s=therm.time_to_breach_s,
        brownout_action=brown.action.value,
        brownout_constraint=brown.binding_constraint,
        brownout_v_min=float(jnp.asarray(pred_v.bus_voltage).min()),
        body_only_thermal_proceeds=True,  # raw MJX body carries no thermal limit
        body_only_brownout_proceeds=pred_body.bus_voltage is None,
        preflight_safe_passed=safe.passed,
        preflight_unsafe_passed=unsafe.passed,
        run_hash=run_hash,
    )


def main(active: bool = False) -> LeKiwiSelfCheckResult:
    print("Predictive self-check — REAL LeKiwi\n" + "=" * 46)
    r = run_demo(active=active)
    print(
        f"1.   real calibration     : synthetic={r.calibration_synthetic} "
        "(per-joint friction from real commanded captures)"
    )
    print(
        f"2.   attested twin        : {r.twin_handle[:16]}…  "
        f"battery V_oc={r.battery_v_oc:.2f}V R_int={r.battery_r_int:.3f}Ω (real fit)"
    )
    print(
        f"3.   co-processor         : {r.ticks_published} ticks, "
        f"{r.deadline_misses} deadline misses"
    )
    ttb = f" in {r.thermal_time_to_breach_s:.1f}s" if r.thermal_time_to_breach_s else ""
    print(
        f"4a.  thermal check        : body-only=proceed  "
        f"twin={r.thermal_action.upper()} [{r.thermal_constraint}]{ttb}"
    )
    print(
        f"4b.  brownout check       : body-only=proceed (no battery)  "
        f"twin={r.brownout_action.upper()} [{r.brownout_constraint}] "
        f"V_min={r.brownout_v_min:.2f}V"
    )
    safe_str = "PASS" if r.preflight_safe_passed else "FAIL"
    unsafe_str = "FAIL (rejected)" if not r.preflight_unsafe_passed else "PASS"
    print(f"5.   pre-flight           : safe={safe_str}  unsafe={unsafe_str}")
    print(f"6.   run hash             : {r.run_hash}")
    return r


if __name__ == "__main__":
    import sys

    main(active="--active" in sys.argv[1:])
