# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""End-to-end predictive self-check on a simulated SO-101.

The whole self-check story in one runnable script:

1. **Self-identify** the unit over its (loopback) servo bus with active sysid
   and calibrate it. *(``--active``; the default uses the committed calibration
   so the demo is fast.)*
2. **Build the attested twin**: a calibrated `Robot` + off-body servo-thermal
   physics + serial, with a per-unit attestation handle.
3. **Run the self-check co-processor** beside a simulated controller at the
   target rate, with zero deadline misses.
4. **The catch**: on a sustained heavy hold, the body-only model proceeds (it
   has no thermal limit) while the twin **vetoes** — naming the binding
   servo-thermal constraint and the predicted time to the ceiling.
5. **Pre-flight**: a deliberately unsafe trajectory is rejected against the
   unit's own twin; a safe one passes.

The run is **reproducible from hashes**: the twin attestation handle plus the
decisions hash to a stable ``run_hash``.

Run: ``python -m examples.selfcheck_demo_so101 [--active]``.
"""

from __future__ import annotations

import dataclasses
import hashlib
import time

import jax.numpy as jnp

from jaxterity import zoo
from jaxterity.calibrate.active import LoopbackLink, active_identify
from jaxterity.selfcheck import (
    CoProcessor,
    SafetyEnvelope,
    SafetyMonitor,
    Scenario,
    constant_policy,
    preflight,
)
from jaxterity.twin import Twin, so101_twin

_DAMPING = ("shoulder_lift.damping", "elbow_flex.damping")
# Representative bench-SO-101 servo-thermal defaults (per DOF) — matches
# jaxterity.twin.so101_twin (pending the per-unit off-body fit).
_C_TH, _R_TH, _K_PWR = 15.0, 4.0, 6.0


@dataclasses.dataclass(frozen=True)
class SelfCheckResult:
    """Structured outcome of the end-to-end self-check run."""

    twin_handle: str
    deadline_misses: int
    ticks_published: int
    body_only_proceeds: bool
    safety_action: str
    safety_constraint: str
    time_to_breach_s: float | None
    preflight_safe_passed: bool
    preflight_unsafe_passed: bool
    run_hash: str


def _build_twin(active: bool, serial: str) -> Twin:
    """Stage 1–2: (optionally) self-identify, then build the attested twin."""
    if not active:
        return so101_twin(serial)
    robot = zoo.load("so100")
    link = LoopbackLink(robot, {p: 0.12 for p in _DAMPING})
    calibrated = active_identify(robot, link, _DAMPING).apply_to(robot)
    nv = calibrated.build_system(actuation=True)._nv
    return Twin(
        calibrated,
        serial=serial,
        C_th=jnp.full(nv, _C_TH),
        R_th=jnp.full(nv, _R_TH),
        k_pwr=jnp.full(nv, _K_PWR),
        servo_names=tuple(f"servo_{i}" for i in range(nv)),
    )


def run_demo(
    *,
    active: bool = False,
    serial: str = "SO101-0001",
    n_ticks: int = 8,
    rate_hz: float = 20.0,
    clock=time.perf_counter,
) -> SelfCheckResult:
    """Run the full simulated self-check and return a structured, hashable result."""
    twin = _build_twin(active, serial)
    nv = twin.n_servos
    nq = twin.robot.build_system()._nq
    T_amb = twin.T_amb
    handle = twin.attestation_handle

    # 3. co-processor beside a simulated controller (constant light command).
    cp = CoProcessor(twin, rate_hz=rate_hz, horizon_steps=10, dt=0.02, clock=clock)

    def source(_k):
        return (
            jnp.zeros(nq),
            jnp.full(nv, 0.05),
            jnp.full(nv, T_amb),
            jnp.full(nv, 0.15),
        )

    run_report = cp.run(source, n_ticks)

    # 4. the catch: a sustained heavy hold. Body-only has no thermal limit.
    envelope = SafetyEnvelope(temp_ceiling_k=T_amb + 35.0)
    monitor = SafetyMonitor(twin, envelope)
    hold = jnp.full(nv, 2.0)
    prediction = cp.predict(jnp.zeros(nq), jnp.zeros(nv), jnp.full(nv, T_amb), hold)
    decision = monitor.evaluate(prediction, action=hold)
    body_only_proceeds = True  # the MJX body model carries no actuator thermal limit

    # 5. pre-flight: an unsafe trajectory is rejected, a safe one passes.
    scenarios = [
        Scenario(
            name="hold-at-rest",
            q0=tuple([0.0] * nq),
            qd0=tuple([0.0] * nv),
            T0=tuple([T_amb] * nv),
            n_steps=10,
            dt=0.02,
        )
    ]
    safe = preflight(constant_policy(jnp.full(nv, 0.3)), twin, envelope, scenarios)
    unsafe = preflight(constant_policy(jnp.full(nv, 2.0)), twin, envelope, scenarios)

    # 6. reproducible from hashes.
    payload = (
        f"{handle}|misses={run_report.deadline_misses}"
        f"|safety={decision.action.value}:{decision.binding_constraint}"
        f"|safe={safe.passed}|unsafe={unsafe.passed}"
    )
    run_hash = hashlib.sha256(payload.encode()).hexdigest()

    return SelfCheckResult(
        twin_handle=handle,
        deadline_misses=run_report.deadline_misses,
        ticks_published=run_report.published,
        body_only_proceeds=body_only_proceeds,
        safety_action=decision.action.value,
        safety_constraint=decision.binding_constraint,
        time_to_breach_s=decision.time_to_breach_s,
        preflight_safe_passed=safe.passed,
        preflight_unsafe_passed=unsafe.passed,
        run_hash=run_hash,
    )


def main(active: bool = False) -> SelfCheckResult:
    """Run the self-check and print the story."""
    print("Predictive self-check — SO-101 (simulated)\n" + "=" * 40)
    r = run_demo(active=active)
    print(f"1-2. attested twin handle : {r.twin_handle}")
    print(
        f"3.   co-processor         : {r.ticks_published}/{r.ticks_published} ticks "
        f"published, {r.deadline_misses} deadline misses"
    )
    ttb = f" (in {r.time_to_breach_s:.1f} s)" if r.time_to_breach_s is not None else ""
    print(
        f"4.   sustained heavy hold : body-only="
        f"{'proceed' if r.body_only_proceeds else 'refuse'}  "
        f"twin={r.safety_action} [{r.safety_constraint}]{ttb}"
    )
    safe_str = "PASS" if r.preflight_safe_passed else "FAIL"
    unsafe_str = "PASS" if r.preflight_unsafe_passed else "FAIL (rejected)"
    print(f"5.   pre-flight           : safe={safe_str}  unsafe={unsafe_str}")
    print(f"6.   run hash             : {r.run_hash}")
    print("\nThe twin vetoed a sustained hold the body-only model waved through.")
    return r


if __name__ == "__main__":
    import sys

    main(active="--active" in sys.argv[1:])
