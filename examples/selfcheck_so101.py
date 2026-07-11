# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Predictive safety vetoing on an SO-101: body-only vs twin.

The whole idea in one script. The body (MJX) model has no actuator thermal
limit — every DOF is force-controllable via ``qfrc_applied``, so a body-only
safety check says *torque available, proceed* for **any** command. The per-unit
twin knows the servo heats as ``tau²`` and, through the self-check co-processor
and the :class:`~jaxterity.selfcheck.SafetyMonitor`, vetoes a sustained hold that
would cook the servo — naming the binding constraint and the predicted time to
the ceiling.

Run: ``python -m examples.selfcheck_so101``.
"""

from __future__ import annotations

import jax.numpy as jnp

from jaxterity.selfcheck import CoProcessor, SafetyEnvelope, SafetyMonitor
from jaxterity.twin import so101_twin


def body_only_proceeds(twin, action) -> bool:
    """The body-only verdict: is the commanded torque deliverable?

    The MJX body model carries no ``<actuator>`` effort limit, so every
    finite torque is 'available' — a body-only check always proceeds.
    """
    return bool(jnp.all(jnp.isfinite(jnp.asarray(action))))


def main() -> None:
    twin = so101_twin("SO101-0001")
    nv = twin.n_servos
    nq = twin.robot.build_system()._nq
    T_amb = twin.T_amb

    # Real-time co-processor (short horizon) + an operator-declared envelope:
    # a 35 K rise ceiling (≈ a warm-but-safe servo) with a 10 K derate band.
    cp = CoProcessor(twin, rate_hz=50.0, horizon_steps=10, dt=0.02)
    monitor = SafetyMonitor(
        twin, SafetyEnvelope(temp_ceiling_k=T_amb + 35.0, temp_band_k=10.0)
    )

    # Measured "now": at rest, at ambient.
    q, qd, T = jnp.zeros(nq), jnp.zeros(nv), jnp.full(nv, T_amb)

    scenarios = [
        ("light reach (0.4 N·m)", 0.4),
        ("firm hold (1.2 N·m)", 1.2),
        ("sustained heavy hold (2.0 N·m)", 2.0),
    ]
    print(f"{'scenario':32s} {'body-only':10s} {'twin self-check'}")
    print("-" * 78)
    for label, tau in scenarios:
        action = jnp.full(nv, tau)
        body = "proceed" if body_only_proceeds(twin, action) else "refuse"
        prediction = cp.predict(q, qd, T, action)
        decision = monitor.evaluate(prediction, action=action)
        print(f"{label:32s} {body:10s} {decision.action.value:7s} — {decision.detail}")

    print(
        "\nThe body-only model proceeds on every command; the twin vetoes the "
        "sustained hold it predicts will breach the servo-thermal ceiling."
    )


if __name__ == "__main__":
    main()
