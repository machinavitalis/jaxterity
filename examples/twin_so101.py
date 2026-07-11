# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Per-unit attested twin of an SO-101 arm.

What this shows, end to end:

1. Build a twin of a *specific unit* — the calibrated SO-101 body plus its
   off-body servo-thermal physics — identified by a serial number.
2. Read its **attestation handle**: a per-unit content hash that moves when the
   unit's parameters, serial, or provenance change, but is stable under cosmetic
   edits. Two units with identical parameters but different serials get distinct
   handles.
3. Roll the twin forward under a sustained holding torque (the fast
   ``functional_dynamics`` path) and read the **predicted servo temperature** —
   the off-body state the body model alone never sees. This forecast is what the
   self-check co-processor consumes to derate/veto before a servo
   actually overheats.

Run: ``python -m examples.twin_so101`` (or ``python examples/twin_so101.py``).
"""

from __future__ import annotations

import jax.numpy as jnp

from jaxterity.twin import Twin, so101_twin


def build_demo_twin(serial: str = "SO101-0001") -> Twin:
    """Build the bench SO-101 twin used by the demo."""
    return so101_twin(serial)


def predict_holding_temperature(
    twin: Twin, *, hold_torque: float = 0.6, seconds: float = 30.0, dt: float = 0.02
) -> dict:
    """Roll the twin forward under a constant holding torque; report the forecast.

    Args
    ----
    twin : Twin
        The unit to roll forward.
    hold_torque : float
        Per-servo holding torque applied throughout, N·m.
    seconds : float
        Horizon, s.
    dt : float
        Step size, s.

    Returns
    -------
    dict
        ``{"T0": ..., "T_final": ..., "rise": ...}`` in kelvin (the hottest
        servo's start, end, and rise).
    """
    nq = twin.robot.build_system()._nq
    nv = twin.n_servos
    n_steps = int(seconds / dt)
    q0 = jnp.zeros(nq)
    qd0 = jnp.zeros(nv)
    T0 = jnp.full(nv, twin.T_amb)
    tau = jnp.tile(jnp.full(nv, hold_torque), (n_steps, 1))
    out = twin.rollout(q0, qd0, T0, tau, dt)
    T_final = float(out.temperature[-1].max())
    return {
        "T0": float(twin.T_amb),
        "T_final": T_final,
        "rise": T_final - float(twin.T_amb),
    }


def main() -> None:
    """Build the twin, print its handle, and forecast its holding temperature."""
    twin = build_demo_twin()
    print(f"twin serial:       {twin.serial}")
    print(f"servos (DOFs):     {twin.n_servos}")
    print(f"attestation handle: {twin.attestation_handle}")

    # A second unit, same parameters, different serial → distinct handle.
    other = twin.with_serial("SO101-0002")
    print(f"sibling handle:     {other.attestation_handle}")
    print(f"handles distinct:   {twin.attestation_handle != other.attestation_handle}")

    forecast = predict_holding_temperature(twin)
    print(
        f"\nholding-torque forecast (off-body, unseen by the body model):\n"
        f"  ambient  T0      = {forecast['T0']:.2f} K\n"
        f"  predicted T_final = {forecast['T_final']:.2f} K\n"
        f"  rise            = {forecast['rise']:.2f} K"
    )


if __name__ == "__main__":
    main()
