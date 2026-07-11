# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Command-line entry points for Jaxterity.

``jaxterity mcp serve`` launches the MCP server on stdio; ``jaxterity calibrate
--active --port …`` runs active (hardware-in-the-loop) system identification
— ``--port loopback`` exercises the whole path against a simulated arm,
with no hardware. ``simulate`` / ``export`` arrive alongside their capabilities.
"""

from __future__ import annotations

import argparse

#: Default damping leaves the ``damping`` recipe identifies (the SO-101's two
#: most observable revolute joints — matches ``examples/sysid_so101.py``).
_DAMPING_TARGETS = ("shoulder_lift.damping", "elbow_flex.damping")


def build_parser() -> argparse.ArgumentParser:
    """Build the ``jaxterity`` argument parser."""
    parser = argparse.ArgumentParser(
        prog="jaxterity",
        description="Jaxterity — JAX-native robotics layer on top of Jaxonomy.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    mcp = sub.add_parser("mcp", help="Model Context Protocol server commands.")
    mcp_sub = mcp.add_subparsers(dest="mcp_command", required=True)
    mcp_sub.add_parser("serve", help="Run the Jaxterity MCP server on stdio.")

    cal = sub.add_parser("calibrate", help="Identify a robot's parameters.")
    cal.add_argument(
        "--active",
        action="store_true",
        required=True,
        help="Active sysid: excite the robot over its servo bus, then fit.",
    )
    cal.add_argument(
        "--port",
        required=True,
        help="Serial port (e.g. /dev/ttyACM0), or 'loopback' for the "
        "hardware-free simulated arm.",
    )
    cal.add_argument(
        "--robot", default="so100", help="Zoo robot name (default: so100 / SO-101)."
    )
    cal.add_argument(
        "--recipe",
        default="damping",
        choices=("damping",),
        help="What to identify (default: damping).",
    )
    cal.add_argument(
        "--speed",
        type=float,
        default=0.1,
        help="Excitation initial joint speed, rad/s (default: 0.1).",
    )
    cal.add_argument(
        "--true-damping",
        type=float,
        default=0.12,
        help="Loopback only: the planted damping the fit must recover.",
    )

    pre = sub.add_parser(
        "preflight", help="Verify a candidate policy on a unit's twin."
    )
    pre.add_argument(
        "--robot", default="so100", help="Zoo robot name (default: so100 / SO-101)."
    )
    pre.add_argument(
        "--serial", default="SO101-0001", help="Unit serial for the twin handle."
    )
    pre.add_argument(
        "--hold-torque",
        type=float,
        default=0.4,
        help="Constant per-joint hold torque the candidate policy commands (N·m). "
        "Minimal stand-in for a loadable --policy trajectory.",
    )
    pre.add_argument(
        "--ceiling-rise",
        type=float,
        default=35.0,
        help="Servo-temperature ceiling as a rise above ambient, K.",
    )

    setup = sub.add_parser(
        "setup", help="First-time bring-up: discover, health-check, calibrate."
    )
    setup.add_argument(
        "--robot", default="lekiwi", help="Layout name (default: lekiwi; or so101)."
    )
    setup.add_argument(
        "--simulate",
        action="store_true",
        help="Exercise the whole touch-off path against a seeded simulator (no "
        "hardware) — recovers the planted offsets.",
    )
    setup.add_argument(
        "--port", default="/dev/ttyACM0", help="Serial port for the real bus."
    )
    setup.add_argument(
        "--mode",
        default="touch-off",
        choices=("touch-off", "single-pose"),
        help="Arm-joint calibration mode (default: touch-off, hands-free).",
    )
    setup.add_argument(
        "--out", default=".", help="Directory for the calibration artifacts."
    )

    return parser


def _run_preflight(args) -> int:
    """Run ``jaxterity preflight`` on a zoo twin with a constant-hold policy."""
    import jax.numpy as jnp

    from ..selfcheck import SafetyEnvelope, Scenario, constant_policy, preflight
    from ..twin import so101_twin

    twin = so101_twin(args.serial) if args.robot == "so100" else None
    if twin is None:
        print(f"preflight: no twin builder for robot {args.robot!r} .")
        return 2
    nv = twin.n_servos
    nq = twin.robot.build_system()._nq
    scenarios = [
        Scenario(
            name="hold-at-rest",
            q0=tuple([0.0] * nq),
            qd0=tuple([0.0] * nv),
            T0=tuple([twin.T_amb] * nv),
            n_steps=10,
            dt=0.02,
        )
    ]
    envelope = SafetyEnvelope(temp_ceiling_k=twin.T_amb + args.ceiling_rise)
    policy = constant_policy(jnp.full(nv, args.hold_torque))
    report = preflight(policy, twin, envelope, scenarios)
    print(report.summary())
    for r in report.results:
        print(f"  {r.name:18s} {r.decision.action.value:7s} — {r.decision.detail}")
    return 0 if report.passed else 1


def _run_calibrate(args) -> int:
    """Run ``jaxterity calibrate --active`` (loopback or real serial bus)."""
    import math

    from .. import zoo
    from ..calibrate.active import (
        ActiveLink,
        LoopbackLink,
        SerialBusLink,
        active_identify,
        design_coast_excitation,
        dof_joint_names,
    )

    robot = zoo.load(args.robot)
    target_params = _DAMPING_TARGETS
    excitation = design_coast_excitation(robot, speed=args.speed)

    link: ActiveLink
    if args.port == "loopback":
        planted = {p: float(args.true_damping) for p in target_params}
        link = LoopbackLink(robot, planted)
        print(
            f"Active sysid on {args.robot} via LOOPBACK (planted damping="
            f"{args.true_damping}) ...",
            flush=True,
        )
    else:
        names = dof_joint_names(robot)
        # Default Feetech STS layout: IDs 1..n, 4096 counts/rev. A real bring-up
        # supplies the unit's true IDs/resolution; this is the hardware path.
        link = SerialBusLink(
            args.port,
            names,
            tuple(range(1, len(names) + 1)),
            tuple(4096.0 / (2.0 * math.pi) for _ in names),
        )
        print(f"Active sysid on {args.robot} via {args.port} ...", flush=True)

    result = active_identify(robot, link, target_params, excitation=excitation)
    link.close()
    calibrated = result.apply_to(robot)
    rms = (result.residuals or {}).get("final_state_rms", float("nan"))
    print("  fitted:")
    for p in target_params:
        print(f"    {p:24s} {float(result.fitted[p]):.5f}")
    print(f"  final-state RMS  : {rms:.3e}")
    print(f"  attestation handle: {calibrated.attestation_handle}")
    return 0


def _run_setup(args) -> int:
    """Run ``jaxterity setup`` — first-time bring-up calibration.

    ``--simulate`` drives a seeded simulator (no hardware), recovering the planted
    offsets through the real touch-off code path; otherwise it drives the servo
    bus on ``--port``. Writes a native ``calibration.json`` and a LeRobot
    ``<robot>_lerobot.json`` into ``--out``.
    """
    import json
    import time
    from pathlib import Path

    from ..calibrate import writer
    from ..calibrate.layout import get_layout
    from ..calibrate.link import ActiveLink, FeetechLink
    from ..calibrate.setup import load_arm_limits, run_setup, simulated_link

    layout = get_layout(args.robot)
    arm_limits = load_arm_limits(layout)
    sim = bool(args.simulate)
    link: ActiveLink
    if sim:
        link = simulated_link(layout, arm_limits)
        print(f"setup: {layout.name} via SIMULATION (no hardware) ...", flush=True)
    else:
        link = FeetechLink(args.port, tuple(s.servo_id for s in layout.joints))
        print(f"setup: {layout.name} via {args.port} ...", flush=True)

    with link:
        result = run_setup(
            link,
            layout,
            arm_limits=arm_limits,
            mode=args.mode,
            sleep=(lambda _t: None) if sim else time.sleep,
        )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    writer.write_native(out / "calibration.json", result)
    lerobot = writer.lerobot_calibration_dict(result.calibrations)
    (out / f"{layout.name}_lerobot.json").write_text(
        json.dumps(lerobot, indent=2) + "\n"
    )
    for warning in result.warnings:
        print(f"  warning: {warning}")
    print(
        f"  calibrated {len(result.calibrations)} joints (ok={result.ok}); wrote "
        f"{out / 'calibration.json'} + {out / (layout.name + '_lerobot.json')}"
    )
    return 0 if result.ok else 1


def main(argv: list[str] | None = None) -> int:
    """Entry point for the ``jaxterity`` console script.

    Args
    ----
    argv : list[str] | None
        Argument vector excluding the program name. ``None`` uses ``sys.argv``.

    Returns
    -------
    int
        Process exit code (``0`` on success).
    """
    args = build_parser().parse_args(argv)

    if args.command == "mcp" and args.mcp_command == "serve":
        from ..mcp.server import serve

        serve()
        return 0

    if args.command == "calibrate":
        return _run_calibrate(args)

    if args.command == "preflight":
        return _run_preflight(args)

    if args.command == "setup":
        return _run_setup(args)

    # argparse's required subparsers make this unreachable, but keep a defined
    # return for type-checkers and future commands.
    return 2
