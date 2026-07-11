# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""First-run bring-up of a LeKiwi (SO-101 arm + 3 wheels), in simulation.

This is the "magic" command a beginner runs once on a new robot, shown here
against the built-in simulator so it runs with no hardware attached:

1. **Discover** — scan the servo bus and match it to the LeKiwi layout
   (arm IDs 1-6, wheels 7-9).
2. **Health** — read each servo's voltage and temperature.
3. **Calibrate** — find every arm joint's tick ↔ radian map *hands-free* by
   gently touching each mechanical hard stop. No manual joint-by-joint sweep:
   the stops sit at the URDF limits Jaxterity already knows, so only the
   per-joint offset has to be found. Wheels get a recorded zero.

On real hardware the same code runs through ``FeetechLink`` instead of the
simulator — that is exactly what ``jaxterity setup --port /dev/ttyACM0
--execute`` does. The offset calibration here is *kinematic* (which count is
``q = 0``); the *dynamic* parameters (friction, inertia) come later from
``jaxterity.sysid`` and are not needed just to teleoperate and collect data.

Run:  ``python -m examples.setup_lekiwi``  (needs the ``[mujoco]`` extra for the
zoo URDF).
"""

from __future__ import annotations

from jaxterity.calibrate import layout as _layout
from jaxterity.calibrate import setup as _setup
from jaxterity.calibrate.types import SetupResult


def run_simulated_setup(robot: str = "lekiwi") -> SetupResult:
    """Run the full bring-up against the simulator and return the result.

    Args
    ----
    robot : str
        Layout key (``"lekiwi"`` or ``"so101"``).

    Returns
    -------
    SetupResult
        The discovery, health, and calibration outcome.
    """
    layout = _layout.get_layout(robot)
    arm_limits = _setup.load_arm_limits(layout)
    handle = _setup.robot_handle(layout)
    link = _setup.simulated_link(layout, arm_limits)
    with link:
        return _setup.run_setup(
            link,
            layout,
            arm_limits=arm_limits,
            handle=handle,
            sleep=lambda _seconds: None,  # instant in simulation
        )


def main() -> None:
    """Bring up a simulated LeKiwi and print the calibration report."""
    result = run_simulated_setup("lekiwi")
    print(f"robot:     {result.robot_name}")
    print(f"bus found: {list(result.discovery.found_ids)} (ok={result.discovery.ok})")
    print(f"health ok: {result.health.ok}")
    print("calibration (offset = the encoder count that means q = 0):")
    for cal in result.calibrations:
        span = (
            f"  stops[{cal.range_min_ticks}..{cal.range_max_ticks}]"
            if cal.range_min_ticks is not None
            else ""
        )
        print(
            f"  {cal.joint:14s} id={cal.servo_id} offset={cal.offset_ticks:8.1f} "
            f"ticks  ({cal.method}){span}"
        )
    print(f"anchored to model handle: {result.robot_handle}")
    print(f"\nall clear: {result.ok}")


if __name__ == "__main__":
    main()
