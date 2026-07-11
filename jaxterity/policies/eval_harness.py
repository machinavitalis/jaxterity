# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Agent-driven control evaluation harness.

Defines a small canned test bank and a runner that scores an LLM-authored
policy on it. The agent receives only the S1-S4 docstrings (the public
API surface, no implementation) and is asked to write a small Python
function ``policy(s4, s3) -> bool`` per task. The harness:

1. Constructs a fresh ``S4`` + ``S3`` for the right zoo robot.
2. Calls the policy.
3. Reports per-task success + a baseline score, optionally surfacing the
   exception when a policy raises (``strict=True`` re-raises so
   debugging an LLM-written policy doesn't have to grep stdout).

The shipped task bank is intentionally minimal: tasks 1–3 work on the
two-DOF cartpole (slide + balance) and the six-DOF SO-100 manipulator
(reach a joint config). Hover (Crazyflie) and pick-and-place (full
manipulation pipeline) depend on the deferred floating-base / contact /
grasp primitives and will land once
those primitives exist.

The runner can also be called on a deterministic *reference* policy
(:func:`reference_policy_for`) to produce a baseline number for the
README — the policy-quality eval pins the LLM's score *relative to* this
baseline rather than to an absolute threshold.
"""

from __future__ import annotations

import dataclasses
from typing import Callable

from .. import zoo
from .s2 import move_joint
from .s3 import S3 as _S3
from .s4 import S4


@dataclasses.dataclass(frozen=True)
class EvalTask:
    """One agent-eval task.

    Fields
    ------
    name : str
        Unique task id (e.g. ``"cartpole_slide_to_x"``).
    robot : str
        Zoo robot name.
    success_predicate : Callable[[S4], bool]
        Called after the policy runs; True iff the policy succeeded.
    description : str
        Plain-English task description shown to the LLM.
    setup : Callable[[S4], None] or None
        Optional hook applied by the harness *before* the policy runs.
        Used to put the robot in a non-trivial initial state (e.g.
        deflect a pole off-upright) so the success predicate is meaningful:
        a do-nothing policy must fail. None means "start from the zoo's
        default state".
    """

    name: str
    robot: str
    success_predicate: Callable
    description: str
    setup: Callable | None = None


def _slider_at(s4: S4, target: float, tolerance: float = 0.01) -> bool:
    return abs(s4.read_encoder("slider_to_cart") - target) < tolerance


def _pole_upright(s4: S4, tolerance: float = 0.1) -> bool:
    """Cartpole pole within ``tolerance`` rad of upright (q=0 in our zoo)."""
    return abs(s4.read_encoder("cart_to_pole")) < tolerance


# The cartpole pole starts upright (q=0) in the zoo, so a pole-upright
# predicate is vacuously satisfied off the initial condition. This setup hook
# deflects the pole well outside the success tolerance before the policy runs,
# so only a policy that closes a real control loop can pass.
_POLE_DEFLECTION = 0.15  # rad; > the 0.05 success tolerance below


def _deflect_pole(s4: S4) -> None:
    """Perturb the cartpole pole ``_POLE_DEFLECTION`` rad off-upright."""
    q, qd = s4.state()
    idx = s4.dof_index("cart_to_pole")
    s4.set_state(q.at[idx].set(_POLE_DEFLECTION), qd)


def _so100_shoulder_at(s4: S4, target: float, tolerance: float = 0.05) -> bool:
    """SO-100 ``shoulder_pan`` joint within ``tolerance`` rad of ``target``.

    Addresses the joint by its real name. (SO-100's joints are
    ``gripper_frame_joint, gripper, wrist_roll, wrist_flex, elbow_flex,
    shoulder_lift, shoulder_pan`` — there is no joint ``"1"``.) ``KeyError``
    propagates rather than silently falling back to dof 0.
    """
    return abs(s4.read_encoder("shoulder_pan") - target) < tolerance


CANNED_TASKS: tuple[EvalTask, ...] = (
    EvalTask(
        name="cartpole_slide_to_0.05",
        robot="cartpole",
        success_predicate=lambda s4: _slider_at(s4, 0.05, tolerance=0.02),
        description=(
            "Drive the cartpole's slider joint to position 0.05 rad. Use S2's "
            "move_joint or assemble a posture task via S3 + step_wbc."
        ),
    ),
    EvalTask(
        name="cartpole_pole_upright",
        robot="cartpole",
        success_predicate=lambda s4: _pole_upright(s4, tolerance=0.05),
        setup=_deflect_pole,
        description=(
            "The cartpole pole starts deflected ~0.15 rad off vertical. Drive it "
            "back to within 0.05 rad of upright. Use S3 + posture task + step_wbc "
            "(or S2.move_joint on the cart_to_pole joint)."
        ),
    ),
    EvalTask(
        name="so100_shoulder_to_0.3",
        robot="so100",
        success_predicate=lambda s4: _so100_shoulder_at(s4, 0.3, tolerance=0.05),
        description=(
            "Drive the SO-100's shoulder_pan joint to position 0.3 rad while "
            "keeping the rest of the arm at rest. Use S1.reach with a per-joint "
            "configuration target, or assemble S2.move_joint on the shoulder_pan "
            "DOF."
        ),
    ),
)


def reference_policy_for(task: EvalTask) -> Callable:
    """Return a deterministic baseline policy for ``task``.

    Used to compute the per-task score the LLM evaluation is benchmarked
    against. Each baseline assembles the minimum-viable S2/S3 calls the
    task needs.
    """
    if task.name == "cartpole_slide_to_0.05":

        def baseline(s4: S4, s3: _S3) -> None:
            move_joint(s3, "slider_to_cart", 0.05, max_steps=300, dt=0.01)

        return baseline

    if task.name == "cartpole_pole_upright":

        def baseline(s4: S4, s3: _S3) -> None:
            # The harness deflects the pole (_deflect_pole) before this runs, so
            # the baseline must actually close a control loop: drive the
            # cart_to_pole joint back to 0 via a posture task + WBC.
            move_joint(s3, "cart_to_pole", 0.0, max_steps=400, dt=0.01, tolerance=0.02)

        return baseline

    if task.name == "so100_shoulder_to_0.3":

        def baseline(s4: S4, s3: _S3) -> None:
            # Drive the real shoulder_pan joint (no "1" guess / fallback).
            move_joint(s3, "shoulder_pan", 0.3, max_steps=400, dt=0.01, tolerance=0.02)

        return baseline

    raise KeyError(f"no reference policy for task {task.name!r}")


@dataclasses.dataclass(frozen=True)
class EvalResult:
    """One row of an evaluation run.

    Fields
    ------
    task_name : str
        Identifier of the task.
    success : bool
        True iff the policy passed its predicate (and didn't raise).
    error : str or None
        ``repr(exception)`` when the policy raised, else None.
        Surfaces bugs in LLM-written policies without requiring strict mode.
    """

    task_name: str
    success: bool
    error: str | None = None


def run_eval(
    policies: dict[str, Callable],
    *,
    tasks: tuple[EvalTask, ...] = CANNED_TASKS,
    strict: bool = False,
) -> tuple[list[EvalResult], float]:
    """Run every task's policy and return per-task results + the success rate.

    Args
    ----
    policies : dict[str, Callable]
        Map of ``task_name -> policy(s4, s3)``. Missing tasks count as failures.
    tasks : tuple[EvalTask, ...]
        Tasks to evaluate. Defaults to :data:`CANNED_TASKS`.
    strict : bool
        If True, re-raise any exception a policy throws instead of recording it
        as a failure. Useful when debugging an LLM-written policy; the
        default False is appropriate for scoring runs where a crashing policy
        is just a failure to score.

    Returns
    -------
    (list[EvalResult], float)
        Per-task results and the overall success rate in [0, 1].
    """
    results = []
    for task in tasks:
        s4 = S4(zoo.load(task.robot))
        s3 = _S3(s4)
        if task.setup is not None:
            task.setup(s4)  # put the robot in a non-trivial start state
        policy = policies.get(task.name)
        if policy is None:
            results.append(EvalResult(task.name, False, error="no policy supplied"))
            continue
        err = None
        ok = False
        try:
            policy(s4, s3)
            ok = bool(task.success_predicate(s4))
        except Exception as e:
            if strict:
                raise
            err = repr(e)
        results.append(EvalResult(task.name, ok, error=err))
    rate = sum(r.success for r in results) / max(1, len(results))
    return results, float(rate)


__all__ = [
    "EvalTask",
    "EvalResult",
    "CANNED_TASKS",
    "reference_policy_for",
    "run_eval",
]
