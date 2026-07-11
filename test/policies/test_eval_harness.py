# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Agent eval harness."""

from __future__ import annotations

import pytest

from jaxterity.policies.eval_harness import (
    CANNED_TASKS,
    EvalResult,
    reference_policy_for,
    run_eval,
)

pytest.importorskip("mujoco")


def test_canned_tasks_have_documentation():
    """Every task in the bank has a description an LLM can read cold."""
    for task in CANNED_TASKS:
        assert task.description
        assert len(task.description) > 30
        assert callable(task.success_predicate)


def test_empty_policies_dict_yields_zero_success_rate():
    results, rate = run_eval({})
    assert rate == 0.0
    for r in results:
        assert isinstance(r, EvalResult)
        assert r.success is False


def test_reference_baselines_pass_their_tasks():
    """Each task's reference policy should drive the predicate to True."""
    policies = {task.name: reference_policy_for(task) for task in CANNED_TASKS}
    results, rate = run_eval(policies)
    # Baselines are constructed exactly to satisfy each predicate, so all should
    # pass; this pins the harness's "successful run" path.
    assert rate == 1.0, [r for r in results if not r.success]


def test_policy_exception_counts_as_failure():
    """A policy that raises is reported as failed rather than propagating."""

    def bad_policy(s4, s3):
        raise RuntimeError("intentional")

    results, rate = run_eval({CANNED_TASKS[0].name: bad_policy})
    assert results[0].success is False
    assert rate == 0.0


# --- error surfaced + strict mode ---------------------------------


def test_policy_exception_records_repr_on_result():
    """Error string preserves repr(exception) for debugging."""

    def bad_policy(s4, s3):
        raise RuntimeError("specific bug message")

    results, _ = run_eval({CANNED_TASKS[0].name: bad_policy})
    assert results[0].error is not None
    assert "RuntimeError" in results[0].error
    assert "specific bug message" in results[0].error


def test_strict_mode_reraises_exceptions():
    """strict=True bubbles the exception instead of catching it."""

    def bad_policy(s4, s3):
        raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        run_eval({CANNED_TASKS[0].name: bad_policy}, strict=True)


def test_missing_policy_records_no_policy_supplied_error():
    """An absent policy records the diagnostic instead of just False."""
    results, _ = run_eval({})
    assert all(r.error == "no policy supplied" for r in results)


# --- third canned task on SO-100 -----------------------------------


def test_canned_tasks_include_so100_shoulder():
    """The task bank now spans cartpole + SO-100, not just cartpole."""
    robots = {t.robot for t in CANNED_TASKS}
    assert "cartpole" in robots
    assert "so100" in robots, (
        "expected an SO-100 task in CANNED_TASKS (the bank now includes the "
        "SO-100 shoulder reach; pick-and-place + hover stay deferred behind "
        "grasp / floating-base primitives)"
    )


# --- non-vacuous tasks -------------------------------------


def _by_name(name):
    return next(t for t in CANNED_TASKS if t.name == name)


def test_do_nothing_policy_fails_nonvacuous_tasks():
    """A no-op policy must FAIL both the pole-upright and the
    shoulder tasks. If it passed, the task would be testing nothing — exactly
    the bug the audit flagged for cartpole_pole_upright (the pole used to start
    at the goal) and so100 (it used to read a 1e-9-mass dummy joint)."""

    def do_nothing(s4, s3):
        return None

    for name in ("cartpole_pole_upright", "so100_shoulder_to_0.3"):
        results, rate = run_eval({name: do_nothing}, tasks=(_by_name(name),))
        assert rate == 0.0, f"{name} passed a do-nothing policy — vacuous task"


def test_pole_upright_setup_deflects_pole_off_goal():
    """The harness puts the pole outside the success tolerance before
    the policy runs, so success requires a real control loop."""
    import jaxterity.zoo as zoo
    from jaxterity.policies.s4 import S4

    task = _by_name("cartpole_pole_upright")
    assert task.setup is not None
    s4 = S4(zoo.load(task.robot))
    task.setup(s4)
    # Deflected beyond the 0.05-rad tolerance the predicate checks.
    assert abs(s4.read_encoder("cart_to_pole")) > 0.05
    assert not task.success_predicate(s4)


def test_so100_task_addresses_shoulder_pan_by_name():
    """The SO-100 predicate reads the real shoulder_pan joint;
    a missing joint name raises rather than silently falling back to dof 0."""
    import jaxterity.zoo as zoo
    from jaxterity.policies.s4 import S4

    s4 = S4(zoo.load("so100"))
    assert "shoulder_pan" in s4.joint_names
    # The predicate reads shoulder_pan (still at 0 here) — far from the 0.3 goal.
    assert not _by_name("so100_shoulder_to_0.3").success_predicate(s4)
