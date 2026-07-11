# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""nanoLeRobot — SO-100: calibrated sim → LeRobotDataset → behaviour cloning.

The SO-100/SO-101 arm is the canonical robot of HuggingFace's open-source
`LeRobot <https://github.com/huggingface/lerobot>`_ stack. This is the
Karpathy-style end-to-end demo for it — everything top to bottom, no hidden
machinery:

1. Load the SO-100 from the robot zoo and apply its *committed calibration*
   plus a small physically-grounded override (``calibrated_so100`` — joint
   damping + floored distal-link inertia standing in for the geared Feetech
   servos' reflected rotor inertia, the sim-to-real correction calibration
   recovers; cf. the cartpole pole inertia in ``nanojaxterity``).
2. Build a fast, functional **position-controlled** environment on top of
   ``Robot.functional_dynamics`` — a stiff PD servo (the Feetech STS3215 is a
   high-gain position source) wrapping the calibrated MJX dynamics, ``jit`` +
   ``vmap`` over episodes.
3. Generate goal-reaching **demonstrations** with a scripted min-jerk expert and
   serialise them to disk in the **LeRobotDataset** format (the open-source
   piece we lean on — Jaxterity ships no trainer or policy, CONTEXT.md). The
   columns (``observation.state``, ``action``) are the SO-100 joint vector in
   LeRobot's own motor order, so the dataset is the same shape a real
   teleoperation recording would have.
4. Read the dataset back — through the real ``lerobot`` library if it is
   installed, otherwise a dependency-light reader — and train a small
   goal-conditioned **behaviour-cloning** policy (an MLP with LeRobot-style
   mean/std normalisation). *Self-contained here*, by the same design choice as
   ``nanojaxterity``'s PPO.
5. Reference check: the cloned policy, driving the calibrated sim closed-loop,
   reaches at least ``success_frac`` of held-out goal joint configurations
   within ``reach_tol`` (behavioural) and reproduces a committed rollout golden
   (reproducibility).

This stops short of deployment — that is the Jaxility lane.

Run::

    python -m examples.nanolerobot                 # demos → dataset → BC → check
    JAXTERITY_SAVE_GOLDEN=1 python -m examples.nanolerobot   # also (re)write goldens

The goldens live next to this file (``nanolerobot_policy.eqx``,
``nanolerobot_golden.npz``) and are what ``test/examples/test_nanolerobot.py``
loads, so the test does not retrain. A fresh LeRobotDataset is (re)written under
``nanolerobot_dataset/`` on every run; it is regenerable and not committed.
"""

from __future__ import annotations

import dataclasses
import os
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from jaxterity import zoo
from jaxterity.integrations import lerobot as lr

_HERE = Path(__file__).resolve().parent
POLICY_PATH = _HERE / "nanolerobot_policy.eqx"
GOLDEN_PATH = _HERE / "nanolerobot_golden.npz"
DATASET_DIR = _HERE / "nanolerobot_dataset"

#: SO-100/SO-101 revolute joints, in MuJoCo/MJX (= LeRobot) motor order.
JOINTS = lr.SO100_JOINTS

#: Per-joint position limits (rad), read from the compiled MJX model. The
#: gripper range is intentionally asymmetric — sampling goals outside a limit
#: would have MJX clamp the joint and the "reach" could never close.
Q_LO = jnp.asarray([-1.92, -1.745, -1.69, -1.658, -2.744, -0.175])
Q_HI = jnp.asarray([1.92, 1.745, 1.69, 1.658, 2.841, 1.745])

NJ = 6  # joints / action width


@dataclasses.dataclass(frozen=True)
class Config:
    """All knobs for the demo, in one place."""

    seed: int = 0
    # --- environment (a stiff position servo on the calibrated MJX dynamics) ---
    fps: int = 30  # LeRobot SO-100 control rate
    n_substep: int = 8  # RK4 substeps per control step (stiff servo + small inertia)
    kp: float = 150.0  # servo position gain (N·m/rad)
    kd: float = 2.8  # servo velocity gain (N·m·s/rad)
    tau_max: float = 3.5  # per-joint torque clamp (N·m) — ~STS3215 stall torque
    horizon: int = 150  # control frames per episode (= 5 s at 30 fps)
    cmd_step: float = 0.25  # expert per-frame command lead toward the goal (rad)
    # goals/starts are sampled within this fraction of each joint's half-range,
    # centred on the range midpoint (keeps every commanded goal inside limits).
    goal_frac: float = 0.6
    start_frac: float = 0.2
    # --- demonstrations ---
    n_demos: int = 128  # expert episodes written to the LeRobotDataset
    explore_noise: float = 0.12  # rad of command noise injected while demonstrating
    #   (DART-style: the recorded states fan out into a recovery tube; each is
    #   labelled with the *clean* reactive command, so the cloned policy learns
    #   to steer back to the goal from off-nominal states instead of compounding)
    # --- behaviour cloning ---
    hidden: int = 192
    depth: int = 2
    lr: float = 3e-4
    n_epochs: int = 400
    batch: int = 1024
    # --- evaluation / reference check ---
    n_eval: int = 24  # held-out (start, goal) pairs
    reach_tol: float = 0.15  # a goal is "reached" if every joint is within this (rad)
    success_frac: float = 0.85  # fraction of held-out goals that must be reached


# --------------------------------------------------------------------------- #
# Calibrated SO-100 — committed calibration + a physically-grounded override.
# --------------------------------------------------------------------------- #


def calibrated_so100():
    """SO-100 with the committed calibration plus servo-grounded overrides.

    The vendored URDF stamps near-zero inertia on the small distal links
    (wrist roll ``~4e-5``, gripper ``~1e-5`` kg·m²) and zero joint damping. For
    the geared Feetech STS3215 servos the *reflected rotor inertia* (gearbox
    ~345:1) dominates those tiny link inertias, and the servo has real internal
    friction. We floor the arm/distal-link inertia to a plausible ``3e-3`` kg·m²
    and add light joint damping — the same kind of sim-to-real correction that
    ``calibrating against real telemetry`` would recover, and the override that
    keeps an explicit RK4 stable for a stiff position servo at a sane substep
    (cf. the cartpole pole inertia in ``nanojaxterity``).
    """
    # Pin both damping AND friction so this demo owns its dynamics deterministically:
    # zoo.calibrated("so100") now carries real per-unit Coulomb friction, but
    # this self-contained BC demo uses its own controlled stand-in (damping 0.08,
    # friction 0) for golden stability.
    overrides: dict[str, object] = {f"{j}.damping": jnp.asarray(0.08) for j in JOINTS}
    overrides.update({f"{j}.friction": jnp.asarray(0.0) for j in JOINTS})
    for link in (
        "upper_arm_link",
        "lower_arm_link",
        "wrist_link",
        "gripper_link",
        "moving_jaw_so101_v1_link",
    ):
        overrides[f"{link}.inertia"] = jnp.diag(jnp.asarray([3e-3, 3e-3, 3e-3]))
    return zoo.calibrated("so100").with_parameters(overrides)


def make_env(robot, cfg: Config):
    """Return ``(sample_pose, step, rollout)`` over the calibrated servo dynamics.

    * State is ``[q(6), q̇(6)]``.
    * The **action is a joint-position command** ``q_cmd(6)`` (LeRobot SO-100
      semantics), clipped to the joint limits: the servo applies a
      torque-limited PD law ``τ = clip(kp·(q_cmd − q) − kd·q̇, ±τ_max)`` and the
      calibrated MJX dynamics integrate it with fixed-step RK4. The gains are
      stiff (the Feetech STS3215 is a high-gain position source); the small
      residual gravity droop on the load-bearing joints is part of the
      calibrated plant the cloned policy must cope with. Clipping the command to
      limits keeps a mis-extrapolating policy from driving the servo to a wild
      setpoint.
    * ``rollout(q0, command_fn)`` runs a whole episode; ``command_fn(state, t)``
      returns the commanded ``q_cmd`` at frame ``t`` and is where the expert or
      a cloned policy plugs in. It returns ``(states, commands)`` stacked over
      the horizon — exactly the ``observation.state`` / ``action`` columns a
      LeRobotDataset stores.
    """
    f = robot.functional_dynamics()
    sub_dt = (1.0 / cfg.fps) / cfg.n_substep

    def servo_step(state, q_cmd):
        q_cmd = jnp.clip(q_cmd, Q_LO, Q_HI)

        def one(s, _):
            q, qd = s[:NJ], s[NJ:]
            tau = jnp.clip(
                cfg.kp * (q_cmd - q) - cfg.kd * qd, -cfg.tau_max, cfg.tau_max
            )
            k1 = f(s, tau)
            k2 = f(s + 0.5 * sub_dt * k1, tau)
            k3 = f(s + 0.5 * sub_dt * k2, tau)
            k4 = f(s + sub_dt * k3, tau)
            return s + sub_dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4), None

        state, _ = jax.lax.scan(one, state, None, length=cfg.n_substep)
        return state

    mid = 0.5 * (Q_LO + Q_HI)
    half = 0.5 * (Q_HI - Q_LO)

    def sample_pose(key, frac):
        """Uniform joint configuration within ``frac`` of each joint half-range."""
        u = jax.random.uniform(key, (NJ,), minval=-1.0, maxval=1.0)
        return mid + frac * half * u

    def rollout(q0, command_fn):
        def scan_step(state, t):
            q_cmd = command_fn(state, t)
            nxt = servo_step(state, q_cmd)
            return nxt, (state[:NJ], q_cmd)

        s0 = jnp.concatenate([q0, jnp.zeros(NJ)])
        final, (states, commands) = jax.lax.scan(scan_step, s0, jnp.arange(cfg.horizon))
        return final, states, commands

    return sample_pose, servo_step, rollout


def reactive_expert(q, goal, cfg: Config):
    """The demonstration controller: a rate-limited move toward the goal.

    A *reactive* law — a pure function of the current measured joints ``q`` and
    the ``goal`` — that commands a point ``cmd_step`` rad ahead toward the goal
    (saturating to the goal when within a step). This is the deliberate design
    choice that makes behaviour cloning work here: a memoryless policy can
    represent it exactly, and because it is closed-loop feedback (contractive
    toward the goal), the cloned policy's small errors self-correct instead of
    compounding — the failure mode of cloning an open-loop/time-indexed
    trajectory like a min-jerk feedforward.
    """
    return q + jnp.clip(goal - q, -cfg.cmd_step, cfg.cmd_step)


# --------------------------------------------------------------------------- #
# Demonstrations → LeRobotDataset.
# --------------------------------------------------------------------------- #


def collect_demonstrations(robot, cfg: Config):
    """Roll out the reactive expert with exploration noise; return
    ``(starts, goals, states, commands)`` batched over ``n_demos`` episodes.

    During the rollout the executed command is the reactive law **plus**
    ``explore_noise`` Gaussian noise, so the recorded ``states`` fan out into a
    tube of slightly-off-nominal configurations. Each recorded state is then
    labelled with the *clean* reactive command — the action a relabelling expert
    would give from that exact state. ``states`` are ``observation.state``;
    ``commands`` are ``action``.
    """
    _, _, rollout = make_env(robot, cfg)
    key = jax.random.PRNGKey(cfg.seed)
    mid = 0.5 * (Q_LO + Q_HI)
    half = 0.5 * (Q_HI - Q_LO)
    k0, kg, kr = jax.random.split(key, 3)
    u0 = jax.random.uniform(k0, (cfg.n_demos, NJ), minval=-1.0, maxval=1.0)
    ug = jax.random.uniform(kg, (cfg.n_demos, NJ), minval=-1.0, maxval=1.0)
    starts = mid + cfg.start_frac * half * u0
    goals = mid + cfg.goal_frac * half * ug
    ep_keys = jax.random.split(kr, cfg.n_demos)

    def one_demo(q0, goal, ep_key):
        def noisy_expert(state, t):
            noise = cfg.explore_noise * jax.random.normal(
                jax.random.fold_in(ep_key, t), (NJ,)
            )
            return reactive_expert(state[:NJ], goal, cfg) + noise

        _, states, _ = rollout(q0, noisy_expert)
        # Relabel: the clean expert command from each *visited* state.
        commands = jax.vmap(lambda q: reactive_expert(q, goal, cfg))(states)
        return states, commands

    states, commands = jax.jit(jax.vmap(one_demo))(starts, goals, ep_keys)
    return starts, goals, states, commands


def to_episodes(goals, states, commands) -> list[lr.Episode]:
    """Pack batched rollouts into LeRobot :class:`~...lerobot.Episode` objects."""
    goals = np.asarray(goals)
    states = np.asarray(states)
    commands = np.asarray(commands)
    horizon = states.shape[1]
    episodes = []
    for i in range(states.shape[0]):
        goal_col = np.broadcast_to(goals[i], (horizon, NJ)).astype(np.float32)
        episodes.append(
            lr.Episode(
                data={
                    "observation.state": states[i].astype(np.float32),
                    "observation.goal": goal_col,
                    "action": commands[i].astype(np.float32),
                },
                task="reach the goal joint configuration",
            )
        )
    return episodes


def write_demo_dataset(root, robot, cfg: Config, episodes) -> Path:
    """Write the demonstrations to ``root`` as a LeRobotDataset (overwrites)."""
    import shutil

    root = Path(root)
    if root.exists():
        shutil.rmtree(root)
    features = lr.features_from_robot(
        robot, joint_names=list(JOINTS), include_goal=True
    )
    return lr.write_lerobot_dataset(
        root, episodes, features, fps=cfg.fps, robot_type="so100"
    )


# --------------------------------------------------------------------------- #
# Behaviour-cloning policy — a tiny normalised MLP (Jaxterity ships none).
# --------------------------------------------------------------------------- #


class Norm(eqx.Module):
    """Mean/std normalisation stats (LeRobot's default policy preprocessing).

    An ``eqx.Module`` so its ``mean``/``std`` are dynamic PyTree leaves — it
    lives inside :class:`BCPolicy` and rides through ``eqx.filter_jit``.
    """

    mean: jnp.ndarray
    std: jnp.ndarray

    def to(self, x):
        return (x - self.mean) / self.std

    def back(self, x):
        return x * self.std + self.mean


def _norm_of(arr) -> Norm:
    arr = jnp.asarray(arr)
    return Norm(mean=arr.mean(0), std=arr.std(0) + 1e-6)


def _as_float32(tree):
    """Cast every inexact-array leaf of a PyTree to float32 (dtype-pin a policy)."""
    return jax.tree_util.tree_map(
        lambda x: x.astype(jnp.float32) if eqx.is_inexact_array(x) else x, tree
    )


class BCPolicy(eqx.Module):
    """Goal-conditioned **residual** MLP: ``q_cmd = goal + net(q, goal)``.

    The command is parameterised as the goal plus a learned residual (the net
    predicts the normalised ``action − goal``). This is a deliberate inductive
    bias: the reactive expert commands ≈ the goal once it is near it, so the
    residual target →0 there — even an imperfect net commands ≈ the goal in the
    goal's neighbourhood, which makes the closed-loop fixed point land on the
    goal (up to the servo's gravity droop) instead of wherever a free-form
    prediction happens to be. Inputs and the residual are mean/std-normalised
    with stats taken from the dataset (LeRobot's default policy preprocessing).
    """

    net: eqx.nn.MLP
    obs_norm: Norm
    goal_norm: Norm
    res_norm: Norm

    def __init__(self, key, cfg: Config, obs_norm, goal_norm, res_norm):
        self.net = eqx.nn.MLP(
            2 * NJ, NJ, cfg.hidden, cfg.depth, activation=jax.nn.gelu, key=key
        )
        self.obs_norm = obs_norm
        self.goal_norm = goal_norm
        self.res_norm = res_norm

    def __call__(self, q, goal):
        x = jnp.concatenate([self.obs_norm.to(q), self.goal_norm.to(goal)])
        return goal + self.res_norm.back(self.net(x))


def train_bc(dataset: lr.LocalLeRobotDataset, cfg: Config, *, verbose: bool = True):
    """Train the BC policy on the LeRobotDataset; return the trained policy.

    Reads ``observation.state`` / ``observation.goal`` / ``action`` straight from
    the dataset (whatever reader :func:`load` returned) and fits the command map
    by mean-squared error — plain supervised behaviour cloning.
    """
    obs = jnp.asarray(dataset.frames("observation.state"))
    goal = jnp.asarray(dataset.frames("observation.goal"))
    act = jnp.asarray(dataset.frames("action"))
    # The policy predicts the residual ``action − goal``; normalise on that.
    obs_norm, goal_norm = _norm_of(obs), _norm_of(goal)
    res_norm = _norm_of(act - goal)

    key = jax.random.PRNGKey(cfg.seed + 1)
    key, mk = jax.random.split(key)
    model = BCPolicy(mk, cfg, obs_norm, goal_norm, res_norm)
    opt = optax.adam(cfg.lr)

    # Train only the MLP; the normalisation stats are fixed dataset statistics,
    # so freeze them (mark just ``model.net``'s arrays as differentiable).
    trainable = jax.tree_util.tree_map(lambda _: False, model)
    trainable = eqx.tree_at(
        lambda m: m.net,
        trainable,
        replace=jax.tree_util.tree_map(eqx.is_inexact_array, model.net),
    )
    # Split into differentiable arrays (the MLP weights) and everything else
    # (activations, frozen norm stats). Only ``diff`` rides the scan carry — the
    # activation function is a non-array leaf and is closed over via ``static``.
    diff, static = eqx.partition(model, trainable)
    opt_state = opt.init(diff)

    n = obs.shape[0]
    n_batch = max(1, n // cfg.batch)

    def loss_fn(diff, q, g, a):
        m = eqx.combine(diff, static)
        pred = jax.vmap(m)(q, g)
        return jnp.mean((pred - a) ** 2)

    @eqx.filter_jit
    def epoch(diff, opt_state, key):
        key, pk = jax.random.split(key)
        perm = jax.random.permutation(pk, n)

        def body(carry, i):
            diff, opt_state = carry
            idx = jax.lax.dynamic_slice_in_dim(perm, i * cfg.batch, cfg.batch)
            loss, grads = jax.value_and_grad(loss_fn)(
                diff, obs[idx], goal[idx], act[idx]
            )
            updates, opt_state = opt.update(grads, opt_state, diff)
            diff = eqx.apply_updates(diff, updates)
            return (diff, opt_state), loss

        (diff, opt_state), losses = jax.lax.scan(
            body, (diff, opt_state), jnp.arange(n_batch)
        )
        return diff, opt_state, key, jnp.mean(losses)

    for it in range(cfg.n_epochs):
        diff, opt_state, key, loss = epoch(diff, opt_state, key)
        if verbose and (it % 50 == 0 or it == cfg.n_epochs - 1):
            print(f"epoch {it:3d}  bc_mse {float(loss):.5f}")
    return eqx.combine(diff, static)


# --------------------------------------------------------------------------- #
# Reference check — closed-loop reach + committed golden.
# --------------------------------------------------------------------------- #


def eval_starts_goals(cfg: Config):
    """Held-out (start, goal) pairs, disjoint from training by a different key."""
    key = jax.random.PRNGKey(cfg.seed + 99)
    mid = 0.5 * (Q_LO + Q_HI)
    half = 0.5 * (Q_HI - Q_LO)
    k0, kg = jax.random.split(key)
    starts = mid + cfg.start_frac * half * jax.random.uniform(
        k0, (cfg.n_eval, NJ), minval=-1.0, maxval=1.0
    )
    goals = mid + cfg.goal_frac * half * jax.random.uniform(
        kg, (cfg.n_eval, NJ), minval=-1.0, maxval=1.0
    )
    return starts, goals


def closed_loop_reach(model, robot, cfg: Config, starts, goals):
    """Drive the calibrated sim with the cloned policy; return final joint error.

    Returns ``(final_q, max_err)`` where ``max_err[i]`` is the worst per-joint
    error of episode ``i`` at the final frame — the reach metric.
    """
    _, _, rollout = make_env(robot, cfg)

    def run_one(q0, goal):
        def policy_cmd(state, _t):
            return model(state[:NJ], goal)

        final, _, _ = rollout(q0, policy_cmd)
        return final[:NJ]

    finals = jax.jit(jax.vmap(run_one))(starts, goals)
    max_err = jnp.max(jnp.abs(finals - goals), axis=1)
    return np.asarray(finals), np.asarray(max_err)


def success_rate(max_err: np.ndarray, cfg: Config) -> float:
    """Fraction of held-out goals reached within ``reach_tol``."""
    return float(np.mean(max_err < cfg.reach_tol))


def reaches(max_err: np.ndarray, cfg: Config) -> bool:
    """True iff at least ``success_frac`` of the held-out goals were reached.

    A success-rate criterion (the standard one for imitation policies) rather
    than all-or-nothing: a single hard goal in the held-out set should not flip
    the verdict. The committed golden additionally pins the exact final joint
    configurations, so the behaviour is reproduced deterministically.
    """
    return success_rate(max_err, cfg) >= cfg.success_frac


def main() -> None:
    cfg = Config()
    robot = calibrated_so100()
    print(f"calibrated SO-100 loaded; collecting {cfg.n_demos} expert demos…")
    _, goals, states, commands = collect_demonstrations(robot, cfg)
    episodes = to_episodes(goals, states, commands)

    write_demo_dataset(DATASET_DIR, robot, cfg, episodes)
    dataset = lr.load_lerobot_dataset(DATASET_DIR)
    backend = "lerobot" if lr.HAS_LEROBOT else "local pyarrow reader"
    print(
        f"wrote LeRobotDataset → {DATASET_DIR.name}/ "
        f"({len(episodes)} episodes, {dataset.frames('action').shape[0]} frames); "
        f"loaded via {backend}. Training BC…"
    )
    model = train_bc(dataset, cfg)
    # Pin the policy to float32 *before* evaluating, so the committed golden
    # corresponds exactly to the float32 weights the test loads (MJX may enable
    # JAX x64, which would otherwise make the trained net float64 and the golden
    # irreproducible after a float32 round-trip). The env still runs in its own
    # dtype; only the policy parameters are pinned.
    model = _as_float32(model)

    starts, eval_goals = eval_starts_goals(cfg)
    finals, max_err = closed_loop_reach(model, robot, cfg, starts, eval_goals)
    ok = reaches(max_err, cfg)
    rate = success_rate(max_err, cfg)
    mean_e, worst_e = float(np.mean(max_err)), float(np.max(max_err))
    print(
        f"\nclosed-loop reach on {cfg.n_eval} held-out goals: "
        f"reached {rate:.0%} within {cfg.reach_tol} rad (need {cfg.success_frac:.0%}); "
        f"mean err={mean_e:.4f}, worst={worst_e:.4f} rad → "
        f"{'REACHED' if ok else 'MISSED'}"
    )

    if os.environ.get("JAXTERITY_SAVE_GOLDEN"):
        eqx.tree_serialise_leaves(POLICY_PATH, model)
        np.savez(
            GOLDEN_PATH,
            finals=finals,
            starts=np.asarray(starts),
            goals=np.asarray(eval_goals),
            max_err=max_err,
        )
        print(f"saved {POLICY_PATH.name} and {GOLDEN_PATH.name}")

    if not ok:
        raise SystemExit("nanolerobot: cloned policy did not reach the held-out goals.")


if __name__ == "__main__":
    main()
