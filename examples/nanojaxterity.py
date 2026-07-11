# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""nanojaxterity — cartpole: URDF → calibration → PPO → balance, in one file.

The Karpathy-style end-to-end demo. Everything is here, top to bottom, no hidden
machinery:

1. Load the cartpole from the robot zoo and apply its *committed calibration*
   (``zoo.calibrated`` — a `Robot` whose MJX dynamics carry the fitted
   parameters).
2. Build a fast, functional environment on top of ``Robot.functional_dynamics``
   (a pure ``jit``/``vmap``-able MJX step — ``policies.S4.step`` rebuilds the
   simulator every call and is ~10⁶× too slow for RL).
3. Train a small PPO agent — *self-contained here*, because Jaxterity ships no
   policy code by design (CONTEXT.md) — to keep the pole upright (θ ≈ 0 is the
   unstable equilibrium).
4. Reference check: a trained agent keeps |θ| within ``theta_bound`` over an
   episode (behavioural), and reproduces a committed greedy-rollout golden
   (reproducibility).

This stops short of deployment — that is handled via Jaxility.

Run::

    python -m examples.nanojaxterity                 # train + report + check
    JAXTERITY_SAVE_GOLDEN=1 python -m examples.nanojaxterity   # also (re)write goldens

The agent goldens live next to this file (`nanojaxterity_policy.eqx`,
`nanojaxterity_golden.npz`) and are what `test/examples/test_nanojaxterity.py`
loads, so the test does not retrain.
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

_HERE = Path(__file__).resolve().parent
POLICY_PATH = _HERE / "nanojaxterity_policy.eqx"
GOLDEN_PATH = _HERE / "nanojaxterity_golden.npz"

#: Discrete action set: multiples of ``Config.force`` applied to the cart.
ACTIONS = (-1.0, 0.0, 1.0)

#: Observation scaling, applied before the policy net. The raw state is tiny
#: (θ ~ 0.05, x ~ 0.05), so an unscaled tanh-MLP barely varies its output and
#: the policy stays near-uniform — normalizing each dim to ~unit scale (θ by the
#: failure angle, x by the position limit) is what lets the agent actually learn.
OBS_SCALE = (1.0 / 2.4, 1.0 / 0.2, 0.5, 0.5)


@dataclasses.dataclass(frozen=True)
class Config:
    """All knobs for the demo, in one place."""

    seed: int = 0
    # --- environment ---
    force: float = 10.0  # N applied to the cart for a ±1 action
    dt: float = 0.02  # control timestep (s)
    n_substep: int = 2  # RK4 substeps per control step (0.01 s each — golden step)
    theta_fail: float = 0.20  # |pole angle| episode-failure threshold (rad)
    x_fail: float = 2.4  # |cart position| episode-failure threshold (m)
    # Initial state ~ Uniform(±spread), per dim [cart_x, θ, ẋ, θ̇]. A strong
    # angular-velocity kick (±0.6 rad/s) makes episodes dynamic so the agent
    # learns real corrective control (without it a near-upright start survives a
    # whole window doing nothing and there is no gradient).
    reset_spread: tuple = (0.05, 0.05, 0.4, 0.3)
    max_ep_len: int = 100_000  # effectively no truncation
    # Reward weights. Keeping the pole up dominates; a modest cart-velocity
    # penalty discourages the "run the cart away" strategy (which holds θ≈0 but
    # drifts off-screen) so the cart stays near centre, without the aggressive
    # position-regulation that destabilised training.
    w_theta: float = 1.0  # × (θ/θ_fail)²
    w_cart: float = 0.15  # × (cart_x/x_fail)²  (gentle position pull)
    w_cartvel: float = 0.05  # × ẋ²  (anti-drift: penalise cart speed)
    # --- PPO ---
    n_envs: int = 512
    n_rollout: int = 256  # ~5 s episodes in the buffer — long enough that the
    gamma: float = 0.997  # agent must value *long-term* balance, not just the
    lam: float = 0.95  # next couple of seconds (the eval runs 10 s).
    clip_eps: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.02
    max_grad_norm: float = 0.5
    lr: float = 1e-3
    n_epochs: int = 4
    n_minibatch: int = 8
    n_iters: int = 250
    # --- evaluation / reference check ---
    eval_start_theta: float = 0.05  # fixed tilted start for the golden rollout
    eval_steps: int = 500
    theta_bound: float = 0.10  # behavioural success: stay within this the whole episode


# --------------------------------------------------------------------------- #
# Environment — a functional cartpole on the calibrated MJX dynamics.
# --------------------------------------------------------------------------- #


def calibrated_cartpole():
    """Cartpole with *physically sensible* inertia and joint friction.

    The vendored pybullet URDF stamps a placeholder unit inertia
    (``1 kg·m²``) on every body and zero joint friction/damping. For the
    0.1 kg, ~1 m pole that inertia is ~120× too large, so the pole rotates in
    slow motion (it barely tilts and looks "floaty"), and the frictionless cart
    glides forever after each push — both visibly unphysical.

    Here we override the pole inertia to a real thin-rod value
    (``I ≈ m L²/12 ≈ 8.3e-3 kg·m²``) and add light joint damping. This is a
    hand-set stand-in for what **calibrating against real telemetry** would
    recover — and it is exactly the sim-to-real correction the rest of the
    stack exists to automate (the committed zoo calibration is synthetic, so
    it carries the URDF placeholders).
    """
    return zoo.calibrated("cartpole").with_parameters(
        {
            "pole.inertia": jnp.diag(jnp.asarray([8.3e-3, 8.3e-3, 8.3e-3])),
            "slider_to_cart.damping": jnp.asarray(0.8),  # cart rolling friction
            "cart_to_pole.damping": jnp.asarray(5e-3),  # small hinge loss
        }
    )


def make_env(robot, cfg: Config):
    """Return ``(reset, step)`` closures over the calibrated MJX dynamics.

    State is ``[cart_x, pole_θ, ẋ, θ̇]``; the action indexes :data:`ACTIONS`
    (a force on the prismatic ``slider_to_cart`` joint; the pole is passive).
    """
    f = robot.functional_dynamics()
    forces = jnp.asarray(ACTIONS) * cfg.force
    sub_dt = cfg.dt / cfg.n_substep

    def rk4(state, tau):
        def one(s, _):
            k1 = f(s, tau)
            k2 = f(s + 0.5 * sub_dt * k1, tau)
            k3 = f(s + 0.5 * sub_dt * k2, tau)
            k4 = f(s + sub_dt * k3, tau)
            return s + sub_dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4), None

        state, _ = jax.lax.scan(one, state, None, length=cfg.n_substep)
        return state

    spread = jnp.asarray(cfg.reset_spread)

    def reset(key):
        return jax.random.uniform(key, (4,), minval=-spread, maxval=spread)

    def step(state, action):
        tau = jnp.array([forces[action], 0.0])  # force on the cart, none on the pole
        nxt = rk4(state, tau)
        cart_x, theta, cart_v = nxt[0], nxt[1], nxt[2]
        done = (jnp.abs(theta) > cfg.theta_fail) | (jnp.abs(cart_x) > cfg.x_fail)
        # Shaped reward (1 at the upright, centred, still state). Keeping the pole
        # up dominates; the gentle position term and the cart-velocity term keep
        # the cart from drifting away (the cheap way to hold θ≈0) so it stays
        # near centre.
        reward = (
            1.0
            - cfg.w_theta * (theta / cfg.theta_fail) ** 2
            - cfg.w_cart * (cart_x / cfg.x_fail) ** 2
            - cfg.w_cartvel * cart_v**2
        )
        return nxt, reward, done

    return reset, step


# --------------------------------------------------------------------------- #
# Policy — a tiny actor-critic. (Jaxterity ships no policy code; this is the demo.)
# --------------------------------------------------------------------------- #


class ActorCritic(eqx.Module):
    """Separate two-layer tanh MLPs for the categorical actor and the value critic."""

    actor: eqx.nn.MLP
    critic: eqx.nn.MLP

    def __init__(self, key):
        ka, kc = jax.random.split(key)
        self.actor = eqx.nn.MLP(4, len(ACTIONS), 64, 2, activation=jax.nn.tanh, key=ka)
        self.critic = eqx.nn.MLP(4, "scalar", 64, 2, activation=jax.nn.tanh, key=kc)

    def __call__(self, state):
        obs = state * jnp.asarray(OBS_SCALE)
        return self.actor(obs), self.critic(obs)


def _log_prob(logits, action):
    return jnp.take_along_axis(jax.nn.log_softmax(logits), action[:, None], axis=1)[
        :, 0
    ]


# --------------------------------------------------------------------------- #
# PPO — rollout, GAE, clipped-surrogate update.
# --------------------------------------------------------------------------- #


def make_ppo(reset, step, cfg: Config):
    """Return ``(collect, update)``, both ``jit``-compiled, plus a fresh optimizer."""
    opt = optax.chain(optax.clip_by_global_norm(cfg.max_grad_norm), optax.adam(cfg.lr))
    vreset = jax.vmap(reset)
    vstep = jax.vmap(step)

    @eqx.filter_jit
    def collect(model, env_state, ep_len, key):
        def scan_step(carry, _):
            state, ep_len, key = carry
            logits, value = jax.vmap(model)(state)
            key, ak, rk = jax.random.split(key, 3)
            action = jax.random.categorical(ak, logits)
            logp = _log_prob(logits, action)
            nxt, reward, fail = vstep(state, action)
            ep_len = ep_len + 1
            done = fail | (ep_len >= cfg.max_ep_len)  # fail OR truncate
            fresh = vreset(jax.random.split(rk, cfg.n_envs))
            nxt = jnp.where(done[:, None], fresh, nxt)
            ep_len = jnp.where(done, 0, ep_len)
            return (nxt, ep_len, key), (state, action, logp, value, reward, done)

        (final_state, ep_len, key), traj = jax.lax.scan(
            scan_step, (env_state, ep_len, key), None, length=cfg.n_rollout
        )
        _, last_value = jax.vmap(model)(final_state)
        return final_state, ep_len, key, traj, last_value

    def _gae(rewards, values, dones, last_value):
        def scan_step(carry, x):
            adv, next_value = carry
            reward, value, done = x
            mask = 1.0 - done
            delta = reward + cfg.gamma * next_value * mask - value
            adv = delta + cfg.gamma * cfg.lam * mask * adv
            return (adv, value), adv

        _, advs = jax.lax.scan(
            scan_step,
            (jnp.zeros_like(last_value), last_value),
            (rewards, values, dones),
            reverse=True,
        )
        return advs, advs + values

    def _loss(model, batch):
        states, actions, old_logp, advs, returns = batch
        logits, values = jax.vmap(model)(states)
        new_logp = _log_prob(logits, actions)
        ratio = jnp.exp(new_logp - old_logp)
        a = (advs - advs.mean()) / (advs.std() + 1e-8)
        unclipped = ratio * a
        clipped = jnp.clip(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps) * a
        pg = -jnp.mean(jnp.minimum(unclipped, clipped))
        vf = 0.5 * jnp.mean((values - returns) ** 2)
        probs = jax.nn.softmax(logits)
        ent = -jnp.mean(jnp.sum(probs * jax.nn.log_softmax(logits), axis=-1))
        return pg + cfg.vf_coef * vf - cfg.ent_coef * ent

    @eqx.filter_jit
    def update(model, opt_state, traj, last_value, key):
        states, actions, logp, values, rewards, dones = traj
        advs, returns = _gae(rewards, values, dones, last_value)
        n = cfg.n_rollout * cfg.n_envs
        flat = (
            states.reshape(n, 4),
            actions.reshape(n),
            logp.reshape(n),
            advs.reshape(n),
            returns.reshape(n),
        )
        mb = n // cfg.n_minibatch
        # Carry only the differentiable arrays through the scans — an eqx Module
        # holds the activation function as a leaf, which is not a valid scan
        # carry. ``static`` (everything non-array) is closed over and recombined.
        params, static = eqx.partition(model, eqx.is_inexact_array)

        def loss_on_params(params, batch):
            return _loss(eqx.combine(params, static), batch)

        def epoch(carry, _):
            params, opt_state, key = carry
            key, pk = jax.random.split(key)
            perm = jax.random.permutation(pk, n)

            def minibatch(carry, i):
                params, opt_state = carry
                idx = jax.lax.dynamic_slice_in_dim(perm, i * mb, mb)
                batch = tuple(x[idx] for x in flat)
                loss, grads = jax.value_and_grad(loss_on_params)(params, batch)
                updates, opt_state = opt.update(grads, opt_state, params)
                params = eqx.apply_updates(params, updates)
                return (params, opt_state), loss

            (params, opt_state), losses = jax.lax.scan(
                minibatch, (params, opt_state), jnp.arange(cfg.n_minibatch)
            )
            return (params, opt_state, key), jnp.mean(losses)

        (params, opt_state, key), losses = jax.lax.scan(
            epoch, (params, opt_state, key), None, length=cfg.n_epochs
        )
        return eqx.combine(params, static), opt_state, key, jnp.mean(losses)

    return collect, update, opt


def train(robot, cfg: Config, *, verbose: bool = True):
    """Train a PPO agent to balance the cartpole; return the trained model."""
    reset, step = make_env(robot, cfg)
    collect, update, opt = make_ppo(reset, step, cfg)

    key = jax.random.PRNGKey(cfg.seed)
    key, mk, ek = jax.random.split(key, 3)
    model = ActorCritic(mk)
    opt_state = opt.init(eqx.filter(model, eqx.is_inexact_array))
    env_state = jax.vmap(reset)(jax.random.split(ek, cfg.n_envs))
    ep_len = jnp.zeros(cfg.n_envs, dtype=int)

    for it in range(cfg.n_iters):
        env_state, ep_len, key, traj, last_value = collect(
            model, env_state, ep_len, key
        )
        model, opt_state, key, loss = update(model, opt_state, traj, last_value, key)
        if verbose and (it % 10 == 0 or it == cfg.n_iters - 1):
            fail_rate = float(traj[5].mean())  # dones (fail or truncate)
            avg_return = float(traj[4].sum() / cfg.n_envs)  # rewards over the window
            print(
                f"iter {it:3d}  loss {float(loss):+.3f}  "
                f"avg_return/window {avg_return:6.1f}  fail_rate {fail_rate:.3f}"
            )
    return model


# --------------------------------------------------------------------------- #
# Reference check — behavioural bound + committed golden.
# --------------------------------------------------------------------------- #


def greedy_rollout(model, robot, cfg: Config):
    """Deterministic greedy rollout from the fixed tilted start; return the
    state trajectory ``(eval_steps, 4)``."""
    reset, step = make_env(robot, cfg)
    start = jnp.array([0.0, cfg.eval_start_theta, 0.0, 0.0])

    @eqx.filter_jit
    def run(model):
        def scan_step(state, _):
            logits, _ = model(state)
            nxt, _, _ = step(state, jnp.argmax(logits))
            return nxt, nxt

        _, traj = jax.lax.scan(scan_step, start, None, length=cfg.eval_steps)
        return traj

    return np.asarray(run(model))


def balances(traj: np.ndarray, cfg: Config) -> bool:
    """True iff the pole stayed within ``theta_bound`` for the whole rollout."""
    return bool(np.max(np.abs(traj[:, 1])) < cfg.theta_bound)


def main() -> None:
    cfg = Config()
    robot = calibrated_cartpole()
    print(f"calibrated cartpole loaded; training PPO ({cfg.n_iters} iters)…")
    model = train(robot, cfg)

    traj = greedy_rollout(model, robot, cfg)
    max_theta = float(np.max(np.abs(traj[:, 1])))
    ok = balances(traj, cfg)
    print(
        f"\ngreedy rollout from θ₀={cfg.eval_start_theta}: "
        f"max|θ|={max_theta:.4f} rad (bound {cfg.theta_bound}) → "
        f"{'BALANCED' if ok else 'FELL'}"
    )

    if os.environ.get("JAXTERITY_SAVE_GOLDEN"):
        eqx.tree_serialise_leaves(POLICY_PATH, model)
        np.savez(GOLDEN_PATH, rollout=traj)
        print(f"saved {POLICY_PATH.name} and {GOLDEN_PATH.name}")

    if not ok:
        raise SystemExit("nanojaxterity: trained agent did not balance the pole.")


if __name__ == "__main__":
    main()
