# SPDX-License-Identifier: MIT
# Copyright (c) 2026 The Jaxterity Authors

"""Parametric first, learned residual second, and the residual *is* the metric.

A pure learned model of a joint predicts well and explains nothing: ask it whether
the servo is wearing out and it has no answer, because it has no baseline to be
compared against. This module keeps the interpretable fit in front — the viscous
and Coulomb parameters, with their intervals — lets an optional learned residual
absorb what the parametric model cannot (hysteresis, backlash, thermal drift), and
then reports **the residual's size as the health signal**:

* **Small residual** — the physics suffices. The parameters are the model, and
  their intervals mean what they say.
* **Growing residual, same protocol and pose** — something unmodelled is
  developing. That is a reading available to a parametric-first method and *not*
  to a pure-residual one, which has nothing to measure the residual against.

So a residual that absorbs a lot is a **warning, not an improvement**. The
temptation is to report the hybrid's lower prediction error as a better model;
that would be measuring the wrong thing, and :func:`residual_gates` warns on
exactly the case where the hybrid fits much better than the physics.

Why the loss is on position, and Huber
--------------------------------------
The existing closed-loop fit works in the torque domain, which means it has to
double-differentiate a noisy position signal to get an acceleration and then a
torque. There is no torque label on this hardware at all — the load register is a
duty proxy.

Position, however, is measured directly. So the residual is trained by pushing it
**through the model** and comparing *positions*: pose substitutes for the absent
torque label. This is the approach NeuralActuator uses (arXiv 2607.11734), and
the reason it works is that the model supplies the map from torque to motion, so
a torque correction is identifiable from position error alone.

The loss is **Huber**, not squared. Stiction release, a dropped sample and a
backlash crossing all produce a handful of large position errors, and a squared
loss would let those few samples set the residual — which would then be fitting
events rather than structure, and the completeness metric would move for reasons
that have nothing to do with the robot's health.

Scope
-----
The prediction here is **one step ahead**: from each measured
``(q, q̇)``, integrate one step and compare to the next measured position. That
keeps the fit convex-ish and cheap, and it is enough to measure how much structure
the parametric model leaves behind. It is *not* as strong a test as a long
rollout, which compounds error and so punishes a residual that has merely
memorized noise; a multi-step version is the natural next step and would change
the numbers here (upward, for any residual that is overfitting).

Units
-----
Residual torques in N·m; position errors in rad. ``completeness`` and the absorbed
fraction are dimensionless.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from ..errors import CalibrationError

#: Huber transition, rad. Position errors below this are treated quadratically and
#: above it linearly. Set at roughly one encoder tick on a 650 ticks/rad servo
#: (1.5 mrad): below the quantum there is nothing to fit, and above it a large
#: error is far more likely to be a stiction event than a systematic effect.
HUBER_DELTA_RAD = 0.0015

#: Gradient steps for the residual fit. Enough for the small models here to
#: converge on a few thousand samples; the fit reports its final loss so a caller
#: can see whether it did.
_STEPS = 400

#: Adam hyperparameters. Hand-rolled rather than pulling in optax, which is not a
#: dependency of this package.
_LR = 0.01
_BETA1, _BETA2, _EPS = 0.9, 0.999, 1e-8

#: Absorbed fraction above which the learned residual is judged to be carrying
#: structure the physics should have. Half: at that point as much of the joint's
#: predictable behaviour lives in the black box as in the parameters, and the
#: parameters can no longer be quoted as a description of the joint.
ABSORBED_WARN_FRACTION = 0.5


@dataclasses.dataclass(frozen=True)
class ResidualModel:
    """A trained residual torque model: ``(q, q̇, load) → Δτ``.

    Deliberately small — one hidden layer, ``tanh`` — because its job is to
    *measure* leftover structure, not to be the model. A large network would
    absorb noise as readily as physics and the completeness metric would stop
    meaning anything.

    Attributes
    ----------
    weights : tuple
        Flat parameter tuple ``(W1, b1, W2, b2)`` as JAX arrays.
    feature_mean, feature_scale : tuple[float, ...]
        Input standardization recorded from the training capture, so the model can
        be applied to another capture without silently changing its input scale.
    hidden : int
        Hidden width.
    final_loss : float
        Huber loss at the last step, rad — reports whether the fit converged.
    """

    weights: tuple
    feature_mean: tuple[float, ...]
    feature_scale: tuple[float, ...]
    hidden: int
    final_loss: float

    def __call__(self, features: np.ndarray) -> np.ndarray:
        """Residual torque for a stack of ``(q, q̇, load)`` rows, N·m."""
        w1, b1, w2, b2 = self.weights
        mean = jnp.asarray(self.feature_mean)
        scale = jnp.asarray(self.feature_scale)
        x = (jnp.asarray(features) - mean) / scale
        return np.asarray(jnp.tanh(x @ w1 + b1) @ w2 + b2)


@dataclasses.dataclass(frozen=True)
class ModelCompleteness:
    """How much of a joint's behaviour the interpretable physics accounts for.

    Attributes
    ----------
    joint : str
        Joint name.
    parametric_rms_rad : float
        One-step position-prediction error of the **parametric model alone**, rad.
        This is the health metric: it is the size of what the physics does not
        explain, and it is the number to track across a unit's life.
    hybrid_rms_rad : float
        The same error with the learned residual added. Lower by construction —
        which is why it is *not* the metric.
    quantum_rad : float
        The encoder quantum, rad. A parametric RMS at or below this is as good as
        the instrument allows and there is nothing left to explain; reporting a
        residual below the quantum would be fitting the encoder.
    n_samples : int
        Samples the comparison used.
    """

    joint: str
    parametric_rms_rad: float
    hybrid_rms_rad: float
    quantum_rad: float
    n_samples: int

    @property
    def absorbed_fraction(self) -> float:
        """Share of the parametric error the residual removed.

        ``1 − hybrid/parametric``. Near 0 means the physics had already captured
        the structure. Near 1 means most of the joint's predictable behaviour was
        living outside the parameters, and the parameters should not be quoted as
        a description of the joint until it is understood.
        """
        if self.parametric_rms_rad <= 0:
            return float("nan")
        return 1.0 - self.hybrid_rms_rad / self.parametric_rms_rad

    @property
    def at_instrument_floor(self) -> bool:
        """Whether the parametric error is already at the encoder quantum.

        When true, ``absorbed_fraction`` says nothing about physics: the residual
        is fitting quantization, and a smaller hybrid error is not a better model.
        """
        return self.parametric_rms_rad <= self.quantum_rad

    def to_dict(self) -> dict[str, Any]:
        """Plain-JSON form with the derived quantities materialized."""
        d = dataclasses.asdict(self)
        d["absorbed_fraction"] = self.absorbed_fraction
        d["at_instrument_floor"] = self.at_instrument_floor
        return d


def _huber(error, delta: float):
    absolute = jnp.abs(error)
    return jnp.where(
        absolute <= delta, 0.5 * error**2, delta * (absolute - 0.5 * delta)
    )


def _one_step_positions(q, qd, dt, accel):
    """Constant-acceleration one-step position prediction."""
    return q + qd * dt + 0.5 * accel * dt**2


def _prepare(capture, inertia: float):
    """Aligned, leak-free arrays for a one-step position fit.

    Returns ``(q_now, qd_now, load_now, q_next, dt)``, all the same length.

    The velocity is a **backward** difference, and that is not a detail. A central
    difference — ``numpy.gradient`` — computes ``q̇[i]`` from ``q[i+1] − q[i−1]``,
    so the very sample being predicted is inside the feature the residual is given.
    A residual handed that can reconstruct part of its own target directly, and
    the absorbed fraction then measures the differencing scheme rather than the
    joint: on the real ``shoulder_pan`` capture it read 92 % that way. A backward
    difference uses only ``q[i]`` and ``q[i−1]``, so every feature predates the
    target and the number means what it claims.
    """
    time = np.asarray(capture.t, dtype=float)
    q = np.asarray(capture.q, dtype=float)
    load = np.asarray(capture.load_proxy, dtype=float)
    if q.size < 3:
        raise CalibrationError(
            f"fit_residual: the capture has {q.size} sample(s); a one-step "
            "position fit needs at least three."
        )
    dt = float(np.median(np.diff(time)))
    if not np.isfinite(dt) or dt <= 0:
        raise CalibrationError(
            "fit_residual: the capture's timestamps give a non-positive sample "
            f"interval ({dt}); cannot integrate a step."
        )
    if inertia <= 0:
        raise CalibrationError(
            f"fit_residual: the joint's inertia came out {inertia}, which cannot "
            "map a torque to an acceleration. Check the model and the joint name."
        )
    steps = np.diff(time)
    qd_causal = np.diff(q) / np.where(steps > 0, steps, np.nan)  # qd at index i+1
    # Index i runs over 1..n-2: state from q[i], qd[i] (which used q[i-1]), target
    # q[i+1]. Nothing in the state comes from at or after the target.
    return (
        q[1:-1],
        qd_causal[:-1],
        load[1:-1],
        q[2:],
        dt,
    )


def model_completeness(
    joint: str,
    capture: Any,
    *,
    inertia: float,
    damping: float,
    friction: float,
    quantum_rad: float,
    k_t: float = 1.0,
    load_offset: float = 0.0,
    residual: ResidualModel | None = None,
    coulomb_eps: float = 0.01,
) -> ModelCompleteness:
    """Score the parametric model's one-step position prediction, and the hybrid's.

    Args
    ----
    joint : str
        Joint name, carried into the result.
    capture : CommandedCapture
        A parsed capture.
    inertia : float
        The joint's axis inertia at the capture's pose, kg·m² — the map from
        torque to acceleration. Get it from
        :func:`jaxterity.calibrate.axis_inertia_feature`; a wrong value rescales
        both errors equally and so leaves ``absorbed_fraction`` intact.
    damping, friction : float
        The identified viscous (N·m·s/rad) and Coulomb (N·m) parameters.
    quantum_rad : float
        Encoder quantum, rad — the floor below which a residual is fitting the
        instrument.
    k_t, load_offset : float
        The measurement calibration that turns the servo's load *proxy* into a
        torque: ``τ = k_t · load + load_offset``. Both come out of
        :func:`~jaxterity.calibrate.identify_closed_loop` alongside the friction
        parameters. Leaving them at ``1.0`` / ``0.0`` treats the raw register as
        newton-metres, which handicaps the parametric model by an arbitrary scale
        factor and inflates the absorbed fraction — the comparison would then be
        against a straw man.
    residual : ResidualModel | None
        A trained residual. Without one, ``hybrid_rms_rad`` equals the parametric
        RMS and ``absorbed_fraction`` is 0: the honest reading for "no residual
        was fitted", rather than an omitted field.
    coulomb_eps : float
        Velocity scale of the smooth ``tanh`` Coulomb model, rad/s.

    Returns
    -------
    ModelCompleteness

    Raises
    ------
    CalibrationError
        If the capture is too short, its timestamps are unusable, or the inertia
        is non-positive.
    """
    q, qd, load, q_next, dt = _prepare(capture, inertia)
    tau_parametric = (
        k_t * load + load_offset - damping * qd - friction * np.tanh(qd / coulomb_eps)
    )
    predicted = _one_step_positions(q, qd, dt, tau_parametric / inertia)
    parametric_error = np.asarray(q_next - predicted)

    if residual is None:
        hybrid_error = parametric_error
    else:
        correction = np.asarray(residual(np.column_stack([q, qd, load]))).reshape(-1)
        hybrid_predicted = _one_step_positions(
            q, qd, dt, (tau_parametric + correction) / inertia
        )
        hybrid_error = np.asarray(q_next - hybrid_predicted)

    rms = lambda a: float(np.sqrt(np.mean(np.square(a))))  # noqa: E731
    return ModelCompleteness(
        joint=joint,
        parametric_rms_rad=rms(parametric_error),
        hybrid_rms_rad=rms(hybrid_error),
        quantum_rad=float(quantum_rad),
        n_samples=int(parametric_error.size),
    )


def fit_residual(
    joint: str,
    capture: Any,
    *,
    inertia: float,
    damping: float,
    friction: float,
    quantum_rad: float,
    k_t: float = 1.0,
    load_offset: float = 0.0,
    hidden: int = 8,
    steps: int = _STEPS,
    huber_delta: float = HUBER_DELTA_RAD,
    coulomb_eps: float = 0.01,
    seed: int = 0,
) -> tuple[ResidualModel, ModelCompleteness]:
    """Train a residual torque model against measured position, Huber loss.

    The parametric parameters are **held fixed**: this fits only what they leave
    behind, which is the whole point of the ordering. Fitting them jointly would
    let the network's flexibility pull the interpretable parameters off their
    physical values, and the intervals reported for them would no longer hold.

    Args
    ----
    joint : str
        Joint name.
    capture : CommandedCapture
        A parsed capture.
    inertia : float
        Axis inertia at the pose, kg·m².
    damping, friction : float
        The already-identified parameters, held fixed.
    quantum_rad : float
        Encoder quantum, rad.
    k_t, load_offset : float
        Load-proxy-to-torque calibration; see :func:`model_completeness`.
    hidden : int
        Hidden width. Small on purpose — see :class:`ResidualModel`.
    steps : int
        Gradient steps.
    huber_delta : float
        Huber transition, rad.
    coulomb_eps : float
        Coulomb velocity scale, rad/s.
    seed : int
        Initialization seed.

    Returns
    -------
    tuple[ResidualModel, ModelCompleteness]
        The trained residual and the before/after comparison. Read the
        completeness, not the residual: the residual is the instrument, the
        completeness is the reading.

    Raises
    ------
    CalibrationError
        If the capture cannot support a one-step fit (see
        :func:`model_completeness`).
    """
    q, qd, load, q_next, dt = _prepare(capture, inertia)
    tau_parametric = (
        k_t * load + load_offset - damping * qd - friction * np.tanh(qd / coulomb_eps)
    )

    features = np.column_stack([q, qd, load])
    mean = features.mean(axis=0)
    scale = features.std(axis=0)
    scale[scale <= 0] = 1.0  # a constant feature carries no information; do not /0
    x = jnp.asarray((features - mean) / scale)
    base_accel = jnp.asarray(tau_parametric / inertia)
    q0 = jnp.asarray(q)
    qd0 = jnp.asarray(qd)
    target = jnp.asarray(q_next)
    torque_to_position = 0.5 * dt**2 / inertia

    key = jax.random.PRNGKey(seed)
    k1, k2 = jax.random.split(key)
    n_features = x.shape[1]
    params: tuple = (
        jax.random.normal(k1, (n_features, hidden)) * 0.1,
        jnp.zeros(hidden),
        jax.random.normal(k2, (hidden, 1)) * 0.1,
        jnp.zeros(1),
    )

    def loss_fn(p):
        w1, b1, w2, b2 = p
        correction = (jnp.tanh(x @ w1 + b1) @ w2 + b2).reshape(-1)
        predicted = (
            q0 + qd0 * dt + 0.5 * base_accel * dt**2 + torque_to_position * correction
        )
        return jnp.mean(_huber(target - predicted, huber_delta))

    grad_fn = jax.jit(jax.value_and_grad(loss_fn))
    m = tuple(jnp.zeros_like(p) for p in params)
    v = tuple(jnp.zeros_like(p) for p in params)
    value = float("nan")
    for step in range(1, max(1, steps) + 1):
        value, grads = grad_fn(params)
        m = tuple(_BETA1 * mi + (1 - _BETA1) * g for mi, g in zip(m, grads))
        v = tuple(_BETA2 * vi + (1 - _BETA2) * g**2 for vi, g in zip(v, grads))
        m_hat = tuple(mi / (1 - _BETA1**step) for mi in m)
        v_hat = tuple(vi / (1 - _BETA2**step) for vi in v)
        params = tuple(
            p - _LR * mh / (jnp.sqrt(vh) + _EPS)
            for p, mh, vh in zip(params, m_hat, v_hat)
        )

    model = ResidualModel(
        weights=params,
        feature_mean=tuple(float(v_) for v_ in mean),
        feature_scale=tuple(float(v_) for v_ in scale),
        hidden=int(hidden),
        final_loss=float(value),
    )
    completeness = model_completeness(
        joint,
        capture,
        inertia=inertia,
        damping=damping,
        friction=friction,
        quantum_rad=quantum_rad,
        k_t=k_t,
        load_offset=load_offset,
        residual=model,
        coulomb_eps=coulomb_eps,
    )
    return model, completeness


def residual_gates(
    completenesses: tuple[ModelCompleteness, ...] | list[ModelCompleteness],
) -> list:
    """Gates over model completeness.

    One gate, and it fires in the direction people find counter-intuitive:
    ``model_completeness`` **warns when the learned residual fits much better
    than the physics**. A large absorbed fraction is not a better model, it is
    evidence that something the parameters do not represent is carrying the
    joint's behaviour — hysteresis, backlash, thermal drift — and until it is
    identified, the interpretable parameters are not a description of the joint.

    Nothing fires when the parametric error is already at the encoder quantum:
    there the residual is fitting quantization, and warning about it would be
    warning about the instrument.

    Args
    ----
    completenesses : sequence of ModelCompleteness
        Per-joint readings.

    Returns
    -------
    list[Gate]
        In the checkup's own :class:`~jaxterity.checkup.report.Gate` form.
    """
    from ..checkup.report import Gate

    gates = []
    for item in completenesses:
        if item.at_instrument_floor:
            continue
        fraction = item.absorbed_fraction
        if np.isfinite(fraction) and fraction >= ABSORBED_WARN_FRACTION:
            gates.append(
                Gate(
                    gate="model_completeness",
                    scope=item.joint,
                    level="warn",
                    detail=(
                        f"a learned residual removed {fraction:.0%} of the "
                        f"parametric model's position error "
                        f"({item.parametric_rms_rad * 1e3:.2f} → "
                        f"{item.hybrid_rms_rad * 1e3:.2f} mrad), so most of this "
                        "joint's predictable behaviour is not in its parameters."
                    ),
                    remedy="Do not read this as a better model. Look for the "
                    "missing mechanism — hysteresis, backlash or thermal drift — "
                    "and treat the fitted parameters as provisional until it is "
                    "identified.",
                )
            )
    return gates


__all__ = [
    "ABSORBED_WARN_FRACTION",
    "HUBER_DELTA_RAD",
    "ModelCompleteness",
    "ResidualModel",
    "fit_residual",
    "model_completeness",
    "residual_gates",
]
