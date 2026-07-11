// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Jaxterity Authors
//
// Two physics backends for the browser cartpole, behind one tiny interface:
//   reset(state) -> void          set the state [cart_x, theta, x_dot, theta_dot]
//   step(force)  -> state         advance one control step under a cart force (N)
//   get state()  -> number[4]
//
// 1. ClosedFormBackend — the parameter-faithful closed-form point-mass cartpole
//    deployment reduction, fixed-step RK4, augmented with the
//    calibrated joint damping so it matches what the policy trained on. Pure JS,
//    no dependencies; this is the default so the page runs anywhere. Its
//    derivative is parity-checked against a NumPy reference in
//    test/web/test_policy_parity.py.
//
// 2. MujocoWasmBackend — full-fidelity MuJoCo physics in the browser via a
//    MuJoCo-WASM build the user supplies (see MUJOCO_WASM.md). Loads the
//    exported cartpole.xml, writes the cart force to qfrc_applied, and reads
//    qpos/qvel back. This path needs the WASM runtime and is confirmed by
//    opening the page in a browser (it is not exercised by the Node tests).

// --- closed-form dynamics (pure, parity-tested) --------------------------- //

/** Continuous-time derivative of [x, theta, x_dot, theta_dot] under force F. */
export function cartpoleDeriv(s, F, p) {
  const [, th, xd, thd] = s;
  const sin = Math.sin(th);
  const cos = Math.cos(th);
  const tot = p.mc + p.mp;
  const den = tot - p.mp * cos * cos;
  const thdd =
    (p.g * sin * tot -
      cos * (F - p.cart_damping * xd + p.mp * p.L * thd * thd * sin) -
      (p.pole_damping * thd * tot) / (p.mp * p.L)) /
    (p.L * den);
  const xdd =
    (F - p.cart_damping * xd + p.mp * p.L * (thd * thd * sin - thdd * cos)) /
    tot;
  return [xd, thd, xdd, thdd];
}

const axpy = (a, x, y) => x.map((xi, i) => y[i] + a * xi); // y + a*x

/** One control step: n_substep RK4 sub-steps of size dt/n_substep. */
export function rk4Step(s, F, p) {
  const h = p.dt / p.n_substep;
  let st = s;
  for (let i = 0; i < p.n_substep; i++) {
    const k1 = cartpoleDeriv(st, F, p);
    const k2 = cartpoleDeriv(axpy(0.5 * h, k1, st), F, p);
    const k3 = cartpoleDeriv(axpy(0.5 * h, k2, st), F, p);
    const k4 = cartpoleDeriv(axpy(h, k3, st), F, p);
    st = st.map(
      (v, j) => v + (h / 6) * (k1[j] + 2 * k2[j] + 2 * k3[j] + k4[j]),
    );
  }
  return st;
}

export class ClosedFormBackend {
  constructor(physics) {
    this.name = "closed-form";
    this.ready = true;
    this.p = physics;
    this.state = [0, 0, 0, 0];
  }
  reset(state) {
    this.state = state.slice();
  }
  step(force) {
    this.state = rk4Step(this.state, force, this.p);
    return this.state;
  }
  /** Disturbance: add an angular-velocity kick to the pole (rad/s). */
  disturb(dThetaDot) {
    this.state[3] += dThetaDot;
  }
}

// --- MuJoCo-WASM backend (full-fidelity MuJoCo physics) ------------------- //

export class MujocoWasmBackend {
  constructor(physics) {
    this.name = "mujoco-wasm";
    this.ready = false;
    this.p = physics;
    this.state = [0, 0, 0, 0];
  }

  /**
   * Load the official google-deepmind/mujoco WASM build (npm `mujoco`) from
   * ./mujoco/mujoco.js and compile the exported MJCF. Throws if the runtime or
   * model is unavailable so the caller can fall back to the closed-form backend.
   */
  async load(mjcfUrl = "./cartpole.xml") {
    const loadMujoco = (await import("./mujoco/mujoco.js")).default;
    const mujoco = await loadMujoco();
    const xml = await (await fetch(mjcfUrl)).text();
    mujoco.FS.writeFile("/cartpole.xml", xml);
    this.mujoco = mujoco;
    this.model = mujoco.MjModel.mj_loadXML("/cartpole.xml");
    this.data = new mujoco.MjData(this.model);
    this.substeps = Math.max(1, Math.round(this.p.dt / this.model.opt.timestep));
    this.ready = true;
  }

  reset(state) {
    this.state = state.slice();
    if (!this.ready) return;
    // qpos = [cart_x, theta], qvel = [x_dot, theta_dot] (live views into WASM).
    const q = this.data.qpos;
    const v = this.data.qvel;
    q[0] = state[0];
    q[1] = state[1];
    v[0] = state[2];
    v[1] = state[3];
    this.data.qfrc_applied[0] = 0;
    this.data.qfrc_applied[1] = 0;
    this.mujoco.mj_forward(this.model, this.data);
  }

  step(force) {
    if (!this.ready) throw new Error("MuJoCo-WASM backend not loaded");
    const f = this.data.qfrc_applied; // applied generalized force on the slider DOF
    f[0] = force;
    f[1] = 0;
    for (let i = 0; i < this.substeps; i++) {
      this.mujoco.mj_step(this.model, this.data);
    }
    const q = this.data.qpos;
    const v = this.data.qvel;
    this.state = [q[0], q[1], v[0], v[1]];
    return this.state;
  }

  /** Disturbance: add an angular-velocity kick to the pole (rad/s). */
  disturb(dThetaDot) {
    if (!this.ready) return;
    this.data.qvel[1] += dThetaDot;
    this.mujoco.mj_forward(this.model, this.data);
    this.state[3] = this.data.qvel[1];
  }
}

/** Build a backend by name; returns the closed-form one for anything else. */
export function makeBackend(kind, physics) {
  return kind === "mujoco-wasm"
    ? new MujocoWasmBackend(physics)
    : new ClosedFormBackend(physics);
}
