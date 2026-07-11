// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Jaxterity Authors
//
// The trained nanojaxterity actor, ported to a few lines of JavaScript. The
// weights live in policy.json (exported by jaxterity.web.bundle); this is a
// plain dense-MLP forward pass + greedy argmax. test/web/test_policy_parity.py
// runs this under Node and asserts the action it picks equals the JAX policy's
// for the same states, so "the browser runs the trained policy" is verified,
// not asserted.

/** Apply one dense layer: y[i] = dot(W[i], x) + b[i]. */
function dense(layer, x) {
  const { W, b } = layer;
  const y = new Array(W.length);
  for (let r = 0; r < W.length; r++) {
    const row = W[r];
    let s = b[r];
    for (let c = 0; c < row.length; c++) s += row[c] * x[c];
    y[r] = s;
  }
  return y;
}

const ACTIVATIONS = {
  tanh: Math.tanh,
  relu: (v) => (v > 0 ? v : 0),
};

/** Forward pass through the MLP; activation between layers, linear output. */
export function mlpForward(mlp, x) {
  const act = ACTIVATIONS[mlp.activation] || Math.tanh;
  let v = x;
  for (let i = 0; i < mlp.layers.length; i++) {
    v = dense(mlp.layers[i], v);
    if (i < mlp.layers.length - 1) v = v.map(act);
  }
  return v;
}

/** Index of the greedy action for `state` ([cart_x, theta, x_dot, theta_dot]). */
export function selectAction(state, policy) {
  const scaled = state.map((s, i) => s * policy.obs_scale[i]);
  const logits = mlpForward(policy.mlp, scaled);
  let best = 0;
  for (let i = 1; i < logits.length; i++) {
    if (logits[i] > logits[best]) best = i;
  }
  return best;
}

/** Cart force (N) the policy commands for `state`. */
export function commandForce(state, policy) {
  return policy.actions[selectAction(state, policy)] * policy.force;
}
