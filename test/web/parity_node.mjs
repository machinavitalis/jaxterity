// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Jaxterity Authors
//
// Headless harness for test/web/test_policy_parity.py. Loads a generated bundle
// (its policy.json + policy.js + physics.js) and emits, as JSON on stdout:
//   - actions[]: the greedy action index the JS policy picks for each input state
//   - derivs[]:  the closed-form derivative at each input state (force = 5 N)
//   - rollout:   {maxTheta, maxX} over a closed-loop closed-form episode
// The Python test compares these to the JAX policy + a NumPy reference.
//
// Usage: node parity_node.mjs <bundleDir> <statesJsonPath>

import { readFileSync } from "node:fs";
import { pathToFileURL } from "node:url";

const [bundleDir, statesPath] = process.argv.slice(2);
const policy = JSON.parse(readFileSync(`${bundleDir}/policy.json`, "utf8"));
const states = JSON.parse(readFileSync(statesPath, "utf8"));

const policyMod = await import(pathToFileURL(`${bundleDir}/policy.js`).href);
const physicsMod = await import(pathToFileURL(`${bundleDir}/physics.js`).href);
const { selectAction, commandForce } = policyMod;
const { cartpoleDeriv, ClosedFormBackend } = physicsMod;

const TEST_FORCE = 5.0;
const actions = states.map((s) => selectAction(s, policy));
const derivs = states.map((s) => cartpoleDeriv(s, TEST_FORCE, policy.physics));

const backend = new ClosedFormBackend(policy.physics);
backend.reset([0, policy.reset_theta, 0, 0]);
let maxTheta = 0;
let maxX = 0;
for (let i = 0; i < 500; i++) {
  const force = commandForce(backend.state, policy);
  const st = backend.step(force);
  maxTheta = Math.max(maxTheta, Math.abs(st[1]));
  maxX = Math.max(maxX, Math.abs(st[0]));
}

process.stdout.write(JSON.stringify({ actions, derivs, rollout: { maxTheta, maxX } }));
