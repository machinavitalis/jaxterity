// SPDX-License-Identifier: MIT
// Copyright (c) 2026 The Jaxterity Authors
//
// Wiring for the browser cartpole demo: fetch policy.json, build a physics
// backend, run the trained policy closed-loop, and draw the cart + pole + the
// commanded-force arrow on a 2D canvas. All demo-specific numbers come from
// policy.json, so this file is generic over any exported cartpole policy.

import { commandForce, selectAction } from "./policy.js";
import { makeBackend } from "./physics.js";

const canvas = document.getElementById("view");
const ctx = canvas.getContext("2d");
const els = {
  theta: document.getElementById("theta"),
  cart: document.getElementById("cart"),
  force: document.getElementById("force"),
  backend: document.getElementById("backend"),
  status: document.getElementById("status"),
  reset: document.getElementById("reset"),
  toggle: document.getElementById("toggle"),
  shove: document.getElementById("shove"),
  auto: document.getElementById("auto"),
};

let policy = null;
let backend = null;
let running = true;
let flash = 0; // frames remaining to flash the "disturbance" marker

//: Periodic disturbance: every ~3 s, kick the pole's angular velocity by a
//: random amount the trained controller has to reject. This is the point of the
//: demo — not just holding equilibrium, but *recovering* from shoves.
const DISTURB_PERIOD_MS = 3000;
const DISTURB_MIN = 1.0;
const DISTURB_MAX = 2.2;

function shove(sign) {
  if (!backend) return;
  const mag = DISTURB_MIN + Math.random() * (DISTURB_MAX - DISTURB_MIN);
  const s = sign ?? (Math.random() < 0.5 ? -1 : 1);
  backend.disturb(s * mag);
  flash = 18;
}

async function makeBackendOrFallback(kind, physics) {
  const b = makeBackend(kind, physics);
  if (kind === "mujoco-wasm") {
    try {
      await b.load("./cartpole.xml");
      els.status.textContent = "MuJoCo-WASM backend loaded.";
      return b;
    } catch (err) {
      els.status.textContent =
        "MuJoCo-WASM unavailable (" + err.message + ") — using closed-form. See MUJOCO_WASM.md.";
      return makeBackend("closed-form", physics);
    }
  }
  els.status.textContent = "Closed-form backend (ships in the page).";
  return b;
}

function resetEpisode() {
  backend.reset([0, policy.reset_theta, 0, 0]);
}

// --- rendering ------------------------------------------------------------ //

function draw(state, force) {
  const W = canvas.width;
  const H = canvas.height;
  ctx.clearRect(0, 0, W, H);

  const groundY = H * 0.7;
  const pxPerM = W / (2 * policy.physics.x_fail); // map [-x_fail, x_fail] -> width
  const cx = W / 2 + state[0] * pxPerM;
  const poleLenPx = policy.physics.L * pxPerM * 2.0; // L is half-ish; scale for visibility

  // ground
  ctx.strokeStyle = "#445";
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(0, groundY);
  ctx.lineTo(W, groundY);
  ctx.stroke();

  // cart
  const cw = 60;
  const ch = 30;
  ctx.fillStyle = "#3f7fd8";
  ctx.fillRect(cx - cw / 2, groundY - ch, cw, ch);

  // pole (theta measured from upright; +theta tips toward +x)
  const th = state[1];
  const px = cx + Math.sin(th) * poleLenPx;
  const py = groundY - ch - Math.cos(th) * poleLenPx;
  ctx.strokeStyle = "#e6822c";
  ctx.lineWidth = 8;
  ctx.lineCap = "round";
  ctx.beginPath();
  ctx.moveTo(cx, groundY - ch);
  ctx.lineTo(px, py);
  ctx.stroke();
  ctx.fillStyle = "#222";
  ctx.beginPath();
  ctx.arc(cx, groundY - ch, 5, 0, 2 * Math.PI);
  ctx.fill();

  // disturbance flash: a ring at the pole tip + a label
  if (flash > 0) {
    ctx.strokeStyle = `rgba(231, 76, 60, ${flash / 18})`;
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.arc(px, py, 16 + (18 - flash), 0, 2 * Math.PI);
    ctx.stroke();
    ctx.fillStyle = `rgba(231, 76, 60, ${flash / 18})`;
    ctx.font = "bold 14px system-ui";
    ctx.fillText("disturbance!", px + 18, py);
  }

  // force arrow
  if (Math.abs(force) > 1e-6) {
    const ay = groundY - ch / 2;
    const ax = cx + Math.sign(force) * (cw / 2 + 6);
    const len = Math.sign(force) * 40;
    ctx.strokeStyle = "#c0392b";
    ctx.lineWidth = 4;
    ctx.beginPath();
    ctx.moveTo(ax, ay);
    ctx.lineTo(ax + len, ay);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(ax + len, ay);
    ctx.lineTo(ax + len - Math.sign(force) * 8, ay - 5);
    ctx.lineTo(ax + len - Math.sign(force) * 8, ay + 5);
    ctx.closePath();
    ctx.fillStyle = "#c0392b";
    ctx.fill();
  }
}

// --- control loop --------------------------------------------------------- //

function tick() {
  if (running && backend) {
    const force = commandForce(backend.state, policy);
    const state = backend.step(force);
    draw(state, force);
    els.theta.textContent = state[1].toFixed(3);
    els.cart.textContent = state[0].toFixed(3);
    els.force.textContent = force.toFixed(1);
    if (
      Math.abs(state[1]) > policy.physics.theta_fail ||
      Math.abs(state[0]) > policy.physics.x_fail
    ) {
      resetEpisode(); // auto-recover so the demo runs forever
    }
    if (flash > 0) flash -= 1;
  }
  requestAnimationFrame(tick);
}

async function start() {
  policy = await (await fetch("./policy.json")).json();
  els.backend.value = "mujoco-wasm";
  backend = await makeBackendOrFallback(els.backend.value, policy.physics);
  resetEpisode();

  els.reset.addEventListener("click", resetEpisode);
  els.toggle.addEventListener("click", () => {
    running = !running;
    els.toggle.textContent = running ? "Pause" : "Resume";
  });
  els.shove.addEventListener("click", () => shove());
  els.backend.addEventListener("change", async () => {
    backend = await makeBackendOrFallback(els.backend.value, policy.physics);
    resetEpisode();
  });

  // periodic disturbances (the controls-interesting part) — toggleable
  setInterval(() => {
    if (running && els.auto.checked) shove();
  }, DISTURB_PERIOD_MS);

  // expose for the headless parity harness / debugging
  window.jaxterity = { policy, selectAction, commandForce, shove };
  requestAnimationFrame(tick);
}

start();
