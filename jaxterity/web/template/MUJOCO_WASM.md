# The MuJoCo-WASM backend

This bundle runs the trained controller on **MuJoCo physics compiled to
WebAssembly** — the official [`mujoco`](https://www.npmjs.com/package/mujoco)
build from Google DeepMind (Apache-2.0), simulating the exported `cartpole.xml`
client-side. It is the default backend. A dependency-free **closed-form**
backend ships in the page as a fallback (pick it from the *Physics* dropdown),
so the demo still runs if the WASM runtime is missing.

## Provide the runtime

`physics.js` loads the runtime from `./mujoco/mujoco.js` (with its
`./mujoco/mujoco.wasm` next to it). Fetch the official build and copy the two
single-threaded artifacts into a `mujoco/` folder beside `index.html`:

```bash
npm i mujoco
cp node_modules/mujoco/mujoco.js   <bundle>/mujoco/mujoco.js
cp node_modules/mujoco/mujoco.wasm <bundle>/mujoco/mujoco.wasm
```

Resulting layout:

```
<bundle>/
  mujoco/
    mujoco.js          # emscripten ES-module glue (default export = factory)
    mujoco.wasm        # ~8.6 MB engine
  cartpole.xml         # written by jaxterity.web
  index.html  app.js  physics.js  policy.js  policy.json  style.css
```

Use the single-threaded `mujoco.js` (not the `mt/` multithreaded variant) so no
cross-origin-isolation (COOP/COEP) headers are required — a plain static server
works.

## Serve over HTTP

ES-module imports and `fetch` need a server (not `file://`):

```bash
python -m http.server 8000   # from the bundle directory; open http://localhost:8000
```

If the runtime is missing or fails to load, the page logs the reason and falls
back to the closed-form backend.

## API the backend uses

`MujocoWasmBackend` (in `physics.js`) uses the canonical MuJoCo C API surfaced by
the bindings:

- `const mujoco = await (await import("./mujoco/mujoco.js")).default();`
- `mujoco.FS.writeFile("/cartpole.xml", xml)` then
  `model = mujoco.MjModel.mj_loadXML("/cartpole.xml")`
- `data = new mujoco.MjData(model)`
- per control step: write `data.qfrc_applied[0]` (slider force, N), call
  `mujoco.mj_step(model, data)` for `dt / model.opt.timestep` sub-steps, read the
  live views `data.qpos = [cart_x, theta]`, `data.qvel = [x_dot, theta_dot]`
- a disturbance adds to `data.qvel[1]` (pole angular velocity) and calls
  `mujoco.mj_forward`.

The state contract both backends share is `[cart_x, theta, x_dot, theta_dot]`.
