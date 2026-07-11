# Everything about motor calibration & joints (low-cost servo arms)

A consolidated, durable reference for kinematic-offset calibration of low-cost
serial-bus servo arms (SO-101 / LeKiwi class, Feetech STS3215), distilled from
building a fully-automated, deterministic setup pipeline on real hardware. Read
this to recap the *concepts and physical effects* — distilled from a
chronological incident ledger (L-001…L-011) of what first revealed each effect
on real hardware.

> Mental model in one line: **calibration is fitting, per joint, the affine map
> between what the encoder reads (ticks) and what the joint physically is
> (radians in the robot's model frame) — and then proving that map against both
> the model and reality.**

---

## 1. The hardware, precisely

**Feetech STS3215** (representative of the class):
- **12-bit single-turn absolute encoder**: reads `0..4095` ticks over one 360°
  turn. `RESOLUTION = 4096`, so `TICKS_PER_RAD = 4096 / (2π) ≈ 651.9`.
- **Absolute within one turn** — power-cycle safe (no homing needed to know
  position), but it *wraps*: tick 4095 and tick 0 are adjacent. This single
  fact causes a whole class of bugs (§5.1).
- **Serial bus**, daisy-chained, addressed by ID. STS3215 uses
  `protocol_end = 0` (little-endian words); the wrong endianness silently reads
  garbage positions. Baud 1 000 000.
- **Position mode** takes `GOAL_POSITION`, `GOAL_SPEED`, and a `TORQUE_LIMIT`
  (0–1000 = fraction of stall torque the servo may apply to reach the goal).
- **Telemetry registers** that matter: `PRESENT_POSITION` (& 0x0FFF),
  `PRESENT_LOAD` (signed, fraction of max), `PRESENT_VOLTAGE`, `PRESENT_TEMP`.
- **Stall torque** ≈ 30 kgf·cm ≈ **2.94 N·m at 12 V**, and it scales with
  supply voltage — at 10.4 V you have ~2.55 N·m. This voltage→torque coupling
  is not a footnote; it decides whether a sweep succeeds (§5.7).

A **continuous-rotation wheel servo** is the same hardware without meaningful
hard stops: you only record a zero count, not a range.

---

## 2. The map: what "calibration" actually produces

Per joint, a 3-parameter affine map plus its valid interval:

```
tick = offset + sign · TICKS_PER_RAD · q          (q in radians, model frame)
q    = (tick − offset) / (sign · TICKS_PER_RAD)
```

- **offset** — the tick value at the joint's model-zero (`q = 0`).
- **sign** (±1) — does the tick count rise or fall as `q` increases? Encodes
  the joint's mounting direction relative to the model. (In LeRobot this becomes
  `drive_mode`: 0 for +1, 1 for −1.)
- **range_min / range_max** — the usable tick interval, the model joint limits
  pushed through the map and clamped to the encoder.
- **scale** = `TICKS_PER_RAD` — fixed by the encoder, but *measured* as a
  cross-check (§3.4); a measured scale far from datasheet means the calibration
  is lying to itself.

That's the whole deliverable. Everything else in the pipeline exists to find
these numbers safely and prove they're right.

---

## 3. Calibration decomposed (cheapest & safest first)

### 3.1 Offset — from a known landmark, read-only
A **hard stop** is a joint at a *known model angle* (the URDF/MJCF limit). Park
the joint there and you have one `(tick, angle)` pair → solve for offset with no
motor motion (so it cannot collide). This is the "seed" method: fold the whole
arm to its stops once, read every encoder, anchor every offset at once.

### 3.2 Sign — needs one bit of *external* physical truth
An encoder alone can never give the sign: both hypotheses fit any single joint's
readings equally (§5.2, the mirror problem). You must import one external fact.
Deterministic, camera-free options, in order of preference:
- **Stall probe at a stop** (best for hard-stop joints): at a stop, exactly one
  direction is free and one is blocked. A small load-capped nudge reveals the
  free (interior) direction; combined with *which* limit the joint rests at, it
  fixes the sign. Bonus: it also *verifies the seed pose* (a joint not at a stop
  moves freely both ways → abort).
- **Gravity-load sign**: read `PRESENT_LOAD`'s sign while holding the pose; the
  model predicts which q-direction gravity pulls. Works only for joints actually
  loaded by gravity in that pose.
- **Compliance** (for the gripper): the springy finger-contact end deflects
  measurably more under a harder push than a rigid open stop — that asymmetry
  identifies open vs closed without a camera.
- **Vision**: observed image-motion vs model-predicted motion. A wrist (eye-in-
  hand) camera resolves even the global-flip ambiguity. Most general, needs a
  camera.

### 3.3 Range — derived, then validated
Push the model limits through the fitted map and clamp to the encoder. A joint
whose far limit lands *outside* `[0, 4095]` has an **encoder-wrapped** range
(§5.1) — exclude it from motor sweeps and rely on the single reachable stop.

### 3.4 Scale — the truth check, needs both stops
Drive the joint to *both* hard stops; `measured_ticks_per_rad =
|t_hi − t_lo| / |q_hi − q_lo|`. Compare to datasheet (±12 %). A large miss means
the joint did **not** traverse its true range — a false stop (obstacle, floor,
weak battery) short-circuited the sweep. This is the single most informative
validation number.

---

## 4. The vocabulary (used precisely — they are not synonyms)

| Term | Question | Artifact |
|---|---|---|
| **Simulation** | What does the world model say is safe/reachable? | MuJoCo collision model (real meshes, derived floor, ACM) |
| **Choreography** | What exact motions should the hardware play? | home + per-joint stances + sweep arcs |
| **Calibration** | What is each joint's encoder→angle map? | offset / sign / range |
| **Registration** | Is that map in the *planning model's* frame? | sign/zero agreement with the MJCF |
| **Verification** | Did the mechanism do what we *commanded*? (vs spec) | arrival & realized-pose checks |
| **Validation** | Does the result match *physical reality*? (vs ground truth) | full-range both-stop sweeps |

```
SIMULATION → CHOREOGRAPHY → [hardware] → CALIBRATION → REGISTRATION → VALIDATION
                                         └────── VERIFICATION runs throughout ──┘
```

The trap that cost the most time: **"internally-consistent calibration" ≠
"registered to the model frame."** A clean symmetric sweep validates a joint in
isolation yet is blind to a sign/zero flip relative to the model — only
multi-joint poses expose it (§5.3).

---

## 5. The physical effects & gotchas (the part to actually remember)

### 5.1 Encoder wrap & the which-stop trap
A single-turn encoder is circular: tick 4095 ↔ 0 are adjacent. If a joint's
travel crosses that seam, one mechanical stop lies *beyond* the commandable
span, and naively anchoring to "a stop" can pick the wrong one — a clean-looking
~130° error. Defenses: do all offset arithmetic **modularly** (wrap-aware
differences), infer which stop keeps the range representable, cross-check with
the stall probe, and exclude wrapped joints from full sweeps.

### 5.2 Mirror ambiguity (sign ⊗ end are coupled, and proprioception is blind)
Per joint there are two observationally-identical hypotheses: at the lower stop
with sign +, or the upper stop with sign −. Both put the far stop at the same
tick, the same span, the same stalls. **No bus measurement distinguishes them.**
Worse, at the *whole-arm* level, mirroring two joints together (e.g. base-yaw +
shoulder) can produce a fold that *looks* nearly identical (~39° apart) — so
photos, encoders, and span-based validation all pass on a mislabeled pose. The
seed-stop end labels must therefore be **measured** (camera-anchored end probe),
not hand-annotated.

### 5.3 Registration ≠ internal consistency
The calibration can be perfect in the robot's own frame yet flipped/shifted
relative to the model the choreography is planned in. Symptom: every commanded
tick is "correct", verification passes, but the realized multi-joint pose is
wrong and sweeps dive into the floor. Fix: a **registration phase** *before* any
choreography — drive each joint to known *model* angles and confirm the realized
motion matches the model's prediction (excursion direction + magnitude), then a
cheap multi-joint home pose that exercises all joints at once.

### 5.4 The compliance effect (a "hard stop" is not a rigid point)
A hard stop's *position depends on contact force*. A joint pushed gently by a
torque-capped motor (e.g. 0.30) rests a few ticks short of where a firm human
hand-fold leaves it — measured directly: **~57 ticks (~5°) of give on the small
plastic wrist joints, vs ~2 ticks on the stiff proximal joints.** Consequences:
- Calibration anchored at gentle-touch force places the limit a few degrees
  *inside* the firm-push limit. Both are "correct" — at their force.
- A **two-force probe** quantifies it: deflection ∝ force, extrapolate to a
  reference. This is also how the gripper's springy end is identified (§3.2).
- When checking round-trip accuracy, expect a few-degree residual on compliant
  joints that is *physics, not calibration error*.
- Pick the force you calibrate at to match how the joint will be driven in use.

### 5.5 Soft stops (the gripper) need their own protocol
The gripper has no trustworthy hard stop: its open side is compliant and its
"closed" side is the fingers pressing together (springy, force-dependent). So it
is excluded from hard-stop seeding and gets a gentler protocol: gentle touches
at two torque caps, identify the finger end by its larger compliance deflection,
anchor with a *loose* scale gate (±30 %), and **functionally test** open/close
through the new map (the only joint actively position-tested, because passive
landmarks aren't trustworthy). Critically: **park the gripper in a known,
model-matching pose (closed) and hold it there before moving the arm** — an
unconstrained jaw changes the collision geometry every later step depends on
(L-011: a left-open gripper splays the fingertips into the floor mid-sweep).

### 5.6 Stops aren't the only false stops — obstacles, floor, self
A torque-capped drive stalls on *anything*: a wall, the floor, another
link, or its own gripper. Every "stop" is a hypothesis; the scale check (§3.4)
is what tells a true mechanical limit from a false one. Plan all motion against
the embodiment's real geometry first (§6) and keep ≥ the arm's reach of clear
space (measured: 49 cm reach → require ≥60 cm).

### 5.7 Voltage ↔ torque ↔ "can't move" vs "can't sweep"
Servo torque scales with supply voltage. At a depleted pack a gravity-loaded
joint can't lift the arm, so a **sweep** (which must drive *into* a stop and hold
torque there) stalls into false stops — while a **position move** to the same
configuration succeeds, because position control decelerates into the goal and
needs only transient torque. Observed directly: at 10.4 V shoulder_lift failed
its sweep (95 % scale error) yet cleanly reached the seed pose under a computed
torque cap. Lesson: *"can't reach a position" and "can't complete a stall-into-
stop sweep" are different failures* — distinguish them. Gate the battery
(≥ 11 V for a 12 V pack) before trusting sweeps.

### 5.8 Gravity load is computable — drive with "just enough" torque
The model gives the gravity-hold torque per joint at any configuration
(`qfrc_bias` at zero velocity). Divide by the voltage-derated stall torque to get
the exact `TORQUE_LIMIT` fraction a joint needs to hold/move there, add a small
friction+motion margin, and you get controlled, gentle motion that uses *only
the calculated amount of torque* (no slamming, no guessing a global cap).
shoulder_lift's lift needed 1.07 N·m = 42 % of derated stall — quantified, not
eyeballed.

### 5.9 Backlash / hysteresis
Gear backlash means the tick at a given physical angle depends on approach
direction. It shows up as a small offset difference between sweeping up vs down,
and as non-repeatable re-anchoring. Keep it bounded by always approaching a
landmark from the same direction, and by gating stop **re-anchor repeatability**
(we use ≤ 25 ticks).

### 5.10 Floor fidelity & collision honesty (sim must not lie)
- **Derive the floor, never hard-code it**: the floor is where the static
  wheels/base rest (lowest non-arm geom), not a guessed plane. A phantom-low
  floor lets the planner use clearance that doesn't exist.
- **Require a clearance *margin*, not just "no contact"** — a budget that
  absorbs held-joint sag, floor covering standing above the wheel-contact plane,
  and transit tracking lag (we use 8 cm).
- **Maximize clearance** among equal-coverage stances (free sag headroom).
- **Model the end-effector honestly**: if the real gripper opens wider than the
  model's jaw range, clearance computed on the model under-reads (L-011).

### 5.11 Hold *at target*, not *at present* (don't freeze the shortfall)
A gentle approach arrives a little short of target (within an arrival tolerance
but outside a tighter stance tolerance). Holding "at present position" then
freezes that shortfall forever; re-commanding the *target* under a near-rigid
torque cap closes it. Proximal joints bearing a cantilever during a distal sweep
need this stiff hold or they sag into the floor.

### 5.12 Verify the *path*, not just the endpoint
Endpoint arrival checks miss what happened *during* the move. A torque-starved
or obstructed transit can graze the floor or bump a wall while still arriving at
the endpoint. Monitor `PRESENT_LOAD` *throughout* every move and flag sustained
saturation — and note the still-open gap: the `touch_off` sweep itself also
needs in-motion monitoring (L-011). The logical endpoint of this idea is a
real-time telemetry-driven digital twin that watches the whole trajectory.

### 5.13 LeRobot homing-offset overflow (a format gotcha)
LeRobot writes `homing_offset` into a **signed 11-bit** register (±2047). The
naive `2048 − mid` overflows when the encoder zero is off-centre (issue #3193).
For a single-turn encoder the correction is modular: **wrap, then clamp** the one
extreme count, and assert the exported value is register-writable.

### 5.14 Wheel identity needs eyes
Which servo drives which physical wheel cannot be read off the bus — spin one at
a time and watch, or trust the assembly convention and verify once.

---

## 6. Why simulate first (the a-priori, per-embodiment plan)

Calibrating distal joints while proximal joints sit in arbitrary poses is how
you jam the gripper between base and floor on run one. Instead, in a faithful
model of *this* embodiment:
- find a compact collision-free **home**;
- for each joint, search a **stance** of the *others* from which it can sweep
  its whole range collision-free (a joint blocked from one shared pose is not
  blocked from its own tailored stance — never assume otherwise);
- route between them through the home (hub-and-spoke, easy to verify);
- check **every** transit and sweep against the derived floor + margin;
- prefer stances with the most floor headroom.

The output (the **choreography**) is what the hardware plays. It is embodiment-
specific: re-derive it (and re-measure the seed end labels) for a new arm.

---

## 7. Drift & recalibration

Hard stops don't move, so they're a stable reference: re-run the validation
sweeps periodically and watch the offset delta and measured scale. A growing
offset shift is mechanical (a loosened horn, a shifted zero); a growing scale
error is a sweep being obstructed or a weakening battery. Either is your
recalibration trigger. (Distinguish *kinematic-offset drift* — what this guide
covers — from *dynamic* parameter drift like friction/inertia, which is system
identification, a separate discipline.)

---

## 8. Deterministic gating (so it runs blind)

Every decision is a numeric threshold compared to a measurement — no human or
LLM judgment mid-run. Hard gates abort safely (torque off); soft gates fall back
(refuse a suspect sweep, keep the seed value) and are reported. Every failure
prints a likely cause + fix-and-rerun remedy, so a blind run ends with
instructions, not a stack trace. The standing thresholds we converged on:
voltage ≥ 11 V (12 V pack), scale ±12 % (gripper ±30 %), stance ≤ 50 ticks,
re-anchor ≤ 25 ticks, floor clearance 8 cm, coverage ≥ 98.5 %, clear space
≥ 60 cm.

---

## 9. Pointers
- [`lekiwi_setup.ipynb`](lekiwi_setup.ipynb) — the executable, self-documenting
  pipeline that implements all of this.
- `lekiwi_autosetup.py` (bus, touch-off, seed inference, LeRobot export) ·
  `lekiwi_calibrate_full.py` (closed-loop executor, transit monitor) ·
  `calib_planner_lekiwi.py` (collision model, derived floor) ·
  `calib_choreography_lekiwi.py` (stance search + sim verification).
