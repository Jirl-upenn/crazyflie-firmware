# app_nnpol — onboard neural-network policy

Runs a trained actor network (mjx-drone-trainer checkpoint) on the STM32,
replacing the ground-station policy loop for the `traj_lissajous/body`
figure-eight task. Phase 1 of
`mjc_dronetests/docs/onboard_inference_plan.md`; the ground-station side is
crazyflie-ros' `exec: onboard` (branch `onboard`).

## How it works

- `CONFIG_CONTROLLER_OOT` registers this app as `stabilizer.controller = 6`.
  The checkpoint's `action_type` (exported as `NNPOL_ACTION_KIND`) selects
  the inner controller the app wraps: firmware Mellinger for `mellinger`/
  `mellinger30`, the stock PID for `ctbr` (rate mode) and for the rpm family
  (`attitude`/`rpm`/`rpm_direct`, through controller_pid.c's per-motor
  `modeMotorPwm` bypass). With `nnpol.enable = 0` (default) the radio
  setpoint passes straight through to that controller — identical flight
  behaviour to `stabilizer.controller = 2` or `1`, which is validation
  step 1. The driver therefore needs `onboard_controller: oot` for a
  Mellinger checkpoint and `oot_rate` for the other two (controller 6 with
  the legacy packet read as rates).
- On every policy tick (`nnpol.ctrlHz`, a writable param defaulting to the
  checkpoint's `ctrl_freq`; the host pushes `policy.inference_hz` from
  `config.yaml` before engaging) the controller snapshots the EKF state + gyro and notifies
  the app task, which builds the observation, runs the network
  (`policy.c`, weights `const` in flash), decodes the action into an
  setpoint (attitude, body rates, or per-motor PWM) and publishes it into
  a sequence-locked buffer. The controller applies the newest complete
  action as a zero-order hold, synthesizing the `setpoint_t` the same
  command would have produced arriving by radio — INCLUDING cflib's pitch
  negation on the wire, which the phase-1 decoder missed (it wrote +pitch;
  bench-check `nnpol.spPitch` against a `/attitude_cmd` pitch step before
  the first onboard flight). Inference never runs in the 1 kHz stabilizer
  context.
- The host keeps streaming `/attitude_cmd` as the commander-watchdog
  keepalive and shadow reference; clearing `nnpol.enable` mid-flight is an
  instantaneous, format-compatible takeover by that stream.
- If the app task stops delivering actions, `nnpol.stale` counts and after
  5 policy periods the controller falls back to the radio setpoint.
- The rate becomes a stabilizer tick divider (`1000 / ctrlHz`, integer) at
  the `enable` rising edge, so only divisors of 1000 (100, 125, 200, 250,
  500 ...) run exactly; a non-divisor runs at the next divisor up (48 ->
  50 Hz) and read-only `nnpol.runHz` reports what actually engaged. The
  curve clock is tick-based, so changing the rate never shifts the
  reference in time — it only changes how often the network is stepped.

## Files

| file | role |
|---|---|
| `src/nnpol_controller.c` | firmware glue: OOT controller, app task, params, logs, scheduling |
| `src/nnpol_obs_lissajous.c` | the `traj_lissajous/body` observation (pure C, host-testable) |
| `src/nnpol_action.c` | action decode for all three kinds: setpoint box + thrust→PWM map, or mixer + rpm→PWM (pure C, host-testable) |
| `src/policy.c/.h` | GENERATED — the network, `const` fp32 weights |
| `src/firmware_export.h` | GENERATED — dims, rate, action box, task constants, hash |
| `test/` | host parity harness (plain gcc) + python runner |

## Regenerating the policy

```bash
cd mjc_dronetests
venv-crazyflow/bin/python export_policy_c.py \
    --run-dir runs/traj_lissajous/ppo_body_yawpen \
    --out-dir ../crazyflie-firmware/examples/app_nnpol/src
```

This also writes `firmware_export.json` (the hash record the ROS node
checks against `nnpol.hash0-3`) and `policy_reference.npz` (parity
vectors) into the run directory. Re-flash after regenerating — the hash
exists precisely so a stale flash refuses to fly a fresh checkpoint.

## Building and flashing

```bash
cd examples/app_nnpol
pixi run --manifest-path ../../pixi.toml make cf21bl_defconfig   # once
pixi run --manifest-path ../../pixi.toml make -j                 # build
pixi run --manifest-path ../../pixi.toml make cload              # flash
```

Measured build (cf21bl, 40→128→128→4 network): flash 392 KB / 1 MB (38%,
~89 KB of it weights), RAM 96 KB / 128 KB (73%), CCM 89%.

## Host parity test (before any flight)

```bash
python3 examples/app_nnpol/test/run_host_parity.py
```

Compiles the flight observation/action/network C with host gcc and checks
it against the trainer's numpy reference (`policy_reference.npz`) and
numpy transcriptions: network < 1e-5, observation < 2e-4, decode < 1e-3.
`--bag-npz` adds mocap-derived snapshots from the offboard baseline bag.

## Param / log reference

Params (`nnpol.`): `enable` (u8, the engage switch), `startPhase` (s),
`offX/offY/offZ` (m, the host's curve pin), `vnom` (V, thrust-map battery
voltage), `rpmScale` (MOTOR kind's rpm multiplier), `ctrlHz` (policy rate,
default = trained `ctrl_freq`) — all
latched at the `enable` rising edge; read-only: identity `hash0-3`,
`obsDim`, `actDim`, and `runHz` (the rate the engaged run steps at).

Logs (`nnpol.`): `t`, `act0-3` (clipped network output), `spRoll/Pitch/
Yaw/Thrust` (decoded setpoint in the host's wire units: deg or deg/s, +
legacy thrust), `pwm0-3` (MOTOR kind's per-motor PWM), `vbx/y/z`,
`wbx/y/z`, `e0x/y/z` (observation excerpts), `us` (inference time),
`stale`, `seq`.

## Engage protocol (what the ROS side does)

1. verify `nnpol.hash0-3` == the run dir's `firmware_export.json` (the
   hash covers the trained `ctrl_freq`), and that the configured
   inference rate divides 1000;
2. take off, hover at the curve's engage point, keep streaming
   `/attitude_cmd` (shadow);
3. push `startPhase`, `offX/Y/Z` (from the host's pin), `vnom`, `rpmScale`,
   `ctrlHz`;
4. set `enable = 1`, confirm by readback → onboard flight;
5. abort/land: clear `enable` first — the host stream takes over on the
   next controller tick.

## Bench checklist (props off) — WP2.7

- hash params match `firmware_export.json`; `nnpol.us` < 800;
- toggling `enable` while level yields sane `act*`/`sp*` logs;
- moving the vehicle by hand changes `e0*` consistently with the pinned
  curve (push toward +x ⇒ `e0x` shrinks when level).
