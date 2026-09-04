# app_nnpol — onboard neural-network policy

Runs a trained actor network (mjx-drone-trainer checkpoint) on the STM32,
replacing the ground-station policy loop for the `traj_lissajous/body`
figure-eight task. Phase 1 of
`mjc_dronetests/docs/onboard_inference_plan.md`; the ground-station side is
crazyflie-ros' `exec: onboard` (branch `onboard`).

## How it works

- `CONFIG_CONTROLLER_OOT` registers this app as `stabilizer.controller = 6`.
  With `nnpol.enable = 0` (default) the radio setpoint passes straight
  through to `controllerMellingerFirmware` — identical flight behaviour to
  `stabilizer.controller = 2`, which is validation step 1.
- On every policy tick (`nnpol.ctrlHz`, compiled in from the checkpoint's
  `ctrl_freq`) the controller snapshots the EKF state + gyro and notifies
  the app task, which builds the observation, runs the network
  (`policy.c`, weights `const` in flash), decodes the action into an
  attitude/thrust setpoint and publishes it into a sequence-locked buffer.
  The controller applies the newest complete action as a zero-order hold.
  Inference never runs in the 1 kHz stabilizer context.
- The host keeps streaming `/attitude_cmd` as the commander-watchdog
  keepalive and shadow reference; clearing `nnpol.enable` mid-flight is an
  instantaneous, format-compatible takeover by that stream.
- If the app task stops delivering actions, `nnpol.stale` counts and after
  5 policy periods the controller falls back to the radio setpoint.

## Files

| file | role |
|---|---|
| `src/nnpol_controller.c` | firmware glue: OOT controller, app task, params, logs, scheduling |
| `src/nnpol_obs_lissajous.c` | the `traj_lissajous/body` observation (pure C, host-testable) |
| `src/nnpol_action_mellinger.c` | action box decode + thrust→PWM map (pure C, host-testable) |
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
voltage) — all latched at the `enable` rising edge; read-only identity:
`hash0-3`, `obsDim`, `actDim`, `ctrlHz`.

Logs (`nnpol.`): `t`, `act0-3` (clipped network output), `spRoll/Pitch/
Yaw/Thrust` (decoded setpoint, deg + legacy thrust), `vbx/y/z`, `wbx/y/z`,
`e0x/y/z` (observation excerpts), `us` (inference time), `stale`, `seq`.

## Engage protocol (what the ROS side does)

1. verify `nnpol.hash0-3` == the run dir's `firmware_export.json`, and
   `nnpol.ctrlHz` == the checkpoint's `ctrl_freq`;
2. take off, hover at the curve's engage point, keep streaming
   `/attitude_cmd` (shadow);
3. push `startPhase`, `offX/Y/Z` (from the host's pin), `vnom`;
4. set `enable = 1`, confirm by readback → onboard flight;
5. abort/land: clear `enable` first — the host stream takes over on the
   next controller tick.

## Bench checklist (props off) — WP2.7

- hash params match `firmware_export.json`; `nnpol.us` < 800;
- toggling `enable` while level yields sane `act*`/`sp*` logs;
- moving the vehicle by hand changes `e0*` consistently with the pinned
  curve (push toward +x ⇒ `e0x` shrinks when level).
