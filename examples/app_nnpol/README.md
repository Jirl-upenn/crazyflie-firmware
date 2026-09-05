# app_nnpol — onboard neural-network policy (out-of-tree controller)

Runs a trained actor network on the Crazyflie's STM32F405 instead of on
the ground station, at the policy's control rate, from a policy the ground
station uploaded over the radio into one of five flash slots. Design and
validation ladder: `mjc_dronetests/docs/onboard_inference_plan.md`.

## What it does

- `CONFIG_CONTROLLER_OOT` registers this app as `stabilizer.controller = 6`.
  With `nnpol.enable = 0` (default) the radio setpoint passes straight
  through to the inner controller its shape asks for — an ANGLE setpoint
  (the driver's `oot` mode) to the firmware Mellinger, a RATE or per-motor
  setpoint (`oot_rate`) to the stock PID — so behaviour is identical to
  `stabilizer.controller = 2` or `1`, which is validation step 1.
- **The app is checkpoint-agnostic: build and flash it once.** Policies
  are `policy_slot.bin` blobs written by `mjc_dronetests/export_policy_c.py`
  (a 256-byte header — identity hash, dims, activation, decode kind, action
  box or mixer anchors, thrust identification, task constants — plus the
  fp32 weights) that live in the internal-flash sectors 7-11, one 128 KB
  slot each, uploaded through the `MEM_TYPE_APP` memory handler and
  selected with `nnpol.slot`. The app task validates the selected slot
  (CRC) and publishes its identity in the read-only `nnpol.hash0-3` /
  `obsDim` / `actDim` / `kind` / `taskId` / `ctrlHz` params, which the
  ground station checks before engaging. crazyflie-ros does all of this
  from `config.yaml`: a drone's run directory is found in a slot or
  uploaded into a free one, then selected — no reflash.
- On every policy tick (`nnpol.ctrlHz`, a writable param defaulting to the
  slot's trained `ctrl_freq`; the host pushes `policy.inference_hz` before
  engaging) the controller snapshots the EKF state + gyro and notifies the
  app task, which builds the observation, runs the network through the
  interpreter (`nnpol_policy.c`, weights read straight from the flash
  slot — no RAM), decodes the action along the slot's kind (attitude,
  body rates, or per-motor PWM) and publishes it into a sequence-locked
  buffer. The controller applies the newest complete action as a
  zero-order hold, synthesizing the `setpoint_t` the same command would
  have produced arriving by radio — INCLUDING cflib's pitch negation on
  the wire (the phase-1 decoder missed it; bench-check `nnpol.spPitch`
  against a `/attitude_cmd` pitch step before the first onboard flight).
  Inference never runs in the 1 kHz stabilizer context.
- The host keeps streaming its command topic as the commander-watchdog
  keepalive and shadow reference; clearing `nnpol.enable` mid-flight is an
  instantaneous, format-compatible takeover by that stream.
- Observations: `traj_lissajous` (the fixed eight, constants in the slot
  header) and `traj_easy` (the run's curve pushed at engage as `nnpol.c*`),
  each with `body` or `body_motorobs` — the latter appends the four
  rotor speeds the app reads from its own bidirectional-DSHOT telemetry
  (`motorsGetRPM`), held through dropouts from a hover start and
  normalized by 15896.3 exactly as the host's `set_motor_rpm` does.
- If the app task stops delivering actions, `nnpol.stale` counts and after
  5 policy periods the controller falls back to the radio setpoint. A slot
  whose task has no observation builder onboard (`taskId` 0xFFFF, the
  world-frame `default` observation) never produces an action:
  `nnpol.obsErr` counts and the fallback holds.
- The rate becomes a stabilizer tick divider (`1000 / ctrlHz`, integer) at
  the `enable` rising edge, so only divisors of 1000 (100, 125, 200, 250,
  500 ...) run exactly; read-only `nnpol.runHz` reports what engaged.

## Files

| file | role |
|---|---|
| `src/nnpol_controller.c` | firmware glue: OOT controller, app task, slot select/erase housekeeping, params, logs |
| `src/nnpol_slot_bank.c` | the flash slot bank: MEM_TYPE_APP upload handler, sector erase (watchdog widened), validity scan |
| `src/nnpol_slot.h` / `nnpol_slot_format.c` | the blob format and its validation (pure C, host-testable) |
| `src/nnpol_policy.c` | the MLP interpreter, blocked output-major kernel + fast tanh (pure C) |
| `src/nnpol_obs_lissajous.c` | the Lissajous-family body observations: the fixed eight or a pushed per-run curve (traj_easy), with the yaw anchor and the optional rotor-speed tail (pure C) |
| `src/nnpol_action.c` | action decode for all three kinds from the header's constants (pure C) |
| `test/` | host parity harness (plain gcc) + python runner, driven by a run directory's `policy_slot.bin` |

## Slot protocol (what the ROS driver does)

1. `nnpol.slotErase = i` — the app task erases sector 7+i on the ground,
   disengaged and disarmed (the erase stalls the whole MCU for 1-2 s; the
   independent watchdog is widened around it). `nnpol.slotState` goes
   1 (erasing) → 2 (erased) or 3 (failed).
2. Stream `policy_slot.bin` with cflib's memory write (`MEM_TYPE_APP`,
   offset `i * 131072`): sequential acknowledged 24-byte chunks, each
   whole words, programmed as they arrive. A word is programmed only if
   it reads erased or already holds the same data (retransmits), so a
   stale slot can never be half-overwritten. `nnpol.writeSlot` /
   `writtenBytes` show progress.
3. `nnpol.slot = i` — the app task validates (CRC over the whole blob) and
   publishes the identity; `slotSel` confirms, `slotStatus` says why not.
   `nnpol.slotValid` is the bitmask of slots holding a valid blob.

## Building and flashing (once)

```
cd examples/app_nnpol
pixi run --manifest-path ../../pixi.toml make cf21bl_defconfig   # once
pixi run --manifest-path ../../pixi.toml make -j                 # build
pixi run --manifest-path ../../pixi.toml make cload              # flash
```

The image must end below `0x08080000` (sector 7): ~308 KB today against a
496 KB ceiling. Check the map before growing the firmware.

## Host parity

```
python3 test/run_host_parity.py --run-dir <run dir exported with export_policy_c.py>
```

Compiles the flight observation/action/interpreter C with host gcc,
loads the run's `policy_slot.bin` exactly as a flash slot, validates it,
and checks the network against the trainer's `policy_reference.npz` and
the observation/decode against numpy transcriptions of the deployed host
path. `--bag-npz` adds mocap-derived snapshots from a flight bag.

## Param / log reference

Params (`nnpol.`): `enable` (u8, the engage switch), `startPhase` (s),
`offX/offY/offZ` (m, the host's curve pin), `yaw0` (rad, the run's yaw
anchor), `cAmp0-2/cOmg0-2/cPh0-2/cCen0-2/cYaw` (a traj_easy run's curve),
`vnom` (V, thrust-map battery voltage), `rpmScale` (MOTOR kind's
rpm multiplier), `ctrlHz` (policy rate, default = the slot's `ctrl_freq`)
— all latched at the `enable` rising edge; `slot` (select), `slotErase`;
read-only: `slotSel`, `slotState`, `slotStatus`, `slotValid`, `nSlots`,
`writeSlot`, `writtenBytes`, `runHz`, and the selected slot's identity
`hash0-3`, `obsDim`, `actDim`, `kind`, `taskId`.

Logs (`nnpol.`): `t`, `act0-3` (clipped network output), `spRoll/Pitch/
Yaw/Thrust` (decoded setpoint in the host's wire units: deg or deg/s, +
legacy thrust), `pwm0-3` (MOTOR kind's per-motor PWM), `vbx/y/z`,
`wbx/y/z`, `e0x/y/z` (observation excerpts), `us` (inference time),
`stale`, `obsErr`, `seq`.

## Engage protocol (what the ROS side does)

1. find the run directory's hash in a slot or upload `policy_slot.bin`
   into a free one; `nnpol.slot = i`; verify `nnpol.hash0-3` == the run
   dir's `firmware_export.json`, and that the configured inference rate
   divides 1000;
2. take off, hover at the curve's engage point, keep streaming the
   checkpoint's command topic (shadow);
3. push `startPhase`, `offX/Y/Z`, `yaw0` (from the host's pin and
   anchor), the curve (`c*`, traj_easy only), `vnom`, `rpmScale`, `ctrlHz`;
4. set `enable = 1`, confirm by readback → onboard flight;
5. abort/land: clear `enable` first — the host stream takes over on the
   next controller tick.
