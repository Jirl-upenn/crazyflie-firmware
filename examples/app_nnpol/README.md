# app_nnpol — onboard racing policy (out-of-tree controller)

Branch `onboard-rpm`. Runs a trained actor network on the Crazyflie's
STM32F405 at the policy's control rate, from a policy the ground station
uploaded over the radio into one of five flash slots. The vehicle builds
the observation from its own EKF and gyro, runs the network and drives the
motors; the ground station (`ros_quad_sim2real`, branch `onboard-rpm`)
forwards mocap, manages policies and supervises.

Scope: the `race` task with the `v3` / `v3_motorobs` observation, and the
rpm action family (`rpm_hover`, `rpm_direct`). Trajectory tracking, the
setpoint action types and offboard policy execution are not on this branch
(see `nnpol` and `onboard_only`).

## Test ladder

Every row is ONE commit on this branch, tagged. To run a phase, check out
its tag, build, flash, and do what the row says; do not climb past a row
that fails. Host tests apply at every commit. Rows marked "ROS" need the
matching tag on `ros_quad_sim2real`.

| tag | phase | what to do | go |
|---|---|---|---|
| `orpm-p0-baseline` | host tests, and nothing changed in the air | `python3 test/run_host_parity.py`; build; flash; take off, hover and land on the SE3 cascade | parity passes; image ends below sector 7; hover as before |
| `orpm-p1-mocap-bench` | mocap bench, props off (ROS `orpm-p1-mocap-bench`) | `python -m pytest test_python/test_pose_delay.py`; flash; stream mocap; log `kalman.poseAgeMs`, `poseHostMs`, `poseDrop` | applied age appears and tracks the driver's value; no drops at rest; no "Kalman prediction rate off" on the console |
| (same commit) | hover, compensation off then on | hover on the SE3 cascade with `kalman.poseComp` 0, then 1 (driver config `pose_delay_comp`) | no change in hover position or heading error between the two; no rate warnings |
| (same commit) | latency flight (ROS `orpm-p3-latency-flight`) | straight passes at stepped speeds, compensation off then on; run the ROS calibration script on both bags | the off bag gives a stable `d0`; with `kalman.poseD0Ms` set, the on bag's slope is within 2 ms of zero (target) |
| `orpm-p4-thrust-bench` | thrust bench, as flown so far (ROS `orpm-p4-thrust-bench`). PROPS ON, VEHICLE RESTRAINED | flash; run the ROS repo's `bin/bench/thrust_sweep.py --convention voltage` | the gain and offset of delivered against commanded rotor speed are on record for the per-motor path |
| `orpm-p5-thrust-force` | thrust bench, force convention (ROS `orpm-p5-thrust-force`). PROPS ON, VEHICLE RESTRAINED | `python3 test/run_host_parity.py`; flash; `bin/bench/thrust_sweep.py --convention force` | delivered thrust tracks commanded within 5 % across the range (target), at a full and at a low battery |

## Mocap latency compensation

The stock pose update compares a mocap pose with the estimator's CURRENT
state, so a moving vehicle's estimate trails by speed × latency. On this
branch:

- The ground station sends `EXT_POSE_AGED` (localization port, generic
  channel, type 32): `<fffIH` = x, y, z in metres, the compressed quaternion
  of the packed pose, and how long it held the sample before the radio, in
  units of 0.1 ms. The stock pose packets still work and behave as before.
- The estimator keeps 64 ms of position and attitude history
  (`kalman_core/pose_history.c`) and, with `kalman.poseComp = 1`, forms the
  residual against the state it held when the pose was captured, applies it
  to the current state, and carries the correction into the history
  (`kalmanCoreUpdateWithPoseDelayed`).
- The age it uses is: the packet's age + the time the packet waited on the
  vehicle (measured) + `kalman.poseD0Ms`, the fixed part nobody can time
  (mocap capture latency, radio hop), calibrated once from a flight.
- A pose older than the history is dropped and counted (`kalman.poseDrop`).
  An age of zero is the stock update bit for bit.
- Built with `CONFIG_ESTIMATOR_KALMAN_POSE_DELAY_COMP` (on in `app-config`);
  `kalman.poseComp` defaults to 0, so a freshly flashed vehicle behaves as
  stock until the ground station turns it on.

Logs (`kalman.`): `poseAgeMs` (total age applied), `poseHostMs` (the
packet's part), `poseDrop`, `poseResX/Y/Z` (position residual as applied).

## Motor values are forces

The firmware passes every 16-bit motor value, on every control path, through
`motorsCompensateBatteryVoltage`: the value is read as a force, value / 65535
× `THRUST_MAX` (0.2 N per motor on the brushless platform), and turned into a
motor voltage at the measured battery voltage. From `orpm-p5-thrust-force`
on, the app sends exactly that: rotor speed → force by the drone's thrust
curve (the slot header's `rpm2thrust`) → fraction of `THRUST_MAX`. Before it,
the app and the host sent a voltage fraction, which the firmware then
inverted a second time (about 0.53 × commanded + 0.17 N delivered on the
October CTBR flights). The build refuses to compile without
`CONFIG_ENABLE_THRUST_BAT_COMPENSATED`, and reports `nnpol.batComp` /
`nnpol.thrustMax` so the ground station can check what it is talking to.

Two limits follow: a commanded force above `THRUST_MAX` saturates, and below
`THRUST_MIN` (0.021 N) the firmware commands idle.

## What it does

- `CONFIG_CONTROLLER_OOT` registers this app as `stabilizer.controller = 6`.
  With `nnpol.enable = 0` (default) the radio setpoint passes straight
  through to the stock PID, so behaviour is identical to
  `stabilizer.controller = 1`. The only thing the ground station streams is
  the SE3 cascade's rate setpoint (takeoff, hover, landing, and the
  hold-here keepalive during a race).
- **The app is checkpoint-agnostic: build and flash it once.** A policy is
  a slot blob (a 256-byte header — identity, dims, activation, action map
  anchors, motor identification, task constants — plus the fp32 weights;
  `test/nnpol_blob.py` is the reference writer). Blobs live in internal-flash
  sectors 7-11, one 128 KB slot each, uploaded through the `MEM_TYPE_APP`
  memory handler and selected with `nnpol.slot`. The app task validates the
  selected slot (CRC) and publishes its identity in the read-only
  `nnpol.hash0-3` / `obsDim` / `actDim` / `kind` / `taskId` / `ctrlHz`
  params, which the ground station checks before engaging.
- On every policy tick (`nnpol.ctrlHz`, a writable param defaulting to the
  slot's trained rate; it must divide 1000 to run exactly, `nnpol.runHz`
  reports what engaged) the stabilizer snapshots EKF state, gyro and rotor
  speeds on one tick, advances the gate state, and wakes the app task, which
  builds the observation, runs the network and decodes the action. The
  stabilizer applies the latest complete action as a zero-order hold
  through the PID's per-motor bypass: no onboard controller runs while
  engaged.
- If the app task stops delivering actions (`nnpol.stale` counts), or the
  ground station clears `nnpol.enable`, the vehicle flies the radio
  setpoint again on the next controller tick.
- Interpreter limits, enforced by slot validation: observation up to 64
  wide, up to three hidden layers of up to 128 units, tanh or relu, clip or
  tanh output map, and the blob must fit one 128 KB slot.

## Files

| file | role |
|---|---|
| `src/nnpol_controller.c` | firmware glue: OOT controller, app task, slot select/erase housekeeping, params, logs |
| `src/nnpol_slot_bank.c` | the flash slot bank: MEM_TYPE_APP upload handler, sector erase (watchdog widened), validity scan |
| `src/nnpol_slot.h` / `nnpol_slot_format.c` | the blob format and its validation (pure C, host-testable) |
| `src/nnpol_policy.c` | the MLP interpreter, blocked output-major kernel + fast tanh, output map from the header (pure C) |
| `src/nnpol_obs_race.c` | the race v3 observation and the gate-crossing test (pure C) |
| `src/nnpol_action.c` | the rpm family's decode: action map, rpm_scale, rotor speed to motor value (pure C) |
| `test/nnpol_blob.py` | the blob format in Python: reference writer and parser |
| `test/` | host parity harness (plain gcc) + python runner |

## Slot protocol (what the ground station does)

1. `nnpol.slotErase = i` — the app task erases sector 7+i on the ground,
   disengaged and disarmed (the erase stalls the whole MCU for 1-2 s; the
   independent watchdog is widened around it). `nnpol.slotState` goes
   1 (erasing) → 2 (erased) or 3 (failed).
2. Stream the blob with cflib's memory write (`MEM_TYPE_APP`, offset
   `i * 131072`): sequential acknowledged 24-byte chunks, each whole words,
   programmed as they arrive. A word is programmed only if it reads erased
   or already holds the same data (retransmits), so a stale slot can never
   be half-overwritten. `nnpol.writeSlot` / `writtenBytes` show progress.
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

The image must end below `0x08060000` (sector 7): ~309 KB today against a
368 KB ceiling (the bootloader takes the first 16 KB). `nnpolSlotBankInit`
checks this at boot and disables the slot bank if the image overlaps it. Check the size line of the build before growing the firmware.

## Host parity

```
python3 test/run_host_parity.py            # synthetic blobs, no run directory needed
python3 test/run_host_parity.py --blob policy_slot.bin
```

Compiles the flight observation / action / interpreter C with host gcc,
loads a blob exactly as a flash slot, validates it, and checks the network,
the observation, the gate test and the decode against numpy transcriptions
of the host path. Needs numpy.

## Param / log reference

Params (`nnpol.`): `enable` (u8, the engage switch); `nGates`, `gate0`,
`g0x..g7nz` (the track), `rpmScale`, `ctrlHz` (policy rate,
default = the slot's trained rate) — all latched at the `enable` rising
edge; `slot` (select), `slotErase`; read-only: `slotSel`, `slotState`,
`slotStatus`, `slotValid`, `nSlots`, `writeSlot`, `writtenBytes`, `runHz`,
the selected slot's identity `hash0-3`, `obsDim`, `actDim`, `kind`,
`taskId`, and the build's thrust convention `batComp` (1 = motor values are
forces, battery compensated) and `thrustMax` (N per motor at 65535).
`benchPwm` (u16): the bench sweep's motor value, applied to all four motors
while armed and not engaged, for 300 ms after each write.

Logs (`nnpol.`): `t` (seconds since engage), `act0-3` (clipped network
output), `pwm0-3` (the decoded per-motor value), `vbx/y/z`, `wbx/y/z`,
`e0x/y/z` (observation excerpts: body velocity, body rate, target gate
offset), `us` (inference time), `stale`, `obsErr`, `gateIdx`, `seq`, `bench` (the bench value being applied).
