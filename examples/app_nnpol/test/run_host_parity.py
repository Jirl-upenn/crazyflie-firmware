#!/usr/bin/env python3
"""Host parity check for app_nnpol — WP2.6 of the onboard inference plan:

    python3 examples/app_nnpol/test/run_host_parity.py \
        [--run-dir .../runs/traj_lissajous/ppo_body_yawpen] [--snapshots N]

Builds test_policy_host.c with plain gcc against the SAME generated
policy.c and pure-C observation/action modules the firmware compiles, then
checks, in order of what would hurt most in the air:

  1. network parity — the 1000 reference observations from the trainer's
     policy_reference.npz through the C policyForward, asserting
     max |act_C - act_numpy| < 1e-5 against the recorded numpy actions.
  2. observation parity — random plausible EKF snapshots through
     nnpolBuildObs vs a numpy transcription of the trainer's body
     observation (quaternion sign, R^T rotations, lookahead window).
  3. action-decode parity — random clipped actions through
     nnpolDecodeAction vs a numpy transcription of the deployed host path
     for the build's NNPOL_ACTION_KIND: box decode, degrees (or deg/s),
     rpm2thrust inverse, vmotor2rpm, np.rint, the legacy radio clamp for
     the setpoint kinds; the checkpoint's mixer, rpm_scale and rpm_to_pwm
     for the MOTOR kind.
  4. end to end — snapshot -> obs -> network -> clip -> decode in one C
     pass vs the same chain in numpy.

Optionally --bag-npz FILE adds a fifth pass: the shadow policy's own
observations from a real flight bag (keys: obs (N, OBS_DIM) float32,
expected_act (N, 4) clipped) through the C network — produced by
crazyflie-ros/bin/compare_shadow.py --export-snapshots once a baseline bag
exists. This closes the loop with observations a real flight produced.

Exit code 0 = all parity bounds met.
"""
import argparse
import re
import shutil
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "src"
DEFAULT_RUN = (HERE.parent.parent.parent.parent
               / "mjc_dronetests/runs/traj_lissajous/ppo_body_yawpen")

SNAP_FLOATS = 17
CMD_FLOATS = 14   # nnpolCmd_t flattened, see test_policy_host.c
VNOM = 4.0
RPM_SCALE = 1.0
KIND_MELLINGER, KIND_CTBR, KIND_MOTOR = 0, 1, 2


def parse_defines(path):
    """The NNPOL_* numeric defines out of firmware_export.h, so the numpy
    mirrors below use exactly what the firmware compiles."""
    d = {}
    src = path.read_text()
    for m in re.finditer(r"#define (NNPOL_\w+) ([-\d.eE+f]+)\n", src):
        d[m.group(1)] = float(m.group(2).rstrip("f"))
    for m in re.finditer(r"#define (NNPOL_ACTION_(?:LOW|HIGH|MID|HALF))\s+\{ ([^}]+) \}", src):
        d[m.group(1)] = np.array([float(v.strip().rstrip("f"))
                                  for v in m.group(2).split(",")])
    return d


def build(tmp):
    cc = shutil.which("gcc") or shutil.which("cc")
    if cc is None:
        sys.exit("no host C compiler")
    exe = tmp / "nnpol_host"
    subprocess.run(
        [cc, "-O2", "-Wall", "-Wextra", "-Werror", "-o", str(exe),
         str(HERE / "test_policy_host.c"),
         str(SRC / "policy.c"), str(SRC / "nnpol_obs_lissajous.c"),
         str(SRC / "nnpol_action.c"),
         "-I", str(SRC), "-lm"], check=True)
    return exe


def run(exe, mode, data, out_floats):
    r = subprocess.run([str(exe), mode], input=data.astype(np.float32).tobytes(),
                       capture_output=True, check=True)
    return np.frombuffer(r.stdout, dtype=np.float32).reshape(-1, out_floats)


# -- numpy mirrors ----------------------------------------------------------

def quat_to_R(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def np_ref(t, C):
    w = 2.0 * np.pi / C["NNPOL_T_PERIOD_S"]
    t = np.asarray(t)
    return np.stack([C["NNPOL_AMPLITUDE_M"] * np.sin(w * t),
                     np.zeros_like(t),
                     0.5 * C["NNPOL_AMPLITUDE_M"] * np.sin(2 * w * t)
                     + C["NNPOL_CENTER_Z_M"]], axis=-1)


def np_obs(snap, C):
    t, off, pos = snap[0], snap[1:4], snap[4:7]
    q, v_w, gyro = snap[7:11].copy(), snap[11:14], snap[14:17]
    if q[0] < 0:
        q = -q
    R = quat_to_R(q)
    n = int(C["NNPOL_N_SAMPLES"])
    ts = t + np.arange(n) * C["NNPOL_SAMPLES_DT_S"]
    e_w = np_ref(ts, C) + off - pos
    return np.concatenate([q, R.T @ v_w, np.radians(gyro), (e_w @ R).reshape(-1)])


def np_rpm_to_pwm(rpm, vnom, C):
    v = (np.asarray(rpm, dtype=np.float64) - C["NNPOL_VMOTOR2RPM_K0"]) / C["NNPOL_VMOTOR2RPM_K1"]
    return np.clip(np.rint(65535.0 * v / vnom), 0, 65535)


def np_mix(a, C):
    """crazyflow_interface/_mixer.py, by the build's NNPOL_MIXER."""
    hover, max_rpm = C["NNPOL_HOVER_RPM"], C["NNPOL_MAX_RPM"]

    def mix_rpm_action(x):
        return np.where(x <= 0, (x + 1.0) * hover, hover + (max_rpm - hover) * x)

    mixer = int(C["NNPOL_MIXER"])
    if mixer == 0:   # attitude
        collective = mix_rpm_action(a[0])
        s = C["NNPOL_DIFFERENTIAL_FRAC"] * hover
        roll, pitch, yaw = a[1], a[2], a[3]
        rpm = np.array([collective + roll * s - pitch * s - yaw * s,
                        collective - roll * s - pitch * s + yaw * s,
                        collective - roll * s + pitch * s - yaw * s,
                        collective + roll * s + pitch * s + yaw * s])
        return np.clip(rpm, 0.0, max_rpm)
    if mixer == 2:   # rpm_direct
        return max_rpm * (a + 1.0) * 0.5
    return mix_rpm_action(a)


def np_decode(action, vnom, rpm_scale, C):
    """nnpolCmd_t as numpy, for the build's NNPOL_ACTION_KIND: the host
    decoders transcribed (MellingerAttitudePolicy / BodyRatePolicy /
    MotorPwmRacingPolicy.decode + the wire and radio-clamp steps)."""
    a = np.clip(action, -1, 1)
    out = np.zeros(CMD_FLOATS)
    if int(C["NNPOL_ACTION_KIND"]) == KIND_MOTOR:
        rpm = np_mix(a, C) * rpm_scale
        out[6:10] = rpm
        out[10:14] = np_rpm_to_pwm(rpm, vnom, C)
        return out
    sp = C["NNPOL_ACTION_MID"] + a * C["NNPOL_ACTION_HALF"]
    a0, a1, a2 = (C["NNPOL_RPM2THRUST_A0"], C["NNPOL_RPM2THRUST_A1"],
                  C["NNPOL_RPM2THRUST_A2"])
    f = max(sp[3], 0.0) / 4.0
    disc = a1 * a1 - 4 * a2 * (a0 - f)
    rpm = 0.0 if disc < 0 else (-a1 + np.sqrt(disc)) / (2 * a2)
    pwm = float(np_rpm_to_pwm(rpm, vnom, C))
    applied = 0.0 if pwm < 1000.0 else min(pwm, 60000.0)
    out[:6] = [np.degrees(sp[0]), np.degrees(sp[1]), np.degrees(sp[2]),
               sp[3], pwm, applied]
    return out


def np_forward(layers, obs):
    x = obs.astype(np.float32)
    for i, (W, b) in enumerate(layers):
        x = x @ W + b
        if i < len(layers) - 1:
            x = np.tanh(x)
    return x


def random_snapshots(rng, n, C):
    """Plausible engaged-flight snapshots: near-curve positions, moderate
    attitude/velocity, the pushed offset spanning a few metres."""
    snaps = np.zeros((n, SNAP_FLOATS), dtype=np.float32)
    snaps[:, 0] = rng.uniform(0, 3 * C["NNPOL_T_PERIOD_S"], n)          # t
    snaps[:, 1:4] = rng.uniform(-3, 3, (n, 3))                          # off
    ref0 = np_ref(snaps[:, 0], C) + snaps[:, 1:4]
    snaps[:, 4:7] = ref0 + rng.uniform(-0.5, 0.5, (n, 3))               # pos
    q = rng.normal(size=(n, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    # half with w<0 on purpose: the C side must sign-normalize
    snaps[:, 7:11] = q
    snaps[:, 11:14] = rng.uniform(-2.5, 2.5, (n, 3))                    # vel_w
    snaps[:, 14:17] = rng.uniform(-250, 250, (n, 3))                    # gyro deg/s
    return snaps


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    p.add_argument("--snapshots", type=int, default=500)
    p.add_argument("--bag-npz", type=Path, default=None)
    args = p.parse_args()

    import tempfile
    tmp = Path(tempfile.mkdtemp(prefix="nnpol_parity_"))
    exe = build(tmp)
    C = parse_defines(SRC / "firmware_export.h")
    kind = int(subprocess.run([str(exe), "kind"], capture_output=True, check=True).stdout)
    assert kind == int(C["NNPOL_ACTION_KIND"]), (kind, C["NNPOL_ACTION_KIND"])
    print(f"[info] build decodes as kind {kind} "
          f"({['MELLINGER', 'CTBR', 'MOTOR'][kind]}, action_type in policy.h)")
    rng = np.random.default_rng(0)
    failures = []
    # The columns each kind fills, and their tolerances: angles/rates and
    # newtons are float math; PWM columns are integers where rintf and
    # np.rint can differ by one count at an exact .5 boundary.
    if kind == KIND_MOTOR:
        # rpm columns are O(1e4): float32 carries ~1e-3 rpm of rounding
        col_f, col_i, tol_f, tol_e2e = slice(6, 10), slice(10, 14), 0.05, 0.5
    else:
        col_f, col_i, tol_f, tol_e2e = slice(0, 4), slice(4, 6), 1e-3, 5e-3

    def check(name, err, bound):
        ok = err < bound
        print(f"[{'ok' if ok else 'FAIL'}] {name}: max err {err:.3g} (bound {bound:g})")
        if not ok:
            failures.append(name)

    # 1. network parity against the trainer's own reference
    ref = np.load(args.run_dir / "policy_reference.npz")
    act = run(exe, "policy", ref["obs"], 4)
    check("network vs policy_reference.npz", np.abs(act - ref["act_raw"]).max(), 1e-5)

    # 2. observation parity
    snaps = random_snapshots(rng, args.snapshots, C)
    obs_c = run(exe, "obs", snaps, int(C["NNPOL_OBS_DIM"]))
    obs_np = np.stack([np_obs(s.astype(np.float64), C) for s in snaps])
    check("observation vs numpy transcription", np.abs(obs_c - obs_np).max(), 2e-4)

    # 3. action decode parity
    acts = rng.uniform(-1.2, 1.2, (args.snapshots, 4))  # beyond the box: clip must match
    tail = np.tile([VNOM, RPM_SCALE], (len(acts), 1))
    cmd_c = run(exe, "act", np.concatenate([acts, tail], axis=1), CMD_FLOATS)
    cmd_np = np.stack([np_decode(a, VNOM, RPM_SCALE, C) for a in acts])
    check("decode float channels", np.abs(cmd_c[:, col_f] - cmd_np[:, col_f]).max(), tol_f)
    check("decode pwm", np.abs(cmd_c[:, col_i] - cmd_np[:, col_i]).max(), 1.001)
    other = [i for i in range(CMD_FLOATS) if i not in range(*col_f.indices(CMD_FLOATS))
             and i not in range(*col_i.indices(CMD_FLOATS))]
    check("decode leaves the other kind's columns zero", np.abs(cmd_c[:, other]).max(), 0.5)

    # 4. end to end
    import json
    with open(args.run_dir / "config.json") as f:
        cfg = json.load(f)
    layers = load_layers(args.run_dir, cfg)
    tail = np.tile([VNOM, RPM_SCALE], (len(snaps), 1))
    full_c = run(exe, "full", np.concatenate([snaps, tail], axis=1), CMD_FLOATS)
    full_np = np.stack([
        np_decode(np.clip(np_forward(layers, obs_np[i].astype(np.float32)), -1, 1),
                  VNOM, RPM_SCALE, C)
        for i in range(len(snaps))])
    check("end-to-end float channels", np.abs(full_c[:, col_f] - full_np[:, col_f]).max(), tol_e2e)
    check("end-to-end pwm", np.abs(full_c[:, col_i] - full_np[:, col_i]).max(), 2.001)

    # 5. optional: real-flight observations from the shadow's bag export
    if args.bag_npz is not None:
        bag = np.load(args.bag_npz)
        act_bag = run(exe, "policy", bag["obs"], 4)
        check("bag observations vs shadow actions",
              np.abs(np.clip(act_bag, -1, 1) - bag["expected_act"]).max(), 1e-5)

    sys.exit(1 if failures else 0)


def load_layers(run_dir, cfg):
    import pickle
    with open(run_dir / "params.pkl", "rb") as f:
        tree = pickle.load(f)
    root = tree["actor"] if "actor" in tree else tree
    params = root["params"] if "params" in root else root
    return [(np.asarray(params[f"Dense_{i}"]["kernel"], dtype=np.float32),
             np.asarray(params[f"Dense_{i}"]["bias"], dtype=np.float32))
            for i in range(len(cfg["hidden"]) + 1)]


if __name__ == "__main__":
    main()
