#!/usr/bin/env python3
"""Host parity check for app_nnpol — WP2.6 of the onboard inference plan:

    python3 examples/app_nnpol/test/run_host_parity.py \
        --run-dir <run dir exported with export_policy_c.py> [--snapshots N]

Builds test_policy_host.c with plain gcc against the SAME pure-C modules
the firmware compiles (slot format, interpreter, observation, decode) and
loads the run directory's policy_slot.bin exactly as the firmware reads a
flash slot, then checks, in order of what would hurt most in the air:

  1. slot validation — magic, sizes, dims, CRC, as the firmware's select
     does; and the header's identity words equal firmware_export.json's.
  2. network parity — the reference observations from the trainer's
     policy_reference.npz through the C interpreter, asserting
     max |act_C - act_numpy| < 1e-5 against the recorded numpy actions.
  3. observation parity — random plausible EKF snapshots (including a yaw
     anchor) through nnpolBuildObs vs a numpy transcription of the host's
     TrajLissajousBody.observe (anchored quaternion, R^T rotations, the
     yawed lookahead window).
  4. action-decode parity — random clipped actions through
     nnpolDecodeAction vs a numpy transcription of the deployed host path
     for the slot's action kind: box decode, the Mellinger yaw anchor,
     degrees (or deg/s), rpm2thrust inverse, vmotor2rpm, np.rint, the
     legacy radio clamp for the setpoint kinds; the checkpoint's mixer,
     rpm_scale and rpm_to_pwm for the MOTOR kind.
  5. end to end — snapshot -> obs -> network -> clip -> decode in one C
     pass vs the same chain in numpy.

Optionally --bag-npz FILE adds a sixth pass: the shadow policy's own
observations from a real flight bag (keys: obs (N, OBS_DIM) float32,
expected_act (N, 4) clipped) through the C network — produced by
crazyflie-ros/bin/compare_shadow.py --export-snapshots once a baseline bag
exists. This closes the loop with observations a real flight produced.

Exit code 0 = all parity bounds met.
"""
import argparse
import json
import shutil
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "src"

SNAP_FLOATS = 85   # + curve(13) + rpm(4) + race(2 + 8*3 + 8*3), see test_policy_host.c
MAX_GATES = 8
NOMINAL_HOVER_RPM = 15896.296489245326
CMD_FLOATS = 14   # nnpolCmd_t flattened, see test_policy_host.c
VNOM = 4.0
RPM_SCALE = 1.0
KIND_MELLINGER, KIND_CTBR, KIND_MOTOR = 0, 1, 2
TASK_LISSAJOUS_BODY = 0
TASK_EASY_BODY = 1
TASK_RACE_V3 = 2

# nnpol_slot.h, mirrored (export_policy_c._SLOT_HEADER_FORMAT).
HEADER_FORMAT = "<4I4I4H5HH4H4f4f3f3f2f8f64sI32s"
HEADER_BYTES = 256
assert struct.calcsize(HEADER_FORMAT) == HEADER_BYTES


def parse_header(blob):
    """The slot header as the dict of NNPOL_* names the numpy mirrors use."""
    v = struct.unpack(HEADER_FORMAT, blob[:HEADER_BYTES])
    i = 0
    magic, header_bytes, total_bytes, crc = v[i:i + 4]; i += 4
    hash_words = list(v[i:i + 4]); i += 4
    obs_dim, action_dim, num_layers, activation = v[i:i + 4]; i += 4
    layer_dims = list(v[i:i + 5]); i += 5
    i += 1  # pad
    kind, mixer, ctrl_hz, task_id = v[i:i + 4]; i += 4
    mid = np.array(v[i:i + 4]); i += 4
    half = np.array(v[i:i + 4]); i += 4
    hover, max_rpm, diff = v[i:i + 3]; i += 3
    rpm2thrust = v[i:i + 3]; i += 3
    vmotor2rpm = v[i:i + 2]; i += 2
    task = v[i:i + 8]; i += 8
    name = v[i].split(b"\0", 1)[0].decode(); i += 1
    payload_floats = v[i]; i += 1
    return {
        "magic": magic, "header_bytes": header_bytes, "total_bytes": total_bytes, "crc": crc,
        "hash_words": hash_words, "name": name, "payload_floats": payload_floats,
        "NNPOL_OBS_DIM": obs_dim, "NNPOL_ACTION_DIM": action_dim,
        "num_layers": num_layers, "activation": activation,
        "layer_dims": layer_dims[:num_layers + 1],
        "NNPOL_ACTION_KIND": kind, "NNPOL_MIXER": mixer,
        "NNPOL_CTRL_FREQ_HZ": ctrl_hz, "task_id": task_id,
        "NNPOL_ACTION_MID": mid, "NNPOL_ACTION_HALF": half,
        "NNPOL_HOVER_RPM": hover, "NNPOL_MAX_RPM": max_rpm, "NNPOL_DIFFERENTIAL_FRAC": diff,
        "NNPOL_RPM2THRUST_A0": rpm2thrust[0], "NNPOL_RPM2THRUST_A1": rpm2thrust[1],
        "NNPOL_RPM2THRUST_A2": rpm2thrust[2],
        "NNPOL_VMOTOR2RPM_K0": vmotor2rpm[0], "NNPOL_VMOTOR2RPM_K1": vmotor2rpm[1],
        "NNPOL_T_PERIOD_S": task[0], "NNPOL_AMPLITUDE_M": task[1], "NNPOL_CENTER_Z_M": task[2],
        "NNPOL_N_SAMPLES": task[3], "NNPOL_SAMPLES_DT_S": task[4], "motorobs": task[6] != 0.0,
        "gate_side": task[7],
    }


def build(tmp):
    cc = shutil.which("gcc") or shutil.which("cc")
    if cc is None:
        sys.exit("no host C compiler")
    exe = tmp / "nnpol_host"
    subprocess.run(
        [cc, "-O2", "-Wall", "-Wextra", "-Werror", "-o", str(exe),
         str(HERE / "test_policy_host.c"),
         str(SRC / "nnpol_slot_format.c"), str(SRC / "nnpol_policy.c"),
         str(SRC / "nnpol_obs_lissajous.c"), str(SRC / "nnpol_obs_race.c"),
         str(SRC / "nnpol_action.c"),
         "-I", str(SRC), "-lm"], check=True)
    return exe


def run(exe, blob, mode, data, out_floats):
    r = subprocess.run([str(exe), mode, str(blob)], input=data.astype(np.float32).tobytes(),
                       capture_output=True, check=True)
    return np.frombuffer(r.stdout, dtype=np.float32).reshape(-1, out_floats)


# -- numpy mirrors ----------------------------------------------------------

def quat_to_R(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def R_to_quat_wxyz(R):
    """crazyflie-ros controller_tasks._quat_wxyz: scalar-first, w >= 0."""
    from scipy.spatial.transform import Rotation
    x, y, z, w = Rotation.from_matrix(R).as_quat()
    q = np.array([w, x, y, z])
    return -q if q[0] < 0 else q


def rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def np_ref(t, C, curve=None):
    """controller_trajectory.ref for an EASY slot (the pushed curve), the
    fixed eight otherwise."""
    t = np.asarray(t)
    if C["task_id"] == TASK_EASY_BODY:
        amp, omega, phase, center, yaw = (curve[0:3], curve[3:6], curve[6:9], curve[9:12],
                                          float(curve[12]))
        local = amp * np.sin(omega * t[..., None] + phase)
        return local @ rot_z(yaw).T + center
    w = 2.0 * np.pi / C["NNPOL_T_PERIOD_S"]
    return np.stack([C["NNPOL_AMPLITUDE_M"] * np.sin(w * t),
                     np.zeros_like(t),
                     0.5 * C["NNPOL_AMPLITUDE_M"] * np.sin(2 * w * t)
                     + C["NNPOL_CENTER_Z_M"]], axis=-1)


def np_race_obs(snap, C):
    """RaceV3.observe: body velocity/rate, gravity in body, the current and
    next gate's offset and approach normal in body, [+ rotor speeds]."""
    pos, q, v_w, gyro, rpm = snap[4:7], snap[7:11], snap[11:14], snap[14:17], snap[31:35]
    n_gates, idx = int(snap[35]), int(snap[36])
    gates = snap[37:37 + 3 * MAX_GATES].reshape(MAX_GATES, 3)
    normals = snap[61:61 + 3 * MAX_GATES].reshape(MAX_GATES, 3)
    R = quat_to_R(q / np.linalg.norm(q))
    Rt = R.T
    nxt = (idx + 1) % n_gates
    parts = [Rt @ v_w, np.radians(gyro), R[2, :],
             Rt @ (gates[idx] - pos), Rt @ (gates[nxt] - pos),
             Rt @ normals[idx], Rt @ normals[nxt]]
    if C["motorobs"]:
        parts.append(rpm)
    return np.concatenate(parts)


def np_update_gate(track, gate_side, pos, gate_idx, prev_gate_x):
    """RaceV3._update_gate / RacePlugin.step's crossing test."""
    n_gates, gates, normals = track
    g, n = gates[gate_idx], normals[gate_idx]
    rel = pos - g
    x = float(np.dot(rel, n))
    y = rel[1] * n[0] - rel[0] * n[1]
    z = rel[2]
    half = gate_side * 0.5
    if x < 0.0 and prev_gate_x > 0.0 and abs(y) < half and abs(z) < half:
        return (gate_idx + 1) % n_gates, 1.0
    return gate_idx, x


def np_obs(snap, C):
    """TrajLissajousBody.observe with the anchor: quaternion of R_tw^T R_wb,
    body velocity and rate, lookahead of the curve yawed by R_tw."""
    if C["task_id"] == TASK_RACE_V3:
        return np_race_obs(snap, C)
    t, off, pos = snap[0], snap[1:4], snap[4:7]
    q, v_w, gyro, yaw0 = snap[7:11].copy(), snap[11:14], snap[14:17], snap[17]
    curve, rpm = snap[18:31], snap[31:35]
    R = quat_to_R(q / np.linalg.norm(q))
    R_tw = rot_z(yaw0)
    n = int(C["NNPOL_N_SAMPLES"])
    ts = t + np.arange(n) * C["NNPOL_SAMPLES_DT_S"]
    e_w = np_ref(ts, C, curve) @ R_tw.T + off - pos
    parts = [R_to_quat_wxyz(R_tw.T @ R), R.T @ v_w, np.radians(gyro), (e_w @ R).reshape(-1)]
    if C["motorobs"]:
        parts.append(rpm)   # TrajEasyBodyMotor.observe: the held, normalized speeds
    return np.concatenate(parts)


def np_rpm_to_pwm(rpm, vnom, C):
    v = (np.asarray(rpm, dtype=np.float64) - C["NNPOL_VMOTOR2RPM_K0"]) / C["NNPOL_VMOTOR2RPM_K1"]
    return np.clip(np.rint(65535.0 * v / vnom), 0, 65535)


def np_mix(a, C):
    """crazyflow_interface/_mixer.py, by the slot's mixer."""
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


def np_decode(action, vnom, rpm_scale, yaw0, C):
    """nnpolCmd_t as numpy, for the slot's action kind: the host decoders
    transcribed (MellingerAttitudePolicy / BodyRatePolicy /
    MotorPwmRacingPolicy.decode + the wire and radio-clamp steps)."""
    a = np.clip(action, -1, 1)
    out = np.zeros(CMD_FLOATS)
    kind = int(C["NNPOL_ACTION_KIND"])
    if kind == KIND_MOTOR:
        rpm = np_mix(a, C) * rpm_scale
        out[6:10] = rpm
        out[10:14] = np_rpm_to_pwm(rpm, vnom, C)
        return out
    sp = C["NNPOL_ACTION_MID"] + a * C["NNPOL_ACTION_HALF"]
    if kind == KIND_MELLINGER and yaw0:
        sp[2] = (sp[2] + yaw0 + np.pi) % (2.0 * np.pi) - np.pi
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


def np_forward(layers, activation, obs):
    act = {0: np.tanh, 1: lambda x: np.maximum(x, 0.0)}[activation]
    x = obs.astype(np.float32)
    for i, (W, b) in enumerate(layers):
        x = x @ W + b
        if i < len(layers) - 1:
            x = act(x)
    return x


def random_snapshots(rng, n, C):
    """Plausible engaged-flight snapshots: near-curve positions, moderate
    attitude/velocity, the pushed offset spanning a few metres, a yaw
    anchor anywhere on the circle."""
    snaps = np.zeros((n, SNAP_FLOATS), dtype=np.float32)
    snaps[:, 0] = rng.uniform(0, 3 * (C["NNPOL_T_PERIOD_S"] or 16.0), n) # t
    snaps[:, 1:4] = rng.uniform(-3, 3, (n, 3))                          # off
    snaps[:, 17] = rng.uniform(-np.pi, np.pi, n)                        # yaw0
    snaps[::4, 17] = 0.0                                                # and unanchored runs
    # a per-run curve like controller_trajectory.sample draws: amplitudes,
    # harmonics 1-2 of an 8-16 s period, phases, centre height, yaw
    T = rng.uniform(8.0, 16.0, n)
    harm = rng.integers(1, 3, (n, 3))
    snaps[:, 18:21] = np.stack([rng.uniform(0.2, 1.0, n), rng.uniform(0.2, 1.0, n),
                                rng.uniform(0.0, 0.5, n)], axis=1)    # amp
    snaps[:, 21:24] = harm * (2.0 * np.pi / T)[:, None]                 # omega
    snaps[:, 24:27] = rng.uniform(0.0, 2.0 * np.pi, (n, 3))            # phase
    snaps[:, 27:30] = np.stack([np.zeros(n), np.zeros(n), rng.uniform(0.8, 1.2, n)], axis=1)
    snaps[:, 30] = rng.uniform(-np.pi, np.pi, n)                        # curve yaw
    snaps[:, 31:35] = rng.uniform(0.6, 1.4, (n, 4))                     # rpm / nominal
    # a race track per snapshot: 2-8 gates on a loop, horizontal normals
    # against the travel direction (RacePlugin.from_config), the drone
    # somewhere near one of them targeting a random gate
    for i in range(n):
        n_gates = int(rng.integers(2, MAX_GATES + 1))
        thetas = np.sort(rng.uniform(-np.pi, np.pi, n_gates))
        radius = rng.uniform(1.0, 3.0)
        gates = np.stack([radius * np.cos(thetas), radius * np.sin(thetas),
                          rng.uniform(0.5, 1.5, n_gates)], axis=1)
        travel = thetas + np.pi / 2 + rng.uniform(-0.3, 0.3, n_gates)
        normals = -np.stack([np.cos(travel), np.sin(travel), np.zeros(n_gates)], axis=1)
        snaps[i, 35] = n_gates
        snaps[i, 36] = rng.integers(0, n_gates)
        snaps[i, 37:37 + 3 * n_gates] = gates.reshape(-1)
        snaps[i, 61:61 + 3 * n_gates] = normals.reshape(-1)
    if C["task_id"] == TASK_RACE_V3:
        for i in range(n):
            g = snaps[i, 37 + 3 * int(snaps[i, 36]):40 + 3 * int(snaps[i, 36])]
            snaps[i, 4:7] = g + rng.uniform(-2.0, 2.0, 3)               # pos near the target
    else:
        for i in range(n):
            ref0 = np_ref(snaps[i, 0], C, snaps[i, 18:31]) @ rot_z(snaps[i, 17]).T + snaps[i, 1:4]
            snaps[i, 4:7] = ref0 + rng.uniform(-0.5, 0.5, 3)            # pos
    q = rng.normal(size=(n, 4))
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    # half with w<0 on purpose: the C side must sign-normalize after anchoring
    snaps[:, 7:11] = q
    snaps[:, 11:14] = rng.uniform(-2.5, 2.5, (n, 3))                    # vel_w
    snaps[:, 14:17] = rng.uniform(-250, 250, (n, 3))                    # gyro deg/s
    return snaps


def load_layers(run_dir, cfg):
    import pickle
    with open(run_dir / "params.pkl", "rb") as f:
        tree = pickle.load(f)
    root = tree["actor"] if "actor" in tree else tree
    params = root["params"] if "params" in root else root
    layers = []
    for i in range(len(cfg["hidden"]) + 1):
        d = params[f"Dense_{i}"]
        layers.append((np.asarray(d["kernel"], dtype=np.float32),
                       np.asarray(d["bias"], dtype=np.float32)))
    return layers


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--blob", type=Path, default=None,
                   help="policy_slot.bin (default: <run-dir>/policy_slot.bin)")
    p.add_argument("--snapshots", type=int, default=500)
    p.add_argument("--bag-npz", type=Path, default=None)
    args = p.parse_args()
    blob_path = args.blob or (args.run_dir / "policy_slot.bin")

    import tempfile
    tmp = Path(tempfile.mkdtemp(prefix="nnpol_parity_"))
    exe = build(tmp)
    blob = blob_path.read_bytes()
    C = parse_header(blob)
    failures = []

    def check(name, err, bound):
        ok = err < bound
        print(f"[{'ok' if ok else 'FAIL'}] {name}: max err {err:.3g} (bound {bound:g})")
        if not ok:
            failures.append(name)

    # 1. the slot validates, and its identity is the record's
    info = subprocess.run([str(exe), "info", str(blob_path)], capture_output=True, text=True)
    status = int(info.stdout.split()[0])
    print(f"[{'ok' if status == 0 else 'FAIL'}] slot validation: status {status} "
          f"({info.stdout.strip()}) — {C['name']!r}, kind {C['NNPOL_ACTION_KIND']}, "
          f"task {C['task_id']}, {C['NNPOL_OBS_DIM']}-dim obs, layers {C['layer_dims']}")
    if status != 0:
        failures.append("slot validation")
    with open(args.run_dir / "firmware_export.json") as f:
        record = json.load(f)
    if list(record["hash_words"]) != C["hash_words"]:
        print("[FAIL] slot hash words differ from firmware_export.json")
        failures.append("slot identity")
    else:
        print("[ok] slot hash words match firmware_export.json")
    kind = int(C["NNPOL_ACTION_KIND"])
    # The columns each kind fills, and their tolerances: angles/rates and
    # newtons are float math; rpm columns are O(1e4) with ~1e-3 rpm of
    # float32 rounding; PWM columns are integers where rintf and np.rint
    # can differ by one count at an exact .5 boundary.
    if kind == KIND_MOTOR:
        col_f, col_i, tol_f, tol_e2e = slice(6, 10), slice(10, 14), 0.05, 0.5
    else:
        col_f, col_i, tol_f, tol_e2e = slice(0, 4), slice(4, 6), 1e-3, 5e-3
    rng = np.random.default_rng(0)

    # 2. network parity against the trainer's own reference
    ref = np.load(args.run_dir / "policy_reference.npz")
    act = run(exe, blob_path, "policy", ref["obs"], 4)
    check("network vs policy_reference.npz", np.abs(act - ref["act_raw"]).max(), 1e-5)

    if C["task_id"] not in (TASK_LISSAJOUS_BODY, TASK_EASY_BODY, TASK_RACE_V3):
        print("[skip] observation/end-to-end: no host mirror for task id %d" % C["task_id"])
        snaps = obs_np = None
    else:
        # 3. observation parity
        snaps = random_snapshots(rng, args.snapshots, C)
        obs_c = run(exe, blob_path, "obs", snaps, int(C["NNPOL_OBS_DIM"]))
        obs_np = np.stack([np_obs(s.astype(np.float64), C) for s in snaps])
        check("observation vs numpy transcription", np.abs(obs_c - obs_np).max(), 2e-4)

    # 3b. race gate state: one crossing-test step per record, positions
    # drawn to land on every branch (approach side, crossing inside the
    # square, crossing outside it, already past)
    if C["task_id"] == TASK_RACE_V3:
        recs, expect = [], []
        for i in range(args.snapshots):
            s = snaps[i].astype(np.float64)
            n_gates, idx = int(s[35]), int(s[36])
            gates = s[37:37 + 3 * MAX_GATES].reshape(MAX_GATES, 3)
            normals = s[61:61 + 3 * MAX_GATES].reshape(MAX_GATES, 3)
            side = float(C["gate_side"])
            n_vec = normals[idx]
            tang = np.array([-n_vec[1], n_vec[0], 0.0])
            pos = (gates[idx] - rng.uniform(-0.3, 0.3) * n_vec
                   + rng.uniform(-side, side) * tang + np.array([0, 0, rng.uniform(-side, side)]))
            prev_x = rng.choice([1.0, 0.2, -0.1])
            rec = np.concatenate([pos, [side, idx, prev_x, n_gates], gates.reshape(-1),
                                  normals.reshape(-1)])
            recs.append(rec)
            expect.append(np_update_gate((n_gates, gates, normals), side, pos, idx, prev_x))
        out = run(exe, blob_path, "gate", np.stack(recs), 2)
        exp = np.array(expect)
        check("gate crossing test (index)", np.abs(out[:, 0] - exp[:, 0]).max(), 0.5)
        check("gate crossing test (prev x)", np.abs(out[:, 1] - exp[:, 1]).max(), 1e-4)
        print("[info] gate test: %d of %d records crossed" % (int((exp[:, 1] == 1.0).sum()), len(exp)))

    # 4. action decode parity
    acts = rng.uniform(-1.2, 1.2, (args.snapshots, 4))  # beyond the box: clip must match
    yaw0s = rng.uniform(-np.pi, np.pi, args.snapshots)
    yaw0s[::3] = 0.0
    tail = np.stack([np.full(len(acts), VNOM), np.full(len(acts), RPM_SCALE), yaw0s], axis=1)
    cmd_c = run(exe, blob_path, "act", np.concatenate([acts, tail], axis=1), CMD_FLOATS)
    cmd_np = np.stack([np_decode(a, VNOM, RPM_SCALE, y, C) for a, y in zip(acts, yaw0s)])
    # The wrapped yaw can land on either side of +-pi for the same command
    # up to float32 rounding; compare it on the circle.
    diff = cmd_c[:, col_f] - cmd_np[:, col_f]
    if kind == KIND_MELLINGER:
        diff[:, 2] = (diff[:, 2] + 180.0) % 360.0 - 180.0
    check("decode float channels", np.abs(diff).max(), tol_f)
    check("decode pwm", np.abs(cmd_c[:, col_i] - cmd_np[:, col_i]).max(), 1.001)
    other = [i for i in range(CMD_FLOATS) if i not in range(*col_f.indices(CMD_FLOATS))
             and i not in range(*col_i.indices(CMD_FLOATS))]
    check("decode leaves the other kind's columns zero", np.abs(cmd_c[:, other]).max(), 0.5)

    # 5. end to end
    if snaps is not None:
        with open(args.run_dir / "config.json") as f:
            cfg = json.load(f)
        layers = load_layers(args.run_dir, cfg)
        tail = np.tile([VNOM, RPM_SCALE], (len(snaps), 1))
        full_c = run(exe, blob_path, "full", np.concatenate([snaps, tail], axis=1), CMD_FLOATS)
        full_np = np.stack([
            np_decode(np.clip(np_forward(layers, C["activation"], obs_np[i].astype(np.float32)), -1, 1),
                      VNOM, RPM_SCALE, float(snaps[i, 17]), C)
            for i in range(len(snaps))])
        diff = full_c[:, col_f] - full_np[:, col_f]
        if kind == KIND_MELLINGER:
            diff[:, 2] = (diff[:, 2] + 180.0) % 360.0 - 180.0
        check("end-to-end float channels", np.abs(diff).max(), tol_e2e)
        check("end-to-end pwm", np.abs(full_c[:, col_i] - full_np[:, col_i]).max(), 2.001)

    # 6. optional: real-flight observations from the shadow's bag export
    if args.bag_npz is not None:
        bag = np.load(args.bag_npz)
        act_bag = run(exe, blob_path, "policy", bag["obs"], 4)
        check("bag observations vs shadow actions",
              np.abs(np.clip(act_bag, -1, 1) - bag["expected_act"]).max(), 1e-5)

    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
