#!/usr/bin/env python3
"""Host parity check for app_nnpol:

    python3 examples/app_nnpol/test/run_host_parity.py [--blob policy_slot.bin] [--snapshots N]

Builds test_policy_host.c with plain gcc against the SAME pure-C modules the
firmware compiles (slot format, interpreter, observation, decode), loads a
slot blob exactly as the firmware reads a flash slot, and checks, in order
of what would hurt most in the air:

  1. slot validation - magic, sizes, dims, CRC, as the firmware's select
     does; the identity words equal the digest of the blob; blobs the
     firmware must refuse (bad CRC, wrong action kind, unknown action map,
     over the interpreter limits) are refused.
  2. network parity - random observations through the C interpreter against
     a numpy forward pass over the weights parsed out of the same blob.
  3. observation parity - random EKF snapshots and tracks through
     nnpolBuildObs against a numpy transcription of the host's RaceV3 /
     RaceV3MotorObs.observe.
  4. gate state - the crossing test against its numpy transcription.
  5. action-decode parity - random actions through nnpolDecodeAction against
     a numpy transcription of the host's MotorPwmRacingPolicy.decode.
  6. end to end - snapshot -> obs -> network -> clip -> decode in one C pass
     against the same chain in numpy.

Without --blob the checks run over synthetic random-weight blobs covering
both observations (v3, v3_motorobs), both action maps (rpm_hover,
rpm_direct), both activations and both output maps, so the suite needs no
trainer run directory. With --blob they run on that blob alone.

Exit code 0 = all parity bounds met.
"""
import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SRC = HERE.parent / "src"
sys.path.insert(0, str(HERE))
import nnpol_blob  # noqa: E402

MAX_GATES = 8
SNAP_FLOATS = 19                    # test_policy_host.c
TRACK_FLOATS = 1 + 6 * MAX_GATES
CMD_FLOATS = 8
VNOM = 4.0
RPM_SCALE = 1.0
C_SOURCES = ("nnpol_slot_format.c", "nnpol_policy.c", "nnpol_obs_race.c", "nnpol_action.c")


def build(tmp):
    cc = shutil.which("gcc") or shutil.which("cc")
    if cc is None:
        sys.exit("no host C compiler")
    exe = tmp / "nnpol_host"
    subprocess.run(
        [cc, "-O2", "-Wall", "-Wextra", "-Werror", "-o", str(exe),
         str(HERE / "test_policy_host.c"), *[str(SRC / s) for s in C_SOURCES],
         "-I", str(SRC), "-lm"], check=True)
    return exe


def run(exe, blob, mode, data, out_floats):
    r = subprocess.run([str(exe), mode, str(blob)], input=data.astype(np.float32).tobytes(),
                       capture_output=True, check=True)
    return np.frombuffer(r.stdout, dtype=np.float32).reshape(-1, out_floats)


def status_of(exe, blob_bytes, tmp):
    path = tmp / "probe.bin"
    path.write_bytes(blob_bytes)
    out = subprocess.run([str(exe), "info", str(path)], capture_output=True, text=True).stdout
    return int(out.split()[0])


# -- numpy mirrors ----------------------------------------------------------

def quat_to_R(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def split_track(track):
    n = int(track[0])
    return (n, track[1:1 + 3 * MAX_GATES].reshape(MAX_GATES, 3),
            track[1 + 3 * MAX_GATES:1 + 6 * MAX_GATES].reshape(MAX_GATES, 3))


def np_obs(snap, track, H):
    """RaceV3.observe: body velocity and rate, gravity in body, the current and
    next gate's offset and approach normal in body, [+ rotor speeds]."""
    pos, q, v_w, gyro, rpm = snap[1:4], snap[4:8], snap[8:11], snap[11:14], snap[14:18]
    n_gates, gates, normals = split_track(track)
    idx = int(snap[18]) % n_gates
    nxt = (idx + 1) % n_gates
    R = quat_to_R(q / np.linalg.norm(q))
    Rt = R.T
    parts = [Rt @ v_w, np.radians(gyro), R[2, :],
             Rt @ (gates[idx] - pos), Rt @ (gates[nxt] - pos),
             Rt @ normals[idx], Rt @ normals[nxt]]
    if H["motorobs"]:
        parts.append(rpm)
    return np.concatenate(parts)


def np_update_gate(track, gate_side, pos, gate_idx, prev_gate_x):
    """RacePlugin.step's crossing test."""
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


def np_rpm_to_pwm(rpm, vnom, H):
    k0, k1 = H["vmotor2rpm"]
    v = (np.asarray(rpm, dtype=np.float64) - k0) / k1
    return np.clip(np.rint(65535.0 * v / vnom), 0, 65535)


def np_mix(a, H):
    """controller/policy/motor.py rpm_hover / rpm_direct."""
    hover, max_rpm = H["hover_rpm"], H["max_rpm"]
    if H["mixer"] == nnpol_blob.MIXER_ID["rpm_direct"]:
        return max_rpm * (a + 1.0) * 0.5
    return np.where(a <= 0, (a + 1.0) * hover, hover + (max_rpm - hover) * a)


def np_decode(action, vnom, rpm_scale, H):
    a = np.clip(action, -1, 1)
    out = np.zeros(CMD_FLOATS)
    rpm = np_mix(a, H) * rpm_scale
    out[0:4] = rpm
    out[4:8] = np_rpm_to_pwm(rpm, vnom, H)
    return out


def np_forward(H, obs):
    act = {0: np.tanh, 1: lambda x: np.maximum(x, 0.0)}[H["activation"]]
    x = obs.astype(np.float32)
    for i, (W, b) in enumerate(H["layers"]):
        x = x @ W + b
        if i < len(H["layers"]) - 1:
            x = act(x)
    return np.tanh(x) if H["output_squash"] else x


def random_cases(rng, n):
    """Plausible engaged-flight snapshots, each with its own track: 2-8 gates
    on a loop, horizontal normals against the travel direction, the drone
    somewhere near the gate it is targeting."""
    snaps = np.zeros((n, SNAP_FLOATS), dtype=np.float32)
    tracks = np.zeros((n, TRACK_FLOATS), dtype=np.float32)
    snaps[:, 0] = rng.uniform(0, 30, n)
    for i in range(n):
        n_gates = int(rng.integers(2, MAX_GATES + 1))
        thetas = np.sort(rng.uniform(-np.pi, np.pi, n_gates))
        radius = rng.uniform(1.0, 3.0)
        gates = np.stack([radius * np.cos(thetas), radius * np.sin(thetas),
                          rng.uniform(0.5, 1.5, n_gates)], axis=1)
        travel = thetas + np.pi / 2 + rng.uniform(-0.3, 0.3, n_gates)
        normals = -np.stack([np.cos(travel), np.sin(travel), np.zeros(n_gates)], axis=1)
        idx = int(rng.integers(0, n_gates))
        tracks[i, 0] = n_gates
        tracks[i, 1:1 + 3 * n_gates] = gates.reshape(-1)
        tracks[i, 1 + 3 * MAX_GATES:1 + 3 * MAX_GATES + 3 * n_gates] = normals.reshape(-1)
        snaps[i, 18] = idx
        snaps[i, 1:4] = gates[idx] + rng.uniform(-2.0, 2.0, 3)
    q = rng.normal(size=(n, 4))
    snaps[:, 4:8] = q / np.linalg.norm(q, axis=1, keepdims=True)
    snaps[:, 8:11] = rng.uniform(-8.0, 8.0, (n, 3))                     # vel_w
    snaps[:, 11:14] = rng.uniform(-600, 600, (n, 3))                    # gyro deg/s
    snaps[:, 14:18] = rng.uniform(0.3, 1.4, (n, 4))                     # rpm / hover
    return snaps, tracks


class Report:
    def __init__(self):
        self.failures = []

    def check(self, name, err, bound):
        ok = err < bound
        print(f"  [{'ok' if ok else 'FAIL'}] {name}: max err {err:.3g} (bound {bound:g})")
        if not ok:
            self.failures.append(name)

    def expect(self, name, ok):
        print(f"  [{'ok' if ok else 'FAIL'}] {name}")
        if not ok:
            self.failures.append(name)


def check_blob(exe, blob, label, n, tmp, rep):
    print(f"== {label}")
    path = tmp / "slot.bin"
    path.write_bytes(blob)
    H = nnpol_blob.parse(blob)
    rng = np.random.default_rng(0)

    # 1. the slot validates and its identity is the digest of what flies
    rep.expect(f"slot validation ({H['name']!r}, {H['obs_dim']}-dim obs, layers {H['layer_dims']})",
               status_of(exe, blob, tmp) == 0)
    rep.expect("identity words are the blob digest", H["hash_words"] == nnpol_blob.hash_words(blob))

    # 2. network
    obs = rng.uniform(-3, 3, (n, H["obs_dim"])).astype(np.float32)
    act = run(exe, path, "policy", obs, 4)
    rep.check("network vs numpy", np.abs(act - np.stack([np_forward(H, o) for o in obs])).max(), 2e-5)

    # 3. observation
    snaps, tracks = random_cases(rng, n)
    both = np.concatenate([snaps, tracks], axis=1)
    obs_c = run(exe, path, "obs", both, H["obs_dim"])
    obs_np = np.stack([np_obs(s.astype(np.float64), t.astype(np.float64), H)
                       for s, t in zip(snaps, tracks)])
    rep.check("observation vs numpy transcription", np.abs(obs_c - obs_np).max(), 2e-4)

    # 4. gate state: positions drawn to land on every branch (approach side,
    # crossing inside the square, crossing outside it, already past)
    recs, expect = [], []
    side = float(H["gate_side"])
    for s, t in zip(snaps.astype(np.float64), tracks.astype(np.float64)):
        n_gates, gates, normals = split_track(t)
        idx = int(s[18])
        n_vec = normals[idx]
        tang = np.array([-n_vec[1], n_vec[0], 0.0])
        pos = (gates[idx] - rng.uniform(-0.3, 0.3) * n_vec
               + rng.uniform(-side, side) * tang + np.array([0, 0, rng.uniform(-side, side)]))
        prev_x = rng.choice([1.0, 0.2, -0.1])
        recs.append(np.concatenate([pos, [side, idx, prev_x], t]))
        expect.append(np_update_gate((n_gates, gates, normals), side, pos, idx, prev_x))
    out = run(exe, path, "gate", np.stack(recs), 2)
    exp = np.array(expect)
    rep.check("gate crossing test (index)", np.abs(out[:, 0] - exp[:, 0]).max(), 0.5)
    rep.check("gate crossing test (prev x)", np.abs(out[:, 1] - exp[:, 1]).max(), 1e-4)
    print("  [info] gate test: %d of %d records crossed" % (int((exp[:, 1] == 1.0).sum()), len(exp)))

    # 5. action decode. rpm columns are O(1e4) with ~1e-3 rpm of float32
    # rounding; PWM columns are integers where rintf and np.rint can differ by
    # one count at an exact .5 boundary.
    acts = rng.uniform(-1.2, 1.2, (n, 4))               # beyond the box: the clip must match
    tail = np.tile([VNOM, RPM_SCALE], (n, 1))
    cmd_c = run(exe, path, "act", np.concatenate([acts, tail], axis=1), CMD_FLOATS)
    cmd_np = np.stack([np_decode(a, VNOM, RPM_SCALE, H) for a in acts])
    rep.check("decode rpm", np.abs(cmd_c[:, :4] - cmd_np[:, :4]).max(), 0.05)
    rep.check("decode pwm", np.abs(cmd_c[:, 4:] - cmd_np[:, 4:]).max(), 1.001)

    # 6. end to end
    full_c = run(exe, path, "full", np.concatenate([both, tail], axis=1), CMD_FLOATS)
    full_np = np.stack([
        np_decode(np.clip(np_forward(H, o.astype(np.float32)), -1, 1), VNOM, RPM_SCALE, H)
        for o in obs_np])
    rep.check("end-to-end rpm", np.abs(full_c[:, :4] - full_np[:, :4]).max(), 0.5)
    rep.check("end-to-end pwm", np.abs(full_c[:, 4:] - full_np[:, 4:]).max(), 2.001)


def check_refusals(exe, tmp, rep):
    """Blobs the firmware must not select."""
    print("== refusals")
    rng = np.random.default_rng(1)
    good = bytearray(nnpol_blob.make_blob(nnpol_blob.random_layers(rng, 21)))

    def patched(offset, fmt, value, fix_crc=True):
        import struct
        import zlib
        b = bytearray(good)
        struct.pack_into(fmt, b, offset, value)
        if fix_crc:
            struct.pack_into("<I", b, 12, zlib.crc32(bytes(b[16:])) & 0xFFFFFFFF)
        return bytes(b)

    flipped = bytearray(good)
    flipped[300] ^= 0x01
    rep.expect("a flipped payload bit fails the CRC (status 5)", status_of(exe, bytes(flipped), tmp) == 5)
    rep.expect("a setpoint action kind is refused (status 6)",
               status_of(exe, patched(52, "<H", 1), tmp) == 6)
    rep.expect("the retired attitude mixer is refused (status 6)",
               status_of(exe, patched(54, "<H", 0), tmp) == 6)
    rep.expect("an unknown activation is refused (status 4)",
               status_of(exe, patched(38, "<H", 7), tmp) == 4)
    rep.expect("an erased slot reads as empty (status 1)", status_of(exe, b"\xff" * 512, tmp) == 1)
    wide = nnpol_blob.random_layers(rng, 21, hidden=(64, 64))
    wide[0] = (np.zeros((21, 200), np.float32), np.zeros(200, np.float32))
    wide[1] = (np.zeros((200, 64), np.float32), np.zeros(64, np.float32))
    rep.expect("a hidden layer over 128 wide is refused (status 4)",
               status_of(exe, nnpol_blob.make_blob(wide), tmp) == 4)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--blob", type=Path, default=None, help="check this blob instead of the synthetic set")
    p.add_argument("--snapshots", type=int, default=300)
    args = p.parse_args()

    tmp = Path(tempfile.mkdtemp(prefix="nnpol_parity_"))
    exe = build(tmp)
    rep = Report()

    if args.blob is not None:
        check_blob(exe, args.blob.read_bytes(), str(args.blob), args.snapshots, tmp, rep)
    else:
        rng = np.random.default_rng(0)
        cases = [
            ("v3, rpm_direct, tanh, clip head, 2x64", dict(), 21, (64, 64)),
            ("v3_motorobs, rpm_hover, tanh, tanh head, 2x128",
             dict(motorobs=True, action_type="rpm_hover", output_map="tanh"), 25, (128, 128)),
            ("v3_motorobs, rpm_direct, relu, clip head, 128-96-64",
             dict(motorobs=True, activation="relu"), 25, (128, 96, 64)),
            ("v3, rpm_hover, tanh, clip head, 1x32",
             dict(action_type="rpm_hover", ctrl_freq=50), 21, (32,)),
        ]
        for label, kwargs, obs_dim, hidden in cases:
            blob = nnpol_blob.make_blob(nnpol_blob.random_layers(rng, obs_dim, hidden),
                                        name=label[:40], **kwargs)
            check_blob(exe, blob, label, args.snapshots, tmp, rep)
        check_refusals(exe, tmp, rep)

    if rep.failures:
        print("FAILED: " + ", ".join(rep.failures))
    else:
        print("all parity bounds met")
    sys.exit(1 if rep.failures else 0)


if __name__ == "__main__":
    main()
