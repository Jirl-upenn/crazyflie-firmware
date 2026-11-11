#!/usr/bin/env python3
"""Host parity check for app_nnpol:

    python3 examples/app_nnpol/test/run_host_parity.py [--blob policy_slot.bin] [--cases N]

Builds libnnpol_core.so - the SAME pure-C files the firmware compiles, behind
the flat entry point of nnpol_host.c - and runs policy steps through it
exactly as the vehicle's app task does (nnpolCoreStep), comparing every stage
with numpy transcriptions of the host implementation. In order of what would
hurt most in the air:

  1. slot validation - magic, sizes, dims, CRC, as the firmware's select
     does; the identity words equal the digest of the blob; blobs the
     firmware must refuse (bad CRC, wrong action kind, unknown action map,
     over the interpreter limits) are refused.
  2. engage context - a track round-trips; a bad CRC, an empty or oversized
     track, a first gate out of range and a non-unit axis are refused, and a
     policy without a track produces no command.
  3. network - random observations through the C interpreter against a numpy
     forward pass over the weights parsed out of the same blob.
  4. one step - random states on random tracks (level and pitched gates):
     the observation against RaceV3 / RaceV3MotorObs.observe, the clipped
     action, and the decode (action map, thrust curve, motor value) against
     MotorPwmRacingPolicy.decode.
  5. gate state - random two-step paths around the target gate against
     RaceV3._update_gate (the trainer's plane_crossings test), and whole
     laps through every gate of a track, dive gates included.

Without --blob the checks run over synthetic random-weight blobs covering
both observations (v3, v3_motorobs), both action maps (rpm_hover,
rpm_direct), both activations and both output maps, so the suite needs no
trainer run directory. With --blob they run on that blob alone.

Exit code 0 = all parity bounds met.
"""
import argparse
import struct
import sys
import zlib
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import nnpol_blob  # noqa: E402
import nnpol_core  # noqa: E402

THRUST_MAX = 0.2                    # platform_defaults_cf21bl.h, N per motor
RPM_SCALE = 1.0


# -- numpy mirrors ----------------------------------------------------------

def quat_to_R(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


class Track:
    """A track as the host holds it (RaceV3): positions and the gate frames."""

    def __init__(self, pos, normals, tangents, ups, side, gate0=0):
        self.pos, self.normals, self.tangents, self.ups = pos, normals, tangents, ups
        self.side, self.gate0, self.n = float(side), int(gate0), len(pos)

    def context(self):
        return nnpol_blob.make_context(self.pos, self.normals, self.tangents, self.ups,
                                       self.side, self.gate0)

    def update_gate(self, idx, prev, pos):
        """RaceV3._update_gate's pass test (GateTrack.plane_crossings)."""
        rel0, rel1 = prev - self.pos[idx], pos - self.pos[idx]
        x0, x1 = float(rel0 @ self.normals[idx]), float(rel1 @ self.normals[idx])
        if not (x0 > 0.0 and x1 <= 0.0):
            return idx, False
        s = x0 / (x0 - x1)
        at = rel0 + s * (rel1 - rel0)
        half = self.side * 0.5
        if abs(at @ self.tangents[idx]) < half and abs(at @ self.ups[idx]) < half:
            return (idx + 1) % self.n, True
        return idx, False


def random_track(rng, n_gates=None, max_gates=None):
    """Gates on a loop. Most are level (normal horizontal, up = world z); some
    are pitched, down to a horizontal dive gate whose normal is vertical."""
    n = int(rng.integers(2, (max_gates or 8) + 1)) if n_gates is None else n_gates
    thetas = np.sort(rng.uniform(-np.pi, np.pi, n))
    radius = rng.uniform(1.5, 4.0)
    pos = np.stack([radius * np.cos(thetas), radius * np.sin(thetas),
                    rng.uniform(0.6, 2.0, n)], axis=1)
    heading = thetas + np.pi / 2 + rng.uniform(-0.3, 0.3, n)
    pitch = np.where(rng.random(n) < 0.3, rng.choice([-np.pi / 2, -0.6, 0.5], n), 0.0)
    travel = np.stack([np.cos(heading) * np.cos(pitch), np.sin(heading) * np.cos(pitch),
                       np.sin(pitch)], axis=1)
    tangents = np.stack([-np.sin(heading), np.cos(heading), np.zeros(n)], axis=1)
    ups = np.cross(travel, tangents)
    return Track(pos, -travel, tangents, ups, side=rng.choice([0.4, 1.0]),
                 gate0=int(rng.integers(0, n)))


def np_obs(H, track, idx, pos, q, v_w, gyro_deg, rpm):
    """RaceV3.observe: body velocity and rate, gravity in body, the current and
    next gate's offset and approach normal in body, [+ rotor speeds / hover]."""
    R = quat_to_R(q / np.linalg.norm(q))
    Rt = R.T
    nxt = (idx + 1) % track.n
    parts = [Rt @ v_w, np.radians(gyro_deg), R[2, :],
             Rt @ (track.pos[idx] - pos), Rt @ (track.pos[nxt] - pos),
             Rt @ track.normals[idx], Rt @ track.normals[nxt]]
    if H["motorobs"]:
        parts.append(rpm / H["hover_rpm"])
    return np.concatenate(parts)


def np_rpm_to_thrust(rpm, H):
    """crazyflow rpm2thrust: the force the simulator gives a rotor speed."""
    a0, a1, a2 = H["rpm2thrust"]
    rpm = np.asarray(rpm, dtype=np.float64)
    return np.maximum(a0 + a1 * rpm + a2 * rpm * rpm, 0.0)


def np_thrust_to_motor(thrust, thrust_max):
    """The 16-bit motor value: the force as a fraction of the build's THRUST_MAX."""
    return np.clip(np.rint(65535.0 * np.asarray(thrust) / thrust_max), 0, 65535)


def np_mix(a, H):
    """controller/policy/motor.py rpm_hover / rpm_direct."""
    hover, max_rpm = H["hover_rpm"], H["max_rpm"]
    if H["mixer"] == nnpol_blob.MIXER_ID["rpm_direct"]:
        return max_rpm * (a + 1.0) * 0.5
    return np.where(a <= 0, (a + 1.0) * hover, hover + (max_rpm - hover) * a)


def np_decode(action, thrust_max, rpm_scale, H):
    rpm = np_mix(np.clip(action, -1, 1), H) * rpm_scale
    thrust = np_rpm_to_thrust(rpm, H)
    return rpm, thrust, np_thrust_to_motor(thrust, thrust_max)


def np_forward(H, obs):
    act = {0: np.tanh, 1: lambda x: np.maximum(x, 0.0)}[H["activation"]]
    x = obs.astype(np.float32)
    for i, (W, b) in enumerate(H["layers"]):
        x = x @ W + b
        if i < len(H["layers"]) - 1:
            x = act(x)
    return np.tanh(x) if H["output_squash"] else x


def random_state(rng, near):
    q = rng.normal(size=4)
    return (near + rng.uniform(-2.0, 2.0, 3), q / np.linalg.norm(q), rng.uniform(-8.0, 8.0, 3),
            rng.uniform(-600, 600, 3), rng.uniform(3000, 22000, 4))


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


def check_blob(lib, blob, label, n, rep):
    print(f"== {label}")
    H = nnpol_blob.parse(blob)
    rng = np.random.default_rng(0)
    max_gates = lib.nnpolHostMaxGates()

    # 1. the slot validates and its identity is the digest of what flies
    rep.expect(f"slot validation ({H['name']!r}, {H['obs_dim']}-dim obs, layers {H['layer_dims']})",
               nnpol_core.slot_status(lib, blob)[0] == 0)
    rep.expect("identity words are the blob digest", H["hash_words"] == nnpol_blob.hash_words(blob))

    with nnpol_core.Session(lib, blob) as s:
        # 3. network
        obs = rng.uniform(-3, 3, (n, H["obs_dim"])).astype(np.float32)
        act = np.stack([s.forward(o) for o in obs])
        rep.check("network vs numpy", np.abs(act - np.stack([np_forward(H, o) for o in obs])).max(), 2e-5)

        rep.expect("no command without a track", s.step(*random_state(rng, np.zeros(3))) is None)

        # 4. one step on a fresh run: the first step never passes a gate, so
        # the observation targets the track's first gate
        e_obs = e_act = e_rpm = e_thrust = e_motor = 0.0
        for _ in range(n):
            track = random_track(rng, max_gates=max_gates)
            assert s.set_context(track.context())[0] == 0
            s.engage(THRUST_MAX, RPM_SCALE)
            state = random_state(rng, track.pos[track.gate0])
            out = s.step(*state)
            ref_obs = np_obs(H, track, track.gate0, *state)
            ref_act = np.clip(np_forward(H, ref_obs.astype(np.float32)), -1, 1)
            rpm, thrust, motor = np_decode(out["act"].astype(np.float64), THRUST_MAX, RPM_SCALE, H)
            e_obs = max(e_obs, np.abs(out["obs"][:H["obs_dim"]] - ref_obs).max())
            e_act = max(e_act, np.abs(out["act"] - ref_act).max())
            e_rpm = max(e_rpm, np.abs(out["rpm"] - rpm).max())
            e_thrust = max(e_thrust, np.abs(out["thrust"] - thrust).max())
            e_motor = max(e_motor, np.abs(out["motor"] - motor).max())
        rep.check("observation vs numpy transcription", e_obs, 2e-4)
        rep.check("clipped action end to end", e_act, 2e-4)
        # rpm is O(1e4) with ~1e-3 rpm of float32 rounding; motor values are
        # integers where rintf and np.rint can differ by one at an exact .5
        rep.check("decode rpm", e_rpm, 0.05)
        rep.check("decode thrust (N)", e_thrust, 2e-6)
        rep.check("decode motor value", e_motor, 1.001)
        hover_a = (np.zeros(4) if H["mixer"] == nnpol_blob.MIXER_ID["rpm_hover"]
                   else np.full(4, 2.0 * H["hover_rpm"] / H["max_rpm"] - 1.0))
        rpm, thrust, motor = np_decode(hover_a, THRUST_MAX, 1.0, H)
        print("  [info] hover: %.0f rpm -> %.4f N per motor -> motor value %d of 65535"
              % (rpm[0], thrust[0], motor[0]))

        # 5a. gate state: two-step paths drawn to land on every branch
        # (through the opening, through the plane outside it, short of the
        # plane, backwards)
        wrong, crossed = 0, 0
        zeros = (np.array([1.0, 0, 0, 0]), np.zeros(3), np.zeros(3), np.full(4, 15000.0))
        for _ in range(4 * n):
            track = random_track(rng, max_gates=max_gates)
            i = track.gate0
            s.set_context(track.context())
            s.engage(THRUST_MAX, RPM_SCALE)

            def point(axial):
                return (track.pos[i] + axial * track.normals[i]
                        + rng.uniform(-track.side, track.side) * track.tangents[i]
                        + rng.uniform(-track.side, track.side) * track.ups[i])
            p0 = point(rng.choice([0.4, 0.05, -0.2]))
            p1 = point(rng.choice([-0.3, -0.01, 0.2]))
            s.step(p0, *zeros)
            out = s.step(p1, *zeros)
            idx, passed = track.update_gate(i, p0, p1)
            crossed += passed
            wrong += (out["gate_idx"] != idx) or (out["passed"] != passed)
        rep.expect("gate crossing test on %d paths (%d passed)" % (4 * n, crossed), wrong == 0)

        # 5b. whole laps: fly gate to gate through every opening
        wrong, passes = 0, 0
        for _ in range(20):
            track = random_track(rng, n_gates=int(rng.integers(3, max_gates + 1)))
            s.set_context(track.context())
            s.engage(THRUST_MAX, RPM_SCALE)
            idx = track.gate0
            prev = None
            for lap_gate in range(2 * track.n):
                g = (track.gate0 + lap_gate) % track.n
                for axial in (0.6, 0.2, -0.2, -0.6):
                    p = (track.pos[g] + axial * track.normals[g]
                         + rng.uniform(-0.3, 0.3) * track.side * track.tangents[g]
                         + rng.uniform(-0.3, 0.3) * track.side * track.ups[g])
                    out = s.step(p, *zeros)
                    if prev is not None:
                        idx, passed = track.update_gate(idx, prev, p)
                        passes += passed
                    prev = p
                    wrong += out["gate_idx"] != idx
        rep.expect("gate index over 20 two-lap flights (%d gates passed)" % passes,
                   wrong == 0 and passes > 100)


def check_refusals(lib, rep):
    """Blobs and contexts the firmware must not adopt."""
    print("== refusals")
    rng = np.random.default_rng(1)
    good = bytearray(nnpol_blob.make_blob(nnpol_blob.random_layers(rng, 21)))

    def patched(offset, fmt, value):
        b = bytearray(good)
        struct.pack_into(fmt, b, offset, value)
        struct.pack_into("<I", b, 12, zlib.crc32(bytes(b[16:])) & 0xFFFFFFFF)
        return bytes(b)

    def status(blob):
        return nnpol_core.slot_status(lib, blob)[0]

    flipped = bytearray(good)
    flipped[300] ^= 0x01
    rep.expect("a flipped payload bit fails the CRC (status 5)", status(bytes(flipped)) == 5)
    rep.expect("a setpoint action kind is refused (status 6)", status(patched(52, "<H", 1)) == 6)
    rep.expect("the retired attitude mixer is refused (status 6)", status(patched(54, "<H", 0)) == 6)
    rep.expect("an unknown activation is refused (status 4)", status(patched(38, "<H", 7)) == 4)
    rep.expect("an erased slot reads as empty (status 1)", status(b"\xff" * 512) == 1)
    wide = nnpol_blob.random_layers(rng, 21, hidden=(64, 64))
    wide[0] = (np.zeros((21, 200), np.float32), np.zeros(200, np.float32))
    wide[1] = (np.zeros((200, 64), np.float32), np.zeros(64, np.float32))
    rep.expect("a hidden layer over 128 wide is refused (status 4)",
               status(nnpol_blob.make_blob(wide)) == 4)

    max_gates = lib.nnpolHostMaxGates()
    track = random_track(rng, n_gates=5)
    ctx = bytearray(track.context())
    with nnpol_core.Session(lib, bytes(good)) as s:
        def ctx_status(c):
            return s.set_context(bytes(c))[0]

        def repacked(n=None, gate0=None, side=None, gates=None):
            body = bytearray(ctx[8:])
            if n is not None:
                body[2] = n
            if gate0 is not None:
                body[3] = gate0
            if side is not None:
                struct.pack_into("<f", body, 4, side)
            if gates is not None:
                body[8:] = gates
                struct.pack_into("<H", body, 0, 16 + len(gates))
            return struct.pack("<II", nnpol_blob.CTX_MAGIC, zlib.crc32(bytes(body)) & 0xFFFFFFFF) + body

        rep.expect("a track round-trips (status 0)", ctx_status(ctx) == 0)
        full = random_track(rng, n_gates=max_gates)
        rep.expect("a full %d-gate track is accepted" % max_gates, ctx_status(full.context()) == 0)
        bad = bytearray(ctx)
        bad[40] ^= 0x10
        rep.expect("a flipped context bit fails the CRC (status 4)", ctx_status(bad) == 4)
        rep.expect("half a context is refused", ctx_status(ctx[:len(ctx) // 2]) in (3, 4))
        rep.expect("nothing written reads as empty (status 1)", ctx_status(b"\0" * 64) == 1)
        rep.expect("a track with no gates is refused (status 5)",
                   ctx_status(repacked(n=0, gates=b"")) == 5)
        rep.expect("a first gate past the last is refused (status 5)", ctx_status(repacked(gate0=5)) == 5)
        rep.expect("a zero gate side is refused (status 5)", ctx_status(repacked(side=0.0)) == 5)
        gates = bytearray(ctx[16:])
        struct.pack_into("<3f", gates, 12, 0.5, 0.0, 0.0)       # gate 0's normal, not a unit vector
        rep.expect("a non-unit gate axis is refused (status 5)", ctx_status(repacked(gates=gates)) == 5)
        rep.expect("a gate count the size does not match is refused (status 3)",
                   ctx_status(repacked(n=4)) == 3)
        over = struct.pack("<II", nnpol_blob.CTX_MAGIC, 0) + struct.pack(
            "<HBBf", 16 + (max_gates + 1) * 48, max_gates + 1, 0, 1.0) + b"\0" * ((max_gates + 1) * 48)
        rep.expect("more gates than the vehicle holds are refused (status 3)", ctx_status(over) == 3)
    rep.expect("the writer refuses %d gates too" % (max_gates + 1), _raises(
        lambda: nnpol_blob.make_context(*[np.zeros((max_gates + 1, 3))] * 4, 1.0)))
    rep.expect("writer and library agree on the limits",
               (nnpol_blob.MAX_GATES, nnpol_blob.CTX_MAX_BYTES, nnpol_blob.SLOT_BYTES,
                nnpol_blob.MAX_OBS_DIM, nnpol_blob.MAX_HIDDEN, nnpol_blob.MAX_LAYERS)
               == (max_gates, lib.nnpolHostContextMaxBytes(), lib.nnpolHostSlotBytes(),
                   lib.nnpolHostMaxObsDim(), lib.nnpolHostMaxHidden(), lib.nnpolHostMaxLayers()))


def _raises(fn):
    try:
        fn()
    except ValueError:
        return True
    return False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--blob", type=Path, default=None, help="check this blob instead of the synthetic set")
    p.add_argument("--cases", type=int, default=200)
    args = p.parse_args()

    lib = nnpol_core.load()
    print("libnnpol_core: core version %d, %d gates" % (lib.nnpolHostVersion(), lib.nnpolHostMaxGates()))
    rep = Report()

    if args.blob is not None:
        check_blob(lib, args.blob.read_bytes(), str(args.blob), args.cases, rep)
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
            check_blob(lib, blob, label, args.cases, rep)
        check_refusals(lib, rep)

    if rep.failures:
        print("FAILED: " + ", ".join(rep.failures))
    else:
        print("all parity bounds met")
    sys.exit(1 if rep.failures else 0)


if __name__ == "__main__":
    main()
