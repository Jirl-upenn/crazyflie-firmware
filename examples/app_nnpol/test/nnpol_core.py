"""ctypes loader for libnnpol_core.so, the firmware's policy step built for the
host (test/nnpol_host.c, test/Makefile).

    lib = nnpol_core.load()                    # builds it if needed
    with nnpol_core.Session(lib, blob) as s:
        s.set_context(ctx)
        s.engage(thrust_max, rpm_scale)
        out = s.step(pos, quat_wxyz, vel_w, gyro_deg, rpm)

Importable from anywhere: the ground station adds this directory to its path.
"""
import ctypes
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
STATE_FLOATS = 17


def build(out_dir=None):
    """Build the library (make if present, otherwise the compiler directly) and
    return its path. out_dir defaults to test/build; pass a writable directory
    when the source tree is not."""
    out = Path(out_dir) if out_dir else HERE / "build"
    out.mkdir(parents=True, exist_ok=True)
    lib = out / "libnnpol_core.so"
    src = HERE.parent / "src"
    sources = [HERE / "nnpol_host.c"] + [src / n for n in (
        "nnpol_slot_format.c", "nnpol_policy.c", "nnpol_obs_race.c", "nnpol_action.c",
        "nnpol_core.c")]
    deps = sources + [src / "nnpol.h", src / "nnpol_slot.h"]
    if lib.exists() and lib.stat().st_mtime >= max(d.stat().st_mtime for d in deps):
        return lib
    cc = os.environ.get("CC") or shutil.which("gcc") or shutil.which("cc")
    if cc is None:
        raise RuntimeError("no host C compiler to build libnnpol_core.so")
    subprocess.run([cc, "-O2", "-Wall", "-Wextra", "-Werror", "-fPIC", "-shared", f"-I{src}",
                    "-o", str(lib), *map(str, sources), "-lm"], check=True)
    return lib


def load(path=None, out_dir=None):
    lib = ctypes.CDLL(str(path or build(out_dir)))
    u32, f32p, u8p = ctypes.c_uint32, ctypes.POINTER(ctypes.c_float), ctypes.c_char_p
    for name in ("Version", "MaxGates", "MaxObsDim", "MaxHidden", "MaxLayers", "SlotBytes",
                 "ContextMaxBytes", "OutFloats"):
        fn = getattr(lib, "nnpolHost" + name)
        fn.restype, fn.argtypes = u32, []
    for name in ("SlotStatusName", "CtxStatusName"):
        fn = getattr(lib, "nnpolHost" + name)
        fn.restype, fn.argtypes = ctypes.c_char_p, [ctypes.c_int]
    lib.nnpolHostOpen.restype = ctypes.c_void_p
    lib.nnpolHostOpen.argtypes = [u8p, u32, ctypes.POINTER(ctypes.c_int)]
    lib.nnpolHostClose.restype, lib.nnpolHostClose.argtypes = None, [ctypes.c_void_p]
    lib.nnpolHostSetContext.restype = ctypes.c_int
    lib.nnpolHostSetContext.argtypes = [ctypes.c_void_p, u8p, u32]
    lib.nnpolHostEngage.restype = None
    lib.nnpolHostEngage.argtypes = [ctypes.c_void_p, ctypes.c_float, ctypes.c_float]
    lib.nnpolHostStep.restype = ctypes.c_int
    lib.nnpolHostStep.argtypes = [ctypes.c_void_p, f32p, f32p]
    lib.nnpolHostForward.restype = None
    lib.nnpolHostForward.argtypes = [ctypes.c_void_p, f32p, f32p]
    return lib


def slot_status(lib, blob):
    """The firmware's verdict on a blob: (status code, its name). 0 = ok."""
    status = ctypes.c_int(0)
    s = lib.nnpolHostOpen(bytes(blob), len(blob), ctypes.byref(status))
    if s:
        lib.nnpolHostClose(s)
    return status.value, lib.nnpolHostSlotStatusName(status.value).decode()


class Session:
    """One policy on one track, stepped as the vehicle steps it."""

    def __init__(self, lib, blob):
        self.lib = lib
        status = ctypes.c_int(0)
        self._s = lib.nnpolHostOpen(bytes(blob), len(blob), ctypes.byref(status))
        if not self._s:
            raise ValueError("the firmware refuses this blob: %s (status %d)"
                             % (lib.nnpolHostSlotStatusName(status.value).decode(), status.value))
        self.max_obs = lib.nnpolHostMaxObsDim()
        self._out = np.zeros(lib.nnpolHostOutFloats(), dtype=np.float32)

    def close(self):
        if self._s:
            self.lib.nnpolHostClose(self._s)
            self._s = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def set_context(self, ctx):
        """Adopt a track. Returns (status code, name); 0 = ok."""
        st = self.lib.nnpolHostSetContext(self._s, bytes(ctx), len(ctx))
        return st, self.lib.nnpolHostCtxStatusName(st).decode()

    def engage(self, thrust_max, rpm_scale=1.0):
        self.lib.nnpolHostEngage(self._s, float(thrust_max), float(rpm_scale))

    def step(self, pos, quat_wxyz, vel_w, gyro_deg, rpm):
        """One policy step. Returns a dict, or None when no command was
        produced (no track, or an observation the network cannot take)."""
        state = np.concatenate([pos, quat_wxyz, vel_w, gyro_deg, rpm]).astype(np.float32)
        assert state.shape == (STATE_FLOATS,)
        ok = self.lib.nnpolHostStep(
            self._s, state.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            self._out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
        if not ok:
            return None
        o, n = self._out, self.max_obs
        return {"obs": o[:n].copy(), "act_raw": o[n:n + 4].copy(), "act": o[n + 4:n + 8].copy(),
                "rpm": o[n + 8:n + 12].copy(), "thrust": o[n + 12:n + 16].copy(),
                "motor": o[n + 16:n + 20].copy(), "gate_idx": int(o[n + 20]),
                "passed": bool(o[n + 21])}

    def forward(self, obs):
        obs = np.ascontiguousarray(obs, dtype=np.float32)
        act = np.zeros(4, dtype=np.float32)
        self.lib.nnpolHostForward(self._s, obs.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                                  act.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
        return act
