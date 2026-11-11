"""The policy slot blob (src/nnpol_slot.h), written and parsed in Python.

This is the reference writer for the format: the host tests build synthetic
blobs with it, and the ground station's compiler (ros_quad_sim2real
controller/onboard/compile.py) must produce byte-identical output for the
same inputs. Nothing here depends on a trainer run directory.

Layout (little-endian, natural alignment, 256-byte header + fp32 payload):

    magic, headerBytes, totalBytes, crc32 | hash[4] | obsDim, actionDim,
    numLayers, activation | layerDims[5], outputSquash | actionKind, mixer,
    ctrlFreqHz, taskId | actionMid[4] | actionHalf[4] | hoverRpm, maxRpm,
    differentialFrac | rpm2thrust[3] | vmotor2rpm[2] | task[8] | name[64] |
    payloadFloats | reserved[32]

    payload: per dense layer, the kernel TRANSPOSED to (out, in) row-major,
    then the bias.

    crc32  zlib crc32 over bytes [16, totalBytes)
    hash   the identity: the first 16 bytes of sha256 over bytes
           [32, totalBytes) - everything that flies - as four LE uint32.
"""
import hashlib
import struct
import zlib

import numpy as np

HEADER_FORMAT = "<4I4I4H5HH4H4f4f3f3f2f8f64sI32s"
HEADER_BYTES = 256
MAGIC = 0x3153504E                # "NPS1"
SLOT_BYTES = 0x20000              # one 128 KB flash sector per slot
MAX_OBS_DIM, MAX_HIDDEN, MAX_LAYERS, ACTION_DIM = 64, 128, 4, 4

KIND_MOTOR = 2
MIXER_ID = {"rpm_hover": 1, "rpm_direct": 2}
ACTIVATION_ID = {"tanh": 0, "relu": 1}
SQUASH_ID = {"clip": 0, "tanh": 1}
TASK_RACE_V3 = 2
TASK_MOTOROBS, TASK_GATE_SIDE = 6, 7

# crazyflow drones/params.toml cf21B_500
RPM2THRUST = (0.0, -3.133427287299859e-7, 4.407354891648379e-10)
VMOTOR2RPM = (2938.3995608848436, 6001.834195381014)

assert struct.calcsize(HEADER_FORMAT) == HEADER_BYTES


def hash_words(blob):
    """The identity of a finished blob: sha256 over everything after the hash field."""
    return list(struct.unpack("<4I", hashlib.sha256(bytes(blob[32:])).digest()[:16]))


def make_blob(layers, *, action_type="rpm_direct", motorobs=False, activation="tanh",
              output_map="clip", ctrl_freq=100, hover_rpm=15896.296489245326,
              max_rpm=21666.0, gate_side=1.0, name="synthetic",
              rpm2thrust=RPM2THRUST, vmotor2rpm=VMOTOR2RPM):
    """layers: [(kernel (in, out), bias (out,)), ...] as flax stores them."""
    dims = [layers[0][0].shape[0]] + [k.shape[1] for k, _ in layers]
    payload = np.concatenate(
        [np.concatenate([np.ascontiguousarray(k.T).reshape(-1), b.reshape(-1)]) for k, b in layers]
    ).astype("<f4")
    task = [0.0] * 8
    task[TASK_MOTOROBS] = 1.0 if motorobs else 0.0
    task[TASK_GATE_SIDE] = float(gate_side)
    after_hash = struct.pack(
        "<" + HEADER_FORMAT[5:],
        dims[0], ACTION_DIM, len(layers), ACTIVATION_ID[activation],
        *(dims + [0] * (5 - len(dims))), SQUASH_ID[output_map],
        KIND_MOTOR, MIXER_ID[action_type], int(ctrl_freq), TASK_RACE_V3,
        *([0.0] * 8),
        float(hover_rpm), float(max_rpm), 0.0,
        *rpm2thrust, *vmotor2rpm,
        *task, name.encode()[:63], int(payload.size), b"\0" * 32,
    ) + payload.tobytes()
    ident = hashlib.sha256(after_hash).digest()[:16]
    body = ident + after_hash
    total = 16 + len(body)
    head = struct.pack("<4I", MAGIC, HEADER_BYTES, total, zlib.crc32(body) & 0xFFFFFFFF)
    return head + body


def parse(blob):
    """Header fields and the layers [(kernel (in, out), bias)] of a blob."""
    v = struct.unpack(HEADER_FORMAT, blob[:HEADER_BYTES])
    num_layers = v[10]
    dims = list(v[12:17])[:num_layers + 1]
    h = {
        "magic": v[0], "header_bytes": v[1], "total_bytes": v[2], "crc": v[3],
        "hash_words": list(v[4:8]), "obs_dim": v[8], "action_dim": v[9],
        "num_layers": num_layers, "activation": v[11], "layer_dims": dims,
        "output_squash": v[17], "kind": v[18], "mixer": v[19], "ctrl_hz": v[20],
        "task_id": v[21], "hover_rpm": v[30], "max_rpm": v[31],
        "rpm2thrust": v[33:36], "vmotor2rpm": v[36:38], "task": v[38:46],
        "name": v[46].split(b"\0", 1)[0].decode(), "payload_floats": v[47],
    }
    h["motorobs"] = h["task"][TASK_MOTOROBS] != 0.0
    h["gate_side"] = h["task"][TASK_GATE_SIDE]
    w = np.frombuffer(blob, dtype="<f4", offset=HEADER_BYTES, count=h["payload_floats"])
    layers, at = [], 0
    for i in range(num_layers):
        n_in, n_out = dims[i], dims[i + 1]
        kernel = w[at:at + n_in * n_out].reshape(n_out, n_in).T
        at += n_in * n_out
        layers.append((kernel, w[at:at + n_out]))
        at += n_out
    h["layers"] = layers
    return h


def random_layers(rng, obs_dim, hidden=(64, 64)):
    dims = [obs_dim, *hidden, ACTION_DIM]
    return [(rng.normal(0.0, 1.0 / np.sqrt(dims[i]), (dims[i], dims[i + 1])).astype(np.float32),
             rng.normal(0.0, 0.1, dims[i + 1]).astype(np.float32)) for i in range(len(dims) - 1)]
