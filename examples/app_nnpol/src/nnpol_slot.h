/**
 * nnpol_slot.h — the policy SLOT format: one self-describing blob per
 * checkpoint, written by mjx-drone-trainer/export_policy_c.py as
 * policy_slot.bin, uploaded over the radio into the app's RAM slot
 * (nnpol_slot_bank.c) at every controller start and selected by the
 * ground station by identity hash — so switching checkpoints is a
 * config.yaml edit, not a reflash. Everything the app needs is in the
 * 256-byte header; the network's weights follow it as IEEE fp16 - half the
 * RAM of fp32, and the Cortex-M4F converts a half in one instruction - in
 * the output-major layout the interpreter (nnpol_policy.c) walks: per
 * layer the kernel as [out][in], then the bias. payloadFloats counts them;
 * the payload is padded to a multiple of 4 bytes.
 *
 * Pure C, host-buildable: the parity harness validates a blob from a file
 * with exactly the code the firmware validates the RAM slot with.
 *
 * LAYOUT IS A CONTRACT with export_policy_c._write_slot_blob (struct
 * format '<4I4I4H5HH4H4f4f3f3f2f8f64sI32s', 256 bytes, little-endian).
 * Every field sits at its natural alignment, so the struct below matches
 * it without packing; the static asserts pin the offsets.
 */
#ifndef NNPOL_SLOT_H
#define NNPOL_SLOT_H

#include <stddef.h>
#include <stdint.h>

#define NNPOL_SLOT_MAGIC 0x3253504Eu   /* "NPS2" as a little-endian u32: fp16 payload */
#define NNPOL_SLOT_HEADER_BYTES 256u

/* Decode paths (export_policy_c.ACTION_KIND_*). */
#define NNPOL_ACTION_KIND_MELLINGER 0  /* attitude box -> firmware Mellinger */
#define NNPOL_ACTION_KIND_CTBR 1       /* body-rate box -> firmware rate PID */
#define NNPOL_ACTION_KIND_MOTOR 2      /* mixer -> per-motor PWM, no onboard loop */
/* Mixers for the MOTOR kind (export_policy_c.MIXER_FOR_TYPE). */
#define NNPOL_MIXER_ATTITUDE 0
#define NNPOL_MIXER_RPM 1
#define NNPOL_MIXER_RPM_DIRECT 2
/* Activations (export_policy_c.ACTIVATION_ID). */
#define NNPOL_ACT_TANH 0
#define NNPOL_ACT_RELU 1
/* Observation builders (export_policy_c.TASK_ID), all in nnpol_obs_lissajous.c. */
#define NNPOL_TASK_LISSAJOUS_BODY 0    /* traj_lissajous / body[_motorobs]: the fixed eight */
#define NNPOL_TASK_EASY_BODY 1         /* traj_easy / body[_motorobs]: a per-run curve
                                          pushed at engage (nnpol.c*) */
#define NNPOL_TASK_RACE_V3 2           /* race / v3[_motorobs]: the pinned track is
                                          pushed at engage (nnpol.g*); nnpol_obs_race.c */
#define NNPOL_TASK_UNSUPPORTED 0xFFFF  /* exported, uploadable, but cannot engage */
/* task[] indices shared by both builders (export_policy_c.slot_task). */
#define NNPOL_TASK_T_PERIOD 0          /* LISSAJOUS: fundamental period, s */
#define NNPOL_TASK_AMPLITUDE 1         /* LISSAJOUS: x amplitude, m (z is half) */
#define NNPOL_TASK_CENTER_Z 2          /* LISSAJOUS: centre height, m */
#define NNPOL_TASK_N_SAMPLES 3         /* lookahead samples (as a float) */
#define NNPOL_TASK_SAMPLES_DT 4        /* lookahead spacing, s */
#define NNPOL_TASK_START_PHASE 5       /* LISSAJOUS: default engage phase, s */
#define NNPOL_TASK_MOTOROBS 6          /* != 0: append 4 normalized rotor speeds */
#define NNPOL_TASK_GATE_SIDE 7         /* RACE: gate side length, m (the crossing square) */
#define NNPOL_MAX_GATES 8              /* RACE: the pushed track's capacity */
/* Rotor-speed normalizer of the motorobs observations: the trainer's
 * NOMINAL_HOVER_RPM (every tasks package: 15896.3; crazyflie-ros TaskObs). */
#define NNPOL_NOMINAL_HOVER_RPM 15896.296489245326f
#define NNPOL_CURVE_FLOATS 13          /* EASY: amp(3) omega(3) phase(3) center(3) yaw */

/* Interpreter limits: a header outside these fails validation. */
#define NNPOL_MAX_OBS_DIM 64
#define NNPOL_MAX_HIDDEN 128
#define NNPOL_MAX_LAYERS 4             /* up to three hidden layers */
#define NNPOL_ACTION_DIM 4

typedef struct {
  uint32_t magic;            /*   0 NNPOL_SLOT_MAGIC */
  uint32_t headerBytes;      /*   4 NNPOL_SLOT_HEADER_BYTES */
  uint32_t totalBytes;       /*   8 header + fp16 payload (padded), multiple of 4 */
  uint32_t crc32;            /*  12 zlib crc32 over bytes [16, totalBytes) */
  uint32_t hash[4];          /*  16 identity hash words (firmware_export.json) */
  uint16_t obsDim;           /*  32 */
  uint16_t actionDim;        /*  34 always 4 */
  uint16_t numLayers;        /*  36 dense layers incl. the output layer */
  uint16_t activation;       /*  38 NNPOL_ACT_* (hidden layers only) */
  uint16_t layerDims[5];     /*  40 [obs, h1, .., out]; unused trailing = 0 */
  uint16_t pad0;             /*  50 */
  uint16_t actionKind;       /*  52 NNPOL_ACTION_KIND_* */
  uint16_t mixer;            /*  54 NNPOL_MIXER_* (MOTOR kind) */
  uint16_t ctrlFreqHz;       /*  56 trained control rate; nnpol.ctrlHz default */
  uint16_t taskId;           /*  58 NNPOL_TASK_* */
  float actionMid[4];        /*  60 setpoint kinds: box midpoint */
  float actionHalf[4];       /*  76 setpoint kinds: box half-range */
  float hoverRpm;            /*  92 MOTOR kind mixer anchors */
  float maxRpm;              /*  96 */
  float differentialFrac;    /* 100 */
  float rpm2thrust[3];       /* 104 crazyflow drone identification (a0, a1, a2) */
  float vmotor2rpm[2];       /* 116 (k0, k1) */
  float task[8];             /* 124 task constants, NNPOL_TASK_* indices:
                                    [T_period_s, amplitude_m, center_z_m,
                                    n_samples, samples_dt_s,
                                    default_start_phase_s, motorobs, gate_side] */
  char name[64];             /* 156 run directory name, NUL-padded */
  uint32_t payloadFloats;    /* 220 weights (fp16 each) after the header */
  uint8_t reserved[32];      /* 224 zeros */
} nnpolSlotHeader_t;         /* 256 */

_Static_assert(sizeof(nnpolSlotHeader_t) == NNPOL_SLOT_HEADER_BYTES, "slot header size");
_Static_assert(offsetof(nnpolSlotHeader_t, hash) == 16, "slot header layout");
_Static_assert(offsetof(nnpolSlotHeader_t, actionKind) == 52, "slot header layout");
_Static_assert(offsetof(nnpolSlotHeader_t, actionMid) == 60, "slot header layout");
_Static_assert(offsetof(nnpolSlotHeader_t, task) == 124, "slot header layout");
_Static_assert(offsetof(nnpolSlotHeader_t, name) == 156, "slot header layout");
_Static_assert(offsetof(nnpolSlotHeader_t, payloadFloats) == 220, "slot header layout");

typedef enum {
  NNPOL_SLOT_OK = 0,
  NNPOL_SLOT_EMPTY = 1,      /* erased flash (magic 0xFFFFFFFF) */
  NNPOL_SLOT_BAD_MAGIC = 2,
  NNPOL_SLOT_BAD_SIZE = 3,   /* header/total bytes inconsistent or > capacity */
  NNPOL_SLOT_BAD_DIMS = 4,   /* outside the interpreter's limits */
  NNPOL_SLOT_BAD_CRC = 5,
} nnpolSlotStatus_t;

/** zlib/binascii-compatible CRC-32 (reflected 0xEDB88320, init/xor
 * 0xFFFFFFFF), bytewise; ~10 ms for a 90 KB blob on the F405. `seed` is
 * a previous call's result for chaining, 0 to start. */
uint32_t nnpolCrc32(const uint8_t* data, uint32_t len, uint32_t seed);

/** Check a candidate blob (flash slot or file image) of at most
 * `capacityBytes`: magic, sizes, dims against the interpreter limits,
 * layer bookkeeping (payloadFloats == sum of kernel + bias sizes), CRC. */
nnpolSlotStatus_t nnpolSlotValidate(const void* base, uint32_t capacityBytes);

/** The weights of a validated blob: fp16 bit patterns, payloadFloats of them. */
static inline const uint16_t* nnpolSlotWeights(const nnpolSlotHeader_t* hdr)
{
  return (const uint16_t*)((const uint8_t*)hdr + hdr->headerBytes);
}

/** Bytes the payload occupies: 2 per weight, padded to a word. */
static inline uint32_t nnpolSlotPayloadBytes(uint32_t payloadFloats)
{
  return (payloadFloats * 2u + 3u) & ~3u;
}

/** Human-readable status, for logs and the host harness. */
const char* nnpolSlotStatusName(nnpolSlotStatus_t s);

#endif /* NNPOL_SLOT_H */
