/**
 * nnpol.h — the pure-C core of the onboard-policy app: one policy step, from
 * the estimator's state to four motor values, for whatever policy slot is
 * selected (nnpol_slot.h) on whatever track was pushed (the engage context).
 *
 * The app is CHECKPOINT-AGNOSTIC: nothing here is generated. Dimensions,
 * the action map, the motor identification and the weights all come from the
 * selected slot's header at run time, so the same flash image flies every
 * exported checkpoint (build once, upload blobs over the radio, select by
 * hash). onboard-rpm flies the race task only: the observation is
 * hard-coded for it.
 *
 * Everything declared here is host-buildable, and the firmware's app task
 * runs a step through the same nnpolCoreStep the host library exports
 * (test/Makefile builds libnnpol_core.so; the ground station loads it and
 * compares it with its own policy before every launch). What is checked on
 * the host is therefore the flight code itself, not a transcription of it.
 * NNPOL_CORE_VERSION names the behaviour: bump it on any change to the blob
 * format, the context format or what a step computes. The vehicle reports it
 * (nnpol.coreVer) and the ground station refuses a library that differs.
 *
 * Conventions shared with the trainer (mjc_dronetests) and the ground
 * station (ros_quad_sim2real):
 *   - quaternion scalar-first (w, x, y, z);
 *   - velocities are metres/second; the state's is WORLD frame and the
 *     observation rotates it into the body frame;
 *   - body rates arrive in degrees/second (the firmware's gyro) and the
 *     observation converts them to rad/s;
 *   - rotor speeds are rpm, M1..M4;
 *   - the action is [-1, 1]^4, clipped exactly as the sim clips it.
 */
#ifndef NNPOL_H
#define NNPOL_H

#include <stdbool.h>
#include <stdint.h>

#include "nnpol_slot.h"

#define NNPOL_CORE_VERSION 1u

// --------------------------------------------------------------------------
// The track and the gate state
// --------------------------------------------------------------------------

/** One gate's frame (the trainer's tasks/race/geometry.py GateTrack): the
 * normal alone is ambiguous for a pitched gate, so the three axes travel
 * together. All unit vectors, world frame. */
typedef struct {
  float pos[3];      /**< centre, m */
  float normal[3];   /**< approach axis, pointing AGAINST the travel direction */
  float tangent[3];  /**< lateral axis of the opening */
  float up[3];       /**< vertical axis of the opening */
} nnpolGate_t;

/** The track as the host placed it for this run. */
typedef struct {
  uint8_t nGates;                     /**< gates in use, 1..NNPOL_MAX_GATES */
  uint8_t gate0;                      /**< first target gate */
  float gateSide;                     /**< side of the PHYSICAL opening, m: the
                                           window a crossing is scored against */
  nnpolGate_t gates[NNPOL_MAX_GATES];
} nnpolRaceTrack_t;

/** Gate state of a run (tasks/race/plugin.py RacePlugin: gate_idx and the
 * step's path). */
typedef struct {
  uint8_t gateIdx;     /**< the gate the policy is targeting */
  uint8_t havePrev;    /**< prevPos is valid (false on the first step of a run) */
  float prevPos[3];    /**< position at the previous policy step */
} nnpolRaceState_t;

/** The engage context: the track, as one checksummed blob the host writes
 * before it engages (see nnpolContextParse for the layout). */
#define NNPOL_CTX_MAGIC 0x3143504Eu   /* "NPC1" as a little-endian u32 */
#define NNPOL_CTX_HEADER_BYTES 16u
#define NNPOL_CTX_GATE_BYTES 48u      /* pos, normal, tangent, up: 12 floats */
#define NNPOL_CTX_MAX_BYTES (NNPOL_CTX_HEADER_BYTES + NNPOL_MAX_GATES * NNPOL_CTX_GATE_BYTES)

typedef enum {
  NNPOL_CTX_OK = 0,
  NNPOL_CTX_EMPTY = 1,       /* nothing written yet (magic 0) */
  NNPOL_CTX_BAD_MAGIC = 2,
  NNPOL_CTX_BAD_SIZE = 3,    /* totalBytes inconsistent with nGates or the buffer */
  NNPOL_CTX_BAD_CRC = 4,
  NNPOL_CTX_BAD_TRACK = 5,   /* no gates, too many, first gate out of range, a
                                non-positive side, or an axis that is not a unit vector */
} nnpolCtxStatus_t;

/**
 * Parse a context blob (little-endian):
 *
 *     u32 magic "NPC1" | u32 crc32 over bytes [8, totalBytes) |
 *     u16 totalBytes | u8 nGates | u8 gate0 | f32 gateSide |
 *     nGates x { f32 pos[3], normal[3], tangent[3], up[3] }
 *
 * On NNPOL_CTX_OK *track is the parsed track; otherwise it is untouched.
 * *crc, if not NULL, receives the blob's CRC field when the status is OK.
 */
nnpolCtxStatus_t nnpolContextParse(const uint8_t* buf, uint32_t len, nnpolRaceTrack_t* track,
                                   uint32_t* crc);
const char* nnpolCtxStatusName(nnpolCtxStatus_t s);

/** Start of a run: gate_idx = the track's first gate, no previous position
 * (RacePlugin.reset_task_state / the host's RaceV3.reset). */
void nnpolRaceReset(const nnpolRaceTrack_t* track, nnpolRaceState_t* race);

/**
 * RacePlugin.step's pass test on the step's path, as the host's
 * RaceV3._update_gate runs it: the target gate advances when the segment
 * from the previous position to this one crosses the gate's plane from the
 * approach side (x0 > 0, x1 <= 0) at a point inside the opening (|lateral|,
 * |vertical| < side / 2, in the gate's own frame - so pitched gates work).
 * Returns true on a pass.
 */
bool nnpolRaceUpdateGate(const nnpolRaceTrack_t* track, nnpolRaceState_t* race, const float pos[3]);

// --------------------------------------------------------------------------
// One policy step
// --------------------------------------------------------------------------

/** Everything a step is a function of besides the policy and the track,
 * sampled on one stabilizer tick. */
typedef struct {
  float pos[3];        /**< world position, m */
  float quat_wxyz[4];  /**< attitude quaternion, scalar first */
  float vel_w[3];      /**< world-frame linear velocity, m/s */
  float gyro_deg[3];   /**< body rates, deg/s */
  float rpm[4];        /**< measured rotor speeds M1..M4, rpm */
} nnpolState_t;

/** One decoded command, every stage kept for logging and host parity. */
typedef struct {
  float rpm[4];          /**< action map output x rpmScale, M1..M4 */
  float thrust[4];       /**< the force that rotor speed stands for, N per motor */
  float motorPwm[4];     /**< the 16-bit motor value: thrust / thrustMax x 65535,
                              rounded as the host rounds (np.rint), clipped */
} nnpolCmd_t;

/** A validated slot: its header and the weights behind it. */
typedef struct {
  const nnpolSlotHeader_t* hdr;
  const float* weights;
} nnpolPolicy_t;

/** The width the header's taskId builds: 21 [+ 4] for RACE_V3, 0 for any
 * other task. Equal to hdr->obsDim for a usable slot. */
int nnpolObsDim(const nnpolSlotHeader_t* hdr);

/** The race v3 / v3_motorobs observation into obs (at most
 * NNPOL_MAX_OBS_DIM floats). Returns the number written, which the caller
 * checks against hdr->obsDim; 0 for a taskId this build has no builder for,
 * or an empty track. The rotor-speed tail is normalized by the header's
 * hoverRpm, the run's own (the host's TaskObs.set_motor_rpm). */
int nnpolBuildObs(const nnpolSlotHeader_t* hdr, const nnpolRaceTrack_t* track, uint8_t gateIdx,
                  const nnpolState_t* state, float obs[NNPOL_MAX_OBS_DIM]);

/** Deterministic actor forward pass over the slot's weights: obs[obsDim]
 * -> act[4], the RAW (unsquashed) mean — the caller clips to [-1, 1]. */
void nnpolForward(const nnpolPolicy_t* p, const float* obs, float act[NNPOL_ACTION_DIM]);

/** Clipped action in [-1, 1]^4 -> per-motor rotor speed, force and motor
 * value. `thrustMax` is the force a motor value of 65535 stands for in this
 * build (THRUST_MAX, N per motor): the firmware reads the value as a force
 * and compensates for battery voltage itself. `rpmScale` is the hover-point
 * calibration knob (nnpol.rpmScale, the host's motor_pwm.rpm_scale). */
void nnpolDecodeAction(const nnpolSlotHeader_t* hdr, const float action[NNPOL_ACTION_DIM],
                       float thrustMax, float rpmScale, nnpolCmd_t* out);

/** A policy on a track, mid-run: what the app task owns, and what the host
 * library hands out as an opaque session. */
typedef struct {
  nnpolPolicy_t policy;      /**< hdr == NULL: no policy */
  nnpolRaceTrack_t track;    /**< nGates == 0: no track */
  nnpolRaceState_t race;
  float thrustMax;
  float rpmScale;
} nnpolCore_t;

/** Everything one step produced. */
typedef struct {
  float obs[NNPOL_MAX_OBS_DIM];
  float actRaw[NNPOL_ACTION_DIM];   /**< network output before the clip */
  float act[NNPOL_ACTION_DIM];      /**< clipped to [-1, 1] */
  nnpolCmd_t cmd;
  uint8_t gateIdx;                  /**< the gate the observation targeted */
  uint8_t passed;                   /**< this step's position crossed a gate */
} nnpolStepOut_t;

void nnpolCoreInit(nnpolCore_t* c);

/** Adopt a blob (a flash slot or a file image of at most capacityBytes).
 * Anything but NNPOL_SLOT_OK leaves the core without a policy. */
nnpolSlotStatus_t nnpolCoreSetPolicy(nnpolCore_t* c, const void* blob, uint32_t capacityBytes);

/** Adopt a track. */
void nnpolCoreSetTrack(nnpolCore_t* c, const nnpolRaceTrack_t* track);

/** Start a run: latch the scale factors and reset the gate state. */
void nnpolCoreEngage(nnpolCore_t* c, float thrustMax, float rpmScale);

/**
 * One policy step: advance the gate state with this step's position, build
 * the observation, run the network, clip, decode. Returns false - and
 * produces no command - without a policy and a track, or when the
 * observation the slot's task builds is not the width its network takes.
 */
bool nnpolCoreStep(nnpolCore_t* c, const nnpolState_t* state, nnpolStepOut_t* out);

#endif /* NNPOL_H */
