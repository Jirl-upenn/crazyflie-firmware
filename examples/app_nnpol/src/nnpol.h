/**
 * nnpol.h — the pure-C contract of the onboard-policy app: the state
 * snapshot the observation is built from, the decoded command, and the
 * three functions that turn a snapshot into a command for whatever
 * policy slot is selected (nnpol_slot.h). onboard-rpm flies the race task
 * only: the observation is hard-coded for it.
 *
 * The app is CHECKPOINT-AGNOSTIC: nothing here is generated. Dimensions,
 * the decode path, the action box or mixer, the task constants and the
 * weights all come from the selected slot's header at run time, so the
 * same flash image flies every exported checkpoint (build once, upload
 * blobs over the radio, select by hash).
 *
 * Everything declared here is host-buildable (test/run_host_parity.py
 * compiles these modules with plain gcc against a policy_slot.bin), which
 * is what makes the flight code itself, not a transcription of it, the
 * thing that is checked.
 *
 * Conventions shared with the trainer (mjc_dronetests crazyflow_interface)
 * and the ground station (crazyflie-ros):
 *   - quaternion scalar-first (w, x, y, z), sign-normalized to w >= 0;
 *   - velocities are metres/second; the snapshot's is WORLD frame and the
 *     observation rotates it into the body frame;
 *   - body rates arrive in degrees/second (the firmware's gyro) and the
 *     observation converts them to rad/s;
 *   - the action is [-1, 1]^4, clipped exactly as the sim clips it.
 */
#ifndef NNPOL_H
#define NNPOL_H

#include <stdbool.h>
#include <stdint.h>

#include "nnpol_slot.h"

/** The race track as the host placed it for this run (pushed before
 * engage), fixed for as long as the policy is engaged. */
typedef struct {
  uint8_t nGates;                         /**< gates in use, <= NNPOL_MAX_GATES */
  float gates[NNPOL_MAX_GATES][3];        /**< world positions, m */
  float normals[NNPOL_MAX_GATES][3];      /**< unit approach axes, against travel, horizontal */
} nnpolRaceTrack_t;

/** Everything the observation is a function of besides the track,
 * snapshotted on one stabilizer tick so the app task computes from a
 * consistent state. */
typedef struct {
  float t;             /**< seconds since engage */
  float pos[3];        /**< world position, m */
  float quat_wxyz[4];  /**< attitude quaternion, scalar first, mocap/EKF frame */
  float vel_w[3];      /**< world-frame linear velocity, m/s */
  float gyro_deg[3];   /**< body rates, deg/s */
  float rpm[4];        /**< measured rotor speeds M1..M4 / NOMINAL_HOVER_RPM,
                            held through ESC dropouts (the host's
                            set_motor_rpm hold); 1.0 = hover before the
                            first reading */
  uint8_t gateIdx;     /**< the gate the policy is targeting on this tick */
} nnpolSnapshot_t;

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
 * other task (this branch flies the race task only). Equal to hdr->obsDim
 * for a usable slot. */
int nnpolObsDim(const nnpolSlotHeader_t* hdr);

/** The gate-crossing test of RacePlugin.step, run once per policy tick
 * with the current position; advances *gateIdx and maintains *prevGateX
 * (re-armed to 1.0 on a crossing, as at reset). */
void nnpolRaceUpdateGate(const nnpolRaceTrack_t* track, float gateSide, const float pos[3],
                         uint8_t* gateIdx, float* prevGateX);

/** The race v3 / v3_motorobs observation into obs (at most
 * NNPOL_MAX_OBS_DIM floats). Returns the number written, which the caller
 * checks against hdr->obsDim; 0 for a taskId this build has no builder
 * for, or an empty track. */
int nnpolBuildObs(const nnpolSlotHeader_t* hdr, const nnpolRaceTrack_t* track,
                  const nnpolSnapshot_t* snap, float obs[NNPOL_MAX_OBS_DIM]);

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

#endif /* NNPOL_H */
