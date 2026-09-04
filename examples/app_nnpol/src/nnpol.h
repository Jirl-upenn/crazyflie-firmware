/**
 * nnpol.h — shared declarations for the onboard-policy app (app_nnpol).
 *
 * The observation and action modules declared here are PURE C: math.h plus
 * the generated firmware_export.h/policy.h only, no firmware includes — so
 * test/test_policy_host.c compiles them with plain host gcc and checks
 * them bit-for-bit against the trainer's numpy reference
 * (policy_reference.npz) before anything flies. All the firmware glue
 * (params, logging, task scheduling, setpoint synthesis) lives in
 * nnpol_controller.c, which is the only file that cannot build on the host.
 *
 * Conventions, fixed by the trainer and the deployed ROS stack (see
 * mjc_dronetests/docs/onboard_inference_plan.md section 2):
 *   - quaternion is scalar-first [w, x, y, z], sign-normalized to w >= 0
 *     (the EKF stores x, y, z, w — the CALLER reorders; nnpolBuildObs only
 *     fixes the sign).
 *   - velocities are metres/second; the snapshot's is WORLD frame and the
 *     observation's is body frame (v_B = R^T v_W).
 *   - gyro is body-frame degrees/second (the firmware's own unit); the
 *     observation wants radians/second.
 *   - angles in the decoded setpoint are DEGREES (the legacy RPYT packet's
 *     unit); the action box is radians.
 */
#ifndef NNPOL_H
#define NNPOL_H

#include "firmware_export.h"

/** Everything the observation is a function of, snapshotted on one
 * stabilizer tick so the app task computes from a consistent state. */
typedef struct {
  float t;             /**< seconds along the curve: startPhase + elapsed */
  float off[3];        /**< curve translation (nnpol.offX/Y/Z, pushed at engage) */
  float pos[3];        /**< world position, m */
  float quat_wxyz[4];  /**< attitude quaternion, scalar first */
  float vel_w[3];      /**< world-frame linear velocity, m/s */
  float gyro_deg[3];   /**< body rates, deg/s */
} nnpolSnapshot_t;

/** One decoded attitude command, every stage kept for logging. */
typedef struct {
  float rollDeg;         /**< legacy attitude setpoint, degrees */
  float pitchDeg;        /**< degrees, TRAINER sign (the firmware Mellinger
                              negates internally — same convention the radio
                              path delivers, FIRMWARE_PITCH_SIGN = +1) */
  float yawDeg;          /**< degrees */
  float thrustN;         /**< decoded collective, newtons */
  float thrustPwm;       /**< 0..65535, the host-parity pre-decoder value */
  float thrustSetpoint;  /**< what setpoint->thrust gets: the crtp_commander_rpyt
                              clamp applied (0 below MIN_THRUST, capped at
                              MAX_THRUST) so radio and onboard paths agree */
} nnpolAttitudeCmd_t;

/** Reference position at time t (seconds), world frame, BEFORE the pushed
 * offset — the tasks/traj_lissajous figure-eight. */
void nnpolRef(float t, float out[3]);

/** The traj_lissajous/body observation, verbatim from the trainer:
 * [quat_wxyz, v_B, omega_B, e_B (NNPOL_N_SAMPLES x 3, row-major)]. */
void nnpolBuildObs(const nnpolSnapshot_t* snap, float obs[NNPOL_OBS_DIM]);

/** Clipped action in [-1, 1]^4 -> attitude/thrust command. `vnom` is the
 * battery voltage the thrust map assumes (nnpol.vnom, pushed at engage —
 * the same fixed voltage the host uses, for shadow parity). */
void nnpolDecodeAction(const float action[NNPOL_ACTION_DIM], float vnom,
                       nnpolAttitudeCmd_t* out);

#endif /* NNPOL_H */
