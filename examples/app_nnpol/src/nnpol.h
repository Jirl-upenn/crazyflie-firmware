/**
 * nnpol.h — the pure-C contract of the onboard-policy app: the state
 * snapshot the observation is built from, the decoded command, and the
 * decode path a checkpoint's action_type selects.
 *
 * Everything declared here is host-buildable (test/run_host_parity.py
 * compiles the observation and action modules with plain gcc against the
 * SAME generated policy.c the firmware links), which is what makes the
 * flight code itself, not a transcription of it, the thing that is checked.
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

#include "firmware_export.h"

/* Decode paths, numbered as export_policy_c.ACTION_KIND_* (the generated
 * firmware_export.h carries the checkpoint's as NNPOL_ACTION_KIND). */
#define NNPOL_ACTION_KIND_MELLINGER 0  /* attitude box -> firmware Mellinger */
#define NNPOL_ACTION_KIND_CTBR 1       /* body-rate box -> firmware rate PID */
#define NNPOL_ACTION_KIND_MOTOR 2      /* mixer -> per-motor PWM, no onboard loop */

/* Mixers for the MOTOR kind, numbered as export_policy_c.MIXER_FOR_TYPE. */
#define NNPOL_MIXER_ATTITUDE 0     /* action_type "attitude" */
#define NNPOL_MIXER_RPM 1          /* action_type "rpm" */
#define NNPOL_MIXER_RPM_DIRECT 2   /* action_type "rpm_direct" */

#ifndef NNPOL_ACTION_KIND
#error "firmware_export.h carries no NNPOL_ACTION_KIND - regenerate it with export_policy_c.py"
#endif
#if NNPOL_ACTION_KIND == NNPOL_ACTION_KIND_MOTOR && !defined(NNPOL_MIXER)
#error "a MOTOR-kind export needs NNPOL_MIXER - regenerate with export_policy_c.py"
#endif

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

/** One decoded command, every stage kept for logging and host parity.
 * Which fields mean anything depends on NNPOL_ACTION_KIND; the rest are 0.
 *
 * ch[3] are the three non-thrust channels in the HOST's wire units and
 * sign — exactly what crazyflie-ros hands cflib's send_setpoint:
 *   MELLINGER  [roll, pitch, yaw] degrees (attitude_cmd_clbk,
 *              FIRMWARE_PITCH_SIGN = +1)
 *   CTBR       [roll_rate, pitch_rate, yaw_rate] degrees/second
 *              (send_ctbr_policy_command; the driver's own yaw negation
 *              and the firmware's undo it, so the sign is the trainer's)
 * The controller then applies what the radio path applies between the
 * host and setpoint_t — cflib negates pitch on the wire — when it
 * synthesizes the setpoint, so onboard and radio flights agree. */
typedef struct {
  float ch[3];
  float thrustN;         /**< setpoint kinds: decoded collective, newtons */
  float thrustPwm;       /**< setpoint kinds: 0..65535, the host's pre-clamp value */
  float thrustSetpoint;  /**< setpoint kinds: what setpoint->thrust gets — the
                              crtp_commander_rpyt clamp applied (0 below
                              MIN_THRUST, capped at MAX_THRUST) */
  float rpm[4];          /**< MOTOR: mixer output x rpmScale, M1..M4 */
  float motorPwm[4];     /**< MOTOR: 0..65535 per motor, rounded as the host's
                              rpm_to_pwm rounds (np.rint) */
} nnpolCmd_t;

/** Reference position at time t (seconds), world frame, BEFORE the pushed
 * offset — the tasks/traj_lissajous figure-eight. */
void nnpolRef(float t, float out[3]);

/** The traj_lissajous/body observation, verbatim from the trainer:
 * [quat_wxyz, v_B, omega_B, e_B (NNPOL_N_SAMPLES x 3, row-major)]. */
void nnpolBuildObs(const nnpolSnapshot_t* snap, float obs[NNPOL_OBS_DIM]);

/** Clipped action in [-1, 1]^4 -> command, along the checkpoint's decode
 * path. `vnom` is the battery voltage every thrust/rpm-to-PWM map assumes
 * (nnpol.vnom, pushed at engage — the same fixed voltage the host uses,
 * for shadow parity); `rpmScale` is the MOTOR kind's hover-point
 * calibration knob (nnpol.rpmScale, the host's motor_pwm.rpm_scale),
 * ignored by the setpoint kinds. */
void nnpolDecodeAction(const float action[NNPOL_ACTION_DIM], float vnom,
                       float rpmScale, nnpolCmd_t* out);

#endif /* NNPOL_H */
