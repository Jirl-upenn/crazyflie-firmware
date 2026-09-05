/**
 * nnpol.h — the pure-C contract of the onboard-policy app: the state
 * snapshot the observation is built from, the decoded command, and the
 * three functions that turn a snapshot into a command for whatever
 * policy slot is selected (nnpol_slot.h).
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
 *   - the action is [-1, 1]^4, clipped exactly as the sim clips it;
 *   - the run's yaw anchor (crazyflie-ros TrajLissajousBody.anchor): the
 *     engage heading yaw0 is the run's yaw zero. The curve is yawed by it
 *     about the engage point, the observed attitude is relative to it,
 *     and a commanded yaw ANGLE gets it added back (rates need nothing).
 */
#ifndef NNPOL_H
#define NNPOL_H

#include <stdbool.h>
#include <stdint.h>

#include "nnpol_slot.h"

/** RACE_V3: the track as the host pinned it for this run (nnpol.g*,
 * pushed at engage) and the gate the policy is targeting, advanced by the
 * controller with nnpolRaceUpdateGate on every policy tick. */
typedef struct {
  uint8_t nGates;                         /**< gates in use, <= NNPOL_MAX_GATES */
  uint8_t gateIdx;                        /**< current target gate */
  float gates[NNPOL_MAX_GATES][3];        /**< world positions, m */
  float normals[NNPOL_MAX_GATES][3];      /**< unit approach axes, against travel, horizontal */
} nnpolRaceTrack_t;

/** Everything the observation is a function of, snapshotted on one
 * stabilizer tick so the app task computes from a consistent state. */
typedef struct {
  float t;             /**< seconds along the curve: startPhase + elapsed */
  float off[3];        /**< curve translation (nnpol.offX/Y/Z, pushed at engage) */
  float pos[3];        /**< world position, m */
  float quat_wxyz[4];  /**< attitude quaternion, scalar first, mocap/EKF frame */
  float vel_w[3];      /**< world-frame linear velocity, m/s */
  float gyro_deg[3];   /**< body rates, deg/s */
  float yaw0;          /**< yaw anchor, rad (nnpol.yaw0, pushed at engage) */
  float curve[NNPOL_CURVE_FLOATS]; /**< EASY_BODY: this run's curve (nnpol.c*,
                                    pushed at engage): amp(3), omega(3),
                                    phase(3), center(3), yaw */
  float rpm[4];        /**< measured rotor speeds M1..M4 / NOMINAL_HOVER_RPM,
                            held through ESC dropouts (the host's
                            set_motor_rpm hold); 1.0 = hover before the
                            first reading */
  nnpolRaceTrack_t race; /**< RACE_V3: the pinned track and the target gate */
} nnpolSnapshot_t;

/** One decoded command, every stage kept for logging and host parity.
 * Which fields mean anything depends on the slot's actionKind; the rest
 * are 0.
 *
 * ch[3] are the three non-thrust channels in the HOST's wire units and
 * sign — exactly what crazyflie-ros hands cflib's send_setpoint:
 *   MELLINGER  [roll, pitch, yaw] degrees (attitude_cmd_clbk,
 *              FIRMWARE_PITCH_SIGN = +1), the yaw already anchored back
 *              into the mocap frame (MellingerAttitudePolicy.decode)
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

/** A validated slot: its header and the weights behind it. */
typedef struct {
  const nnpolSlotHeader_t* hdr;
  const uint16_t* weights;   /**< fp16, see nnpol_slot.h */
} nnpolPolicy_t;

/** Reference position at time t (seconds), world frame, BEFORE the yaw
 * anchor and the pushed offset: the fixed figure-eight from the header's
 * constants (LISSAJOUS_BODY) or this run's curve from the snapshot
 * (EASY_BODY). */
void nnpolRef(const nnpolSlotHeader_t* hdr, const nnpolSnapshot_t* snap, float t, float out[3]);

/** The width the header's taskId/task[] build, 0 when this build has no
 * builder for it (or it would exceed NNPOL_MAX_OBS_DIM). Equal to
 * hdr->obsDim for a usable slot. */
int nnpolObsDim(const nnpolSlotHeader_t* hdr);

/** RACE_V3 (nnpol_obs_race.c): the gate-crossing test of RacePlugin.step,
 * run once per policy tick with the current position; advances *gateIdx
 * and maintains *prevGateX (re-armed to 1.0 on a crossing, as at reset). */
void nnpolRaceUpdateGate(const nnpolRaceTrack_t* track, float gateSide, const float pos[3],
                         uint8_t* gateIdx, float* prevGateX);
int nnpolRaceObsDim(const nnpolSlotHeader_t* hdr);
int nnpolRaceBuildObs(const nnpolSlotHeader_t* hdr, const nnpolSnapshot_t* snap,
                      float obs[NNPOL_MAX_OBS_DIM]);

/** The observation the header's taskId names, into obs (at most
 * NNPOL_MAX_OBS_DIM floats). Returns the number written, which the caller
 * checks against hdr->obsDim; 0 for a taskId this build has no builder
 * for. */
int nnpolBuildObs(const nnpolSlotHeader_t* hdr, const nnpolSnapshot_t* snap,
                  float obs[NNPOL_MAX_OBS_DIM]);

/** Deterministic actor forward pass over the slot's weights: obs[obsDim]
 * -> act[4], the RAW (unsquashed) mean — the caller clips to [-1, 1]. */
void nnpolForward(const nnpolPolicy_t* p, const float* obs, float act[NNPOL_ACTION_DIM]);

/** Clipped action in [-1, 1]^4 -> command, along the header's decode
 * path. `vnom` is the battery voltage every thrust/rpm-to-PWM map assumes
 * (nnpol.vnom, pushed at engage — the same fixed voltage the host uses,
 * for shadow parity); `rpmScale` is the MOTOR kind's hover-point
 * calibration knob (nnpol.rpmScale, the host's motor_pwm.rpm_scale);
 * `yaw0` is the run's yaw anchor, added back to a MELLINGER yaw command.
 * Each is ignored by the kinds it does not concern. */
void nnpolDecodeAction(const nnpolSlotHeader_t* hdr, const float action[NNPOL_ACTION_DIM],
                       float vnom, float rpmScale, float yaw0, nnpolCmd_t* out);

#endif /* NNPOL_H */
