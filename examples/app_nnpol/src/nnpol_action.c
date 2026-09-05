/**
 * nnpol_action.c — clipped action -> command, along the decode path the
 * checkpoint's action_type selects (NNPOL_ACTION_KIND in the generated
 * firmware_export.h). Each path is a transcription of the deployed HOST
 * path in crazyflie-ros, so the onboard and shadow commands agree to
 * float precision:
 *
 *   MELLINGER  MellingerAttitudePolicy.decode: mid + a * half (rad, N; no
 *              yaw offset — the lissajous tasks never rotate their
 *              observation frame); angles to degrees
 *              (send_attitude_command + attitude_cmd_clbk,
 *              FIRMWARE_PITCH_SIGN = +1); collective through
 *              mellinger_thrust_to_pwm; the radio clamp.
 *   CTBR       BodyRatePolicy.decode: the same affine map on the body-rate
 *              box (rad/s, N); rates to degrees/second
 *              (send_ctbr_policy_command); the SAME thrust map and clamp
 *              — that path deliberately does not use the SE3 cascade's
 *              c1/c2/c3 fit.
 *   MOTOR      MotorPwmRacingPolicy.decode: the checkpoint's mixer
 *              (crazyflow_interface/_mixer.py, mirrored in
 *              controller_motor_policy.py) -> rpm x rpm_scale ->
 *              rpm_to_pwm (vmotor2rpm inverted at vnom, np.rint, clip).
 *
 *   thrust map (setpoint kinds)   F/4 through the positive root of
 *              rpm2thrust, then vmotor2rpm at the fixed engage voltage,
 *              then pwm16 = 65535 * V / vnom, rounded to nearest.
 *   radio clamp                   crtp_commander_rpyt.c: thrust below
 *              MIN_THRUST (1000) becomes 0, above MAX_THRUST (60000) is
 *              capped. Applied here too so the synthesized setpoint equals
 *              what the same command would have produced arriving by radio.
 *
 * What is NOT here: cflib's pitch negation on the wire. ch[] carries the
 * host's numbers; nnpol_controller.c applies the negation where the radio
 * path would (see nnpol.h nnpolCmd_t).
 *
 * Pure C: host-testable, no firmware includes.
 */
#include <math.h>
#include <string.h>

#include "nnpol.h"

/* crtp_commander_rpyt.c's legacy thrust window, mirrored (values, not
 * included — that header is not host-buildable). */
#define NNPOL_LEGACY_MIN_THRUST 1000.0f
#define NNPOL_LEGACY_MAX_THRUST 60000.0f

static float clip1(float v)
{
  if (v < -1.0f) { return -1.0f; }
  if (v > 1.0f) { return 1.0f; }
  return v;
}

/* Per-motor rpm -> 16-bit ratio at the pushed nominal voltage: the host's
 * rpm_to_pwm (controller_motor_policy.py), np.rint then clip. */
static float rpmToPwm(float rpm, float vnom)
{
  const float vMotor = (rpm - NNPOL_VMOTOR2RPM_K0) / NNPOL_VMOTOR2RPM_K1;
  float pwm = rintf(65535.0f * vMotor / vnom);
  if (pwm < 0.0f) { pwm = 0.0f; }
  if (pwm > 65535.0f) { pwm = 65535.0f; }
  return pwm;
}

#if NNPOL_ACTION_KIND == NNPOL_ACTION_KIND_MOTOR

#if NNPOL_MIXER != NNPOL_MIXER_RPM_DIRECT
/* _mixer.mix_rpm_action: two hover-anchored segments, action 0 = exact hover. */
static float mixRpmAction(float a)
{
  if (a <= 0.0f) {
    return (a + 1.0f) * NNPOL_HOVER_RPM;
  }
  return NNPOL_HOVER_RPM + (NNPOL_MAX_RPM - NNPOL_HOVER_RPM) * a;
}
#endif

static void mixer(const float a[NNPOL_ACTION_DIM], float rpm[4])
{
#if NNPOL_MIXER == NNPOL_MIXER_ATTITUDE
  /* _mixer.mix_attitude_rpm: [thrust_norm, roll, pitch, yaw_rate] ->
   * hover-anchored collective plus fixed-gain differentials, clipped. */
  const float collective = mixRpmAction(a[0]);
  const float scale = NNPOL_DIFFERENTIAL_FRAC * NNPOL_HOVER_RPM;
  const float roll = a[1], pitch = a[2], yawRate = a[3];
  rpm[0] = collective + roll * scale - pitch * scale - yawRate * scale;
  rpm[1] = collective - roll * scale - pitch * scale + yawRate * scale;
  rpm[2] = collective - roll * scale + pitch * scale - yawRate * scale;
  rpm[3] = collective + roll * scale + pitch * scale + yawRate * scale;
  for (int i = 0; i < 4; i++) {
    if (rpm[i] < 0.0f) { rpm[i] = 0.0f; }
    if (rpm[i] > NNPOL_MAX_RPM) { rpm[i] = NNPOL_MAX_RPM; }
  }
#elif NNPOL_MIXER == NNPOL_MIXER_RPM_DIRECT
  /* _mixer.rpm_direct: one line through [0, max_rpm], no hover anchor. */
  for (int i = 0; i < 4; i++) {
    rpm[i] = NNPOL_MAX_RPM * (a[i] + 1.0f) * 0.5f;
  }
#else
  /* _mixer.mix_rpm_action per motor. */
  for (int i = 0; i < 4; i++) {
    rpm[i] = mixRpmAction(a[i]);
  }
#endif
}

#else /* setpoint kinds */

static const float kActionMid[NNPOL_ACTION_DIM] = NNPOL_ACTION_MID;
static const float kActionHalf[NNPOL_ACTION_DIM] = NNPOL_ACTION_HALF;

/* Collective newtons -> legacy thrust setpoint: the host's
 * mellinger_thrust_to_pwm. F/4 -> per-motor rpm as the positive root of
 * a2*rpm^2 + a1*rpm + (a0 - F/4) = 0 (a thrust below what the curve can
 * produce has no real root, so it maps to zero rpm rather than
 * misbehaving), then rpm -> V -> pwm16, rounded as the host rounds. */
static float collectiveToPwm(float thrustN, float vnom)
{
  const float a0 = NNPOL_RPM2THRUST_A0;
  const float a1 = NNPOL_RPM2THRUST_A1;
  const float a2 = NNPOL_RPM2THRUST_A2;
  const float f = (thrustN > 0.0f ? thrustN : 0.0f) * 0.25f;
  const float disc = a1 * a1 - 4.0f * a2 * (a0 - f);
  const float rpm = disc < 0.0f ? 0.0f : (-a1 + sqrtf(disc)) / (2.0f * a2);
  return rpmToPwm(rpm, vnom);
}

#endif

void nnpolDecodeAction(const float action[NNPOL_ACTION_DIM], float vnom,
                       float rpmScale, nnpolCmd_t* out)
{
  memset(out, 0, sizeof(*out));
  float a[NNPOL_ACTION_DIM];
  for (int i = 0; i < NNPOL_ACTION_DIM; i++) {
    a[i] = clip1(action[i]);
  }

#if NNPOL_ACTION_KIND == NNPOL_ACTION_KIND_MOTOR
  float rpm[4];
  mixer(a, rpm);
  for (int i = 0; i < 4; i++) {
    out->rpm[i] = rpm[i] * rpmScale;
    out->motorPwm[i] = rpmToPwm(out->rpm[i], vnom);
  }
#else
  (void)rpmScale;
  const float RAD2DEG = 180.0f / (float)M_PI;
  float sp[NNPOL_ACTION_DIM];
  for (int i = 0; i < NNPOL_ACTION_DIM; i++) {
    sp[i] = kActionMid[i] + a[i] * kActionHalf[i];
  }
  /* MELLINGER: rad -> deg; CTBR: rad/s -> deg/s. Same factor, host sign. */
  out->ch[0] = sp[0] * RAD2DEG;
  out->ch[1] = sp[1] * RAD2DEG;
  out->ch[2] = sp[2] * RAD2DEG;
  out->thrustN = sp[3];
  out->thrustPwm = collectiveToPwm(sp[3], vnom);
  if (out->thrustPwm < NNPOL_LEGACY_MIN_THRUST) {
    out->thrustSetpoint = 0.0f;
  } else if (out->thrustPwm > NNPOL_LEGACY_MAX_THRUST) {
    out->thrustSetpoint = NNPOL_LEGACY_MAX_THRUST;
  } else {
    out->thrustSetpoint = out->thrustPwm;
  }
#endif
}
