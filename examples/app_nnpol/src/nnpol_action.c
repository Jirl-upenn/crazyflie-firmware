/**
 * nnpol_action.c — clipped action -> command, along the decode path the
 * slot header's actionKind selects. Each path is a transcription of the
 * deployed HOST path in crazyflie-ros, so the onboard and shadow commands
 * agree to float precision:
 *
 *   MELLINGER  MellingerAttitudePolicy.decode: mid + a * half (rad, N),
 *              the yaw ANGLE anchored back into the mocap frame (+ yaw0,
 *              wrapped to [-pi, pi) — only when yaw0 != 0, exactly as the
 *              host's `if offset:`); angles to degrees
 *              (send_attitude_command + attitude_cmd_clbk,
 *              FIRMWARE_PITCH_SIGN = +1); collective through
 *              mellinger_thrust_to_pwm; the radio clamp.
 *   CTBR       BodyRatePolicy.decode: the same affine map on the body-rate
 *              box (rad/s, N), no anchor (rates are body-frame); rates to
 *              degrees/second (send_ctbr_policy_command); the SAME thrust
 *              map and clamp — that path deliberately does not use the
 *              SE3 cascade's c1/c2/c3 fit.
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
 * Every constant comes from the slot header (box, mixer anchors, the
 * drone's thrust identification), so one build serves every checkpoint.
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

/* Python's (v + pi) % (2 pi) - pi: the modulo is non-negative, so the
 * result lies in [-pi, pi). */
static float wrapPi(float v)
{
  const float twoPi = 2.0f * (float)M_PI;
  float r = fmodf(v + (float)M_PI, twoPi);
  if (r < 0.0f) { r += twoPi; }
  return r - (float)M_PI;
}

/* Per-motor rpm -> 16-bit ratio at the pushed nominal voltage: the host's
 * rpm_to_pwm (controller_motor_policy.py), np.rint then clip. */
static float rpmToPwm(const nnpolSlotHeader_t* h, float rpm, float vnom)
{
  const float vMotor = (rpm - h->vmotor2rpm[0]) / h->vmotor2rpm[1];
  float pwm = rintf(65535.0f * vMotor / vnom);
  if (pwm < 0.0f) { pwm = 0.0f; }
  if (pwm > 65535.0f) { pwm = 65535.0f; }
  return pwm;
}

/* _mixer.mix_rpm_action: two hover-anchored segments, action 0 = exact hover. */
static float mixRpmAction(const nnpolSlotHeader_t* h, float a)
{
  if (a <= 0.0f) {
    return (a + 1.0f) * h->hoverRpm;
  }
  return h->hoverRpm + (h->maxRpm - h->hoverRpm) * a;
}

static void mixer(const nnpolSlotHeader_t* h, const float a[NNPOL_ACTION_DIM], float rpm[4])
{
  switch (h->mixer) {
    case NNPOL_MIXER_ATTITUDE: {
      /* _mixer.mix_attitude_rpm: [thrust_norm, roll, pitch, yaw_rate] ->
       * hover-anchored collective plus fixed-gain differentials, clipped. */
      const float collective = mixRpmAction(h, a[0]);
      const float scale = h->differentialFrac * h->hoverRpm;
      const float roll = a[1], pitch = a[2], yawRate = a[3];
      rpm[0] = collective + roll * scale - pitch * scale - yawRate * scale;
      rpm[1] = collective - roll * scale - pitch * scale + yawRate * scale;
      rpm[2] = collective - roll * scale + pitch * scale - yawRate * scale;
      rpm[3] = collective + roll * scale + pitch * scale + yawRate * scale;
      for (int i = 0; i < 4; i++) {
        if (rpm[i] < 0.0f) { rpm[i] = 0.0f; }
        if (rpm[i] > h->maxRpm) { rpm[i] = h->maxRpm; }
      }
      break;
    }
    case NNPOL_MIXER_RPM_DIRECT:
      /* _mixer.rpm_direct: one line through [0, max_rpm], no hover anchor. */
      for (int i = 0; i < 4; i++) {
        rpm[i] = h->maxRpm * (a[i] + 1.0f) * 0.5f;
      }
      break;
    default:
      /* _mixer.mix_rpm_action per motor. */
      for (int i = 0; i < 4; i++) {
        rpm[i] = mixRpmAction(h, a[i]);
      }
      break;
  }
}

/* Collective newtons -> legacy thrust setpoint: the host's
 * mellinger_thrust_to_pwm. F/4 -> per-motor rpm as the positive root of
 * a2*rpm^2 + a1*rpm + (a0 - F/4) = 0 (a thrust below what the curve can
 * produce has no real root, so it maps to zero rpm rather than
 * misbehaving), then rpm -> V -> pwm16, rounded as the host rounds. */
static float collectiveToPwm(const nnpolSlotHeader_t* h, float thrustN, float vnom)
{
  const float a0 = h->rpm2thrust[0];
  const float a1 = h->rpm2thrust[1];
  const float a2 = h->rpm2thrust[2];
  const float f = (thrustN > 0.0f ? thrustN : 0.0f) * 0.25f;
  const float disc = a1 * a1 - 4.0f * a2 * (a0 - f);
  const float rpm = disc < 0.0f ? 0.0f : (-a1 + sqrtf(disc)) / (2.0f * a2);
  return rpmToPwm(h, rpm, vnom);
}

void nnpolDecodeAction(const nnpolSlotHeader_t* h, const float action[NNPOL_ACTION_DIM],
                       float vnom, float rpmScale, float yaw0, nnpolCmd_t* out)
{
  memset(out, 0, sizeof(*out));
  float a[NNPOL_ACTION_DIM];
  for (int i = 0; i < NNPOL_ACTION_DIM; i++) {
    a[i] = clip1(action[i]);
  }

  if (h->actionKind == NNPOL_ACTION_KIND_MOTOR) {
    float rpm[4];
    mixer(h, a, rpm);
    for (int i = 0; i < 4; i++) {
      out->rpm[i] = rpm[i] * rpmScale;
      out->motorPwm[i] = rpmToPwm(h, out->rpm[i], vnom);
    }
    return;
  }

  const float RAD2DEG = 180.0f / (float)M_PI;
  float sp[NNPOL_ACTION_DIM];
  for (int i = 0; i < NNPOL_ACTION_DIM; i++) {
    sp[i] = h->actionMid[i] + a[i] * h->actionHalf[i];
  }
  if (h->actionKind == NNPOL_ACTION_KIND_MELLINGER && yaw0 != 0.0f) {
    /* Roll and pitch are body-frame and survive a change of world heading
     * untouched; the yaw is an absolute heading in the observation's
     * (anchored) frame and is rotated back into the mocap frame. */
    sp[2] = wrapPi(sp[2] + yaw0);
  }
  /* MELLINGER: rad -> deg; CTBR: rad/s -> deg/s. Same factor, host sign. */
  out->ch[0] = sp[0] * RAD2DEG;
  out->ch[1] = sp[1] * RAD2DEG;
  out->ch[2] = sp[2] * RAD2DEG;
  out->thrustN = sp[3];
  out->thrustPwm = collectiveToPwm(h, sp[3], vnom);
  if (out->thrustPwm < NNPOL_LEGACY_MIN_THRUST) {
    out->thrustSetpoint = 0.0f;
  } else if (out->thrustPwm > NNPOL_LEGACY_MAX_THRUST) {
    out->thrustSetpoint = NNPOL_LEGACY_MAX_THRUST;
  } else {
    out->thrustSetpoint = out->thrustPwm;
  }
}
