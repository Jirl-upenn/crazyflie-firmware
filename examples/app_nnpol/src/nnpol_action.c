/**
 * nnpol_action.c — clipped action -> per-motor rotor speed and PWM, the
 * rpm family's decode path transcribed from the deployed HOST path in
 * crazyflie-ros (MotorPwmRacingPolicy.decode), so the onboard and shadow
 * commands agree to float precision:
 *
 *   mixer        the checkpoint's (crazyflow_interface/_mixer.py, mirrored in
 *                controller_motor_policy.py), selected by the slot header's
 *                mixer field: attitude (hover-anchored collective plus
 *                fixed-gain differentials, clipped), rpm (two hover-anchored
 *                segments per motor), rpm_direct (one line through
 *                [0, max_rpm])
 *   rpm_scale    the host's hover-point calibration knob (nnpol.rpmScale)
 *   rpm_to_pwm   vmotor2rpm inverted at vnom, np.rint, clip to [0, 65535]
 *
 * onboard_only: this branch flies the rpm family only. The setpoint kinds
 * (Mellinger angles, ctbr rates) and the inner controllers they need live
 * on the nnpol branch; a slot of another kind fails validation here.
 *
 * Every constant comes from the slot header (mixer anchors, the drone's
 * motor identification), so one build serves every rpm-family checkpoint.
 * Pure C: host-testable, no firmware includes.
 */
#include <math.h>
#include <string.h>

#include "nnpol.h"

static float clip1(float v)
{
  if (v < -1.0f) { return -1.0f; }
  if (v > 1.0f) { return 1.0f; }
  return v;
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

void nnpolDecodeAction(const nnpolSlotHeader_t* h, const float action[NNPOL_ACTION_DIM],
                       float vnom, float rpmScale, nnpolCmd_t* out)
{
  memset(out, 0, sizeof(*out));
  float a[NNPOL_ACTION_DIM];
  for (int i = 0; i < NNPOL_ACTION_DIM; i++) {
    a[i] = clip1(action[i]);
  }
  float rpm[4];
  mixer(h, a, rpm);
  for (int i = 0; i < 4; i++) {
    out->rpm[i] = rpm[i] * rpmScale;
    out->motorPwm[i] = rpmToPwm(h, out->rpm[i], vnom);
  }
}
