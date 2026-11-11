/**
 * nnpol_action.c — clipped action -> per-motor rotor speed and PWM, the
 * rpm family's decode path transcribed from the HOST path in
 * ros_quad_sim2real (controller/policy/motor.py MotorPwmRacingPolicy.decode),
 * so the onboard and shadow commands agree to float precision:
 *
 *   action map   the checkpoint's (crazyflow_interface/_actions.py), selected
 *                by the slot header's mixer field: rpm_hover (two
 *                hover-anchored segments per motor) or rpm_direct (one line
 *                through [0, max_rpm])
 *   rpm_scale    the host's hover-point calibration knob (nnpol.rpmScale)
 *   rpm_to_pwm   vmotor2rpm inverted at vnom, np.rint, clip to [0, 65535]
 *
 * Every constant comes from the slot header (map anchors, the drone's
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

/* _RpmHoverAdapter: two hover-anchored segments, action 0 = exact hover. */
static float mixRpmAction(const nnpolSlotHeader_t* h, float a)
{
  if (a <= 0.0f) {
    return (a + 1.0f) * h->hoverRpm;
  }
  return h->hoverRpm + (h->maxRpm - h->hoverRpm) * a;
}

static void mixer(const nnpolSlotHeader_t* h, const float a[NNPOL_ACTION_DIM], float rpm[4])
{
  if (h->mixer == NNPOL_MIXER_RPM_DIRECT) {
    /* rpm_direct: one line through [0, max_rpm], no hover anchor. */
    for (int i = 0; i < 4; i++) {
      rpm[i] = h->maxRpm * (a[i] + 1.0f) * 0.5f;
    }
    return;
  }
  /* rpm_hover, per motor. */
  for (int i = 0; i < 4; i++) {
    rpm[i] = mixRpmAction(h, a[i]);
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
