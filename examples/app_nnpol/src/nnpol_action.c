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
 *   thrust       the rotor speed as a force, by the drone's thrust curve
 *   motor value  that force as a fraction of the build's THRUST_MAX
 *
 * THE MOTOR VALUE IS A FORCE, NOT A PWM RATIO. With
 * CONFIG_ENABLE_THRUST_BAT_COMPENSATED (required here) the firmware reads
 * every 16-bit motor value as value / 65535 x THRUST_MAX newtons and turns
 * it into a motor voltage at the measured battery voltage. Sending a voltage
 * fraction instead - what the host and this app did before - inverts the
 * motor curve twice: delivered thrust came out near 0.53 x commanded +
 * 0.17 N on the October flights. Sending the force makes the delivered
 * thrust independent of battery voltage, as training assumes.
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

/* Per-motor rotor speed -> force, by the drone's thrust curve (crazyflow
 * rpm2thrust: f = a0 + a1 rpm + a2 rpm^2, the curve the simulator turns a
 * commanded rotor speed into force with). */
static float rpmToThrust(const nnpolSlotHeader_t* h, float rpm)
{
  const float f = h->rpm2thrust[0] + (h->rpm2thrust[1] + h->rpm2thrust[2] * rpm) * rpm;
  return f > 0.0f ? f : 0.0f;
}

/* Force -> the 16-bit motor value: the fraction of thrustMax, np.rint, clip.
 * A force beyond thrustMax saturates here, as the motor would. */
static float thrustToMotor(float thrust, float thrustMax)
{
  float value = rintf(65535.0f * thrust / thrustMax);
  if (value < 0.0f) { value = 0.0f; }
  if (value > 65535.0f) { value = 65535.0f; }
  return value;
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
                       float thrustMax, float rpmScale, nnpolCmd_t* out)
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
    out->thrust[i] = rpmToThrust(h, out->rpm[i]);
    out->motorPwm[i] = thrustToMotor(out->thrust[i], thrustMax);
  }
}
