/**
 * nnpol_action_mellinger.c — clipped action -> legacy attitude setpoint,
 * transcribed from the deployed host path so the onboard and shadow
 * commands agree to float precision:
 *
 *   box decode     crazyflie-ros MellingerAttitudePolicy.decode
 *                  (mid + a * half, radians/newtons; no yaw offset — the
 *                  lissajous tasks never rotate their observation frame,
 *                  world_yaw_offset is always 0)
 *   angle units    controller_utils.send_attitude_command +
 *                  crazyradio_driver attitude_cmd_clbk: degrees, pitch
 *                  MULTIPLIED BY FIRMWARE_PITCH_SIGN = +1 (bench-measured;
 *                  the firmware Mellinger's internal negation is a frame
 *                  conversion, not an interface convention — see the long
 *                  comment on FIRMWARE_PITCH_SIGN in the driver)
 *   thrust map     controller_utils.mellinger_thrust_to_pwm: F/4 through
 *                  the positive root of rpm2thrust, then vmotor2rpm at the
 *                  fixed engage voltage, then pwm16 = 65535 * V / vnom
 *   radio clamp    crtp_commander_rpyt.c: thrust below MIN_THRUST (1000)
 *                  becomes 0, above MAX_THRUST (60000) is capped. Applied
 *                  here too so the synthesized setpoint equals what the
 *                  same command would have produced arriving by radio.
 *
 * Pure C: host-testable, no firmware includes.
 */
#include <math.h>

#include "nnpol.h"

/* crtp_commander_rpyt.c's legacy thrust window, mirrored (values, not
 * included — that header is not host-buildable). */
#define NNPOL_LEGACY_MIN_THRUST 1000.0f
#define NNPOL_LEGACY_MAX_THRUST 60000.0f

static const float kActionMid[NNPOL_ACTION_DIM] = NNPOL_ACTION_MID;
static const float kActionHalf[NNPOL_ACTION_DIM] = NNPOL_ACTION_HALF;

static float clip1(float v)
{
  if (v < -1.0f) { return -1.0f; }
  if (v > 1.0f) { return 1.0f; }
  return v;
}

void nnpolDecodeAction(const float action[NNPOL_ACTION_DIM], float vnom,
                       nnpolAttitudeCmd_t* out)
{
  const float RAD2DEG = 180.0f / (float)M_PI;

  float sp[NNPOL_ACTION_DIM];
  for (int i = 0; i < NNPOL_ACTION_DIM; i++) {
    sp[i] = kActionMid[i] + clip1(action[i]) * kActionHalf[i];
  }

  out->rollDeg = sp[0] * RAD2DEG;
  out->pitchDeg = sp[1] * RAD2DEG;   /* FIRMWARE_PITCH_SIGN = +1 */
  out->yawDeg = sp[2] * RAD2DEG;
  out->thrustN = sp[3];

  /* Collective newtons -> per-motor rpm: positive root of
   * a2*rpm^2 + a1*rpm + (a0 - F/4) = 0. A thrust below what the curve can
   * produce leaves the discriminant negative; that is the bottom of the
   * motor's range, so it maps to zero rpm rather than misbehaving. */
  const float a0 = NNPOL_RPM2THRUST_A0;
  const float a1 = NNPOL_RPM2THRUST_A1;
  const float a2 = NNPOL_RPM2THRUST_A2;
  const float fMotor = (sp[3] > 0.0f ? sp[3] : 0.0f) / 4.0f;
  const float disc = a1 * a1 - 4.0f * a2 * (a0 - fMotor);
  const float rpm = disc < 0.0f ? 0.0f : (-a1 + sqrtf(disc)) / (2.0f * a2);

  /* rpm -> motor voltage -> 16-bit ratio at the pushed nominal voltage.
   * Rounded to the nearest integer exactly as the host's rpm_to_pwm does
   * (np.rint), so shadow and onboard produce the identical value. */
  const float vMotor = (rpm - NNPOL_VMOTOR2RPM_K0) / NNPOL_VMOTOR2RPM_K1;
  float pwm = 65535.0f * vMotor / vnom;
  if (pwm < 0.0f) { pwm = 0.0f; }
  if (pwm > 65535.0f) { pwm = 65535.0f; }
  pwm = rintf(pwm);
  out->thrustPwm = pwm;

  /* The radio decoder's clamp, so enable=1 flies the same thrust the same
   * command would have flown through /attitude_cmd. */
  if (pwm < NNPOL_LEGACY_MIN_THRUST) {
    out->thrustSetpoint = 0.0f;
  } else if (pwm > NNPOL_LEGACY_MAX_THRUST) {
    out->thrustSetpoint = NNPOL_LEGACY_MAX_THRUST;
  } else {
    out->thrustSetpoint = pwm;
  }
}
