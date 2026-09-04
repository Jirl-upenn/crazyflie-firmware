/**
 * nnpol_obs_lissajous.c — the traj_lissajous/body observation, transcribed
 * from the trainer (tasks/traj_lissajous/observations.py _BodyFrame) via
 * the deployed host implementation (crazyflie-ros controller_tasks.py
 * TrajLissajousBody.observe). Pure C: host-testable, no firmware includes.
 *
 * Observation layout (NNPOL_OBS_DIM = 10 + 3 * NNPOL_N_SAMPLES):
 *
 *     [quat_wxyz(4), v_B(3), omega_B(3),
 *      e_B(NNPOL_N_SAMPLES x 3, row-major)]
 *
 * where e_B[k] = R^T (ref(t + k*SAMPLES_DT) + off - pos): the body-frame
 * offsets to the next lookahead points, offsets starting at 0 (the CURRENT
 * reference point is included — the trainer's sample_offsets = arange(n) *
 * dt, unlike tasks/traj whose window starts at t+dt).
 *
 * The quaternion sign is normalized to w >= 0. The trainer's quaternions
 * come from MuJoCo, which keeps w >= 0, and the host's _quat_wxyz enforces
 * the same — without it the double cover could hand the network two
 * encodings of one attitude across a run.
 */
#include <math.h>

#include "nnpol.h"

void nnpolRef(float t, float out[3])
{
  const float w = 2.0f * (float)M_PI / NNPOL_T_PERIOD_S;
  out[0] = NNPOL_AMPLITUDE_M * sinf(w * t);
  out[1] = 0.0f;
  out[2] = 0.5f * NNPOL_AMPLITUDE_M * sinf(2.0f * w * t) + NNPOL_CENTER_Z_M;
}

/** Body-to-world rotation matrix from a scalar-first unit quaternion.
 * R[i][j], rows are world axes: v_W = R v_B, v_B = R^T v_W. */
static void quatToRotation(const float q[4], float R[3][3])
{
  const float qw = q[0], qx = q[1], qy = q[2], qz = q[3];
  R[0][0] = 1.0f - 2.0f * (qy * qy + qz * qz);
  R[0][1] = 2.0f * (qx * qy - qw * qz);
  R[0][2] = 2.0f * (qx * qz + qw * qy);
  R[1][0] = 2.0f * (qx * qy + qw * qz);
  R[1][1] = 1.0f - 2.0f * (qx * qx + qz * qz);
  R[1][2] = 2.0f * (qy * qz - qw * qx);
  R[2][0] = 2.0f * (qx * qz - qw * qy);
  R[2][1] = 2.0f * (qy * qz + qw * qx);
  R[2][2] = 1.0f - 2.0f * (qx * qx + qy * qy);
}

static void worldToBody(const float R[3][3], const float v_w[3], float v_b[3])
{
  for (int i = 0; i < 3; i++) {
    v_b[i] = R[0][i] * v_w[0] + R[1][i] * v_w[1] + R[2][i] * v_w[2];
  }
}

void nnpolBuildObs(const nnpolSnapshot_t* snap, float obs[NNPOL_OBS_DIM])
{
  const float DEG2RAD = (float)M_PI / 180.0f;

  /* Quaternion, sign-normalized. The magnitude is the estimator's business
   * (the EKF keeps it normalized); only the double cover is fixed here. */
  float q[4] = { snap->quat_wxyz[0], snap->quat_wxyz[1],
                 snap->quat_wxyz[2], snap->quat_wxyz[3] };
  if (q[0] < 0.0f) {
    q[0] = -q[0]; q[1] = -q[1]; q[2] = -q[2]; q[3] = -q[3];
  }

  float R[3][3];
  quatToRotation(q, R);

  int n = 0;
  obs[n++] = q[0]; obs[n++] = q[1]; obs[n++] = q[2]; obs[n++] = q[3];

  float v_b[3];
  worldToBody(R, snap->vel_w, v_b);
  obs[n++] = v_b[0]; obs[n++] = v_b[1]; obs[n++] = v_b[2];

  /* The gyro is already body-frame; only the unit changes. */
  obs[n++] = snap->gyro_deg[0] * DEG2RAD;
  obs[n++] = snap->gyro_deg[1] * DEG2RAD;
  obs[n++] = snap->gyro_deg[2] * DEG2RAD;

  for (int k = 0; k < NNPOL_N_SAMPLES; k++) {
    float ref[3];
    nnpolRef(snap->t + (float)k * NNPOL_SAMPLES_DT_S, ref);
    const float e_w[3] = { ref[0] + snap->off[0] - snap->pos[0],
                           ref[1] + snap->off[1] - snap->pos[1],
                           ref[2] + snap->off[2] - snap->pos[2] };
    float e_b[3];
    worldToBody(R, e_w, e_b);
    obs[n++] = e_b[0]; obs[n++] = e_b[1]; obs[n++] = e_b[2];
  }
}
