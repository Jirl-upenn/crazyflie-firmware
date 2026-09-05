/**
 * nnpol_obs_lissajous.c — the traj_lissajous/body observation, transcribed
 * from the trainer (tasks/traj_lissajous/observations.py _BodyFrame) via
 * the deployed host implementation (crazyflie-ros controller_tasks.py
 * TrajLissajousBody.observe). Pure C: host-testable, no firmware includes.
 *
 * Observation layout (obsDim = 10 + 3 * n_samples):
 *
 *     [quat_wxyz(4), v_B(3), omega_B(3),
 *      e_B(n_samples x 3, row-major)]
 *
 * where e_B[k] = R^T (R_tw ref(t + k*samples_dt) + off - pos): the
 * body-frame offsets to the next lookahead points, offsets starting at 0
 * (the CURRENT reference point is included — the trainer's sample_offsets
 * = arange(n) * dt, unlike tasks/traj whose window starts at t+dt).
 *
 * Yaw anchor (TrajLissajousBody.anchor, YAW_ANCHORED): R_tw = rot_z(yaw0)
 * rotates the task frame — the trainer's world, where every episode starts
 * at yaw 0 — into the mocap frame. The curve is yawed by it (the host's
 * pushed offset already places the yawed curve at the engage point) and
 * the observed attitude is R_tw^T R_wb, i.e. relative to the engage
 * heading. v_B, omega_B and e_B are body-frame already and need nothing.
 *
 * The curve constants (period, amplitude, centre height, n_samples,
 * samples_dt) come from the slot header's task[] block, so the same
 * build serves every lissajous checkpoint.
 *
 * The quaternion sign is normalized to w >= 0 AFTER the anchor rotation,
 * as the host's _quat_wxyz does. The trainer's quaternions come from
 * MuJoCo, which keeps w >= 0 — without it the double cover could hand
 * the network two encodings of one attitude across a run.
 */
#include <math.h>

#include "nnpol.h"

/* task[] indices for NNPOL_TASK_LISSAJOUS_BODY (export_policy_c). */
#define T_PERIOD 0
#define AMPLITUDE 1
#define CENTER_Z 2
#define N_SAMPLES 3
#define SAMPLES_DT 4

void nnpolRef(const nnpolSlotHeader_t* hdr, float t, float out[3])
{
  const float w = 2.0f * (float)M_PI / hdr->task[T_PERIOD];
  out[0] = hdr->task[AMPLITUDE] * sinf(w * t);
  out[1] = 0.0f;
  out[2] = 0.5f * hdr->task[AMPLITUDE] * sinf(2.0f * w * t) + hdr->task[CENTER_Z];
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

/** Hamilton product out = a * b, scalar-first. */
static void quatMul(const float a[4], const float b[4], float out[4])
{
  out[0] = a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3];
  out[1] = a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2];
  out[2] = a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1];
  out[3] = a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0];
}

int nnpolBuildObs(const nnpolSlotHeader_t* hdr, const nnpolSnapshot_t* snap,
                  float obs[NNPOL_MAX_OBS_DIM])
{
  if (hdr->taskId != NNPOL_TASK_LISSAJOUS_BODY) {
    return 0;
  }
  const int nSamples = (int)hdr->task[N_SAMPLES];
  if (nSamples < 0 || 10 + 3 * nSamples > NNPOL_MAX_OBS_DIM) {
    return 0;
  }
  const float DEG2RAD = (float)M_PI / 180.0f;

  /* The attitude in the mocap frame: R_wb for the body-frame rotations
   * (those are anchor-invariant), from the EKF quaternion as it is. */
  float Rwb[3][3];
  quatToRotation(snap->quat_wxyz, Rwb);

  /* The OBSERVED attitude: R_tw^T R_wb, as a quaternion, = q_z(-yaw0) * q,
   * then sign-normalized. The magnitude is the estimator's business (the
   * EKF keeps it normalized); only the double cover is fixed here. */
  const float half = -0.5f * snap->yaw0;
  const float qz[4] = { cosf(half), 0.0f, 0.0f, sinf(half) };
  float q[4];
  quatMul(qz, snap->quat_wxyz, q);
  if (q[0] < 0.0f) {
    q[0] = -q[0]; q[1] = -q[1]; q[2] = -q[2]; q[3] = -q[3];
  }

  int n = 0;
  obs[n++] = q[0]; obs[n++] = q[1]; obs[n++] = q[2]; obs[n++] = q[3];

  float v_b[3];
  worldToBody(Rwb, snap->vel_w, v_b);
  obs[n++] = v_b[0]; obs[n++] = v_b[1]; obs[n++] = v_b[2];

  /* The gyro is already body-frame; only the unit changes. */
  obs[n++] = snap->gyro_deg[0] * DEG2RAD;
  obs[n++] = snap->gyro_deg[1] * DEG2RAD;
  obs[n++] = snap->gyro_deg[2] * DEG2RAD;

  const float c = cosf(snap->yaw0), s = sinf(snap->yaw0);
  for (int k = 0; k < nSamples; k++) {
    float ref[3];
    nnpolRef(hdr, snap->t + (float)k * hdr->task[SAMPLES_DT], ref);
    /* R_tw ref: the nominal curve yawed to the engage heading. */
    const float rx = c * ref[0] - s * ref[1];
    const float ry = s * ref[0] + c * ref[1];
    const float e_w[3] = { rx + snap->off[0] - snap->pos[0],
                           ry + snap->off[1] - snap->pos[1],
                           ref[2] + snap->off[2] - snap->pos[2] };
    float e_b[3];
    worldToBody(Rwb, e_w, e_b);
    obs[n++] = e_b[0]; obs[n++] = e_b[1]; obs[n++] = e_b[2];
  }
  return n;
}
