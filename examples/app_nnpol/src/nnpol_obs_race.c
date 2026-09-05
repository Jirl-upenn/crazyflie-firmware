/**
 * nnpol_obs_race.c — the race task's v3 / v3_motorobs observation and its
 * gate state, transcribed from the trainer (tasks/race/observations.py _V3,
 * tasks/race/plugin.py RacePlugin.step) via the deployed host
 * implementation (crazyflie-ros controller_tasks.py RaceV3 /
 * RaceV3MotorObs). Pure C: host-testable, no firmware includes.
 *
 * Observation layout (obsDim = 21 [+ 4]), entirely body frame:
 *
 *     [v_B(3), omega_B(3), gravity_B(3),
 *      target_B(3), target_B_next(3), normal_B(3), normal_B_next(3)
 *      [, rotor_speed(4) / NOMINAL_HOVER_RPM]]
 *
 * gravity_B is world +Z in the body frame (the third row of R_wb);
 * target_B = R^T (gate - pos) for the current gate and the one after it;
 * normal_B = R^T n for the same two gates, n the gate's approach axis
 * (pointing AGAINST the travel direction, horizontal). No yaw anchor: the
 * track itself is pinned to the start pose by the host (RaceV3
 * .pin_start_gate — a yaw rotation plus a translation of the surveyed
 * layout so the first gate sits ahead of the drone), and the pinned
 * positions and normals are what arrive at engage.
 *
 * Gate state (RacePlugin.step, RaceV3._update_gate): the target gate
 * advances when the drone crosses the gate plane from the approach side
 * (x_wrt_gate < 0 after prev > 0) inside the gate's square (|y|, |z| <
 * side / 2); prev_gate_x re-arms to 1.0 on a crossing. The controller
 * runs this on every policy tick with the EKF position — the same
 * position the host runs it with from mocap — so the two gate counters
 * agree unless the estimates disagree at a plane.
 */
#include <math.h>

#include "nnpol.h"

void nnpolRaceUpdateGate(const nnpolRaceTrack_t* track, float gateSide, const float pos[3],
                         uint8_t* gateIdx, float* prevGateX)
{
  if (track->nGates == 0) {
    return;
  }
  const uint8_t idx = *gateIdx % track->nGates;
  const float* g = track->gates[idx];
  const float* n = track->normals[idx];
  const float rel[3] = { pos[0] - g[0], pos[1] - g[1], pos[2] - g[2] };
  const float xWrt = rel[0] * n[0] + rel[1] * n[1] + rel[2] * n[2];
  /* Lateral distance via the cross product's z component: exact because
   * gate normals are horizontal. */
  const float yWrt = rel[1] * n[0] - rel[0] * n[1];
  const float zWrt = rel[2];
  const float half = gateSide * 0.5f;
  const int justPassed = (xWrt < 0.0f) && (*prevGateX > 0.0f)
      && (fabsf(yWrt) < half) && (fabsf(zWrt) < half);
  if (justPassed) {
    *gateIdx = (uint8_t)((idx + 1) % track->nGates);
    *prevGateX = 1.0f;
  } else {
    *gateIdx = idx;
    *prevGateX = xWrt;
  }
}

int nnpolRaceObsDim(const nnpolSlotHeader_t* hdr)
{
  return 21 + (hdr->task[NNPOL_TASK_MOTOROBS] != 0.0f ? 4 : 0);
}

/** Body-to-world rotation matrix from a scalar-first unit quaternion. */
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

int nnpolRaceBuildObs(const nnpolSlotHeader_t* hdr, const nnpolSnapshot_t* snap,
                      float obs[NNPOL_MAX_OBS_DIM])
{
  const nnpolRaceTrack_t* track = &snap->race;
  if (track->nGates == 0) {
    return 0;
  }
  const float DEG2RAD = (float)M_PI / 180.0f;
  float R[3][3];
  quatToRotation(snap->quat_wxyz, R);

  int n = 0;
  float v_b[3];
  worldToBody(R, snap->vel_w, v_b);
  obs[n++] = v_b[0]; obs[n++] = v_b[1]; obs[n++] = v_b[2];
  obs[n++] = snap->gyro_deg[0] * DEG2RAD;
  obs[n++] = snap->gyro_deg[1] * DEG2RAD;
  obs[n++] = snap->gyro_deg[2] * DEG2RAD;
  /* gravity_B = R^T [0, 0, 1] = the third row of R. */
  obs[n++] = R[2][0]; obs[n++] = R[2][1]; obs[n++] = R[2][2];

  const uint8_t idx = track->gateIdx % track->nGates;
  const uint8_t nxt = (uint8_t)((idx + 1) % track->nGates);
  const uint8_t which[2] = { idx, nxt };
  for (int k = 0; k < 2; k++) {
    const float* g = track->gates[which[k]];
    const float e_w[3] = { g[0] - snap->pos[0], g[1] - snap->pos[1], g[2] - snap->pos[2] };
    float e_b[3];
    worldToBody(R, e_w, e_b);
    obs[n++] = e_b[0]; obs[n++] = e_b[1]; obs[n++] = e_b[2];
  }
  for (int k = 0; k < 2; k++) {
    float n_b[3];
    worldToBody(R, track->normals[which[k]], n_b);
    obs[n++] = n_b[0]; obs[n++] = n_b[1]; obs[n++] = n_b[2];
  }
  if (hdr->task[NNPOL_TASK_MOTOROBS] != 0.0f) {
    for (int i = 0; i < 4; i++) {
      obs[n++] = snap->rpm[i];
    }
  }
  return n;
}
