/**
 * nnpol_obs_race.c — the race task's v3 / v3_motorobs observation, its gate
 * state and the engage context, transcribed from the trainer
 * (tasks/race/observations.py _V3, tasks/race/plugin.py RacePlugin.step,
 * tasks/race/geometry.py GateTrack.plane_crossings) via the host
 * implementation (ros_quad_sim2real controller/tasks/race/observations.py
 * RaceV3 / RaceV3MotorObs). Pure C: host-testable, no firmware includes.
 *
 * Observation layout (obsDim = 21 [+ 4]), entirely body frame:
 *
 *     [v_B(3), omega_B(3), gravity_B(3),
 *      target_B(3), target_B_next(3), normal_B(3), normal_B_next(3)
 *      [, rotor_speed(4) / hover_rpm]]
 *
 * gravity_B is world +Z in the body frame (the third row of R_wb);
 * target_B = R^T (gate - pos) for the current gate and the one after it;
 * normal_B = R^T n for the same two gates, n the gate's approach axis
 * (pointing AGAINST the travel direction).
 *
 * The track is FIXED in the mocap frame: the host places it (its
 * world_offset) and pushes it as the engage context; the start pose does not
 * move it.
 */
#include <math.h>
#include <string.h>

#include "nnpol.h"

// --------------------------------------------------------------------------
// Engage context
// --------------------------------------------------------------------------

static uint16_t rdU16(const uint8_t* p) { return (uint16_t)(p[0] | (p[1] << 8)); }
static uint32_t rdU32(const uint8_t* p)
{
  return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
}
static float rdF32(const uint8_t* p)
{
  float f;
  const uint32_t u = rdU32(p);
  memcpy(&f, &u, sizeof(f));
  return f;
}

static bool isUnit(const float v[3])
{
  const float n2 = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
  return isfinite(n2) && fabsf(n2 - 1.0f) < 1e-3f;
}

const char* nnpolCtxStatusName(nnpolCtxStatus_t s)
{
  switch (s) {
    case NNPOL_CTX_OK: return "ok";
    case NNPOL_CTX_EMPTY: return "empty";
    case NNPOL_CTX_BAD_MAGIC: return "bad magic";
    case NNPOL_CTX_BAD_SIZE: return "bad size";
    case NNPOL_CTX_BAD_CRC: return "bad crc";
    case NNPOL_CTX_BAD_TRACK: return "bad track";
    default: return "?";
  }
}

nnpolCtxStatus_t nnpolContextParse(const uint8_t* buf, uint32_t len, nnpolRaceTrack_t* track,
                                   uint32_t* crc)
{
  if (len < NNPOL_CTX_HEADER_BYTES) {
    return NNPOL_CTX_BAD_SIZE;
  }
  const uint32_t magic = rdU32(buf);
  if (magic == 0u) {
    return NNPOL_CTX_EMPTY;
  }
  if (magic != NNPOL_CTX_MAGIC) {
    return NNPOL_CTX_BAD_MAGIC;
  }
  const uint32_t total = rdU16(buf + 8);
  const uint8_t nGates = buf[10];
  const uint8_t gate0 = buf[11];
  if (nGates > NNPOL_MAX_GATES
      || total != NNPOL_CTX_HEADER_BYTES + (uint32_t)nGates * NNPOL_CTX_GATE_BYTES
      || total > len) {
    return NNPOL_CTX_BAD_SIZE;
  }
  if (nnpolCrc32(buf + 8, total - 8u, 0) != rdU32(buf + 4)) {
    return NNPOL_CTX_BAD_CRC;
  }
  const float side = rdF32(buf + 12);
  if (nGates == 0 || gate0 >= nGates || !(side > 0.0f) || !isfinite(side)) {
    return NNPOL_CTX_BAD_TRACK;
  }
  nnpolRaceTrack_t t;
  memset(&t, 0, sizeof(t));
  t.nGates = nGates;
  t.gate0 = gate0;
  t.gateSide = side;
  for (int i = 0; i < nGates; i++) {
    const uint8_t* g = buf + NNPOL_CTX_HEADER_BYTES + (uint32_t)i * NNPOL_CTX_GATE_BYTES;
    for (int k = 0; k < 3; k++) {
      t.gates[i].pos[k] = rdF32(g + 4 * k);
      t.gates[i].normal[k] = rdF32(g + 12 + 4 * k);
      t.gates[i].tangent[k] = rdF32(g + 24 + 4 * k);
      t.gates[i].up[k] = rdF32(g + 36 + 4 * k);
    }
    const float* p = t.gates[i].pos;
    if (!isfinite(p[0]) || !isfinite(p[1]) || !isfinite(p[2]) || !isUnit(t.gates[i].normal)
        || !isUnit(t.gates[i].tangent) || !isUnit(t.gates[i].up)) {
      return NNPOL_CTX_BAD_TRACK;
    }
  }
  *track = t;
  if (crc != NULL) {
    *crc = rdU32(buf + 4);
  }
  return NNPOL_CTX_OK;
}

// --------------------------------------------------------------------------
// Gate state
// --------------------------------------------------------------------------

void nnpolRaceReset(const nnpolRaceTrack_t* track, nnpolRaceState_t* race)
{
  memset(race, 0, sizeof(*race));
  race->gateIdx = track->nGates ? (uint8_t)(track->gate0 % track->nGates) : 0;
}

static float dot3(const float a[3], const float b[3])
{
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

bool nnpolRaceUpdateGate(const nnpolRaceTrack_t* track, nnpolRaceState_t* race, const float pos[3])
{
  bool passed = false;
  if (track->nGates != 0 && race->havePrev) {
    const uint8_t idx = race->gateIdx % track->nGates;
    const nnpolGate_t* g = &track->gates[idx];
    const float rel0[3] = { race->prevPos[0] - g->pos[0], race->prevPos[1] - g->pos[1],
                            race->prevPos[2] - g->pos[2] };
    const float rel1[3] = { pos[0] - g->pos[0], pos[1] - g->pos[1], pos[2] - g->pos[2] };
    const float x0 = dot3(rel0, g->normal);
    const float x1 = dot3(rel1, g->normal);
    if (x0 > 0.0f && x1 <= 0.0f) {
      /* where the step's path meets the plane (x0 != x1 here) */
      const float s = x0 / (x0 - x1);
      const float at[3] = { rel0[0] + s * (rel1[0] - rel0[0]), rel0[1] + s * (rel1[1] - rel0[1]),
                            rel0[2] + s * (rel1[2] - rel0[2]) };
      const float half = track->gateSide * 0.5f;
      passed = fabsf(dot3(at, g->tangent)) < half && fabsf(dot3(at, g->up)) < half;
    }
    if (passed) {
      race->gateIdx = (uint8_t)((idx + 1) % track->nGates);
    }
  }
  race->prevPos[0] = pos[0];
  race->prevPos[1] = pos[1];
  race->prevPos[2] = pos[2];
  race->havePrev = 1;
  return passed;
}

// --------------------------------------------------------------------------
// Observation
// --------------------------------------------------------------------------

int nnpolObsDim(const nnpolSlotHeader_t* hdr)
{
  if (hdr->taskId != NNPOL_TASK_RACE_V3) {
    return 0;
  }
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

int nnpolBuildObs(const nnpolSlotHeader_t* hdr, const nnpolRaceTrack_t* track, uint8_t gateIdx,
                  const nnpolState_t* state, float obs[NNPOL_MAX_OBS_DIM])
{
  if (hdr->taskId != NNPOL_TASK_RACE_V3 || track->nGates == 0) {
    return 0;
  }
  const float DEG2RAD = (float)M_PI / 180.0f;
  float R[3][3];
  quatToRotation(state->quat_wxyz, R);

  int n = 0;
  float v_b[3];
  worldToBody(R, state->vel_w, v_b);
  obs[n++] = v_b[0]; obs[n++] = v_b[1]; obs[n++] = v_b[2];
  obs[n++] = state->gyro_deg[0] * DEG2RAD;
  obs[n++] = state->gyro_deg[1] * DEG2RAD;
  obs[n++] = state->gyro_deg[2] * DEG2RAD;
  /* gravity_B = R^T [0, 0, 1] = the third row of R. */
  obs[n++] = R[2][0]; obs[n++] = R[2][1]; obs[n++] = R[2][2];

  const uint8_t idx = gateIdx % track->nGates;
  const uint8_t nxt = (uint8_t)((idx + 1) % track->nGates);
  const uint8_t which[2] = { idx, nxt };
  for (int k = 0; k < 2; k++) {
    const float* g = track->gates[which[k]].pos;
    const float e_w[3] = { g[0] - state->pos[0], g[1] - state->pos[1], g[2] - state->pos[2] };
    float e_b[3];
    worldToBody(R, e_w, e_b);
    obs[n++] = e_b[0]; obs[n++] = e_b[1]; obs[n++] = e_b[2];
  }
  for (int k = 0; k < 2; k++) {
    float n_b[3];
    worldToBody(R, track->gates[which[k]].normal, n_b);
    obs[n++] = n_b[0]; obs[n++] = n_b[1]; obs[n++] = n_b[2];
  }
  if (hdr->task[NNPOL_TASK_MOTOROBS] != 0.0f) {
    for (int i = 0; i < 4; i++) {
      obs[n++] = state->rpm[i] / hdr->hoverRpm;
    }
  }
  return n;
}
