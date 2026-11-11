/**
 * pose_history.c - see pose_history.h. Pure C.
 */
#include <math.h>
#include <string.h>

#include "pose_history.h"

void poseHistoryReset(poseHistory_t* h)
{
  h->head = 0;
  h->count = 0;
}

void poseHistoryPush(poseHistory_t* h, uint32_t timeUs, const float pos[3], const float q[4])
{
  if (h->count > 0) {
    h->head = (uint16_t)((h->head + 1) % POSE_HISTORY_LEN);
  }
  if (h->count < POSE_HISTORY_LEN) {
    h->count++;
  }
  poseHistoryEntry_t* e = &h->entries[h->head];
  e->timeUs = timeUs;
  memcpy(e->pos, pos, sizeof(e->pos));
  memcpy(e->q, q, sizeof(e->q));
}

static void interpolate(const float posNew[3], const float qNew[4],
                        const float posOld[3], const float qOld[4], float frac,
                        float pos[3], float q[4])
{
  for (int i = 0; i < 3; i++) {
    pos[i] = posNew[i] + frac * (posOld[i] - posNew[i]);
  }
  // normalized lerp along the short way round
  float dot = 0.0f;
  for (int i = 0; i < 4; i++) {
    dot += qNew[i] * qOld[i];
  }
  const float sign = dot < 0.0f ? -1.0f : 1.0f;
  float norm = 0.0f;
  for (int i = 0; i < 4; i++) {
    q[i] = qNew[i] + frac * (sign * qOld[i] - qNew[i]);
    norm += q[i] * q[i];
  }
  norm = sqrtf(norm);
  if (norm > 1e-9f) {
    for (int i = 0; i < 4; i++) {
      q[i] /= norm;
    }
  } else {
    memcpy(q, qNew, 4 * sizeof(float));
  }
}

bool poseHistoryLookup(const poseHistory_t* h, uint32_t nowUs, uint32_t ageUs,
                       uint32_t curUs, const float curPos[3], const float curQ[4],
                       float pos[3], float q[4])
{
  // Ages are measured back from nowUs with wrapping arithmetic; the newer
  // side of the bracket starts at the current state.
  uint32_t newerAge = nowUs - curUs;
  const float* newerPos = curPos;
  const float* newerQ = curQ;

  if (ageUs <= newerAge) {
    // Not older than the current state: nothing to rewind.
    memcpy(pos, curPos, 3 * sizeof(float));
    memcpy(q, curQ, 4 * sizeof(float));
    return true;
  }

  for (uint16_t k = 0; k < h->count; k++) {
    const poseHistoryEntry_t* e =
        &h->entries[(h->head + POSE_HISTORY_LEN - k) % POSE_HISTORY_LEN];
    const uint32_t age = nowUs - e->timeUs;
    if (age >= ageUs) {
      const uint32_t span = age - newerAge;
      const float frac = span > 0 ? (float)(ageUs - newerAge) / (float)span : 0.0f;
      interpolate(newerPos, newerQ, e->pos, e->q, frac, pos, q);
      return true;
    }
    newerAge = age;
    newerPos = e->pos;
    newerQ = e->q;
  }
  return false;
}

void poseHistoryShift(poseHistory_t* h, const float dPos[3], const float dq[4])
{
  for (uint16_t k = 0; k < h->count; k++) {
    poseHistoryEntry_t* e = &h->entries[(h->head + POSE_HISTORY_LEN - k) % POSE_HISTORY_LEN];
    for (int i = 0; i < 3; i++) {
      e->pos[i] += dPos[i];
    }
    const float w = e->q[0], x = e->q[1], y = e->q[2], z = e->q[3];
    float r[4] = {
      w * dq[0] - x * dq[1] - y * dq[2] - z * dq[3],
      w * dq[1] + x * dq[0] + y * dq[3] - z * dq[2],
      w * dq[2] - x * dq[3] + y * dq[0] + z * dq[1],
      w * dq[3] + x * dq[2] - y * dq[1] + z * dq[0],
    };
    const float norm = sqrtf(r[0] * r[0] + r[1] * r[1] + r[2] * r[2] + r[3] * r[3]);
    if (norm > 1e-9f) {
      for (int i = 0; i < 4; i++) {
        e->q[i] = r[i] / norm;
      }
    }
  }
}
