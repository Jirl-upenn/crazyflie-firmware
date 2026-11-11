/**
 * pose_history.h - a short ring of past estimator position and attitude,
 * for measurements that arrive late (the aged external pose, EXT_POSE_AGED).
 *
 * A pose captured d ago is compared with what the estimator believed d ago,
 * not with what it believes now; the residual is then applied to the current
 * state. That removes the lag of (speed x latency) the plain update has.
 *
 * Pure C, no firmware includes: host-testable.
 */
#pragma once

#include <stdbool.h>
#include <stdint.h>

// One entry per estimator tick. At 1 kHz this is 64 ms of history, which has
// to cover the total age of a pose: mocap latency + radio + queueing.
#define POSE_HISTORY_LEN 64

typedef struct {
  uint32_t timeUs;   // the time the state was valid for (last prediction), wrapping microseconds
  float pos[3];      // world position [m]
  float q[4];        // attitude quaternion [w, x, y, z]
} poseHistoryEntry_t;

typedef struct {
  poseHistoryEntry_t entries[POSE_HISTORY_LEN];
  uint16_t head;     // index of the newest entry
  uint16_t count;    // valid entries
} poseHistory_t;

void poseHistoryReset(poseHistory_t* h);

// Append the state valid at timeUs. Entries must be pushed in time order;
// equal stamps are fine (several ticks between two predictions).
void poseHistoryPush(poseHistory_t* h, uint32_t timeUs, const float pos[3], const float q[4]);

/**
 * The state the estimator held at (nowUs - ageUs).
 *
 * The current state (curPos, curQ, valid at curUs) is the newest sample, so
 * an age of zero returns it unchanged. Position is interpolated linearly
 * between the two bracketing samples, attitude by normalized lerp.
 *
 * Returns false when the requested time is older than the oldest entry.
 */
bool poseHistoryLookup(const poseHistory_t* h, uint32_t nowUs, uint32_t ageUs,
                       uint32_t curUs, const float curPos[3], const float curQ[4],
                       float pos[3], float q[4]);

/**
 * Apply a correction to every entry: pos += dPos and q = q * dq (dq a small
 * body-frame rotation [w, x, y, z]). Called after each measurement update
 * with what the update changed, so that the next late measurement is not
 * compared with a past that predates this correction - without it a
 * latency longer than the measurement interval counts each correction more
 * than once.
 */
void poseHistoryShift(poseHistory_t* h, const float dPos[3], const float dq[4]);
