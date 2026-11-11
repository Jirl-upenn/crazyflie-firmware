/**
 * pose_delay_host.c - host harness for the delay-compensated external pose
 * update (kalman_core/mm_pose.c, pose_history.c), built with plain gcc
 * against the firmware's own Kalman core by test_python/test_pose_delay.py.
 *
 * It runs the estimator loop in the order estimator_kalman.c runs it
 * (predict at 500 Hz, process noise, queued measurements, finalize, history
 * push - all on a 1 kHz tick) for a vehicle flying a straight line at
 * constant speed while yawing at a constant rate, and feeds it mocap poses
 * that are `latency` old when they arrive.
 *
 *   pose_delay_host flight <stock|comp> <speed m/s> <yaw rate rad/s> <latency us> <age us> <offset m> [dump]
 *       prints: pos_err_m yaw_err_rad drops worst_err_m
 *       pos_err is the along-track error (estimate - truth) and yaw_err the
 *       heading error, both averaged over the last second; `age` is what the
 *       update is told the pose's age is. The estimate starts `offset` away
 *       from the truth along the track with a tight initial covariance, and
 *       worst_err is the largest along-track error on the far side of the
 *       truth while it converges (overshoot). With `dump` the final state
 *       follows as raw hex words, for bit-for-bit comparisons.
 *
 *   pose_delay_host ring
 *       prints one line per check of the history ring: name ok(0/1)
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "kalman_core.h"
#include "kalman_core_params_defaults.h"
#include "mm_pose.h"
#include "pose_history.h"

void assertFail(char* exp, char* file, int line)
{
  fprintf(stderr, "ASSERT %s at %s:%d\n", exp, file, line);
  exit(3);
}

#define POSE_INTERVAL_US 8333   /* 120 Hz mocap */
#define PREDICT_EVERY_TICKS 2   /* CONFIG_ESTIMATOR_KALMAN_PREDICT_RATE=500 */

static float wrapPi(float a)
{
  while (a > (float)M_PI) { a -= 2.0f * (float)M_PI; }
  while (a < -(float)M_PI) { a += 2.0f * (float)M_PI; }
  return a;
}

static int flight(int comp, float speed, float yawRate, uint32_t latencyUs, uint32_t ageUs,
                  float offset, int dump)
{
  static kalmanCoreData_t core;
  static poseHistory_t history;
  kalmanCoreParams_t params = { KALMAN_CORE_DEFAULT_PARAMS_INIT };
  params.initialZ = 1.0f;
  params.initialX = offset;
  if (offset != 0.0f) {
    params.stdDevInitialPosition_xy = 0.05f;
  }
  kalmanCoreInit(&core, &params, 0);
  core.S[KC_STATE_PX] = speed;
  poseHistoryReset(&history);

  Axis3f acc = { .x = 0.0f, .y = 0.0f, .z = 9.81f };
  Axis3f gyro = { .x = 0.0f, .y = 0.0f, .z = yawRate };

  const int ticks = 5000;
  uint32_t stateUs = 0;
  uint32_t nextPoseUs = 20000;
  double sumPos = 0.0, sumYaw = 0.0, worst = 0.0;
  int n = 0, drops = 0;

  for (int i = 1; i <= ticks; i++) {
    const uint32_t nowUs = (uint32_t)i * 1000u;
    if (i % PREDICT_EVERY_TICKS == 0) {
      kalmanCorePredict(&core, &params, &acc, &gyro, (uint32_t)i, true);
      stateUs = nowUs;
    }
    kalmanCoreAddProcessNoise(&core, &params, (uint32_t)i);

    if (nowUs >= nextPoseUs) {
      nextPoseUs += POSE_INTERVAL_US;
      /* the truth one latency ago: world velocity is constant along x */
      const double t = ((double)nowUs - (double)latencyUs) * 1e-6;
      const double yaw = (double)yawRate * t;
      poseMeasurement_t pose;
      memset(&pose, 0, sizeof(pose));
      pose.x = (float)(speed * t);
      pose.y = 0.0f;
      pose.z = 1.0f;
      pose.quat.x = 0.0f;
      pose.quat.y = 0.0f;
      pose.quat.z = (float)sin(yaw / 2.0);
      pose.quat.w = (float)cos(yaw / 2.0);
      pose.stdDevPos = 0.01f;
      pose.stdDevQuat = 4.5e-3f;
      float residual[3];
      if (comp) {
        if (!kalmanCoreUpdateWithPoseDelayed(&core, &pose, &history, nowUs, ageUs, stateUs, residual)) {
          drops++;
        }
      } else {
        kalmanCoreUpdateWithPose(&core, &pose);
      }
    }

    kalmanCoreFinalize(&core);
    poseHistoryPush(&history, stateUs, &core.S[KC_STATE_X], core.q);

    const double tState = (double)stateUs * 1e-6;
    const double err = (double)core.S[KC_STATE_X] - speed * tState;
    if (offset != 0.0f && err * offset < 0.0 && fabs(err) > worst) {
      worst = fabs(err);
    }
    if (i > ticks - 1000) {
      sumPos += err;
      const float yawEst = atan2f(2.0f * (core.q[1] * core.q[2] + core.q[0] * core.q[3]),
                                  core.q[0] * core.q[0] + core.q[1] * core.q[1]
                                  - core.q[2] * core.q[2] - core.q[3] * core.q[3]);
      sumYaw += wrapPi(yawEst - (float)fmod(yawRate * tState, 2.0 * M_PI));
      n++;
    }
  }
  printf("%.6f %.6f %d %.6f\n", sumPos / n, sumYaw / n, drops, worst);
  if (dump) {
    uint32_t w;
    for (int k = 0; k < KC_STATE_DIM; k++) { memcpy(&w, &core.S[k], 4); printf("%08x ", w); }
    for (int k = 0; k < 4; k++) { memcpy(&w, &core.q[k], 4); printf("%08x ", w); }
    for (int r = 0; r < KC_STATE_DIM; r++) {
      for (int c = 0; c < KC_STATE_DIM; c++) { memcpy(&w, &core.P[r][c], 4); printf("%08x ", w); }
    }
    printf("\n");
  }
  return 0;
}

static void report(const char* name, int ok) { printf("%s %d\n", name, ok); }

static int ring(void)
{
  static poseHistory_t h;
  const float q0[4] = { 1.0f, 0.0f, 0.0f, 0.0f };
  float pos[3], q[4];

  /* 100 states one millisecond apart, x = time in ms: only the last 64 stay */
  poseHistoryReset(&h);
  const uint32_t base = 0xFFFFFFFFu - 50000u;        /* the stamps wrap mid-way */
  for (int i = 0; i < 100; i++) {
    const float p[3] = { (float)i, 0.0f, 0.0f };
    poseHistoryPush(&h, base + (uint32_t)i * 1000u, p, q0);
  }
  const uint32_t now = base + 99500u;                 /* half a millisecond after the last push */
  const float cur[3] = { 99.5f, 0.0f, 0.0f };

  int ok = poseHistoryLookup(&h, now, 0, now, cur, q0, pos, q);
  report("age_zero_is_current", ok && pos[0] == 99.5f);

  ok = poseHistoryLookup(&h, now, 500, now, cur, q0, pos, q);
  report("lands_on_an_entry", ok && fabsf(pos[0] - 99.0f) < 1e-4f);

  ok = poseHistoryLookup(&h, now, 10250, now, cur, q0, pos, q);
  report("interpolates_between_entries", ok && fabsf(pos[0] - 89.25f) < 1e-3f);

  ok = poseHistoryLookup(&h, now, 63500, now, cur, q0, pos, q);
  report("oldest_entry_reachable", ok && fabsf(pos[0] - 36.0f) < 1e-3f);

  ok = poseHistoryLookup(&h, now, 63501, now, cur, q0, pos, q);
  report("older_than_history_refused", !ok);

  /* a state that has not been propagated for 2 ms: younger ages return it */
  ok = poseHistoryLookup(&h, now + 2000u, 1500, now, cur, q0, pos, q);
  report("younger_than_state_is_current", ok && pos[0] == 99.5f);

  /* a correction reaches every entry */
  const float dPos[3] = { 0.5f, -1.0f, 2.0f };
  const float dq[4] = { 1.0f, 0.0f, 0.0f, 0.01f };
  poseHistoryShift(&h, dPos, dq);
  ok = poseHistoryLookup(&h, now, 20500, now, cur, q0, pos, q);
  report("shift_moves_positions", ok && fabsf(pos[0] - 79.5f) < 1e-3f && fabsf(pos[1] + 1.0f) < 1e-6f
         && fabsf(pos[2] - 2.0f) < 1e-6f);
  report("shift_rotates_attitudes", ok && fabsf(q[3] - 0.01f) < 1e-4f && fabsf(q[0] - 1.0f) < 1e-3f);

  /* attitude interpolation takes the short way round */
  poseHistoryReset(&h);
  const float qa[4] = { 0.9238795f, 0.0f, 0.0f, 0.3826834f };    /* yaw 45 deg */
  const float qb[4] = { -1.0f, 0.0f, 0.0f, 0.0f };               /* yaw 0, other cover */
  const float z[3] = { 0.0f, 0.0f, 0.0f };
  poseHistoryPush(&h, 1000, z, qb);
  ok = poseHistoryLookup(&h, 2000, 500, 2000, z, qa, pos, q);
  const float yaw = 2.0f * atan2f(q[3], q[0]);
  report("attitude_short_way", ok && fabsf(yaw - 0.3927f) < 2e-3f);

  /* an empty history has only the current state */
  poseHistoryReset(&h);
  report("empty_history_refuses_past", !poseHistoryLookup(&h, 5000, 1000, 5000, z, q0, pos, q));
  return 0;
}

int main(int argc, char** argv)
{
  if (argc >= 2 && strcmp(argv[1], "ring") == 0) {
    return ring();
  }
  if (argc >= 8 && strcmp(argv[1], "flight") == 0) {
    return flight(strcmp(argv[2], "comp") == 0, (float)atof(argv[3]), (float)atof(argv[4]),
                  (uint32_t)atol(argv[5]), (uint32_t)atol(argv[6]), (float)atof(argv[7]),
                  argc >= 9);
  }
  fprintf(stderr, "usage: see the file header\n");
  return 2;
}
