/**
 * test_policy_host.c — host-side parity harness for app_nnpol.
 *
 * Compiled with plain gcc against the SAME generated policy.c and the same
 * pure-C observation/action modules the firmware builds, so what is
 * checked is the flight code itself, not a transcription of it. Driven by
 * run_host_parity.py, which feeds binary records on stdin and compares
 * the answers against the trainer's numpy reference
 * (policy_reference.npz) and against its own numpy transcriptions of the
 * observation and decode.
 *
 * Modes (argv[1]); all records are little-endian float32 streams:
 *   policy : obs[OBS_DIM]                        -> act[4]   (raw, unclipped)
 *   obs    : snapshot[17]                        -> obs[OBS_DIM]
 *   act    : action[4], vnom, rpmScale           -> cmd[14]
 *   full   : snapshot[17], vnom, rpmScale        -> cmd[14]  (obs -> net -> clip -> decode)
 *
 * snapshot[17] = [t, off(3), pos(3), quat_wxyz(4), vel_w(3), gyro_deg(3)]
 * cmd[14]      = [ch0, ch1, ch2, thrustN, thrustPwm, thrustSetpoint,
 *                 rpm(4), motorPwm(4)]   — nnpolCmd_t flattened; which
 *                 columns are non-zero depends on NNPOL_ACTION_KIND
 */
#include <stdio.h>
#include <string.h>

#include "nnpol.h"
#include "policy.h"

#define SNAP_FLOATS 17
#define CMD_FLOATS 14
#define ACT_IN_FLOATS (NNPOL_ACTION_DIM + 2)

static void unpackSnapshot(const float* in, nnpolSnapshot_t* s)
{
  s->t = in[0];
  memcpy(s->off, in + 1, 3 * sizeof(float));
  memcpy(s->pos, in + 4, 3 * sizeof(float));
  memcpy(s->quat_wxyz, in + 7, 4 * sizeof(float));
  memcpy(s->vel_w, in + 11, 3 * sizeof(float));
  memcpy(s->gyro_deg, in + 14, 3 * sizeof(float));
}

static void packCmd(const nnpolCmd_t* c, float* out)
{
  out[0] = c->ch[0];
  out[1] = c->ch[1];
  out[2] = c->ch[2];
  out[3] = c->thrustN;
  out[4] = c->thrustPwm;
  out[5] = c->thrustSetpoint;
  memcpy(out + 6, c->rpm, 4 * sizeof(float));
  memcpy(out + 10, c->motorPwm, 4 * sizeof(float));
}

static void clipAction(float* act)
{
  for (int i = 0; i < NNPOL_ACTION_DIM; i++) {
    if (act[i] < -1.0f) { act[i] = -1.0f; }
    if (act[i] > 1.0f) { act[i] = 1.0f; }
  }
}

int main(int argc, char** argv)
{
  if (argc != 2) {
    fprintf(stderr, "usage: %s {policy|obs|act|full|kind}\n", argv[0]);
    return 2;
  }
  const char* mode = argv[1];

  if (strcmp(mode, "kind") == 0) {
    /* What this build decodes as, so the runner can pick its mirror. */
    printf("%d\n", NNPOL_ACTION_KIND);
  } else if (strcmp(mode, "policy") == 0) {
    float obs[NNPOL_OBS_DIM], act[NNPOL_ACTION_DIM];
    while (fread(obs, sizeof(float), NNPOL_OBS_DIM, stdin) == NNPOL_OBS_DIM) {
      policyForward(obs, act);
      fwrite(act, sizeof(float), NNPOL_ACTION_DIM, stdout);
    }
  } else if (strcmp(mode, "obs") == 0) {
    float in[SNAP_FLOATS], obs[NNPOL_OBS_DIM];
    nnpolSnapshot_t snap;
    while (fread(in, sizeof(float), SNAP_FLOATS, stdin) == SNAP_FLOATS) {
      unpackSnapshot(in, &snap);
      nnpolBuildObs(&snap, obs);
      fwrite(obs, sizeof(float), NNPOL_OBS_DIM, stdout);
    }
  } else if (strcmp(mode, "act") == 0) {
    float in[ACT_IN_FLOATS], out[CMD_FLOATS];
    nnpolCmd_t cmd;
    while (fread(in, sizeof(float), ACT_IN_FLOATS, stdin) == ACT_IN_FLOATS) {
      nnpolDecodeAction(in, in[NNPOL_ACTION_DIM], in[NNPOL_ACTION_DIM + 1], &cmd);
      packCmd(&cmd, out);
      fwrite(out, sizeof(float), CMD_FLOATS, stdout);
    }
  } else if (strcmp(mode, "full") == 0) {
    float in[SNAP_FLOATS + 2], obs[NNPOL_OBS_DIM], act[NNPOL_ACTION_DIM], out[CMD_FLOATS];
    nnpolSnapshot_t snap;
    nnpolCmd_t cmd;
    while (fread(in, sizeof(float), SNAP_FLOATS + 2, stdin) == SNAP_FLOATS + 2) {
      unpackSnapshot(in, &snap);
      nnpolBuildObs(&snap, obs);
      policyForward(obs, act);
      clipAction(act);
      nnpolDecodeAction(act, in[SNAP_FLOATS], in[SNAP_FLOATS + 1], &cmd);
      packCmd(&cmd, out);
      fwrite(out, sizeof(float), CMD_FLOATS, stdout);
    }
  } else {
    fprintf(stderr, "unknown mode %s\n", mode);
    return 2;
  }
  return 0;
}
