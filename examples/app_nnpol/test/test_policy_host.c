/**
 * test_policy_host.c — host-side parity harness for app_nnpol.
 *
 * Compiled with plain gcc against the SAME pure-C modules the firmware
 * builds (slot format, interpreter, observation, decode), loading a
 * policy_slot.bin the way the firmware reads a flash slot, so what is
 * checked is the flight code itself, not a transcription of it. Driven by
 * run_host_parity.py, which feeds binary records on stdin and compares
 * the answers against the trainer's numpy reference
 * (policy_reference.npz) and against its own numpy transcriptions of the
 * observation and decode.
 *
 * Usage: nnpol_host <mode> <policy_slot.bin>
 *
 * Modes; all records are little-endian float32 streams:
 *   info   : prints "status obsDim actionDim numLayers kind mixer taskId ctrlHz"
 *   policy : obs[obsDim]                          -> act[4]   (raw, unclipped)
 *   obs    : snapshot[18]                         -> obs[obsDim]
 *   act    : action[4], vnom, rpmScale, yaw0      -> cmd[14]
 *   full   : snapshot[18], vnom, rpmScale         -> cmd[14]  (obs -> net -> clip -> decode)
 *
 * snapshot[18] = [t, off(3), pos(3), quat_wxyz(4), vel_w(3), gyro_deg(3), yaw0]
 * cmd[14]      = [ch0, ch1, ch2, thrustN, thrustPwm, thrustSetpoint,
 *                 rpm(4), motorPwm(4)]   — nnpolCmd_t flattened; which
 *                 columns are non-zero depends on the slot's actionKind
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "nnpol.h"

#define SNAP_FLOATS 18
#define CMD_FLOATS 14
#define ACT_IN_FLOATS (NNPOL_ACTION_DIM + 3)
#define SLOT_CAPACITY 0x20000u

static uint8_t g_slot[SLOT_CAPACITY];

static void unpackSnapshot(const float* in, nnpolSnapshot_t* s)
{
  s->t = in[0];
  memcpy(s->off, in + 1, 3 * sizeof(float));
  memcpy(s->pos, in + 4, 3 * sizeof(float));
  memcpy(s->quat_wxyz, in + 7, 4 * sizeof(float));
  memcpy(s->vel_w, in + 11, 3 * sizeof(float));
  memcpy(s->gyro_deg, in + 14, 3 * sizeof(float));
  s->yaw0 = in[17];
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
  if (argc != 3) {
    fprintf(stderr, "usage: %s {info|policy|obs|act|full} policy_slot.bin\n", argv[0]);
    return 2;
  }
  const char* mode = argv[1];

  /* Load the blob into the slot image exactly as an upload would leave it:
   * erased (0xFF) beyond the blob. */
  memset(g_slot, 0xFF, sizeof(g_slot));
  FILE* f = fopen(argv[2], "rb");
  if (f == NULL) {
    fprintf(stderr, "cannot open %s\n", argv[2]);
    return 2;
  }
  const size_t n = fread(g_slot, 1, sizeof(g_slot), f);
  fclose(f);
  if (n == 0) {
    fprintf(stderr, "empty blob\n");
    return 2;
  }
  const nnpolSlotStatus_t status = nnpolSlotValidate(g_slot, SLOT_CAPACITY);
  const nnpolSlotHeader_t* hdr = (const nnpolSlotHeader_t*)g_slot;

  if (strcmp(mode, "info") == 0) {
    printf("%d %d %d %d %d %d %d %d\n", (int)status, hdr->obsDim, hdr->actionDim,
           hdr->numLayers, hdr->actionKind, hdr->mixer, hdr->taskId, hdr->ctrlFreqHz);
    return status == NNPOL_SLOT_OK ? 0 : 1;
  }
  if (status != NNPOL_SLOT_OK) {
    fprintf(stderr, "blob rejected: %s\n", nnpolSlotStatusName(status));
    return 1;
  }
  const nnpolPolicy_t policy = { hdr, nnpolSlotWeights(hdr) };
  const int obsDim = hdr->obsDim;

  if (strcmp(mode, "policy") == 0) {
    float obs[NNPOL_MAX_OBS_DIM], act[NNPOL_ACTION_DIM];
    while (fread(obs, sizeof(float), obsDim, stdin) == (size_t)obsDim) {
      nnpolForward(&policy, obs, act);
      fwrite(act, sizeof(float), NNPOL_ACTION_DIM, stdout);
    }
  } else if (strcmp(mode, "obs") == 0) {
    float in[SNAP_FLOATS], obs[NNPOL_MAX_OBS_DIM];
    nnpolSnapshot_t snap;
    while (fread(in, sizeof(float), SNAP_FLOATS, stdin) == SNAP_FLOATS) {
      unpackSnapshot(in, &snap);
      const int got = nnpolBuildObs(hdr, &snap, obs);
      if (got != obsDim) {
        fprintf(stderr, "observation builder returned %d, header says %d\n", got, obsDim);
        return 1;
      }
      fwrite(obs, sizeof(float), obsDim, stdout);
    }
  } else if (strcmp(mode, "act") == 0) {
    float in[ACT_IN_FLOATS], out[CMD_FLOATS];
    nnpolCmd_t cmd;
    while (fread(in, sizeof(float), ACT_IN_FLOATS, stdin) == ACT_IN_FLOATS) {
      nnpolDecodeAction(hdr, in, in[NNPOL_ACTION_DIM], in[NNPOL_ACTION_DIM + 1],
                        in[NNPOL_ACTION_DIM + 2], &cmd);
      packCmd(&cmd, out);
      fwrite(out, sizeof(float), CMD_FLOATS, stdout);
    }
  } else if (strcmp(mode, "full") == 0) {
    float in[SNAP_FLOATS + 2], obs[NNPOL_MAX_OBS_DIM], act[NNPOL_ACTION_DIM], out[CMD_FLOATS];
    nnpolSnapshot_t snap;
    nnpolCmd_t cmd;
    while (fread(in, sizeof(float), SNAP_FLOATS + 2, stdin) == SNAP_FLOATS + 2) {
      unpackSnapshot(in, &snap);
      if (nnpolBuildObs(hdr, &snap, obs) != obsDim) {
        return 1;
      }
      nnpolForward(&policy, obs, act);
      clipAction(act);
      nnpolDecodeAction(hdr, act, in[SNAP_FLOATS], in[SNAP_FLOATS + 1], snap.yaw0, &cmd);
      packCmd(&cmd, out);
      fwrite(out, sizeof(float), CMD_FLOATS, stdout);
    }
  } else {
    fprintf(stderr, "unknown mode %s\n", mode);
    return 2;
  }
  return 0;
}
