/**
 * test_policy_host.c — host-side parity harness for app_nnpol.
 *
 * Compiled with plain gcc against the SAME pure-C modules the firmware
 * builds (slot format, interpreter, observation, decode), loading a slot
 * blob the way the firmware reads a flash slot, so what is checked is the
 * flight code itself, not a transcription of it. Driven by
 * run_host_parity.py, which feeds binary records on stdin and compares the
 * answers against its own numpy transcriptions of the host implementation.
 *
 * Usage: nnpol_host <mode> <policy_slot.bin>
 *
 * Modes; all records are little-endian float32 streams:
 *   info   : prints "status obsDim actionDim numLayers kind mixer taskId ctrlHz"
 *   policy : obs[obsDim]                          -> act[4]   (raw, unclipped)
 *   obs    : snapshot[SNAP], track[TRACK]         -> obs[obsDim]
 *   act    : action[4], vnom, rpmScale            -> cmd[8]
 *   full   : snapshot[SNAP], track[TRACK], vnom, rpmScale
 *                                                 -> cmd[8]   (obs -> net -> clip -> decode)
 *   gate   : pos(3), gateSide, gateIdx, prevGateX, track[TRACK]
 *                                                 -> [gateIdx, prevGateX] (one crossing-test step)
 *
 * snapshot[19] = [t, pos(3), quat_wxyz(4), vel_w(3), gyro_deg(3), rpm_norm(4), gateIdx]
 * track[49]    = [nGates, gates(8 x 3), normals(8 x 3)]
 * cmd[8]       = [rpm(4), motorPwm(4)]   — nnpolCmd_t flattened
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "nnpol.h"

#define SNAP_FLOATS 19
#define TRACK_FLOATS (1 + 6 * NNPOL_MAX_GATES)
#define CMD_FLOATS 8
#define ACT_IN_FLOATS (NNPOL_ACTION_DIM + 2)
#define SLOT_CAPACITY 0x20000u

static uint8_t g_slot[SLOT_CAPACITY];

static void unpackSnapshot(const float* in, nnpolSnapshot_t* s)
{
  s->t = in[0];
  memcpy(s->pos, in + 1, 3 * sizeof(float));
  memcpy(s->quat_wxyz, in + 4, 4 * sizeof(float));
  memcpy(s->vel_w, in + 8, 3 * sizeof(float));
  memcpy(s->gyro_deg, in + 11, 3 * sizeof(float));
  memcpy(s->rpm, in + 14, 4 * sizeof(float));
  s->gateIdx = (uint8_t)in[18];
}

static void unpackTrack(const float* in, nnpolRaceTrack_t* t)
{
  t->nGates = (uint8_t)in[0];
  for (int i = 0; i < NNPOL_MAX_GATES; i++) {
    memcpy(t->gates[i], in + 1 + 3 * i, 3 * sizeof(float));
    memcpy(t->normals[i], in + 1 + 3 * NNPOL_MAX_GATES + 3 * i, 3 * sizeof(float));
  }
}

static void packCmd(const nnpolCmd_t* c, float* out)
{
  memcpy(out, c->rpm, 4 * sizeof(float));
  memcpy(out + 4, c->motorPwm, 4 * sizeof(float));
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
    fprintf(stderr, "usage: %s {info|policy|obs|act|full|gate} policy_slot.bin\n", argv[0]);
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
    float in[SNAP_FLOATS + TRACK_FLOATS], obs[NNPOL_MAX_OBS_DIM];
    nnpolSnapshot_t snap;
    nnpolRaceTrack_t track;
    while (fread(in, sizeof(float), SNAP_FLOATS + TRACK_FLOATS, stdin) == SNAP_FLOATS + TRACK_FLOATS) {
      unpackSnapshot(in, &snap);
      unpackTrack(in + SNAP_FLOATS, &track);
      const int got = nnpolBuildObs(hdr, &track, &snap, obs);
      if (got != obsDim) {
        fprintf(stderr, "observation builder returned %d, header says %d\n", got, obsDim);
        return 1;
      }
      fwrite(obs, sizeof(float), obsDim, stdout);
    }
  } else if (strcmp(mode, "gate") == 0) {
    float in[6 + TRACK_FLOATS], out[2];
    const size_t nin = 6 + TRACK_FLOATS;
    while (fread(in, sizeof(float), nin, stdin) == nin) {
      nnpolRaceTrack_t track;
      unpackTrack(in + 6, &track);
      uint8_t gateIdx = (uint8_t)in[4];
      float prevGateX = in[5];
      nnpolRaceUpdateGate(&track, in[3], in, &gateIdx, &prevGateX);
      out[0] = (float)gateIdx;
      out[1] = prevGateX;
      fwrite(out, sizeof(float), 2, stdout);
    }
  } else if (strcmp(mode, "act") == 0) {
    float in[ACT_IN_FLOATS], out[CMD_FLOATS];
    nnpolCmd_t cmd;
    while (fread(in, sizeof(float), ACT_IN_FLOATS, stdin) == ACT_IN_FLOATS) {
      nnpolDecodeAction(hdr, in, in[NNPOL_ACTION_DIM], in[NNPOL_ACTION_DIM + 1], &cmd);
      packCmd(&cmd, out);
      fwrite(out, sizeof(float), CMD_FLOATS, stdout);
    }
  } else if (strcmp(mode, "full") == 0) {
    const size_t nin = SNAP_FLOATS + TRACK_FLOATS + 2;
    float in[SNAP_FLOATS + TRACK_FLOATS + 2], obs[NNPOL_MAX_OBS_DIM], act[NNPOL_ACTION_DIM],
        out[CMD_FLOATS];
    nnpolSnapshot_t snap;
    nnpolRaceTrack_t track;
    nnpolCmd_t cmd;
    while (fread(in, sizeof(float), nin, stdin) == nin) {
      unpackSnapshot(in, &snap);
      unpackTrack(in + SNAP_FLOATS, &track);
      if (nnpolBuildObs(hdr, &track, &snap, obs) != obsDim) {
        return 1;
      }
      nnpolForward(&policy, obs, act);
      clipAction(act);
      nnpolDecodeAction(hdr, act, in[nin - 2], in[nin - 1], &cmd);
      packCmd(&cmd, out);
      fwrite(out, sizeof(float), CMD_FLOATS, stdout);
    }
  } else {
    fprintf(stderr, "unknown mode %s\n", mode);
    return 2;
  }
  return 0;
}
