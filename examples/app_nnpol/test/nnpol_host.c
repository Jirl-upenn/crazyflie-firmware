/**
 * nnpol_host.c — the app's pure-C core as a host shared library
 * (libnnpol_core.so, built by test/Makefile), with a flat C entry point for
 * ctypes. It links the SAME source files the firmware compiles, so a step
 * here is the step the vehicle runs.
 *
 * The ground station loads it at launch, runs the blob it compiled and the
 * track it is about to push through it, and compares the result with its own
 * policy (the shadow) before it uploads anything. nnpolHostVersion() must
 * equal the vehicle's nnpol.coreVer.
 *
 * A session is one policy on one track:
 *
 *     s = nnpolHostOpen(blob, len, &status)       NULL unless status == 0
 *     nnpolHostSetContext(s, ctx, len)            0 = ok (nnpolCtxStatus_t)
 *     nnpolHostEngage(s, thrustMax, rpmScale)     start of a run
 *     nnpolHostStep(s, state, out)                1 = a command was produced
 *     nnpolHostClose(s)
 *
 * state[17] = pos(3) m, quat_wxyz(4), vel_w(3) m/s, gyro(3) DEG/s, rotor rpm(4)
 * out[86]   = obs(64), act_raw(4), act(4), rpm(4), thrust_N(4), motor_value(4),
 *             gate_idx, passed
 */
#include <stdlib.h>
#include <string.h>

#include "nnpol.h"

#define SLOT_CAPACITY 0x20000u
#define NNPOL_HOST_STATE_FLOATS 17
#define NNPOL_HOST_OUT_FLOATS (NNPOL_MAX_OBS_DIM + 22)

typedef struct {
  nnpolCore_t core;
  uint8_t slot[SLOT_CAPACITY] __attribute__((aligned(8)));
} nnpolHost_t;

uint32_t nnpolHostVersion(void) { return NNPOL_CORE_VERSION; }
uint32_t nnpolHostMaxGates(void) { return NNPOL_MAX_GATES; }
uint32_t nnpolHostMaxObsDim(void) { return NNPOL_MAX_OBS_DIM; }
uint32_t nnpolHostMaxHidden(void) { return NNPOL_MAX_HIDDEN; }
uint32_t nnpolHostMaxLayers(void) { return NNPOL_MAX_LAYERS; }
uint32_t nnpolHostSlotBytes(void) { return SLOT_CAPACITY; }
uint32_t nnpolHostContextMaxBytes(void) { return NNPOL_CTX_MAX_BYTES; }
uint32_t nnpolHostOutFloats(void) { return NNPOL_HOST_OUT_FLOATS; }

const char* nnpolHostSlotStatusName(int status) { return nnpolSlotStatusName((nnpolSlotStatus_t)status); }
const char* nnpolHostCtxStatusName(int status) { return nnpolCtxStatusName((nnpolCtxStatus_t)status); }

void* nnpolHostOpen(const uint8_t* blob, uint32_t len, int* status)
{
  nnpolHost_t* s = (nnpolHost_t*)malloc(sizeof(nnpolHost_t));
  if (s == NULL) {
    *status = -1;
    return NULL;
  }
  /* The slot image exactly as an upload leaves it: erased (0xFF) beyond the blob. */
  memset(s->slot, 0xFF, sizeof(s->slot));
  nnpolCoreInit(&s->core);
  if (len > SLOT_CAPACITY) {
    *status = (int)NNPOL_SLOT_BAD_SIZE;
    free(s);
    return NULL;
  }
  memcpy(s->slot, blob, len);
  const nnpolSlotStatus_t st = nnpolCoreSetPolicy(&s->core, s->slot, SLOT_CAPACITY);
  *status = (int)st;
  if (st != NNPOL_SLOT_OK) {
    free(s);
    return NULL;
  }
  return s;
}

void nnpolHostClose(void* session) { free(session); }

int nnpolHostSetContext(void* session, const uint8_t* ctx, uint32_t len)
{
  nnpolHost_t* s = (nnpolHost_t*)session;
  nnpolRaceTrack_t track;
  /* the vehicle parses its whole staging region: zero-filled past the blob */
  uint8_t staged[NNPOL_CTX_MAX_BYTES];
  memset(staged, 0, sizeof(staged));
  if (len > sizeof(staged)) {
    return (int)NNPOL_CTX_BAD_SIZE;
  }
  memcpy(staged, ctx, len);
  const nnpolCtxStatus_t st = nnpolContextParse(staged, sizeof(staged), &track, NULL);
  if (st == NNPOL_CTX_OK) {
    nnpolCoreSetTrack(&s->core, &track);
  }
  return (int)st;
}

void nnpolHostEngage(void* session, float thrustMax, float rpmScale)
{
  nnpolCoreEngage(&((nnpolHost_t*)session)->core, thrustMax, rpmScale);
}

int nnpolHostStep(void* session, const float* state, float* out)
{
  nnpolHost_t* s = (nnpolHost_t*)session;
  nnpolState_t st;
  memcpy(st.pos, state, 3 * sizeof(float));
  memcpy(st.quat_wxyz, state + 3, 4 * sizeof(float));
  memcpy(st.vel_w, state + 7, 3 * sizeof(float));
  memcpy(st.gyro_deg, state + 10, 3 * sizeof(float));
  memcpy(st.rpm, state + 13, 4 * sizeof(float));
  nnpolStepOut_t o;
  memset(&o, 0, sizeof(o));
  const bool ok = nnpolCoreStep(&s->core, &st, &o);
  float* p = out;
  memcpy(p, o.obs, sizeof(o.obs)); p += NNPOL_MAX_OBS_DIM;
  memcpy(p, o.actRaw, sizeof(o.actRaw)); p += 4;
  memcpy(p, o.act, sizeof(o.act)); p += 4;
  memcpy(p, o.cmd.rpm, sizeof(o.cmd.rpm)); p += 4;
  memcpy(p, o.cmd.thrust, sizeof(o.cmd.thrust)); p += 4;
  memcpy(p, o.cmd.motorPwm, sizeof(o.cmd.motorPwm)); p += 4;
  p[0] = (float)o.gateIdx;
  p[1] = (float)o.passed;
  return ok ? 1 : 0;
}

/** The network alone: obs[obsDim] -> act_raw[4]. */
void nnpolHostForward(void* session, const float* obs, float* actRaw)
{
  nnpolForward(&((nnpolHost_t*)session)->core.policy, obs, actRaw);
}
