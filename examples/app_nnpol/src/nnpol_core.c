/**
 * nnpol_core.c — one policy step, end to end (nnpol.h). The firmware's app
 * task and the host library both run this file, which is the point of it.
 * Pure C: host-testable, no firmware includes.
 */
#include <string.h>

#include "nnpol.h"

void nnpolCoreInit(nnpolCore_t* c)
{
  memset(c, 0, sizeof(*c));
  c->rpmScale = 1.0f;
}

nnpolSlotStatus_t nnpolCoreSetPolicy(nnpolCore_t* c, const void* blob, uint32_t capacityBytes)
{
  c->policy.hdr = NULL;
  c->policy.weights = NULL;
  if (blob == NULL) {
    return NNPOL_SLOT_EMPTY;
  }
  const nnpolSlotStatus_t st = nnpolSlotValidate(blob, capacityBytes);
  if (st == NNPOL_SLOT_OK) {
    c->policy.hdr = (const nnpolSlotHeader_t*)blob;
    c->policy.weights = nnpolSlotWeights(c->policy.hdr);
  }
  return st;
}

void nnpolCoreSetTrack(nnpolCore_t* c, const nnpolRaceTrack_t* track)
{
  c->track = *track;
  nnpolRaceReset(&c->track, &c->race);
}

void nnpolCoreEngage(nnpolCore_t* c, float thrustMax, float rpmScale)
{
  c->thrustMax = thrustMax;
  c->rpmScale = rpmScale;
  nnpolRaceReset(&c->track, &c->race);
}

bool nnpolCoreStep(nnpolCore_t* c, const nnpolState_t* state, nnpolStepOut_t* out)
{
  const nnpolSlotHeader_t* h = c->policy.hdr;
  if (h == NULL || c->track.nGates == 0) {
    return false;
  }
  /* The gate state first, with this step's position: the observation then
   * targets the gate the vehicle is flying AT, as RacePlugin.step leaves it
   * for the observation function. */
  out->passed = nnpolRaceUpdateGate(&c->track, &c->race, state->pos) ? 1 : 0;
  out->gateIdx = c->race.gateIdx;

  if (nnpolBuildObs(h, &c->track, c->race.gateIdx, state, out->obs) != (int)h->obsDim) {
    return false;
  }
  nnpolForward(&c->policy, out->obs, out->actRaw);
  for (int i = 0; i < NNPOL_ACTION_DIM; i++) {
    float a = out->actRaw[i];
    if (a < -1.0f) { a = -1.0f; }
    if (a > 1.0f) { a = 1.0f; }
    out->act[i] = a;
  }
  nnpolDecodeAction(h, out->act, c->thrustMax, c->rpmScale, &out->cmd);
  return true;
}
