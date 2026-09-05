/**
 * nnpol_slot_format.c — validation of a policy slot blob (nnpol_slot.h).
 * Pure C, no firmware includes: the host parity harness runs exactly this
 * over a policy_slot.bin file, the firmware runs it over a flash slot.
 */
#include <string.h>

#include "nnpol_slot.h"

uint32_t nnpolCrc32(const uint8_t* data, uint32_t len, uint32_t seed)
{
  uint32_t crc = seed ^ 0xFFFFFFFFu;
  for (uint32_t i = 0; i < len; i++) {
    crc ^= data[i];
    for (int b = 0; b < 8; b++) {
      const uint32_t mask = 0u - (crc & 1u);
      crc = (crc >> 1) ^ (0xEDB88320u & mask);
    }
  }
  return crc ^ 0xFFFFFFFFu;
}

const char* nnpolSlotStatusName(nnpolSlotStatus_t s)
{
  switch (s) {
    case NNPOL_SLOT_OK: return "ok";
    case NNPOL_SLOT_EMPTY: return "empty";
    case NNPOL_SLOT_BAD_MAGIC: return "bad magic";
    case NNPOL_SLOT_BAD_SIZE: return "bad size";
    case NNPOL_SLOT_BAD_DIMS: return "bad dims";
    case NNPOL_SLOT_BAD_CRC: return "bad crc";
    default: return "?";
  }
}

nnpolSlotStatus_t nnpolSlotValidate(const void* base, uint32_t capacityBytes)
{
  const nnpolSlotHeader_t* h = (const nnpolSlotHeader_t*)base;
  if (capacityBytes < NNPOL_SLOT_HEADER_BYTES) {
    return NNPOL_SLOT_BAD_SIZE;
  }
  if (h->magic == 0xFFFFFFFFu) {
    return NNPOL_SLOT_EMPTY;
  }
  if (h->magic != NNPOL_SLOT_MAGIC) {
    return NNPOL_SLOT_BAD_MAGIC;
  }
  if (h->headerBytes != NNPOL_SLOT_HEADER_BYTES || h->totalBytes > capacityBytes
      || h->totalBytes < h->headerBytes || (h->totalBytes & 3u) != 0u) {
    return NNPOL_SLOT_BAD_SIZE;
  }
  if (h->actionDim != NNPOL_ACTION_DIM || h->obsDim == 0 || h->obsDim > NNPOL_MAX_OBS_DIM
      || h->numLayers == 0 || h->numLayers > NNPOL_MAX_LAYERS
      || (h->activation != NNPOL_ACT_TANH && h->activation != NNPOL_ACT_RELU)
      || h->actionKind > NNPOL_ACTION_KIND_MOTOR || h->mixer > NNPOL_MIXER_RPM_DIRECT) {
    return NNPOL_SLOT_BAD_DIMS;
  }
  if (h->layerDims[0] != h->obsDim || h->layerDims[h->numLayers] != h->actionDim) {
    return NNPOL_SLOT_BAD_DIMS;
  }
  uint32_t floats = 0;
  for (int l = 0; l < (int)h->numLayers; l++) {
    const uint32_t in = h->layerDims[l], out = h->layerDims[l + 1];
    if (out == 0 || (l + 1 < (int)h->numLayers && out > NNPOL_MAX_HIDDEN)) {
      return NNPOL_SLOT_BAD_DIMS;
    }
    floats += in * out + out;
  }
  if (floats != h->payloadFloats
      || h->headerBytes + nnpolSlotPayloadBytes(floats) != h->totalBytes) {
    return NNPOL_SLOT_BAD_SIZE;
  }
  const uint8_t* bytes = (const uint8_t*)base;
  if (nnpolCrc32(bytes + 16, h->totalBytes - 16, 0) != h->crc32) {
    return NNPOL_SLOT_BAD_CRC;
  }
  return NNPOL_SLOT_OK;
}
