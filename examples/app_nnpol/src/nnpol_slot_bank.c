/**
 * nnpol_slot_bank.c — see nnpol_slot_bank.h.
 *
 * Upload protocol (the ground station's side is crazyflie-ros
 * crazyradio_driver nnpol_slot_clbk):
 *   1. host writes nnpol.slotErase = i; the app task empties the slot
 *      (instant) and reports through nnpol.slotState;
 *   2. host streams policy_slot.bin with cflib's MEM_TYPE_APP write at
 *      offset i * NNPOL_SLOT_BYTES, in acknowledged chunks; each chunk is
 *      copied into the slot as it arrives (a retransmit rewrites the same
 *      bytes, harmlessly). nnpol.writeSlot / writtenBytes show progress;
 *   3. host writes nnpol.slot = i; the app task validates the slot (CRC
 *      over the whole blob) and publishes its identity in the nnpol.*
 *      params, which the existing hash check then verifies.
 *
 * Writes and erases are refused while a policy is engaged (the interpreter
 * is reading the slot) or the vehicle is armed.
 */
#include <string.h>

#include "mem.h"
#include "supervisor.h"

#include "nnpol_slot_bank.h"

static uint8_t bank[NNPOL_NUM_SLOTS * NNPOL_SLOT_BYTES] __attribute__((aligned(4)));

static uint8_t validMask = 0;
static nnpolSlotStatus_t slotStatus[NNPOL_NUM_SLOTS];
static volatile bool bankLocked = false;
static uint8_t writeSlot = NNPOL_SLOT_NONE;
static uint32_t writtenBytes = 0;

static uint8_t* slotBase(uint8_t slot)
{
  return bank + (uint32_t)slot * NNPOL_SLOT_BYTES;
}

const nnpolSlotHeader_t* nnpolSlotBankHeader(uint8_t slot)
{
  if (slot >= NNPOL_NUM_SLOTS) {
    return NULL;
  }
  return (const nnpolSlotHeader_t*)slotBase(slot);
}

uint8_t nnpolSlotBankRescan(void)
{
  uint8_t mask = 0;
  for (uint8_t i = 0; i < NNPOL_NUM_SLOTS; i++) {
    slotStatus[i] = nnpolSlotValidate(slotBase(i), NNPOL_SLOT_BYTES);
    if (slotStatus[i] == NNPOL_SLOT_OK) {
      mask |= (uint8_t)(1u << i);
    }
  }
  validMask = mask;
  return mask;
}

uint8_t nnpolSlotBankValidMask(void) { return validMask; }

nnpolSlotStatus_t nnpolSlotBankStatus(uint8_t slot)
{
  return slot < NNPOL_NUM_SLOTS ? slotStatus[slot] : NNPOL_SLOT_BAD_SIZE;
}

void nnpolSlotBankSetLocked(bool locked) { bankLocked = locked; }
uint8_t nnpolSlotBankWriteSlot(void) { return writeSlot; }
uint32_t nnpolSlotBankWrittenBytes(void) { return writtenBytes; }

/* The slot may be rewritten whenever the interpreter is not reading it
 * and the vehicle is on the ground. Armed is NOT a reason to refuse: with
 * CONFIG_MOTORS_REQUIRE_ARMING the ground station arms before takeoff,
 * and a controller restart on an armed, landed vehicle must still be able
 * to upload its checkpoint. */
static bool writesAllowed(void)
{
  return !bankLocked && !supervisorIsFlying();
}

bool nnpolSlotBankErase(uint8_t slot)
{
  if (slot >= NNPOL_NUM_SLOTS || !writesAllowed()) {
    return false;
  }
  memset(slotBase(slot), 0xFF, NNPOL_SLOT_BYTES);
  writeSlot = slot;
  writtenBytes = 0;
  nnpolSlotBankRescan();
  return true;
}

// --------------------------------------------------------------------------
// MEM_TYPE_APP handler: the whole bank as one linear memory
// --------------------------------------------------------------------------

static uint32_t handleMemGetSize(const uint8_t internal_id)
{
  (void)internal_id;
  return NNPOL_NUM_SLOTS * NNPOL_SLOT_BYTES;
}

static bool handleMemRead(const uint8_t internal_id, const uint32_t memAddr,
                          const uint8_t readLen, uint8_t* buffer)
{
  (void)internal_id;
  if (memAddr + readLen > NNPOL_NUM_SLOTS * NNPOL_SLOT_BYTES) {
    return false;
  }
  memcpy(buffer, bank + memAddr, readLen);
  return true;
}

static bool handleMemWrite(const uint8_t internal_id, const uint32_t memAddr,
                           const uint8_t writeLen, const uint8_t* buffer)
{
  (void)internal_id;
  if (!writesAllowed()) {
    return false;
  }
  if (memAddr + writeLen > NNPOL_NUM_SLOTS * NNPOL_SLOT_BYTES) {
    return false;
  }
  const uint8_t slot = (uint8_t)(memAddr / NNPOL_SLOT_BYTES);
  if ((memAddr + writeLen - 1u) / NNPOL_SLOT_BYTES != slot) {
    return false;   /* a chunk never straddles two slots */
  }
  memcpy(bank + memAddr, buffer, writeLen);
  if (writeSlot != slot) {
    writeSlot = slot;
    writtenBytes = 0;
  }
  const uint32_t end = memAddr - (uint32_t)slot * NNPOL_SLOT_BYTES + writeLen;
  if (end > writtenBytes) {
    writtenBytes = end;
  }
  /* The slot's validity is unknown mid-upload: the select re-validates. */
  validMask &= (uint8_t)~(1u << slot);
  slotStatus[slot] = NNPOL_SLOT_BAD_CRC;
  return true;
}

static const MemoryHandlerDef_t memDef = {
  .type = MEM_TYPE_APP,
  .getSize = handleMemGetSize,
  .read = handleMemRead,
  .write = handleMemWrite,
};

void nnpolSlotBankInit(void)
{
  static bool isInit = false;
  if (isInit) {
    return;   /* controllerOutOfTreeInit runs again on every controller switch */
  }
  memset(bank, 0xFF, sizeof(bank));
  nnpolSlotBankRescan();
  memoryRegisterHandler(&memDef);   /* must precede crtp_mem's memBlockHandlerRegistration */
  isInit = true;
}
