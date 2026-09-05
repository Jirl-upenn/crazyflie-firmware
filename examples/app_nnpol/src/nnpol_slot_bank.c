/**
 * nnpol_slot_bank.c — see nnpol_slot_bank.h.
 *
 * Upload protocol (the ground station's side is crazyflie-ros
 * crazyradio_driver nnpol_slot_clbk):
 *   1. host writes nnpol.slotErase = i; the app task erases sector 7+i
 *      (nnpolSlotBankErase) and reports through nnpol.slotState;
 *   2. host streams policy_slot.bin with cflib's MEM_TYPE_APP write at
 *      offset i * NNPOL_SLOT_BYTES: strictly sequential 24-byte chunks,
 *      each acknowledged before the next, so every chunk is whole words
 *      at a word-aligned offset (the blob is padded to 4 bytes). Each
 *      word is programmed as it arrives — ~16 us of flash stall per word
 *      inside the CRTP task, invisible to the 1 kHz loop;
 *   3. host writes nnpol.slot = i; the app task validates the slot (CRC
 *      over the whole blob) and publishes its identity in the nnpol.*
 *      params, which the existing hash check then verifies.
 *
 * Flash rules honoured: a word is programmed only if it reads erased, or
 * already holds exactly the data being written (cflib retransmits a chunk
 * whose ack was lost, and re-programming identical bits is harmless);
 * anything else is refused so a stale slot can never be half-overwritten.
 */
#include <string.h>

#include "FreeRTOS.h"
#include "task.h"

#include "stm32fxxx.h"
#include "stm32f4xx_flash.h"
#include "stm32f4xx_iwdg.h"

#include "mem.h"
#include "supervisor.h"
#include "watchdog.h"

#include "nnpol_slot_bank.h"

static uint8_t validMask = 0;
static nnpolSlotStatus_t slotStatus[NNPOL_NUM_SLOTS];
static volatile bool bankLocked = false;
static uint8_t writeSlot = NNPOL_SLOT_NONE;
static uint32_t writtenBytes = 0;

static uint32_t slotAddress(uint8_t slot)
{
  return NNPOL_SLOT_BASE + (uint32_t)slot * NNPOL_SLOT_BYTES;
}

const nnpolSlotHeader_t* nnpolSlotBankHeader(uint8_t slot)
{
  if (slot >= NNPOL_NUM_SLOTS) {
    return NULL;
  }
  return (const nnpolSlotHeader_t*)slotAddress(slot);
}

uint8_t nnpolSlotBankRescan(void)
{
  uint8_t mask = 0;
  for (uint8_t i = 0; i < NNPOL_NUM_SLOTS; i++) {
    slotStatus[i] = nnpolSlotValidate((const void*)slotAddress(i), NNPOL_SLOT_BYTES);
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

static bool writesAllowed(void)
{
  return !bankLocked && !supervisorIsArmed();
}

// --------------------------------------------------------------------------
// Erase
// --------------------------------------------------------------------------

/* watchdog.c: prescaler 32, reload 188 -> 100-353 ms depending on the LSI.
 * A 128 KB sector erase takes 1-2 s (typ.) with the core stalled, so the
 * reload is widened to the maximum (prescaler 256, reload 4095: 22-61 s)
 * for the erase and restored after. IWDG_PR/RLR stay writable after the
 * watchdog is running; the PVU/RVU flags say when the update has landed. */
static void watchdogWiden(bool widen)
{
  IWDG_WriteAccessCmd(IWDG_WriteAccess_Enable);
  IWDG_SetPrescaler(widen ? IWDG_Prescaler_256 : IWDG_Prescaler_32);
  IWDG_SetReload(widen ? 0xFFF : 188);
  while (IWDG_GetFlagStatus(IWDG_FLAG_PVU) == SET || IWDG_GetFlagStatus(IWDG_FLAG_RVU) == SET) {
  }
  IWDG_ReloadCounter();
}

bool nnpolSlotBankErase(uint8_t slot)
{
  if (slot >= NNPOL_NUM_SLOTS || !writesAllowed()) {
    return false;
  }
  /* Slot i lives in sector 8 + i (NNPOL_SLOT_BASE is sector 8's start);
   * the std-periph sector codes step by 8 per sector. */
  const uint32_t sector = FLASH_Sector_8 + (uint32_t)slot * (FLASH_Sector_9 - FLASH_Sector_8);

  watchdogWiden(true);
  FLASH_Unlock();
  FLASH_ClearFlag(FLASH_FLAG_EOP | FLASH_FLAG_OPERR | FLASH_FLAG_WRPERR |
                  FLASH_FLAG_PGAERR | FLASH_FLAG_PGPERR | FLASH_FLAG_PGSERR);
  const FLASH_Status st = FLASH_EraseSector(sector, VoltageRange_3);
  FLASH_Lock();
  watchdogWiden(false);
  watchdogReset();

  writeSlot = slot;
  writtenBytes = 0;
  nnpolSlotBankRescan();
  return st == FLASH_COMPLETE;
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
  memcpy(buffer, (const void*)(NNPOL_SLOT_BASE + memAddr), readLen);
  return true;
}

static bool handleMemWrite(const uint8_t internal_id, const uint32_t memAddr,
                           const uint8_t writeLen, const uint8_t* buffer)
{
  (void)internal_id;
  if (!writesAllowed()) {
    return false;
  }
  if (memAddr + writeLen > NNPOL_NUM_SLOTS * NNPOL_SLOT_BYTES
      || (memAddr & 3u) != 0u || (writeLen & 3u) != 0u) {
    /* Word-granular only (see the protocol note above). */
    return false;
  }
  const uint8_t slot = (uint8_t)(memAddr / NNPOL_SLOT_BYTES);
  if ((memAddr + writeLen - 1u) / NNPOL_SLOT_BYTES != slot) {
    return false;   /* a chunk never straddles two slots */
  }

  FLASH_Unlock();
  FLASH_ClearFlag(FLASH_FLAG_EOP | FLASH_FLAG_OPERR | FLASH_FLAG_WRPERR |
                  FLASH_FLAG_PGAERR | FLASH_FLAG_PGPERR | FLASH_FLAG_PGSERR);
  bool ok = true;
  for (uint32_t off = 0; off < writeLen && ok; off += 4u) {
    uint32_t word;
    memcpy(&word, buffer + off, 4u);
    const uint32_t addr = NNPOL_SLOT_BASE + memAddr + off;
    const uint32_t current = *(const volatile uint32_t*)addr;
    if (current == word) {
      continue;                     /* retransmit of an already-programmed chunk */
    }
    if (current != 0xFFFFFFFFu) {
      ok = false;                   /* not erased: refuse rather than corrupt */
      break;
    }
    ok = (FLASH_ProgramWord(addr, word) == FLASH_COMPLETE);
  }
  FLASH_Lock();

  if (ok) {
    if (writeSlot != slot) {
      writeSlot = slot;
      writtenBytes = 0;
    }
    if (memAddr - (uint32_t)slot * NNPOL_SLOT_BYTES + writeLen > writtenBytes) {
      writtenBytes = memAddr - (uint32_t)slot * NNPOL_SLOT_BYTES + writeLen;
    }
    /* The bank's validity is unknown mid-upload: the select re-validates. */
    validMask &= (uint8_t)~(1u << slot);
    slotStatus[slot] = NNPOL_SLOT_BAD_CRC;
  }
  return ok;
}

static const MemoryHandlerDef_t memDef = {
  .type = MEM_TYPE_APP,
  .getSize = handleMemGetSize,
  .read = handleMemRead,
  .write = handleMemWrite,
};

void nnpolSlotBankInit(void)
{
  nnpolSlotBankRescan();
  memoryRegisterHandler(&memDef);
}
