/**
 * nnpol_slot_bank.h — the flash slot bank: NNPOL_NUM_SLOTS policy blobs
 * (nnpol_slot.h) resident in the internal-flash sectors the firmware
 * image never reaches, uploaded over the radio through a MEM_TYPE_APP
 * memory handler and selected at run time. Firmware-only (flash, IWDG,
 * mem subsystem); the format itself is in nnpol_slot_format.c.
 *
 * Geometry: STM32F405 sectors 7..11 are 128 KB each at 0x08080000..
 * 0x080FFFFF; the linker's FLASH region is 0x08004000 + 1008 KB, and the
 * app image ends well below 0x08080000 (check the map if the firmware
 * ever grows past 496 KB: the bank would then overlap the image and the
 * first erase would brick the flash until a reflash).
 */
#ifndef NNPOL_SLOT_BANK_H
#define NNPOL_SLOT_BANK_H

#include <stdbool.h>
#include <stdint.h>

#include "nnpol_slot.h"

#define NNPOL_SLOT_BASE 0x08080000u
#define NNPOL_SLOT_BYTES 0x20000u
#define NNPOL_NUM_SLOTS 5u
#define NNPOL_SLOT_NONE 255u

/** Register the MEM_TYPE_APP handler and scan the bank. */
void nnpolSlotBankInit(void);

/** The slot's header (raw flash; validate before trusting). */
const nnpolSlotHeader_t* nnpolSlotBankHeader(uint8_t slot);

/** Bit i set = slot i validates (nnpolSlotValidate == OK). Cached; refreshed
 * by nnpolSlotBankRescan, by an erase, and by a select. */
uint8_t nnpolSlotBankValidMask(void);
uint8_t nnpolSlotBankRescan(void);
nnpolSlotStatus_t nnpolSlotBankStatus(uint8_t slot);

/** Erase one slot's sector. TASK CONTEXT ONLY, ON THE GROUND: the erase
 * stalls every instruction fetch from flash for ~1-2 s, so nothing —
 * stabilizer, radio, FreeRTOS tick — runs meanwhile; the independent
 * watchdog is widened around it and restored after. Refuses (false) while
 * the bank is locked (policy engaged) or the vehicle is armed. */
bool nnpolSlotBankErase(uint8_t slot);

/** While locked, radio writes and erases are refused: set by the
 * controller for as long as a policy is engaged or the vehicle armed. */
void nnpolSlotBankSetLocked(bool locked);

/** Upload bookkeeping, for the ground station (params): the slot the last
 * radio write landed in and how many bytes of it have been written since
 * its erase (255 / 0 when nothing is in progress). */
uint8_t nnpolSlotBankWriteSlot(void);
uint32_t nnpolSlotBankWrittenBytes(void);

#endif /* NNPOL_SLOT_BANK_H */
