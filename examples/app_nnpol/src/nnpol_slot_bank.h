/**
 * nnpol_slot_bank.h — the flash slot bank: NNPOL_NUM_SLOTS policy blobs
 * (nnpol_slot.h) resident in the internal-flash sectors the firmware
 * image never reaches, uploaded over the radio through a MEM_TYPE_APP
 * memory handler and selected at run time, plus the RAM region the engage
 * context is written to. Firmware-only (flash, IWDG,
 * mem subsystem); the format itself is in nnpol_slot_format.c.
 *
 * Geometry: STM32F405 sectors 7..11 are 128 KB each at 0x08060000..
 * 0x080FFFFF (sectors 0-3 are 16 KB, 4 is 64 KB, 5.. are 128 KB, so
 * sector 7 starts at 0x08060000, NOT 0x08080000 - that is sector 8, and a
 * base there put slot 4 at 0x08100000, past the end of flash: a bus fault
 * on the first rescan). The linker's FLASH region is 0x08004000 + 1008 KB
 * and the app image must end below 0x08060000 (368 KB of image after the
 * bootloader); nnpolSlotBankInit checks this at boot and disables the bank
 * rather than let an erase eat the image.
 */
#ifndef NNPOL_SLOT_BANK_H
#define NNPOL_SLOT_BANK_H

#include <stdbool.h>
#include <stdint.h>

#include "nnpol_slot.h"

#define NNPOL_SLOT_BASE 0x08060000u /* sector 7 */
#define NNPOL_SLOT_BYTES 0x20000u
#define NNPOL_NUM_SLOTS 5u
#define NNPOL_SLOT_NONE 255u

/** Register the MEM_TYPE_APP handler and scan the bank. Must run before
 * the scheduler starts (memTask closes handler registration as soon as it
 * runs - appMain is too late, it races memTask); called from
 * controllerOutOfTreeInit. Idempotent. */
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

/** The engage context (nnpol.h, the track): a small RAM region mapped
 * right after the bank in the same MEM_TYPE_APP memory, at
 * NNPOL_CTX_MEM_OFFSET. Unlike the flash slots it may be written while
 * armed and flying - a policy is engaged from a hover - but not while a
 * policy is engaged. nnpolContextStaged returns the staged bytes and a
 * counter that changes with every accepted write; the app task parses and
 * validates them (nnpolContextParse) before anything uses them. */
#define NNPOL_CTX_MEM_OFFSET (NNPOL_NUM_SLOTS * NNPOL_SLOT_BYTES)
const uint8_t* nnpolContextStaged(uint32_t* writeCount);
void nnpolContextSetLocked(bool locked);

/** Upload bookkeeping, for the ground station (params): the slot the last
 * radio write landed in and how many bytes of it have been written since
 * its erase (255 / 0 when nothing is in progress). */
uint8_t nnpolSlotBankWriteSlot(void);
uint32_t nnpolSlotBankWrittenBytes(void);

#endif /* NNPOL_SLOT_BANK_H */
