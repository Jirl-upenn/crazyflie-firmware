/**
 * nnpol_slot_bank.h — the policy slot: ONE policy blob (nnpol_slot.h)
 * resident in RAM, uploaded over the radio through a MEM_TYPE_APP memory
 * handler at every controller start and selected at run time. Nothing
 * here touches flash: the app image is flashed once and never rewritten,
 * a checkpoint change is a config.yaml edit, and a power cycle simply
 * empties the slot until the next upload (a few seconds, on the ground).
 *
 * The slot is sized for the 64x64x64 family (fp32 weights, 41-46 KB with
 * the observations flown here); the exporter refuses a blob that would
 * not fit, and so does the validator. Firmware-only (mem subsystem,
 * supervisor); the format itself is in nnpol_slot_format.c.
 */
#ifndef NNPOL_SLOT_BANK_H
#define NNPOL_SLOT_BANK_H

#include <stdbool.h>
#include <stdint.h>

#include "nnpol_slot.h"

/** Bytes one slot holds: 48 KB, plain .bss in the F405's 128 KB SRAM.
 * A 64x64x64 checkpoint needs 256 + 4 * (obs*64 + 64 + 2*(64*64 + 64) +
 * 64*4 + 4) bytes: 41.2 KB at obs 25, 45.1 KB at obs 40, 46.1 KB at
 * obs 44; an observation wider than 55 no longer fits and the exporter
 * refuses it. Raise it only against the build's RAM report. */
#define NNPOL_SLOT_BYTES 0xC000u
#define NNPOL_NUM_SLOTS 1u
#define NNPOL_SLOT_NONE 255u

/** Register the MEM_TYPE_APP handler and empty the slot. */
void nnpolSlotBankInit(void);

/** The slot's header (raw buffer; validate before trusting). */
const nnpolSlotHeader_t* nnpolSlotBankHeader(uint8_t slot);

/** Bit i set = slot i validates (nnpolSlotValidate == OK). Cached; refreshed
 * by nnpolSlotBankRescan, by an erase, and by a select. */
uint8_t nnpolSlotBankValidMask(void);
uint8_t nnpolSlotBankRescan(void);
nnpolSlotStatus_t nnpolSlotBankStatus(uint8_t slot);

/** Empty one slot (fill with 0xFF, the "erased" the validator recognises).
 * Instant; refused while the bank is locked or the vehicle armed. */
bool nnpolSlotBankErase(uint8_t slot);

/** Locked = an engaged policy is reading the slot: uploads and erases are
 * refused until it is released. */
void nnpolSlotBankSetLocked(bool locked);

/** Upload progress, for the nnpol.writeSlot / writtenBytes params. */
uint8_t nnpolSlotBankWriteSlot(void);
uint32_t nnpolSlotBankWrittenBytes(void);

#endif /* NNPOL_SLOT_BANK_H */
