/**
 * nnpol_controller.c — firmware glue for the onboard-policy app.
 *
 * onboard-rpm: the race task only, rpm-family policies only.
 *
 * Architecture (mjc_dronetests/docs/onboard_inference_plan.md section 3):
 * the OOT controller hook runs in the STABILIZER task at 1 kHz. On every
 * policy tick (every RATE_MAIN_LOOP / nnpol.ctrlHz stabilizer ticks) it
 * snapshots the EKF state + gyro + rotor speeds and notifies the APP task,
 * which runs one policy step (nnpolCoreStep, the same function the host
 * parity library exports): the gate state, the observation, the selected
 * slot's network through the
 * interpreter (nnpol_policy.c, 0.3-0.8 ms for a 2x128 — too close to the
 * 1 ms stabilizer budget to run inline, which is the whole reason for the
 * task split), decodes the action and publishes it into a sequence-locked
 * buffer. The controller applies the latest complete action as a
 * zero-order hold — the same hold the sim's 5 substeps per action
 * perform — by synthesizing the setpoint_t the same command would have
 * produced ARRIVING BY RADIO — on this branch always the per-motor shape:
 *
 *   MOTOR      modeMotorPwm setpoint (thrust = M1, attitudeRate = M2..M4)
 *              -> controllerPid's per-motor bypass, no onboard loop at all.
 *              onboard_only flies the rpm family only; a slot of another
 *              kind fails validation (nnpolSlotValidate) and is never
 *              selected. The Mellinger and the rate loop live on nnpol.
 *
 * The passthrough (nnpol.enable = 0, no usable slot, or a stale action)
 * is the stock PID: the only thing the ground station ever streams on
 * this branch is the SE3 cascade's RATE setpoint on /ctbr_cmd — takeoff,
 * hover, landing, and during a race the hold-here keepalive the vehicle
 * falls back to on takeover. There is no offboard policy route: no
 * attitude stream, no per-motor packet (the type-12 decoder is gone from
 * crtp_commander_generic.c on this branch), so controller 6 flies the
 * radio exactly as controller 1 does. Expected added latency: one 1 kHz
 * tick.
 *
 * THE APP IS CHECKPOINT-AGNOSTIC. Policies live in the flash slot bank
 * (nnpol_slot_bank.c): the ground station uploads policy_slot.bin blobs
 * over the radio and selects one with nnpol.slot; the app task validates
 * it (CRC) and publishes its identity in the read-only nnpol.hash0-3 /
 * obsDim / actDim / kind / ctrlHz params, which the ground station's
 * existing hash check verifies before engaging. Erase and select run in
 * the app task, only while nothing is engaged.
 *
 * Engage protocol (the ground station's side is the onboard client of
 * ros_quad_sim2real `onboard-rpm`): select the slot, verify nnpol.hash0-3
 * against the blob the host compiled and nnpol.coreVer against its parity
 * library, write the engage context (the track: MEM_TYPE_APP at
 * NNPOL_CTX_MEM_OFFSET) and verify nnpol.ctxCrc, push rpmScale and ctrlHz,
 * then set enable = 1. enable is ignored without a validated context. The
 * rising edge latches everything, so nothing written mid-run moves the track.
 *
 * Policy rate: nnpol.ctrlHz is writable and defaults to the selected
 * slot's trained ctrl_freq. The rising edge turns it into a tick divider
 * (RATE_MAIN_LOOP / ctrlHz, integer), so only divisors of 1000 run
 * exactly — anything else runs at the next divisor UP (48 -> 50 Hz) and
 * the read-only nnpol.runHz reports what actually engaged. The host
 * validates divisibility before pushing.
 *
 * Staleness: if the app task stops producing actions (overrun, crash),
 * the controller counts policy periods without a fresh sequence number.
 * From 2 periods it logs (nnpol.stale); from NNPOL_STALE_FALLBACK periods
 * it falls back to the radio setpoint — the shadow host command, which is
 * still arriving.
 */
#include <math.h>
#include <string.h>

#include "app.h"

#include "FreeRTOS.h"
#include "task.h"

#define DEBUG_MODULE "NNPOL"
#include "debug.h"

#include "controller.h"
#include "controller_pid.h"
#include "log.h"
#include "motors.h"
#include "param.h"
#include "platform_defaults.h"
#include "stabilizer_types.h"
#include "supervisor.h"
#include "usec_time.h"

#include "nnpol.h"
#include "nnpol_slot_bank.h"

/** Policy periods without a fresh action before the log counter runs. */
#define NNPOL_STALE_LOG_PERIODS 2
/** Policy periods without a fresh action before falling back to the radio
 * setpoint (the shadow host command). */
#define NNPOL_STALE_FALLBACK_PERIODS 5

/** How long a bench motor value is held after the write that set it. The
 * bench script rewrites it several times a second; when it stops, so do the
 * motors. */
#define NNPOL_BENCH_HOLD_TICKS M2T(300)

/* nnpol.slotState values. */
#define SLOT_STATE_IDLE 0
#define SLOT_STATE_ERASING 1
#define SLOT_STATE_ERASED 2
#define SLOT_STATE_ERASE_FAILED 3
#define SLOT_STATE_SELECT_FAILED 4
#define SLOT_STATE_SELECTED 5

// --------------------------------------------------------------------------
// Parameters
// --------------------------------------------------------------------------

static uint8_t nnpolEnable = 0;
static float paramRpmScale = 1.0f;  /* the host's motor_pwm.rpm_scale */
/* Latched from the param at the enable rising edge, so a mid-run param
 * write cannot rescale the motors; read by the app task. */
static float latchedRpmScale = 1.0f;
/* motorobs: the last valid reading per motor, rpm, held through DSHOT
 * telemetry dropouts exactly as the host's set_motor_rpm does. Until the
 * first reading the observation gets the slot's hover speed — a state a
 * flying drone is in, unlike zero. */
static float heldRpm[4];
static bool haveRpm[4];

/* The engage context: the track the host wrote (nnpol_slot_bank.c stages
 * the bytes), parsed and validated by the app task while nothing is
 * engaged. A policy cannot engage without a valid one. */
static nnpolRaceTrack_t pendingTrack;
static volatile bool pendingTrackOk = false;
static uint32_t ctxWriteSeen = 0;
static uint8_t paramCtxStatus = NNPOL_CTX_EMPTY;  /* nnpolCtxStatus_t of the last parse */
static uint32_t paramCtxCrc = 0;                  /* CRC of the validated context, 0 = none */
static uint8_t paramMaxGates = NNPOL_MAX_GATES;
static uint16_t paramCoreVer = NNPOL_CORE_VERSION;
static uint8_t logGateIdx = 0;

/* Thrust convention of this build, read-only, so the ground station can
 * refuse a vehicle whose motor values mean something else than it sends.
 * motorsCompensateBatteryVoltage reads every 16-bit motor value - the
 * per-motor path included - as a FORCE, value / 65535 x THRUST_MAX newtons,
 * and turns it into a motor voltage at the measured battery voltage. The
 * app's decode (nnpol_action.c) sends exactly that, so the build must
 * compensate. */
#ifndef CONFIG_ENABLE_THRUST_BAT_COMPENSATED
#error "app_nnpol sends motor values as forces: build with CONFIG_ENABLE_THRUST_BAT_COMPENSATED"
#endif
static uint8_t paramBatComp = 1;
static float paramThrustMax = THRUST_MAX;

/* Bench sweep (props on, vehicle restrained): one 16-bit value applied to all
 * four motors through the same per-motor path a policy's action takes, while
 * armed and not engaged, for NNPOL_BENCH_HOLD_TICKS after each write. Used to
 * measure delivered rotor speed against the commanded value. */
static uint16_t paramBenchPwm = 0;
static uint32_t benchTick = 0;
static uint16_t logBench = 0;

static void benchWritten(void)
{
  benchTick = xTaskGetTickCount();
}

/* Policy rate. Writable (the host's policy.inference_hz), defaulting to the
 * selected slot's trained rate; latched into a tick divider at the enable
 * rising edge. */
static uint16_t paramCtrlHz = 100;
/* Read-only: the rate the current/last engaged run actually steps at,
 * RATE_MAIN_LOOP / divider. Differs from ctrlHz only for a non-divisor. */
static uint16_t paramRunHz = 100;

/* Slot bank control (writable) and its readback (read-only). */
static uint8_t paramSlot = NNPOL_SLOT_NONE;       /* wanted slot; 255 = none */
static uint8_t paramSlotSel = NNPOL_SLOT_NONE;    /* selected & validated slot */
static uint8_t paramSlotErase = NNPOL_SLOT_NONE;  /* write i to erase slot i */
static uint8_t paramSlotState = SLOT_STATE_IDLE;
static uint8_t paramSlotStatus = NNPOL_SLOT_EMPTY; /* nnpolSlotStatus_t of the last select */
static uint8_t paramSlotValid = 0;                /* bitmask */
static uint8_t paramNumSlots = NNPOL_NUM_SLOTS;
static uint8_t paramWriteSlot = NNPOL_SLOT_NONE;
static uint32_t paramWrittenBytes = 0;

/* Read-only identity/architecture of the SELECTED slot — checked by the
 * ground station before an onboard engage; a vehicle whose selected slot
 * came from a different checkpoint reports different words and the ROS
 * node refuses to engage. All zero while nothing is selected. */
static uint32_t paramHash0 = 0, paramHash1 = 0, paramHash2 = 0, paramHash3 = 0;
static uint16_t paramObsDim = 0;
static uint16_t paramActDim = 0;
static uint16_t paramKind = 0xFFFF;
static uint16_t paramTaskId = 0xFFFF;

// --------------------------------------------------------------------------
// The selected policy (owned by the app task; read by the stabilizer only
// through activeValid, which changes only while disengaged)
// --------------------------------------------------------------------------

/* The policy step's state: the selected policy, the latched track and the
 * gate state. Owned by the app task. */
static nnpolCore_t core;
static volatile bool activeValid = false;
/* Bumped by the stabilizer at every enable rising edge; the app task starts
 * a run (latches the track, resets the gate state) when it sees a new value. */
static volatile uint32_t engageCount = 0;
static uint32_t engageSeen = 0;

static void publishIdentity(const nnpolSlotHeader_t* h)
{
  if (h == NULL) {
    paramHash0 = paramHash1 = paramHash2 = paramHash3 = 0;
    paramObsDim = paramActDim = 0;
    paramKind = paramTaskId = 0xFFFF;
    return;
  }
  paramHash0 = h->hash[0]; paramHash1 = h->hash[1];
  paramHash2 = h->hash[2]; paramHash3 = h->hash[3];
  paramObsDim = h->obsDim;
  paramActDim = h->actionDim;
  paramKind = h->actionKind;
  paramTaskId = h->taskId;
  paramCtrlHz = h->ctrlFreqHz;   /* the trained default; the host may override */
}

/** Validate and adopt slot `slot` (255 = deselect). App task only, while
 * disengaged. */
static void selectSlot(uint8_t slot)
{
  activeValid = false;
  nnpolCoreSetPolicy(&core, NULL, 0);
  if (slot == NNPOL_SLOT_NONE || slot >= NNPOL_NUM_SLOTS) {
    paramSlotSel = NNPOL_SLOT_NONE;
    paramSlotStatus = NNPOL_SLOT_EMPTY;
    paramSlotState = SLOT_STATE_IDLE;
    publishIdentity(NULL);
    return;
  }
  const nnpolSlotHeader_t* h = nnpolSlotBankHeader(slot);
  const nnpolSlotStatus_t st = nnpolCoreSetPolicy(&core, h, NNPOL_SLOT_BYTES);
  paramSlotStatus = (uint8_t)st;
  nnpolSlotBankRescan();
  paramSlotValid = nnpolSlotBankValidMask();
  if (st != NNPOL_SLOT_OK) {
    paramSlotSel = NNPOL_SLOT_NONE;
    paramSlotState = SLOT_STATE_SELECT_FAILED;
    publishIdentity(NULL);
    DEBUG_PRINT("slot %d rejected: %s\n", slot, nnpolSlotStatusName(st));
    return;
  }
  if (nnpolObsDim(h) != (int)h->obsDim) {
    /* Selectable but not flyable: the identity is published so the ground
     * station can say which run it is, and every policy tick then counts
     * an obsErr instead of producing an action. */
    DEBUG_PRINT("slot %d: no observation builder for task %d / %d-dim\n",
                slot, h->taskId, h->obsDim);
  }
  publishIdentity(h);
  activeValid = true;
  paramSlotSel = slot;
  paramSlotState = SLOT_STATE_SELECTED;
  DEBUG_PRINT("slot %d selected: %s obs %d kind %d %d Hz hash %08lx\n", slot, h->name,
              h->obsDim, h->actionKind, h->ctrlFreqHz, (unsigned long)h->hash[0]);
}

// --------------------------------------------------------------------------
// Stabilizer -> app task: the state snapshot (sequence lock)
// --------------------------------------------------------------------------
// Single writer (stabilizer task), single reader (app task). The writer
// bumps the counter to odd, writes, bumps to even; the reader retries
// until it sees the same even value on both sides of its copy.

/** The estimator state of one policy tick and its run clock. */
typedef struct {
  float t;               /* seconds since engage */
  nnpolState_t state;
} nnpolSnapshot_t;

static nnpolSnapshot_t snapBuf;
static volatile uint32_t snapLock = 0;

static void snapshotWrite(const nnpolSnapshot_t* s)
{
  snapLock++;
  __asm volatile ("" ::: "memory");
  snapBuf = *s;
  __asm volatile ("" ::: "memory");
  snapLock++;
}

static void snapshotRead(nnpolSnapshot_t* out)
{
  uint32_t a, b;
  do {
    a = snapLock;
    __asm volatile ("" ::: "memory");
    *out = snapBuf;
    __asm volatile ("" ::: "memory");
    b = snapLock;
  } while (a != b || (a & 1u));
}

// --------------------------------------------------------------------------
// App task -> stabilizer: the decoded action (sequence lock, same pattern)
// --------------------------------------------------------------------------

typedef struct {
  nnpolCmd_t cmd;
  uint32_t seq;              /* 0 = no action computed since boot */
} nnpolActionSlot_t;

static nnpolActionSlot_t actionSlot;
static volatile uint32_t actionLock = 0;

static void actionWrite(const nnpolCmd_t* cmd)
{
  actionLock++;
  __asm volatile ("" ::: "memory");
  actionSlot.cmd = *cmd;
  actionSlot.seq++;
  __asm volatile ("" ::: "memory");
  actionLock++;
}

static uint32_t actionRead(nnpolCmd_t* out)
{
  nnpolActionSlot_t slot;
  uint32_t a, b;
  do {
    a = actionLock;
    __asm volatile ("" ::: "memory");
    slot = actionSlot;
    __asm volatile ("" ::: "memory");
    b = actionLock;
  } while (a != b || (a & 1u));
  *out = slot.cmd;
  return slot.seq;
}

// --------------------------------------------------------------------------
// Logging
// --------------------------------------------------------------------------

static float logT;
static float logAct[NNPOL_ACTION_DIM];      /* clipped network output */
static uint16_t logPwm[4];                  /* the decoded 16-bit motor value */
static float logVb[3], logWb[3], logE0[3];  /* obs excerpts */
static uint16_t logUs;                      /* inference microseconds */
static uint16_t logStale;                   /* periods spent >= stale threshold */
static uint16_t logObsErr;                  /* observations refused (builder/dim) */
static uint32_t logSeq;

// --------------------------------------------------------------------------
// App task: slot housekeeping + observation + network + decode
// --------------------------------------------------------------------------

static TaskHandle_t appTaskHandle = NULL;

/** Erase / select requests from the params. Only while disengaged: the
 * stabilizer reads activeValid and the bank refuses writes
 * while locked, so nothing changes under a flying policy. */
static void housekeeping(void)
{
  if (nnpolEnable) {
    return;
  }
  if (paramSlotErase != NNPOL_SLOT_NONE) {
    const uint8_t slot = paramSlotErase;
    if (slot == paramSlotSel) {
      selectSlot(NNPOL_SLOT_NONE);   /* never erase under the selected policy */
      paramSlot = NNPOL_SLOT_NONE;
    }
    paramSlotState = SLOT_STATE_ERASING;
    const bool ok = nnpolSlotBankErase(slot);
    paramSlotState = ok ? SLOT_STATE_ERASED : SLOT_STATE_ERASE_FAILED;
    paramSlotValid = nnpolSlotBankValidMask();
    paramSlotErase = NNPOL_SLOT_NONE;
    DEBUG_PRINT("slot %d erase %s\n", slot, ok ? "ok" : "FAILED");
  }
  if (paramSlot != paramSlotSel) {
    selectSlot(paramSlot);
    if (paramSlotSel != paramSlot) {
      paramSlot = paramSlotSel;      /* the request failed: reflect it */
    }
  }
  paramWriteSlot = nnpolSlotBankWriteSlot();
  paramWrittenBytes = nnpolSlotBankWrittenBytes();

  /* The engage context arrives in chunks: every pass that sees new bytes
   * parses the whole region again, and only a blob whose CRC and track check
   * out becomes the pending track. A failed parse keeps the engage blocked. */
  uint32_t writeCount;
  const uint8_t* staged = nnpolContextStaged(&writeCount);
  if (writeCount != ctxWriteSeen) {
    ctxWriteSeen = writeCount;
    uint32_t crc = 0;
    pendingTrackOk = false;
    const nnpolCtxStatus_t st = nnpolContextParse(staged, NNPOL_CTX_MAX_BYTES, &pendingTrack, &crc);
    paramCtxStatus = (uint8_t)st;
    paramCtxCrc = (st == NNPOL_CTX_OK) ? crc : 0u;
    pendingTrackOk = (st == NNPOL_CTX_OK);
  }
}

void appMain()
{
  appTaskHandle = xTaskGetCurrentTaskHandle();
  nnpolCoreInit(&core);
  /* nnpolSlotBankInit ran in controllerOutOfTreeInit, before the scheduler. */
  paramSlotValid = nnpolSlotBankValidMask();
  DEBUG_PRINT("nnpol: %d slots, valid mask 0x%02x\n", NNPOL_NUM_SLOTS, paramSlotValid);

  while (1) {
    if (ulTaskNotifyTake(pdTRUE, pdMS_TO_TICKS(20)) == 0) {
      housekeeping();
      continue;
    }
    if (!activeValid) {
      continue;
    }

    nnpolSnapshot_t snap;
    snapshotRead(&snap);

    if (engageSeen != engageCount) {
      /* A new run: the track the host pushed, the gate state as at reset.
       * pendingTrack is only written by housekeeping above, in this task
       * and never while engaged, so it is stable here. */
      engageSeen = engageCount;
      nnpolCoreSetTrack(&core, &pendingTrack);
      nnpolCoreEngage(&core, THRUST_MAX, latchedRpmScale);
    }

    const uint64_t tStart = usecTimestamp();

    static nnpolStepOut_t out;
    if (!nnpolCoreStep(&core, &snap.state, &out)) {
      if (logObsErr < 0xFFFFu) { logObsErr++; }
      continue;                     /* no action: the stale fallback takes over */
    }

    const uint64_t elapsed = usecTimestamp() - tStart;

    actionWrite(&out.cmd);

    /* Log fields: written only here (single writer), read by the log
     * framework whenever a block samples them. */
    logT = snap.t;
    memcpy(logAct, out.act, sizeof(logAct));
    for (int i = 0; i < 4; i++) {
      logPwm[i] = (uint16_t)out.cmd.motorPwm[i];
    }
    memcpy(logVb, &out.obs[0], sizeof(logVb));   /* v_B */
    memcpy(logWb, &out.obs[3], sizeof(logWb));   /* omega_B */
    memcpy(logE0, &out.obs[9], sizeof(logE0));   /* target gate offset, body frame */
    logGateIdx = out.gateIdx;
    logUs = (uint16_t)(elapsed > 0xFFFFu ? 0xFFFFu : elapsed);
    logSeq++;
  }
}

// --------------------------------------------------------------------------
// The OOT controller
// --------------------------------------------------------------------------

/* Engage state, owned by the stabilizer context. */
static bool wasEnabled = false;
static uint32_t t0Tick = 0;
static uint32_t seqAtEngage = 0;      /* only actions newer than this fly */
static uint32_t lastSeqSeen = 0;
static uint16_t periodsWithoutFresh = 0;
static uint32_t activeDivider = RATE_MAIN_LOOP / 100;

/** Stabilizer ticks per policy step for a requested rate: integer
 * division, clamped to [1 Hz, RATE_MAIN_LOOP]. */
static uint32_t tickDividerFor(uint16_t hz)
{
  uint32_t h = hz;
  if (h < 1u) { h = 1u; }
  if (h > RATE_MAIN_LOOP) { h = RATE_MAIN_LOOP; }
  return RATE_MAIN_LOOP / h;
}

void controllerOutOfTreeInit()
{
  nnpolSlotBankInit();   /* here, not in appMain: see nnpol_slot_bank.h */
  controllerPidInit();
  wasEnabled = false;
  periodsWithoutFresh = 0;
}

bool controllerOutOfTreeTest()
{
  return controllerPidTest();
}

/** The inner loop a radio setpoint gets: the stock PID, as
 * stabilizer.controller = 1 would run it. */
static void passthrough(control_t* control, const setpoint_t* setpoint,
                        const sensorData_t* sensors, const state_t* state,
                        const uint32_t tick)
{
  controllerPid(control, setpoint, sensors, state, tick);
}

/** Four 16-bit motor values in the setpoint shape the (former) per-motor
 * radio packet produced: modeMotorPwm on all three axes, M1 in thrust and
 * M2..M4 in attitudeRate, which controllerPid turns into normalized forces
 * and nothing else. */
static void motorSetpoint(setpoint_t* sp, uint32_t timestamp, float m1, float m2, float m3, float m4)
{
  memset(sp, 0, sizeof(*sp));
  sp->timestamp = timestamp;
  sp->mode.x = modeDisable;
  sp->mode.y = modeDisable;
  sp->mode.z = modeDisable;
  sp->mode.quat = modeDisable;
  sp->mode.roll = modeMotorPwm;
  sp->mode.pitch = modeMotorPwm;
  sp->mode.yaw = modeMotorPwm;
  sp->thrust = m1;
  sp->attitudeRate.roll = m2;
  sp->attitudeRate.pitch = m3;
  sp->attitudeRate.yaw = m4;
}

void controllerOutOfTree(control_t* control, const setpoint_t* setpoint,
                         const sensorData_t* sensors, const state_t* state,
                         const uint32_t tick)
{
  /* No policy without a validated track: enable is ignored until the host
   * has written a context that parsed. */
  const bool enabled = (nnpolEnable != 0) && activeValid && pendingTrackOk;
  nnpolSlotBankSetLocked(nnpolEnable != 0 || supervisorIsArmed());
  nnpolContextSetLocked(nnpolEnable != 0);

  if (enabled && !wasEnabled) {
    /* Rising edge: the run clock starts NOW. The rpm scale is latched here
     * and the track by the app task on its first step of the run, so
     * nothing written mid-run moves the track or rescales the motors. */
    t0Tick = tick;
    latchedRpmScale = paramRpmScale;
    engageCount++;
    activeDivider = tickDividerFor(paramCtrlHz);
    paramRunHz = (uint16_t)(RATE_MAIN_LOOP / activeDivider);
    nnpolCmd_t unused;
    seqAtEngage = actionRead(&unused);
    lastSeqSeen = seqAtEngage;
    periodsWithoutFresh = 0;
    logStale = 0;
    logObsErr = 0;
  }
  wasEnabled = enabled;

  if (enabled && (tick % activeDivider) == 0u) {
    /* Policy tick: snapshot a consistent state and wake the app task.
     * The EKF quaternion is stored x, y, z, w; the observation wants
     * scalar-first — reordered here. */
    nnpolSnapshot_t snap;
    snap.t = (float)(tick - t0Tick) * 0.001f;
    snap.state.pos[0] = state->position.x;
    snap.state.pos[1] = state->position.y;
    snap.state.pos[2] = state->position.z;
    snap.state.quat_wxyz[0] = state->attitudeQuaternion.w;
    snap.state.quat_wxyz[1] = state->attitudeQuaternion.x;
    snap.state.quat_wxyz[2] = state->attitudeQuaternion.y;
    snap.state.quat_wxyz[3] = state->attitudeQuaternion.z;
    snap.state.vel_w[0] = state->velocity.x;
    snap.state.vel_w[1] = state->velocity.y;
    snap.state.vel_w[2] = state->velocity.z;
    snap.state.gyro_deg[0] = sensors->gyro.x;
    snap.state.gyro_deg[1] = sensors->gyro.y;
    snap.state.gyro_deg[2] = sensors->gyro.z;
    /* Bidirectional-DSHOT rotor speed (motors.c), sampled on the same tick
     * as the estimator state: UINT16_MAX = invalid or no reply, held per
     * motor. core.policy is stable while engaged (the app task changes it
     * only while disengaged). */
    for (int i = 0; i < 4; i++) {
      const uint16_t raw = motorsGetRPM(i);
      if (raw != UINT16_MAX) {
        heldRpm[i] = (float)raw;
        haveRpm[i] = true;
      }
      snap.state.rpm[i] = haveRpm[i] ? heldRpm[i] : core.policy.hdr->hoverRpm;
    }
    snapshotWrite(&snap);

    if (appTaskHandle != NULL) {
      xTaskNotifyGive(appTaskHandle);
    }

    /* Staleness accounting, one step behind by construction: the action
     * requested by THIS notify lands ~one 1 kHz tick from now, so what is
     * checked here is whether the previous period ever delivered. */
    nnpolCmd_t peek;
    const uint32_t seq = actionRead(&peek);
    if (seq == lastSeqSeen) {
      if (periodsWithoutFresh < 0xFFFFu) { periodsWithoutFresh++; }
      if (periodsWithoutFresh >= NNPOL_STALE_LOG_PERIODS && logStale < 0xFFFFu) {
        logStale++;
      }
    } else {
      lastSeqSeen = seq;
      periodsWithoutFresh = 0;
    }
  }

  nnpolCmd_t cmd;
  const uint32_t seq = actionRead(&cmd);
  const bool haveFreshAction = enabled && seq != seqAtEngage
      && periodsWithoutFresh < NNPOL_STALE_FALLBACK_PERIODS;

  if (!haveFreshAction) {
    const bool bench = (nnpolEnable == 0) && paramBenchPwm > 0 && supervisorIsArmed()
        && (xTaskGetTickCount() - benchTick) < NNPOL_BENCH_HOLD_TICKS;
    if (bench) {
      setpoint_t sp;
      motorSetpoint(&sp, setpoint->timestamp, (float)paramBenchPwm, (float)paramBenchPwm,
                    (float)paramBenchPwm, (float)paramBenchPwm);
      logBench = paramBenchPwm;
      controllerPid(control, &sp, sensors, state, tick);
      return;
    }
    if (paramBenchPwm != 0 && (xTaskGetTickCount() - benchTick) >= NNPOL_BENCH_HOLD_TICKS) {
      paramBenchPwm = 0;            /* expired: reads back as stopped */
    }
    logBench = 0;
    /* Disabled, no slot, no action computed yet, or stale past the
     * fallback threshold: fly the radio setpoint — the shadow host command. */
    passthrough(control, setpoint, sensors, state, tick);
    return;
  }
  logBench = 0;

  /* Zero-order hold of the latest complete action. */
  setpoint_t sp;
  motorSetpoint(&sp, setpoint->timestamp, cmd.motorPwm[0], cmd.motorPwm[1], cmd.motorPwm[2],
                cmd.motorPwm[3]);
  controllerPid(control, &sp, sensors, state, tick);
}

// --------------------------------------------------------------------------
// Param / log groups
// --------------------------------------------------------------------------

/**
 * Onboard policy control, slot bank and identity.
 */
PARAM_GROUP_START(nnpol)
/** @brief 1 = the onboard policy commands the vehicle; 0 = radio setpoint
 * passthrough (offboard/shadow). Clearing mid-flight is an instantaneous,
 * format-compatible takeover by the host stream. Has no effect without a
 * selected, valid slot. */
PARAM_ADD(PARAM_UINT8, enable, &nnpolEnable)
/** @brief nnpolCtxStatus_t of the engage context as last parsed: 0 ok,
 * 1 empty, 2 bad magic, 3 bad size, 4 bad crc, 5 bad track. A policy does
 * not engage unless this is 0. */
PARAM_ADD(PARAM_UINT8 | PARAM_RONLY, ctxStatus, &paramCtxStatus)
/** @brief CRC of the validated engage context (its own crc32 field), 0 when
 * none is valid: the ground station compares it with what it wrote. */
PARAM_ADD(PARAM_UINT32 | PARAM_RONLY, ctxCrc, &paramCtxCrc)
/** @brief Gates a track may have. */
PARAM_ADD(PARAM_UINT8 | PARAM_RONLY, maxGates, &paramMaxGates)
/** @brief NNPOL_CORE_VERSION of this image: the ground station's parity
 * library must report the same. */
PARAM_ADD(PARAM_UINT16 | PARAM_RONLY, coreVer, &paramCoreVer)
/** @brief rpm multiplier before the thrust map (host motor_pwm.rpm_scale). */
PARAM_ADD(PARAM_FLOAT, rpmScale, &paramRpmScale)
/** @brief Policy rate (Hz) latched at the enable rising edge. Defaults to
 * the selected slot's trained ctrl_freq; must divide 1000 to run exactly. */
PARAM_ADD(PARAM_UINT16, ctrlHz, &paramCtrlHz)
/** @brief Rate (Hz) the engaged run actually steps at, read-only. */
PARAM_ADD(PARAM_UINT16 | PARAM_RONLY, runHz, &paramRunHz)
/** @brief Slot to fly (0..nSlots-1, 255 = none). Validated (CRC) and
 * adopted by the app task while disengaged; on failure it snaps back to
 * slotSel and slotStatus says why. */
PARAM_ADD(PARAM_UINT8, slot, &paramSlot)
/** @brief The slot actually selected and validated, read-only (255 = none). */
PARAM_ADD(PARAM_UINT8 | PARAM_RONLY, slotSel, &paramSlotSel)
/** @brief Write a slot index to erase it (on the ground, disengaged, disarmed);
 * reads back 255 when done. Erasing the selected slot deselects it. */
PARAM_ADD(PARAM_UINT8, slotErase, &paramSlotErase)
/** @brief 0 idle, 1 erasing, 2 erased, 3 erase failed, 4 select failed,
 * 5 selected. */
PARAM_ADD(PARAM_UINT8 | PARAM_RONLY, slotState, &paramSlotState)
/** @brief nnpolSlotStatus_t of the last select: 0 ok, 1 empty, 2 bad magic,
 * 3 bad size, 4 bad dims, 5 bad crc. */
PARAM_ADD(PARAM_UINT8 | PARAM_RONLY, slotStatus, &paramSlotStatus)
/** @brief Bit i set = slot i holds a valid blob. */
PARAM_ADD(PARAM_UINT8 | PARAM_RONLY, slotValid, &paramSlotValid)
PARAM_ADD(PARAM_UINT8 | PARAM_RONLY, nSlots, &paramNumSlots)
/** @brief Upload progress: the slot the last radio write landed in and the
 * bytes written to it since its erase. */
PARAM_ADD(PARAM_UINT8 | PARAM_RONLY, writeSlot, &paramWriteSlot)
PARAM_ADD(PARAM_UINT32 | PARAM_RONLY, writtenBytes, &paramWrittenBytes)
/** @brief Identity of the SELECTED slot (firmware_export.json hash words), read-only. */
PARAM_ADD(PARAM_UINT32 | PARAM_RONLY, hash0, &paramHash0)
PARAM_ADD(PARAM_UINT32 | PARAM_RONLY, hash1, &paramHash1)
PARAM_ADD(PARAM_UINT32 | PARAM_RONLY, hash2, &paramHash2)
PARAM_ADD(PARAM_UINT32 | PARAM_RONLY, hash3, &paramHash3)
PARAM_ADD(PARAM_UINT16 | PARAM_RONLY, obsDim, &paramObsDim)
PARAM_ADD(PARAM_UINT16 | PARAM_RONLY, actDim, &paramActDim)
/** @brief Decode path of the selected slot: always 2 (motor) on this branch. */
PARAM_ADD(PARAM_UINT16 | PARAM_RONLY, kind, &paramKind)
/** @brief Observation builder of the selected slot (2 = race v3). */
PARAM_ADD(PARAM_UINT16 | PARAM_RONLY, taskId, &paramTaskId)
/** @brief 1 = this build reads every 16-bit motor value as a force and
 * compensates for battery voltage (always, on this branch). */
PARAM_ADD(PARAM_UINT8 | PARAM_RONLY, batComp, &paramBatComp)
/** @brief The force a motor value of 65535 stands for, N per motor. */
PARAM_ADD(PARAM_FLOAT | PARAM_RONLY, thrustMax, &paramThrustMax)
/** @brief Bench sweep: a 16-bit value applied to all four motors while armed
 * and not engaged, held for 300 ms after each write. PROPS ON, RESTRAINED. */
PARAM_ADD_WITH_CALLBACK(PARAM_UINT16, benchPwm, &paramBenchPwm, &benchWritten)
PARAM_GROUP_STOP(nnpol)

/**
 * Onboard policy telemetry, sampled by the ground station at the policy
 * rate for the shadow diff (bin/compare_shadow.py in crazyflie-ros).
 */
LOG_GROUP_START(nnpol)
/** @brief Seconds since engage of the last observation. */
LOG_ADD(LOG_FLOAT, t, &logT)
/** @brief Clipped network output. */
LOG_ADD(LOG_FLOAT, act0, &logAct[0])
LOG_ADD(LOG_FLOAT, act1, &logAct[1])
LOG_ADD(LOG_FLOAT, act2, &logAct[2])
LOG_ADD(LOG_FLOAT, act3, &logAct[3])
/** @brief The decoded 16-bit motor value M1..M4 (a force, see thrustMax). */
LOG_ADD(LOG_UINT16, pwm0, &logPwm[0])
LOG_ADD(LOG_UINT16, pwm1, &logPwm[1])
LOG_ADD(LOG_UINT16, pwm2, &logPwm[2])
LOG_ADD(LOG_UINT16, pwm3, &logPwm[3])
/** @brief Observation excerpts: body velocity, body rate, target gate offset (body). */
LOG_ADD(LOG_FLOAT, vbx, &logVb[0])
LOG_ADD(LOG_FLOAT, vby, &logVb[1])
LOG_ADD(LOG_FLOAT, vbz, &logVb[2])
LOG_ADD(LOG_FLOAT, wbx, &logWb[0])
LOG_ADD(LOG_FLOAT, wby, &logWb[1])
LOG_ADD(LOG_FLOAT, wbz, &logWb[2])
LOG_ADD(LOG_FLOAT, e0x, &logE0[0])
LOG_ADD(LOG_FLOAT, e0y, &logE0[1])
LOG_ADD(LOG_FLOAT, e0z, &logE0[2])
/** @brief Inference time, microseconds (obs + network + decode). */
LOG_ADD(LOG_UINT16, us, &logUs)
/** @brief Policy periods spent at/past the stale threshold. */
LOG_ADD(LOG_UINT16, stale, &logStale)
/** @brief Observations refused since engage (no builder for the slot's
 * task, or a dimension mismatch) — each one is a missed action. */
LOG_ADD(LOG_UINT16, obsErr, &logObsErr)
/** @brief race: the gate the onboard crossing test is targeting. */
LOG_ADD(LOG_UINT8, gateIdx, &logGateIdx)
/** @brief The bench motor value being applied right now, 0 when none. */
LOG_ADD(LOG_UINT16, bench, &logBench)
/** @brief Actions computed since boot. */
LOG_ADD(LOG_UINT32, seq, &logSeq)
LOG_GROUP_STOP(nnpol)
