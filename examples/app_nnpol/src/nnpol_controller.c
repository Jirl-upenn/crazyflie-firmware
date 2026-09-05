/**
 * nnpol_controller.c — firmware glue for the onboard-policy app.
 *
 * Architecture (mjc_dronetests/docs/onboard_inference_plan.md section 3):
 * the OOT controller hook runs in the STABILIZER task at 1 kHz. On every
 * policy tick (every RATE_MAIN_LOOP / nnpol.ctrlHz stabilizer ticks) it
 * snapshots the EKF state + gyro and notifies the APP task, which builds
 * the observation, runs the selected slot's network through the
 * interpreter (nnpol_policy.c, 0.3-0.8 ms for a 2x128 — too close to the
 * 1 ms stabilizer budget to run inline, which is the whole reason for the
 * task split), decodes the action and publishes it into a sequence-locked
 * buffer. The controller applies the latest complete action as a
 * zero-order hold — the same hold the sim's 5 substeps per action
 * perform — by synthesizing the setpoint_t the same command would have
 * produced ARRIVING BY RADIO and handing it to the inner controller the
 * slot's action kind needs:
 *
 *   MELLINGER  attitude-only setpoint -> controllerMellingerFirmware, the
 *              exact inner loop the offboard Mellinger flights validated
 *   CTBR       legacy RATE-mode setpoint -> controllerPid, the loop the
 *              SE3 takeoff/land cascade and the offboard ctbr policy fly on
 *   MOTOR      modeMotorPwm setpoint (thrust = M1, attitudeRate = M2..M4)
 *              -> controllerPid's per-motor bypass (the generic commander
 *              type-12 path), no onboard loop at all
 *
 * The passthrough (nnpol.enable = 0, or no usable slot) dispatches on the
 * radio setpoint's own shape — an ANGLE setpoint (mode.roll == modeAbs)
 * to the Mellinger, everything else to the PID — so the shadow stream on
 * /attitude_cmd, /ctbr_cmd or /motor_cmd flies through controller 6
 * exactly as it does through 2 or 1. Expected added latency: one 1 kHz
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
 * Radio-path equivalence, including the wire: cflib's send_setpoint packs
 * -pitch, and crtp_commander_rpyt.c copies values->pitch into the
 * setpoint unchanged, so the setpoint_t a host command produces carries
 * the NEGATED host pitch (angle or rate). The synthesized setpoint does
 * the same. Bench-check nnpol.spPitch against a /attitude_cmd pitch step
 * before the first onboard flight.
 *
 * Engage protocol (the ground station's side is crazyflie-ros
 * `exec: onboard`): select the slot, verify nnpol.hash0-3 against the run
 * directory's firmware_export.json, push startPhase/offX/offY/offZ/yaw0/
 * vnom/rpmScale/ctrlHz, then set enable = 1. The rising edge latches t0
 * and the pushed values, so the curve clock starts at the engage instant
 * with the host-pinned offset and heading.
 *
 * Policy rate: nnpol.ctrlHz is writable and defaults to the selected
 * slot's trained ctrl_freq. The rising edge turns it into a tick divider
 * (RATE_MAIN_LOOP / ctrlHz, integer), so only divisors of 1000 run
 * exactly — anything else runs at the next divisor UP (48 -> 50 Hz) and
 * the read-only nnpol.runHz reports what actually engaged. The host
 * validates divisibility before pushing; the curve clock is tick-based,
 * so a rate change never shifts the reference.
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
#include "controller_mellinger.h"
#include "controller_pid.h"
#include "log.h"
#include "motors.h"
#include "param.h"
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
static float paramStartPhase = 0.0f;
static float paramOffX = 0.0f;
static float paramOffY = 0.0f;
static float paramOffZ = 0.0f;
static float paramYaw0 = 0.0f;      /* the run's yaw anchor, rad */
static float paramVnom = 4.0f;      /* same fixed voltage the host maps thrust at */
static float paramRpmScale = 1.0f;  /* MOTOR kind: the host's motor_pwm.rpm_scale */
/* Latched from the params at the enable rising edge (like the pin), so a
 * mid-run param write cannot rescale the motors or re-anchor the run;
 * read by the app task. */
static float activeRpmScale = 1.0f;
static float activeYaw0 = 0.0f;
/* EASY_BODY: this run's curve, pushed by the host before enable (the
 * nominal member of the family; the pin offset and yaw anchor are pushed
 * separately, as for the fixed eight) and latched at the rising edge. */
static float paramCurve[NNPOL_CURVE_FLOATS] = { 0 };
static float activeCurve[NNPOL_CURVE_FLOATS] = { 0 };
/* motorobs: the last valid reading per motor, normalized. Starts at hover
 * (1.0) — a state a flying drone is in, unlike zero — and holds through
 * DSHOT telemetry dropouts, exactly as the host's set_motor_rpm does. */
static float heldRpm[4] = { 1.0f, 1.0f, 1.0f, 1.0f };

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
// through activeValid + activeKind, which change only while disengaged)
// --------------------------------------------------------------------------

static nnpolPolicy_t activePolicy = { NULL, NULL };
static volatile bool activeValid = false;
static volatile uint16_t activeKind = 0xFFFF;

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
  activeKind = 0xFFFF;
  activePolicy.hdr = NULL;
  activePolicy.weights = NULL;
  if (slot == NNPOL_SLOT_NONE || slot >= NNPOL_NUM_SLOTS) {
    paramSlotSel = NNPOL_SLOT_NONE;
    paramSlotStatus = NNPOL_SLOT_EMPTY;
    paramSlotState = SLOT_STATE_IDLE;
    publishIdentity(NULL);
    return;
  }
  const nnpolSlotHeader_t* h = nnpolSlotBankHeader(slot);
  const nnpolSlotStatus_t st = nnpolSlotValidate(h, NNPOL_SLOT_BYTES);
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
  activePolicy.hdr = h;
  activePolicy.weights = nnpolSlotWeights(h);
  if (nnpolObsDim(h) != (int)h->obsDim) {
    /* Selectable but not flyable: the identity is published so the ground
     * station can say which run it is, and every policy tick then counts
     * an obsErr instead of producing an action. */
    DEBUG_PRINT("slot %d: no observation builder for task %d / %d-dim\n",
                slot, h->taskId, h->obsDim);
  }
  publishIdentity(h);
  activeKind = h->actionKind;
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
static float logSpRoll, logSpPitch, logSpYaw, logSpThrust;
static uint16_t logPwm[4];                  /* MOTOR kind: per-motor PWM */
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
 * stabilizer reads activeValid/activeKind and the bank refuses writes
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
}

void appMain()
{
  appTaskHandle = xTaskGetCurrentTaskHandle();
  nnpolSlotBankInit();
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

    const uint64_t tStart = usecTimestamp();

    float obs[NNPOL_MAX_OBS_DIM];
    const int obsDim = nnpolBuildObs(activePolicy.hdr, &snap, obs);
    if (obsDim != (int)activePolicy.hdr->obsDim) {
      if (logObsErr < 0xFFFFu) { logObsErr++; }
      continue;                     /* no action: the stale fallback takes over */
    }

    float act[NNPOL_ACTION_DIM];
    nnpolForward(&activePolicy, obs, act);
    for (int i = 0; i < NNPOL_ACTION_DIM; i++) {
      if (act[i] < -1.0f) { act[i] = -1.0f; }
      if (act[i] > 1.0f) { act[i] = 1.0f; }
    }

    nnpolCmd_t cmd;
    nnpolDecodeAction(activePolicy.hdr, act, paramVnom, activeRpmScale, activeYaw0, &cmd);

    const uint64_t elapsed = usecTimestamp() - tStart;

    actionWrite(&cmd);

    /* Log fields: written only here (single writer), read by the log
     * framework whenever a block samples them. */
    logT = snap.t;
    memcpy(logAct, act, sizeof(logAct));
    logSpRoll = cmd.ch[0];
    logSpPitch = cmd.ch[1];
    logSpYaw = cmd.ch[2];
    logSpThrust = cmd.thrustSetpoint;
    for (int i = 0; i < 4; i++) {
      logPwm[i] = (uint16_t)cmd.motorPwm[i];
    }
    memcpy(logVb, &obs[4], sizeof(logVb));   /* v_B */
    memcpy(logWb, &obs[7], sizeof(logWb));   /* omega_B */
    memcpy(logE0, &obs[10], sizeof(logE0));  /* first lookahead sample */
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
static float activeStartPhase = 0.0f;
static float activeOff[3] = { 0.0f, 0.0f, 0.0f };
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
  controllerMellingerFirmwareInit();
  controllerPidInit();
  wasEnabled = false;
  periodsWithoutFresh = 0;
}

bool controllerOutOfTreeTest()
{
  return controllerMellingerFirmwareTest() && controllerPidTest();
}

/** The inner loop a radio setpoint needs: the ANGLE shape the driver's
 * 'oot' mode produces goes to the Mellinger (as stabilizer.controller = 2
 * would), the RATE and per-motor shapes of 'oot_rate' to the PID (as 1). */
static void passthrough(control_t* control, const setpoint_t* setpoint,
                        const sensorData_t* sensors, const state_t* state,
                        const uint32_t tick)
{
  if (setpoint->mode.roll == modeAbs) {
    controllerMellingerFirmware(control, setpoint, sensors, state, tick);
  } else {
    controllerPid(control, setpoint, sensors, state, tick);
  }
}

void controllerOutOfTree(control_t* control, const setpoint_t* setpoint,
                         const sensorData_t* sensors, const state_t* state,
                         const uint32_t tick)
{
  const bool enabled = (nnpolEnable != 0) && activeValid;
  nnpolSlotBankSetLocked(nnpolEnable != 0 || supervisorIsArmed());

  if (enabled && !wasEnabled) {
    /* Rising edge: the curve clock starts NOW, with the values the host
     * pushed before flipping enable. Latching them here (not reading the
     * params live) means a mid-run param write cannot shift the clock,
     * teleport the curve or re-anchor the heading. */
    t0Tick = tick;
    activeStartPhase = paramStartPhase;
    activeOff[0] = paramOffX;
    activeOff[1] = paramOffY;
    activeOff[2] = paramOffZ;
    activeYaw0 = paramYaw0;
    activeRpmScale = paramRpmScale;
    memcpy(activeCurve, paramCurve, sizeof(activeCurve));
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
     * scalar-first — reordered here, anchored and sign-normalized in
     * nnpolBuildObs. */
    nnpolSnapshot_t snap;
    snap.t = activeStartPhase + (float)(tick - t0Tick) * 0.001f;
    snap.off[0] = activeOff[0];
    snap.off[1] = activeOff[1];
    snap.off[2] = activeOff[2];
    snap.pos[0] = state->position.x;
    snap.pos[1] = state->position.y;
    snap.pos[2] = state->position.z;
    snap.quat_wxyz[0] = state->attitudeQuaternion.w;
    snap.quat_wxyz[1] = state->attitudeQuaternion.x;
    snap.quat_wxyz[2] = state->attitudeQuaternion.y;
    snap.quat_wxyz[3] = state->attitudeQuaternion.z;
    snap.vel_w[0] = state->velocity.x;
    snap.vel_w[1] = state->velocity.y;
    snap.vel_w[2] = state->velocity.z;
    snap.gyro_deg[0] = sensors->gyro.x;
    snap.gyro_deg[1] = sensors->gyro.y;
    snap.gyro_deg[2] = sensors->gyro.z;
    snap.yaw0 = activeYaw0;
    memcpy(snap.curve, activeCurve, sizeof(snap.curve));
    /* Bidirectional-DSHOT rotor speed (motors.c): UINT16_MAX = invalid or
     * no reply, held per motor. Read on every policy tick regardless of the
     * slot's observation so the hold is warm when a motorobs slot engages. */
    for (int i = 0; i < 4; i++) {
      const uint16_t raw = motorsGetRPM(i);
      if (raw != UINT16_MAX) {
        heldRpm[i] = (float)raw / NNPOL_NOMINAL_HOVER_RPM;
      }
      snap.rpm[i] = heldRpm[i];
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
    /* Disabled, no slot, no action computed yet, or stale past the
     * fallback threshold: fly the radio setpoint — the shadow host command. */
    passthrough(control, setpoint, sensors, state, tick);
    return;
  }

  /* Zero-order hold of the latest complete action, in the exact setpoint
   * shape the radio path produces for this kind. cmd.ch[] is in the host's
   * wire units and sign; cflib packs -pitch, and crtp_commander_rpyt copies
   * values->pitch into the setpoint unchanged, so the setpoint gets the
   * negated host pitch. Roll and yaw arrive unchanged (the driver's
   * yaw-rate negation and the decoder's "legacy rate input is inverted"
   * cancel). */
  setpoint_t sp;
  memset(&sp, 0, sizeof(sp));
  sp.timestamp = setpoint->timestamp;
  sp.mode.x = modeDisable;
  sp.mode.y = modeDisable;
  sp.mode.z = modeDisable;
  sp.mode.quat = modeDisable;
  switch (activeKind) {
    case NNPOL_ACTION_KIND_MELLINGER:
      /* crtp_commander_rpyt ANGLE branch: roll/pitch/yaw modeAbs, thrust
       * already through the legacy clamp. */
      sp.mode.roll = modeAbs;
      sp.mode.pitch = modeAbs;
      sp.mode.yaw = modeAbs;
      sp.attitude.roll = cmd.ch[0];
      sp.attitude.pitch = -cmd.ch[1];
      sp.attitude.yaw = cmd.ch[2];
      sp.thrust = cmd.thrustSetpoint;
      controllerMellingerFirmware(control, &sp, sensors, state, tick);
      break;
    case NNPOL_ACTION_KIND_CTBR:
      /* crtp_commander_rpyt RATE branch: roll/pitch/yaw modeVelocity with
       * the rates in attitudeRate (deg/s), attitude zeroed. */
      sp.mode.roll = modeVelocity;
      sp.mode.pitch = modeVelocity;
      sp.mode.yaw = modeVelocity;
      sp.attitudeRate.roll = cmd.ch[0];
      sp.attitudeRate.pitch = -cmd.ch[1];
      sp.attitudeRate.yaw = cmd.ch[2];
      sp.thrust = cmd.thrustSetpoint;
      controllerPid(control, &sp, sensors, state, tick);
      break;
    default:
      /* crtp_commander_generic motorPwmDecoder (type 12): modeMotorPwm on
       * all three axes, M1 in thrust and M2..M4 in attitudeRate, which
       * controllerPid turns into normalized forces and nothing else. */
      sp.mode.roll = modeMotorPwm;
      sp.mode.pitch = modeMotorPwm;
      sp.mode.yaw = modeMotorPwm;
      sp.thrust = cmd.motorPwm[0];
      sp.attitudeRate.roll = cmd.motorPwm[1];
      sp.attitudeRate.pitch = cmd.motorPwm[2];
      sp.attitudeRate.yaw = cmd.motorPwm[3];
      controllerPid(control, &sp, sensors, state, tick);
      break;
  }
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
/** @brief Curve phase (s) the clock starts at on the enable rising edge. */
PARAM_ADD(PARAM_FLOAT, startPhase, &paramStartPhase)
/** @brief World-frame curve translation, pushed by the host's pin. */
PARAM_ADD(PARAM_FLOAT, offX, &paramOffX)
PARAM_ADD(PARAM_FLOAT, offY, &paramOffY)
PARAM_ADD(PARAM_FLOAT, offZ, &paramOffZ)
/** @brief The run's yaw anchor (rad): the engage heading the curve is yawed
 * to and the observed attitude is relative to; added back to a commanded
 * yaw angle. Pushed by the host (world_yaw_offset). */
PARAM_ADD(PARAM_FLOAT, yaw0, &paramYaw0)
/** @brief traj_easy: this run's curve, the nominal member of the Lissajous
 * family the host drew (controller_trajectory.Curve): amplitudes (m),
 * angular rates (rad/s), phases (rad), centre (m), yaw (rad). Latched at
 * the enable rising edge; ignored by the fixed-eight slots. */
PARAM_ADD(PARAM_FLOAT, cAmp0, &paramCurve[0])
PARAM_ADD(PARAM_FLOAT, cAmp1, &paramCurve[1])
PARAM_ADD(PARAM_FLOAT, cAmp2, &paramCurve[2])
PARAM_ADD(PARAM_FLOAT, cOmg0, &paramCurve[3])
PARAM_ADD(PARAM_FLOAT, cOmg1, &paramCurve[4])
PARAM_ADD(PARAM_FLOAT, cOmg2, &paramCurve[5])
PARAM_ADD(PARAM_FLOAT, cPh0, &paramCurve[6])
PARAM_ADD(PARAM_FLOAT, cPh1, &paramCurve[7])
PARAM_ADD(PARAM_FLOAT, cPh2, &paramCurve[8])
PARAM_ADD(PARAM_FLOAT, cCen0, &paramCurve[9])
PARAM_ADD(PARAM_FLOAT, cCen1, &paramCurve[10])
PARAM_ADD(PARAM_FLOAT, cCen2, &paramCurve[11])
PARAM_ADD(PARAM_FLOAT, cYaw, &paramCurve[12])
/** @brief Battery voltage every thrust/rpm-to-PWM map assumes (host parity). */
PARAM_ADD(PARAM_FLOAT, vnom, &paramVnom)
/** @brief MOTOR kind: rpm multiplier before the PWM map (host motor_pwm.rpm_scale). */
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
/** @brief Decode path of the selected slot: 0 Mellinger, 1 ctbr, 2 motor. */
PARAM_ADD(PARAM_UINT16 | PARAM_RONLY, kind, &paramKind)
/** @brief Observation builder of the selected slot (0 lissajous/body). */
PARAM_ADD(PARAM_UINT16 | PARAM_RONLY, taskId, &paramTaskId)
PARAM_GROUP_STOP(nnpol)

/**
 * Onboard policy telemetry, sampled by the ground station at the policy
 * rate for the shadow diff (bin/compare_shadow.py in crazyflie-ros).
 */
LOG_GROUP_START(nnpol)
/** @brief Curve clock (s) of the last observation. */
LOG_ADD(LOG_FLOAT, t, &logT)
/** @brief Clipped network output. */
LOG_ADD(LOG_FLOAT, act0, &logAct[0])
LOG_ADD(LOG_FLOAT, act1, &logAct[1])
LOG_ADD(LOG_FLOAT, act2, &logAct[2])
LOG_ADD(LOG_FLOAT, act3, &logAct[3])
/** @brief Decoded setpoint in the host's wire units: MELLINGER angles
 * (deg), CTBR rates (deg/s), plus the legacy thrust; zero for MOTOR. */
LOG_ADD(LOG_FLOAT, spRoll, &logSpRoll)
LOG_ADD(LOG_FLOAT, spPitch, &logSpPitch)
LOG_ADD(LOG_FLOAT, spYaw, &logSpYaw)
LOG_ADD(LOG_FLOAT, spThrust, &logSpThrust)
/** @brief MOTOR kind: per-motor PWM M1..M4 (zero for the setpoint kinds). */
LOG_ADD(LOG_UINT16, pwm0, &logPwm[0])
LOG_ADD(LOG_UINT16, pwm1, &logPwm[1])
LOG_ADD(LOG_UINT16, pwm2, &logPwm[2])
LOG_ADD(LOG_UINT16, pwm3, &logPwm[3])
/** @brief Observation excerpts: body velocity, body rate, first lookahead. */
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
/** @brief Actions computed since boot. */
LOG_ADD(LOG_UINT32, seq, &logSeq)
LOG_GROUP_STOP(nnpol)
