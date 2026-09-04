/**
 * nnpol_controller.c — firmware glue for the onboard-policy app.
 *
 * Architecture (mjc_dronetests/docs/onboard_inference_plan.md section 3):
 * the OOT controller hook runs in the STABILIZER task at 1 kHz. On every
 * policy tick (every RATE_MAIN_LOOP / nnpol.ctrlHz stabilizer ticks) it
 * snapshots the EKF state + gyro and notifies the APP task, which builds the observation,
 * runs policyForward (0.3-0.8 ms — too close to the 1 ms stabilizer budget
 * to run inline, which is the whole reason for the task split), decodes
 * the action and publishes it into a sequence-locked buffer. The
 * controller applies the latest complete action as a zero-order hold —
 * the same hold the sim's 5 substeps per action perform — by synthesizing
 * the attitude-only setpoint_t shape and calling
 * controllerMellingerFirmware, i.e. the exact inner loop the offboard
 * Mellinger flights already validated. Expected added latency: one 1 kHz
 * tick.
 *
 * With nnpol.enable = 0 (the default) the radio setpoint passes through
 * untouched, which is byte-for-byte today's offboard behaviour behind
 * stabilizer.controller = 6 instead of 2. The host keeps streaming
 * /attitude_cmd either way: its content is ignored while enable = 1, but
 * the stream is the commander watchdog's keepalive (motors stop 500 ms
 * after the last setpoint) and clearing enable is an instantaneous,
 * format-compatible takeover.
 *
 * Engage protocol (the ground station's side is crazyflie-ros
 * `exec: onboard`): verify nnpol.hash0-3 against the run directory's
 * firmware_export.json, push startPhase/offX/offY/offZ/vnom/ctrlHz, then
 * set enable = 1. The rising edge latches t0 and the pushed values, so the
 * curve clock starts at the engage instant with the host-pinned offset.
 *
 * Policy rate: nnpol.ctrlHz is WRITABLE and defaults to the checkpoint's
 * trained ctrl_freq (NNPOL_CTRL_FREQ_HZ). The ground station pushes the
 * configured inference rate before enable; the rising edge turns it into
 * a tick divider (RATE_MAIN_LOOP / ctrlHz, integer), so only divisors of
 * 1000 run exactly — anything else runs at the next divisor UP (48 -> 50
 * Hz) and the read-only nnpol.runHz reports what actually engaged. The
 * host validates divisibility before pushing; the observation's curve
 * clock is tick-based, so a rate change never shifts the reference.
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
#include "log.h"
#include "param.h"
#include "stabilizer_types.h"
#include "usec_time.h"

#include "firmware_export.h"
#include "nnpol.h"
#include "policy.h"

#if POLICY_OBS_DIM != NNPOL_OBS_DIM
#error "policy.h and firmware_export.h disagree on the observation size - regenerate both with export_policy_c.py"
#endif

/** Policy periods without a fresh action before the log counter runs. */
#define NNPOL_STALE_LOG_PERIODS 2
/** Policy periods without a fresh action before falling back to the radio
 * setpoint (the shadow host command). */
#define NNPOL_STALE_FALLBACK_PERIODS 5

// --------------------------------------------------------------------------
// Parameters
// --------------------------------------------------------------------------

static uint8_t nnpolEnable = 0;
static float paramStartPhase = NNPOL_DEFAULT_START_PHASE_S;
static float paramOffX = 0.0f;
static float paramOffY = 0.0f;
static float paramOffZ = 0.0f;
static float paramVnom = 4.0f;   /* same fixed voltage the host maps thrust at */

/* Read-only identity/architecture — checked by the ground station before
 * an onboard engage; a vehicle flashed with a different checkpoint reports
 * different words and the ROS node refuses to start. */
static uint32_t paramHash0 = NNPOL_HASH0;
static uint32_t paramHash1 = NNPOL_HASH1;
static uint32_t paramHash2 = NNPOL_HASH2;
static uint32_t paramHash3 = NNPOL_HASH3;
static uint16_t paramObsDim = NNPOL_OBS_DIM;
static uint16_t paramActDim = NNPOL_ACTION_DIM;

/* Policy rate. Writable (the host's policy.inference_hz), defaulting to the
 * trained rate; latched into a tick divider at the enable rising edge. */
static uint16_t paramCtrlHz = NNPOL_CTRL_FREQ_HZ;
/* Read-only: the rate the current/last engaged run actually steps at,
 * RATE_MAIN_LOOP / divider. Differs from ctrlHz only for a non-divisor. */
static uint16_t paramRunHz = NNPOL_CTRL_FREQ_HZ;

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
  nnpolAttitudeCmd_t cmd;
  uint32_t seq;              /* 0 = no action computed since boot */
} nnpolActionSlot_t;

static nnpolActionSlot_t actionSlot;
static volatile uint32_t actionLock = 0;

static void actionWrite(const nnpolAttitudeCmd_t* cmd)
{
  actionLock++;
  __asm volatile ("" ::: "memory");
  actionSlot.cmd = *cmd;
  actionSlot.seq++;
  __asm volatile ("" ::: "memory");
  actionLock++;
}

static uint32_t actionRead(nnpolAttitudeCmd_t* out)
{
  uint32_t a, b;
  nnpolActionSlot_t slot;
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
static float logVb[3], logWb[3], logE0[3];  /* obs excerpts */
static uint16_t logUs;                      /* inference microseconds */
static uint16_t logStale;                   /* periods spent >= stale threshold */
static uint32_t logSeq;

// --------------------------------------------------------------------------
// App task: observation + network + decode
// --------------------------------------------------------------------------

static TaskHandle_t appTaskHandle = NULL;

void appMain()
{
  appTaskHandle = xTaskGetCurrentTaskHandle();
  DEBUG_PRINT("nnpol: %s/%s, obs %d act %d, %d Hz, hash %08lx %08lx %08lx %08lx\n",
              NNPOL_TASK, NNPOL_OBS_FN, NNPOL_OBS_DIM, NNPOL_ACTION_DIM,
              NNPOL_CTRL_FREQ_HZ,
              (unsigned long)paramHash0, (unsigned long)paramHash1,
              (unsigned long)paramHash2, (unsigned long)paramHash3);

  while (1) {
    ulTaskNotifyTake(pdTRUE, portMAX_DELAY);

    nnpolSnapshot_t snap;
    snapshotRead(&snap);

    const uint64_t tStart = usecTimestamp();

    float obs[NNPOL_OBS_DIM];
    nnpolBuildObs(&snap, obs);

    float act[NNPOL_ACTION_DIM];
    policyForward(obs, act);
    for (int i = 0; i < NNPOL_ACTION_DIM; i++) {
      if (act[i] < -1.0f) { act[i] = -1.0f; }
      if (act[i] > 1.0f) { act[i] = 1.0f; }
    }

    nnpolAttitudeCmd_t cmd;
    nnpolDecodeAction(act, paramVnom, &cmd);

    const uint64_t elapsed = usecTimestamp() - tStart;

    actionWrite(&cmd);

    /* Log fields: written only here (single writer), read by the log
     * framework whenever a block samples them. */
    logT = snap.t;
    memcpy(logAct, act, sizeof(logAct));
    logSpRoll = cmd.rollDeg;
    logSpPitch = cmd.pitchDeg;
    logSpYaw = cmd.yawDeg;
    logSpThrust = cmd.thrustSetpoint;
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
static float activeStartPhase = NNPOL_DEFAULT_START_PHASE_S;
static float activeOff[3] = { 0.0f, 0.0f, 0.0f };
static uint32_t seqAtEngage = 0;      /* only actions newer than this fly */
static uint32_t lastSeqSeen = 0;
static uint16_t periodsWithoutFresh = 0;
static uint32_t activeDivider = RATE_MAIN_LOOP / NNPOL_CTRL_FREQ_HZ;

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
  wasEnabled = false;
  periodsWithoutFresh = 0;
}

bool controllerOutOfTreeTest()
{
  return controllerMellingerFirmwareTest();
}

void controllerOutOfTree(control_t* control, const setpoint_t* setpoint,
                         const sensorData_t* sensors, const state_t* state,
                         const uint32_t tick)
{
  const bool enabled = (nnpolEnable != 0);

  if (enabled && !wasEnabled) {
    /* Rising edge: the curve clock starts NOW, with the values the host
     * pushed before flipping enable. Latching them here (not reading the
     * params live) means a mid-run param write cannot shift the clock or
     * teleport the curve. */
    t0Tick = tick;
    activeStartPhase = paramStartPhase;
    activeOff[0] = paramOffX;
    activeOff[1] = paramOffY;
    activeOff[2] = paramOffZ;
    activeDivider = tickDividerFor(paramCtrlHz);
    paramRunHz = (uint16_t)(RATE_MAIN_LOOP / activeDivider);
    nnpolAttitudeCmd_t unused;
    seqAtEngage = actionRead(&unused);
    lastSeqSeen = seqAtEngage;
    periodsWithoutFresh = 0;
    logStale = 0;
  }
  wasEnabled = enabled;

  if (enabled && (tick % activeDivider) == 0u) {
    /* Policy tick: snapshot a consistent state and wake the app task.
     * The EKF quaternion is stored x, y, z, w; the observation wants
     * scalar-first — reordered here, sign-normalized in nnpolBuildObs. */
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
    snapshotWrite(&snap);

    if (appTaskHandle != NULL) {
      xTaskNotifyGive(appTaskHandle);
    }

    /* Staleness accounting, one step behind by construction: the action
     * requested by THIS notify lands ~one 1 kHz tick from now, so what is
     * checked here is whether the previous period ever delivered. */
    nnpolAttitudeCmd_t peek;
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

  nnpolAttitudeCmd_t cmd;
  const uint32_t seq = actionRead(&cmd);
  const bool haveFreshAction = enabled && seq != seqAtEngage
      && periodsWithoutFresh < NNPOL_STALE_FALLBACK_PERIODS;

  if (haveFreshAction) {
    /* Zero-order hold of the latest complete action, in the exact
     * attitude-only setpoint shape the radio path produces
     * (crtp_commander_rpyt ANGLE branch): roll/pitch/yaw modeAbs,
     * x/y/z modeDisable, thrust already through the legacy clamp. */
    setpoint_t sp;
    memset(&sp, 0, sizeof(sp));
    sp.timestamp = setpoint->timestamp;
    sp.mode.x = modeDisable;
    sp.mode.y = modeDisable;
    sp.mode.z = modeDisable;
    sp.mode.roll = modeAbs;
    sp.mode.pitch = modeAbs;
    sp.mode.yaw = modeAbs;
    sp.mode.quat = modeDisable;
    sp.attitude.roll = cmd.rollDeg;
    sp.attitude.pitch = cmd.pitchDeg;
    sp.attitude.yaw = cmd.yawDeg;
    sp.thrust = cmd.thrustSetpoint;
    controllerMellingerFirmware(control, &sp, sensors, state, tick);
  } else {
    /* Disabled, no action computed yet, or stale past the fallback
     * threshold: fly the radio setpoint — the shadow host command. */
    controllerMellingerFirmware(control, setpoint, sensors, state, tick);
  }
}

// --------------------------------------------------------------------------
// Param / log groups
// --------------------------------------------------------------------------

/**
 * Onboard policy control and identity.
 */
PARAM_GROUP_START(nnpol)
/** @brief 1 = the onboard policy commands the vehicle; 0 = radio setpoint
 * passthrough (offboard/shadow). Clearing mid-flight is an instantaneous,
 * format-compatible takeover by the host stream. */
PARAM_ADD(PARAM_UINT8, enable, &nnpolEnable)
/** @brief Curve phase (s) the clock starts at on the enable rising edge. */
PARAM_ADD(PARAM_FLOAT, startPhase, &paramStartPhase)
/** @brief World-frame curve translation, pushed by the host's pin. */
PARAM_ADD(PARAM_FLOAT, offX, &paramOffX)
PARAM_ADD(PARAM_FLOAT, offY, &paramOffY)
PARAM_ADD(PARAM_FLOAT, offZ, &paramOffZ)
/** @brief Battery voltage the thrust map assumes (host parity, phase 1). */
PARAM_ADD(PARAM_FLOAT, vnom, &paramVnom)
/** @brief Policy rate (Hz) latched at the enable rising edge. Defaults to
 * the checkpoint's trained ctrl_freq; must divide 1000 to run exactly. */
PARAM_ADD(PARAM_UINT16, ctrlHz, &paramCtrlHz)
/** @brief Rate (Hz) the engaged run actually steps at, read-only. */
PARAM_ADD(PARAM_UINT16 | PARAM_RONLY, runHz, &paramRunHz)
/** @brief Policy identity (firmware_export.json hash words), read-only. */
PARAM_ADD(PARAM_UINT32 | PARAM_RONLY, hash0, &paramHash0)
PARAM_ADD(PARAM_UINT32 | PARAM_RONLY, hash1, &paramHash1)
PARAM_ADD(PARAM_UINT32 | PARAM_RONLY, hash2, &paramHash2)
PARAM_ADD(PARAM_UINT32 | PARAM_RONLY, hash3, &paramHash3)
PARAM_ADD(PARAM_UINT16 | PARAM_RONLY, obsDim, &paramObsDim)
PARAM_ADD(PARAM_UINT16 | PARAM_RONLY, actDim, &paramActDim)
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
/** @brief Decoded setpoint (deg, deg, deg, legacy thrust). */
LOG_ADD(LOG_FLOAT, spRoll, &logSpRoll)
LOG_ADD(LOG_FLOAT, spPitch, &logSpPitch)
LOG_ADD(LOG_FLOAT, spYaw, &logSpYaw)
LOG_ADD(LOG_FLOAT, spThrust, &logSpThrust)
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
/** @brief Actions computed since boot. */
LOG_ADD(LOG_UINT32, seq, &logSeq)
LOG_GROUP_STOP(nnpol)
