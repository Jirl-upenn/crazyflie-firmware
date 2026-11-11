/**
 * ,---------,       ____  _ __
 * |  ,-^-,  |      / __ )(_) /_______________ _____  ___
 * | (  O  ) |     / __  / / __/ ___/ ___/ __ `/_  / / _ \
 * | / ,--'  |    / /_/ / / /_/ /__/ /  / /_/ / / /_/  __/
 *    +------`   /_____/_/\__/\___/_/   \__,_/ /___/\___/
 *
 * Crazyflie control firmware
 *
 * Copyright (C) 2021 Bitcraze AB
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, in version 3.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
 *
 */

#include "mm_pose.h"
#include "math3d.h"

void kalmanCoreUpdateWithPose(kalmanCoreData_t* this, poseMeasurement_t *pose)
{
  kalmanCoreUpdateWithPoseAt(this, pose, &this->S[KC_STATE_X], this->q, 0);
}

void kalmanCoreUpdateWithPoseAt(kalmanCoreData_t* this, const poseMeasurement_t *pose,
                                const float refPos[3], const float refQ[4],
                                float posResidual[3])
{
  // The reference as an offset from the current position, taken before the
  // first scalar update. Each axis is then compared with (current + offset):
  // the scalar updates are sequential, so a later axis has to see what the
  // earlier ones already corrected, and the reference moves with them. For
  // the plain update the offset is exactly zero and this is the stock code.
  const float offset[3] = {
    refPos[0] - this->S[KC_STATE_X],
    refPos[1] - this->S[KC_STATE_Y],
    refPos[2] - this->S[KC_STATE_Z],
  };
  // The reference attitude is read now for the same reason: refQ may alias
  // this->q, which the scalar updates below do not touch.
  struct quat const q_ref = mkquat(refQ[1], refQ[2], refQ[3], refQ[0]);

  // a direct measurement of states x, y, and z, and orientation
  // do a scalar update for each state, since this should be faster than updating all together
  for (int i=0; i<3; i++) {
    float h[KC_STATE_DIM] = {0};
    arm_matrix_instance_f32 H = {1, KC_STATE_DIM, h};
    h[KC_STATE_X+i] = 1;
    const float residual = pose->pos[i] - (this->S[KC_STATE_X+i] + offset[i]);
    if (posResidual) {
      posResidual[i] = residual;
    }
    kalmanCoreScalarUpdate(this, &H, residual, pose->stdDevPos);
  }

  // compute orientation error
  struct quat const q_measured = mkquat(pose->quat.x, pose->quat.y, pose->quat.z, pose->quat.w);
  struct quat const q_residual = qqmul(q_measured, qinv(q_ref));
  // small angle approximation, see eq. 141 in http://mars.cs.umn.edu/tr/reports/Trawny05b.pdf
  struct vec const err_quat = vscl(2.0f / q_residual.w, quatimagpart(q_residual));

  // do a scalar update for each state
  {
    float h[KC_STATE_DIM] = {0};
    arm_matrix_instance_f32 H = {1, KC_STATE_DIM, h};
    h[KC_STATE_D0] = 1;
    kalmanCoreScalarUpdate(this, &H, err_quat.x, pose->stdDevQuat);
    h[KC_STATE_D0] = 0;

    h[KC_STATE_D1] = 1;
    kalmanCoreScalarUpdate(this, &H, err_quat.y, pose->stdDevQuat);
    h[KC_STATE_D1] = 0;

    h[KC_STATE_D2] = 1;
    kalmanCoreScalarUpdate(this, &H, err_quat.z, pose->stdDevQuat);
  }
}

bool kalmanCoreUpdateWithPoseDelayed(kalmanCoreData_t* this, const poseMeasurement_t *pose,
                                     poseHistory_t* history, uint32_t nowUs, uint32_t ageUs,
                                     uint32_t stateUs, float posResidual[3])
{
  float refPos[3];
  float refQ[4];
  if (!poseHistoryLookup(history, nowUs, ageUs, stateUs, &this->S[KC_STATE_X], this->q,
                         refPos, refQ)) {
    return false;
  }

  const float before[6] = {
    this->S[KC_STATE_X], this->S[KC_STATE_Y], this->S[KC_STATE_Z],
    this->S[KC_STATE_D0], this->S[KC_STATE_D1], this->S[KC_STATE_D2],
  };
  kalmanCoreUpdateWithPoseAt(this, pose, refPos, refQ, posResidual);

  // What the update changed, applied to the history as well: the next pose
  // is usually older than the interval between poses, and must not be
  // compared with a past this correction has not reached.
  const float dPos[3] = {
    this->S[KC_STATE_X] - before[0],
    this->S[KC_STATE_Y] - before[1],
    this->S[KC_STATE_Z] - before[2],
  };
  // The attitude error states are a body-frame rotation vector, folded into
  // the quaternion as q * dq by the next finalize.
  const float dq[4] = {
    1.0f,
    0.5f * (this->S[KC_STATE_D0] - before[3]),
    0.5f * (this->S[KC_STATE_D1] - before[4]),
    0.5f * (this->S[KC_STATE_D2] - before[5]),
  };
  poseHistoryShift(history, dPos, dq);
  return true;
}
