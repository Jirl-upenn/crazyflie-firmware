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

#pragma once

#include "kalman_core.h"
#include "pose_history.h"

// Direct measurements of Crazyflie pose
void kalmanCoreUpdateWithPose(kalmanCoreData_t* this, poseMeasurement_t *pose);

/**
 * The same update, with the residual formed against a reference state
 * instead of the current one: refPos [x, y, z] and refQ [w, x, y, z] are
 * what the estimator held when the pose was captured (pose_history.h). The
 * residual is applied to the current state with the current gain, i.e. the
 * estimate's error is assumed to change slowly over the pose's age.
 *
 * With refPos = &this->S[KC_STATE_X] and refQ = this->q this IS
 * kalmanCoreUpdateWithPose, bit for bit.
 *
 * posResidual, if not NULL, receives the position residual of each axis as
 * it was applied [m].
 */
void kalmanCoreUpdateWithPoseAt(kalmanCoreData_t* this, const poseMeasurement_t *pose,
                                const float refPos[3], const float refQ[4],
                                float posResidual[3]);

/**
 * The delay-compensated update: look up the state held ageUs before nowUs
 * in the history (stateUs is the time the current state was last propagated
 * to), form the residual against it, apply it to the current state, and
 * carry the correction into the history.
 *
 * Returns false, with nothing changed, when the pose is older than the
 * history.
 */
bool kalmanCoreUpdateWithPoseDelayed(kalmanCoreData_t* this, const poseMeasurement_t *pose,
                                     poseHistory_t* history, uint32_t nowUs, uint32_t ageUs,
                                     uint32_t stateUs, float posResidual[3]);
