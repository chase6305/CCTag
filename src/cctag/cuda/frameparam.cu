/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include "frameparam.h"
#include "frame.h"

#include <cmath>
#include <stdexcept>

namespace cctag {

FrameParam::FrameParam(const cctag::Parameters& params)
  : cannyThrLow_x_256(params._cannyThrLow * 256.0f)
  , cannyThrHigh_x_256(params._cannyThrHigh * 256.0f)
  , ratioVoting(params._ratioVoting)
  , thrGradientMagInVote(params._thrGradientMagInVote)
  , distSearch(params._distSearch)
  , minVotesToSelectCandidate(params._minVotesToSelectCandidate)
  , nCrowns(params._nCrowns)
  , sampleCutLength(params._sampleCutLength)
  , neighbourSize(params._imagedCenterNeighbourSize)
  , gridNSample(params._imagedCenterNGridSample)
  , numCutsInIdentStep(params._numCutsInIdentStep)
{
    if(params._nCrowns > RESERVE_MEM_MAX_CROWNS)
        throw std::invalid_argument("CUDA crown count exceeds RESERVE_MEM_MAX_CROWNS");
    if(params._imagedCenterNGridSample != 5)
        throw std::invalid_argument("CUDA _imagedCenterNGridSample must be 5");
    if(params._numCutsInIdentStep < 2 || params._numCutsInIdentStep > 22)
        throw std::invalid_argument("CUDA _numCutsInIdentStep must be between 2 and 22");
    if(params._sampleCutLength < 2 || params._sampleCutLength > 100)
        throw std::invalid_argument("CUDA _sampleCutLength must be between 2 and 100");
    if(!std::isfinite(params._imagedCenterNeighbourSize) || params._imagedCenterNeighbourSize < 0.f)
        throw std::invalid_argument("CUDA _imagedCenterNeighbourSize must be finite and nonnegative");
}

} // namespace cctag
