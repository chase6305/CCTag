/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef VISION_EDGEPOINT_HPP_
#define VISION_EDGEPOINT_HPP_

#include <cctag/geometry/Point.hpp>
#include <cctag/utils/Defines.hpp>

#include <cstddef>
#include <atomic>
#include <cstdint>
#include <limits>
#include <new>
#include <sys/types.h>
#include <cmath>
#include <iosfwd>

namespace cctag
{

using Vector3s = Eigen::Matrix<short, 3, 1>;

class EdgePoint : public Vector3s
{
public:
  static constexpr std::size_t MAX_PROCESSED_RUNS = std::numeric_limits<std::uint64_t>::digits;

  // Keep unused collection slots uninitialized.
  EdgePoint() = default;

  // Copies retain geometry and start with fresh detection bookkeeping.
  EdgePoint( const EdgePoint& p )
    : Vector3s(p)
    , _grad( p._grad )
    , _normGrad ( p._normGrad )
    , _flowLength (0)
    , _processed( 0 )
    , _isMax( -1 )
    , _nSegmentOut(-1)
  {}

  EdgePoint( const int vx, const int vy, const float vdx, const float vdy )
    : Vector3s( vx, vy, 1 )
    , _grad(vdx, vdy)
    , _normGrad(std::sqrt( vdx * vdx + vdy * vdy ))
    , _flowLength (0)
    , _processed( 0 )
    , _isMax( -1 )
    , _nSegmentOut(-1)
  {
  }

  EdgePoint& operator=(const EdgePoint& point)
  {
    if(this != &point)
    {
      Vector3s::operator=(point);
      _grad = point._grad;
      _normGrad = point._normGrad;
      _flowLength = point._flowLength;
      // Reconstruct the flag to also initialize a default-constructed destination.
      new (&_processed) std::atomic<std::uint64_t>(point._processed.load(std::memory_order_relaxed));
      _isMax = point._isMax;
      _nSegmentOut = point._nSegmentOut;
    }
    return *this;
  }

  Eigen::Vector2f gradient() const
  {
    return _grad;
  }
  
  float dX() const
  {
    return _grad(0);
  }
  
   float dY() const
  {
    return _grad(1);
  }

  float normGradient() const
  {
    return _normGrad ;
  }

  friend std::ostream& operator<<( std::ostream& os, const EdgePoint& eP );

private:
  Eigen::Vector2f _grad;
  float _normGrad;
public:
  float _flowLength;
  std::atomic<std::uint64_t> _processed; // One visited bit per candidate run.
  int _isMax;
  int _nSegmentOut; // Outer-segment label; -1 means unassigned.
};

static_assert(sizeof(EdgePoint) == 40, "EdgePoint host layout must remain 40 bytes");

inline bool receivedMoreVoteThan(const EdgePoint * const p1,  const EdgePoint * const p2)
{
  return (p1->_isMax > p2->_isMax);
}

} // namespace cctag

#endif

