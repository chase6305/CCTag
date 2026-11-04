/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */

#include <cctag/Types.hpp>
#include <algorithm>
#include <cstring>
#include <limits>

namespace cctag
{

namespace
{
size_t checkedArea(size_t width, size_t height, size_t maximumArea)
{
  // EdgePoint stores zero-based coordinates in Scalar.
  constexpr size_t maximumDimension = static_cast<size_t>(std::numeric_limits<EdgePoint::Scalar>::max()) + 1;
  if (width > maximumDimension || height > maximumDimension)
    throw std::length_error("EdgePointCollection: image coordinates do not fit EdgePoint storage");
  if (width != 0 && height > maximumArea / width)
    throw std::length_error("EdgePointCollection: image resolution is too large");
  return width * height;
}
}

EdgePointCollection::EdgePointCollection(size_t w, size_t h) :
  _edgeMap(new int[checkedArea(w, h, MAX_RESOLUTION*MAX_RESOLUTION)]),
  _edgeList(new EdgePoint[std::min(w*h, MAX_POINTS)]),
  _linkList(new int[2*std::min(w*h, MAX_POINTS)]),
  _votersIndex(new int[std::min(w*h, MAX_POINTS)+CUDA_OFFSET+1]),
  _processedIn(new std::atomic<bool>[std::min(w*h, MAX_POINTS)]),
  _processedAux(new std::atomic<bool>[std::min(w*h, MAX_POINTS)])
{
  point_count() = 0;
  _edgeMapShape[0] = w; _edgeMapShape[1] = h;
  memset(_edgeMap.get(), -1, w*h*sizeof(int)); // -1 denotes pixels without an edge point.
  
  // Flags are initialized only when their point is added.
}

void EdgePointCollection::add_point(int vx, int vy, float vdx, float vdy)
{
  if (vx < 0 || vx >= _edgeMapShape[0] || vy < 0 || vy >= _edgeMapShape[1])
    throw std::out_of_range("EdgePointCollection::add_point: coordinate out of range");

  size_t imap = map_index(vx, vy);
  if (_edgeMap[imap] != -1)
    throw std::logic_error("EdgePointCollection::add_point: point already exists");

  if (point_count() >= MAX_POINTS)
    throw std::logic_error(std::string("EdgePointCollection::add_point: too many edge points (nb points: ") + std::to_string(point_count()) + ", max: " + std::to_string(MAX_POINTS) + ")");
  
  size_t ipoint = point_count()++;
  _edgeMap[imap] = ipoint;
  // Replacing this array element with the same type starts its new lifetime.
  new (&_edgeList[ipoint]) EdgePoint(vx, vy, vdx, vdy);
  _linkList[2*ipoint+0] = -1;
  _linkList[2*ipoint+1] = -1;
  std::atomic_init(&_processedIn[ipoint], false);
  std::atomic_init(&_processedAux[ipoint], false);
  // voter lists must be constructed afterwards
}


// The input is suboptimal but we don't care: it matters only for the CPU version;
// CUDA version will directly create the required representation.
void EdgePointCollection::create_voter_lists(const std::vector<std::vector<int>>& voter_lists)
{
  if (voter_lists.size() != point_count())
    throw std::length_error("EdgePointCollection::create_voters_lists: inconsistent sizes");
  
  _votersIndex[0+CUDA_OFFSET] = 0;
  size_t voterCount = 0;
  for (size_t i = 0; i < point_count(); ++i)
  {
    if (voter_lists[i].size() > MAX_VOTERLIST_SIZE - voterCount)
      throw std::length_error("EdgePointCollection::create_voters_lists: too many voters");
    voterCount += voter_lists[i].size();
    _votersIndex[i+1+CUDA_OFFSET] = static_cast<int>(voterCount);
  }

  _votersList.reset(new int[voterCount]);
  int *p = _votersList.get();
  for (const auto& vlist: voter_lists)
    p = std::copy(vlist.begin(), vlist.end(), p);

  if (p != _votersList.get() + voterCount)
    throw std::logic_error("EdgePointCollection::create_voters_lists: invalid count copied");
}

void EdgePointCollection::create_voter_lists_from_targets(const std::vector<int>& vote_targets)
{
  const size_t count = point_count();
  if (vote_targets.size() != count)
    throw std::length_error("EdgePointCollection::create_voter_lists_from_targets: inconsistent sizes");

  int* offsets = _votersIndex.get() + CUDA_OFFSET;
  std::fill(offsets, offsets + count + 1, 0);
  for (int target : vote_targets)
  {
    if (target < -1 || (target >= 0 && size_t(target) >= count))
      throw std::out_of_range("EdgePointCollection::create_voter_lists_from_targets: invalid target");
    if (target >= 0)
      ++offsets[target + 1];
  }
  for (size_t i = 1; i <= count; ++i)
    offsets[i] += offsets[i - 1];

  _votersList.reset(new int[offsets[count]]);
  // Start offsets double as write cursors. Visiting points in their original
  // order preserves the order of each list, including ties in later sampling.
  for (size_t i = 0; i < count; ++i)
    if (vote_targets[i] >= 0)
      _votersList[offsets[vote_targets[i]]++] = static_cast<int>(i);

  // The cursors now hold end offsets; restore the list boundaries in place.
  for (size_t i = count; i > 0; --i)
    offsets[i] = offsets[i - 1];
  offsets[0] = 0;
}

} // namespace cctag
