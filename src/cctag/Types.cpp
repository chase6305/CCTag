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

namespace cctag
{

namespace
{
size_t checkedArea(size_t w, size_t h, size_t maxArea)
{
  if (w != 0 && h > maxArea / w)
    throw std::length_error("EdgePointCollection: image resolution is too large");
  return w*h;
}
}

EdgePointCollection::EdgePointCollection(size_t w, size_t h) :
  _edgeMap(new int[checkedArea(w, h, MAX_RESOLUTION*MAX_RESOLUTION)]),
  _edgeList(new EdgePoint[std::min(w*h, MAX_POINTS)]),
  _linkList(new int[2*std::min(w*h, MAX_POINTS)]),
  _votersIndex(new int[std::min(w*h, MAX_POINTS)+CUDA_OFFSET+1]),
  _processedIn(new unsigned[(std::min(w*h, MAX_POINTS)+3)/4]),
  _processedAux(new unsigned[(std::min(w*h, MAX_POINTS)+3)/4])
{
  point_count() = 0;
  _edgeMapShape[0] = w; _edgeMapShape[1] = h;
  memset(_edgeMap.get(), -1, w*h*sizeof(int));  // XXX@stian: unnecessary for CUDA
  
  // The existing flag layout stores four point indices in each unsigned word.
  const size_t flagWords = (std::min(w*h, size_t(MAX_POINTS)) + 3) / 4;
  memset(_processedIn.get(), 0, flagWords * sizeof(unsigned));
  memset(_processedAux.get(), 0, flagWords * sizeof(unsigned));
}

void EdgePointCollection::add_point(int vx, int vy, float vdx, float vdy)
{
  if (vx < 0 || vx >= _edgeMapShape[0] || vy < 0 || vy >= _edgeMapShape[1])
    throw std::out_of_range("EdgePointCollection::add_point: coordinate out of range");

  size_t imap = map_index(vx, vy);
  if (_edgeMap[imap] != -1)
    throw std::logic_error("EdgePointCollection::add_point: point already exists");

  // XXX@stian: new() below is technically UB, but the class has no defined dtors
  // so it's safe to re-new it in place w/o calling the dtor firs.
  
  if (point_count() >= MAX_POINTS)
    throw std::logic_error(std::string("EdgePointCollection::add_point: too many edge points (nb points: ") + std::to_string(point_count()) + ", max: " + std::to_string(MAX_POINTS) + ")");
  
  size_t ipoint = point_count()++;
  _edgeMap[imap] = ipoint;
  new (&_edgeList[ipoint]) EdgePoint(vx, vy, vdx, vdy);
  _linkList[2*ipoint+0] = -1;
  _linkList[2*ipoint+1] = -1;
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

} // namespace cctag
