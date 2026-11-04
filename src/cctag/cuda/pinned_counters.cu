/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include "pinned_counters.h"
#include "debug_macros.hpp"
#include "onoff.h"

#include <cctag/cuda/cctag_cuda_runtime.h>
#include <iostream>
#include <stdexcept>

namespace cctag {

namespace { std::mutex defaultLimitsMutex; }

using namespace std;

bool PinnedCounters::_max_values_set = false;
int  PinnedCounters::_max_counters   = 100;
int  PinnedCounters::_max_points     = MAX_MARKER_FOR_IDENT;

/* This is system-wide unique allocation
 */
PinnedCounters pinned_counters[MAX_PIPES];

PinnedCounters::PinnedCounters( )
    : _counters( 0 )
    , _allocated_counters( 0 )
    , _nearby_points( 0 )
    , _nearby_point_counter( 0 )
{ }

PinnedCounters::~PinnedCounters( )
{
}

void PinnedCounters::setGlobalMax( int max_counters, int max_points )
{
    std::lock_guard<std::mutex> lock(defaultLimitsMutex);
    if( _max_values_set )
    {
        return;
    }

    _max_counters   = max_counters;
    _max_points     = max_points;
    _max_values_set = true;
}

void PinnedCounters::init(int tagPipe, int maxCounters, int maxPoints)
{
    if(tagPipe < 0 || tagPipe >= MAX_PIPES)
        throw std::out_of_range("PinnedCounters pipeline index is out of range");
    if(maxCounters < 0 || maxPoints < 0)
    {
        std::lock_guard<std::mutex> lock(defaultLimitsMutex);
        _max_values_set = true;
        if(maxCounters < 0)
            maxCounters = _max_counters;
        if(maxPoints < 0)
            maxPoints = _max_points;
    }
    pinned_counters[tagPipe].obj_init(maxCounters, maxPoints);
}

void PinnedCounters::release( int tagPipe )
{
    auto& counters = pinned_counters[tagPipe];
    std::lock_guard<std::mutex> lock(counters._lock);
    if(counters._counters)
        POP_CUDA_FREE_HOST(counters._counters);
    if(counters._nearby_points)
        POP_CUDA_FREE_HOST(counters._nearby_points);
    counters._counters = nullptr;
    counters._nearby_points = nullptr;
    counters._allocated_counters = 0;
    counters._nearby_point_counter = 0;
    counters._counterCapacity = 0;
    counters._pointCapacity = 0;
}

int& PinnedCounters::getCounter( int tagPipe )
{
    return pinned_counters[tagPipe].obj_getCounter( );
}

NearbyPoint& PinnedCounters::getPoint( int tagPipe, const char* file, int line )
{
    return pinned_counters[tagPipe].obj_getPoint( file, line );
}

NearbyPoint* PinnedCounters::getPointPtr( int tagPipe, const char* file, int line )
{
    return pinned_counters[tagPipe].obj_getPointPtr( file, line );
}

void PinnedCounters::obj_init(int maxCounters, int maxPoints)
{
    if(maxCounters <= 0 || maxPoints <= 0)
        throw std::invalid_argument("Pinned memory capacities must be positive");
    std::lock_guard<std::mutex> lock(_lock);
    if((_counters && _counterCapacity != maxCounters) ||
       (_nearby_points && _pointCapacity != maxPoints))
        throw std::logic_error("Release pinned memory before changing its capacities");
    if(!_counters)
        POP_CUDA_MALLOC_HOST(&_counters, maxCounters * sizeof(int));
    if(!_nearby_points)
        POP_CUDA_MALLOC_HOST(&_nearby_points, maxPoints * sizeof(NearbyPoint));
    _counterCapacity = maxCounters;
    _pointCapacity = maxPoints;
}

int& PinnedCounters::obj_getCounter( )
{
    _lock.lock();
    if( _allocated_counters < _counterCapacity ) {
        int idx = _allocated_counters++;
        _lock.unlock();
        return _counters[idx];
    } else {
        _lock.unlock();
        cerr << __FILE__ << ":" << __LINE__
             << "    Hard-coded number of integer counters in pinned memory is too small." << endl
             << "    Increase and recompile." << endl;
        exit( -1 );
    }
}

NearbyPoint& PinnedCounters::obj_getPoint( const char* file, int line )
{
    _lock.lock();
    if( _nearby_point_counter < _pointCapacity ) {
        int idx = _nearby_point_counter++;
        _lock.unlock();
        return _nearby_points[idx];
    } else {
        _lock.unlock();
        cerr << __FILE__ << ":" << __LINE__
             << "    called from " << file << ":" << line
             << "    Hard-coded number of Nearby Points in pinned memory is too small." << endl
             << "    Increase and recompile." << endl;
        exit( -1 );
    }
}

NearbyPoint* PinnedCounters::obj_getPointPtr( const char* file, int line )
{
    _lock.lock();
    if( _nearby_point_counter < _pointCapacity ) {
        int idx = _nearby_point_counter++;
        _lock.unlock();
        return &_nearby_points[idx];
    } else {
        _lock.unlock();
        cerr << __FILE__ << ":" << __LINE__
             << "    called from " << file << ":" << line
             << "    Hard-coded number of Nearyby Points in pinned memory is too small." << endl
             << "    Increase and recompile." << endl;
        return 0;
    }
}

void PinnedCounters::releaseAllPoints( int tagPipe )
{
    pinned_counters[tagPipe]._lock.lock();
    pinned_counters[tagPipe]._nearby_point_counter = 0;
    pinned_counters[tagPipe]._lock.unlock();
}

} // namespace cctag

