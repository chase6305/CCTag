/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#pragma once

#include <chrono>
#include <cstddef>
#include <iosfwd>
#include <string>
#include <vector>

namespace cctag {
namespace logtime {

struct Mgmt
{
    using Clock = std::chrono::steady_clock;

    class Measurement
    {
    public:
        void log( const char* probename, std::chrono::microseconds duration ) {
            if( _count == 0 ) _probe = probename;
            _totalMilliseconds += std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();
            ++_count;
        }

        bool doPrint( ) const;

        void print( std::ostream& ostr ) const;

    private:
        std::string _probe;
        std::chrono::milliseconds::rep _totalMilliseconds = 0;
        std::size_t _count = 0;
    };

    Clock::time_point        _previous_time;
    std::vector<Measurement> _durations;
    int                      _reserved;
    int                      _idx;

    explicit Mgmt( int rsvp );

    void resetStartTime( );

    void log( const char* probename ) {
        // std::cerr << "logging >>>" << probename << "<<<" << std::endl;
        if( _idx >= _reserved ) return;

        const auto now = Clock::now();
        const auto duration = std::chrono::duration_cast<std::chrono::microseconds>(now - _previous_time);
        _previous_time = now;
        _durations[_idx].log( probename, duration );
        _idx++;
    }

    void print( std::ostream& ostr ) const;
};

} // logtime
} // cctag

