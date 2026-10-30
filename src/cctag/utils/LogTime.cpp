/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include "LogTime.hpp"

#include <ostream>

namespace cctag {
namespace logtime {

bool Mgmt::Measurement::doPrint( ) const
{
    return ( _count != 0 );
}

void Mgmt::Measurement::print( std::ostream& ostr ) const
{
    if( ! doPrint() ) return;
    ostr << _probe << ": "
         << static_cast<double>(_totalMilliseconds) / _count << "ms "
         << std::endl;
}

Mgmt::Mgmt( int rsvp )
    : _previous_time( Clock::now() )
    , _durations( rsvp )
    , _reserved( rsvp )
    , _idx( 0 )
{ }

void Mgmt::resetStartTime( )
{
    _previous_time = Clock::now();
    _idx = 0;
}

void Mgmt::print( std::ostream& ostr ) const
{
    int idx = 0;
    for( const Measurement& m : _durations ) {
        if( m.doPrint() ) {
	    ostr << "(" << idx++ << ") ";
	    m.print( ostr );
        }
    }
}

} // logtime
} // cctag

