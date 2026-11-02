/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef VISION_MARKER_CCTAG_MARKERS_BANK_HPP
#define	VISION_MARKER_CCTAG_MARKERS_BANK_HPP

#include <cstddef>
#include <string>
#include <vector>

namespace cctag
{

class CCTagMarkersBank
{
public:
  explicit CCTagMarkersBank( std::size_t nCrowns );
  explicit CCTagMarkersBank( const std::string & file );
  
  virtual ~CCTagMarkersBank() = default;

  void read( const std::string & file );
  std::size_t identify( const std::vector<float> & marker ) const;
  inline const std::vector< std::vector<float> > & getMarkers() const
  {
    return _markers;
  }

private:
  static const float idThreeCrowns[32][5];
  static const float idFourCrowns[128][7];
  
  std::vector< std::vector<float> > _markers;

};

} // namespace cctag

#endif
