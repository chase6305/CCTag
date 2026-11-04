/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef VISION_MARKER_CCTAG_MARKERS_BANK_HPP
#define VISION_MARKER_CCTAG_MARKERS_BANK_HPP

#include <cstddef>
#include <string>
#include <vector>

namespace cctag
{

class CCTagMarkersBank
{
public:
  // Built-in tables support three or four crowns; other counts create an empty bank.
  explicit CCTagMarkersBank( std::size_t nCrowns );
  explicit CCTagMarkersBank( const std::string & file );
  
  virtual ~CCTagMarkersBank() = default;

  CCTagMarkersBank(const CCTagMarkersBank&) = default;
  CCTagMarkersBank& operator=(const CCTagMarkersBank&) = default;
  CCTagMarkersBank(CCTagMarkersBank&&) noexcept = default;
  CCTagMarkersBank& operator=(CCTagMarkersBank&&) noexcept = default;

  // Append one marker per nonblank line, using outer-radius / inner-radius ratios.
  // Ratios may be decimals or unsigned fractions and must be positive and finite.
  // Invalid or empty files throw without changing the existing bank.
  void read( const std::string & file );

  // Return the nearest marker's one-based ID; only equal-length profiles are compared.
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
