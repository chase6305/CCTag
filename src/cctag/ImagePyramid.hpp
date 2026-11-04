/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef _CCTAG_IMAGEPYRAMID_HPP
#define	_CCTAG_IMAGEPYRAMID_HPP

#include "cctag/Level.hpp"

#include <cstddef>
#include <vector>

namespace cctag {

struct Parameters; // forward declaration

class ImagePyramid
{
public:
  ImagePyramid();
  
  ImagePyramid( std::size_t width, std::size_t height, std::size_t nLevels, bool cuda_allocates );
  
  ~ImagePyramid();

  ImagePyramid(const ImagePyramid&) = delete;
  ImagePyramid& operator=(const ImagePyramid&) = delete;

  Level* getLevel( std::size_t level ) const;
  
  std::size_t getNbLevels() const;
  
  // Build CPU levels; CUDA levels are populated from the pipeline instead.
  void build(const cv::Mat & src, float thrLowCanny, float thrHighCanny, const cctag::Parameters* params );

private:
  std::vector<Level*> _levels;
};

// Scale signed image values to [0, 255]; constant images become zero.
void sIntToUchar(const cv::Mat& src, cv::Mat& dst);

}

#endif	/* _CCTAG_IMAGEPYRAMID_HPP */

