/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include <cctag/Canny.hpp>

#include "utils/Defines.hpp"

namespace cctag
{

void edgesPointsFromCanny(
        EdgePointCollection& edgeCollection,
        const cv::Mat & edges,
        const cv::Mat & dx,
        const cv::Mat & dy )
{
  std::size_t width = edges.cols;
  std::size_t height = edges.rows;
  if(width == 0)
    return;
  
  for( int y = 0 ; y < height ; ++y )
  {
    const uchar* row = &edges.at<uchar>(y, 0);
    for( int x = 0 ; x < width ; ++x )
    {
      if ( row[x] == 255 )
      {
        edgeCollection.add_point(x, y, dx.at<short>(y,x), dy.at<short>(y,x));
      }
    }
  }
}

} // namespace cctag


