/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include <cctag/Level.hpp>
#include <cctag/filter/cvRecode.hpp>
#include <cctag/filter/thinning.hpp>
#include "cctag/utils/Talk.hpp"
#ifdef CCTAG_WITH_CUDA
#include "cctag/cuda/tag.h"
#endif
#include <opencv2/imgproc.hpp>
#include <opencv2/imgproc/types_c.h>

#include <iostream>
#include <memory>

namespace cctag {

Level::Level( std::size_t width, std::size_t height, int debug_info_level, bool cuda_allocates )
    : _level( debug_info_level )
    , _cuda_allocates( cuda_allocates )
    , _mat_initialized_from_cuda( false )
    , _cols( width )
    , _rows( height )
    , _dx(nullptr)
    , _dy(nullptr)
    , _mag(nullptr)
    , _src(nullptr)
    , _edges(nullptr)
{
    if(!_cuda_allocates)
    {
        // Keep ownership local until every allocation has succeeded.
        auto source = std::make_unique<cv::Mat>(height, width, CV_8UC1);
        auto derivativeX = std::make_unique<cv::Mat>(height, width, CV_16SC1);
        auto derivativeY = std::make_unique<cv::Mat>(height, width, CV_16SC1);
        auto magnitude = std::make_unique<cv::Mat>(height, width, CV_16SC1);
        auto edges = std::make_unique<cv::Mat>(height, width, CV_8UC1);
        _temp.create(height, width, CV_8UC1);

#ifdef CCTAG_EXTRA_LAYER_DEBUG
        _edgesNotThin.create(height, width, CV_8UC1);
#endif

        _src = source.release();
        _dx = derivativeX.release();
        _dy = derivativeY.release();
        _mag = magnitude.release();
        _edges = edges.release();
    }
}

Level::~Level( )
{
    delete _src;
    delete _dx;
    delete _dy;
    delete _mag;
    delete _edges;
}

void Level::setLevel( const cv::Mat & src,
                      float thrLowCanny,
                      float thrHighCanny,
                      const cctag::Parameters* params )
{
    if( _cuda_allocates ) {
        std::cerr << "This function makes no sense with CUDA in " << __FUNCTION__ << ":" << __LINE__ << std::endl;
        exit( -__LINE__ );
    }

    cv::resize( src, *_src, cv::Size(_src->cols,_src->rows) );
    // Compute derivative and canny edge extraction.
    cvRecodedCanny( *_src, *_edges, *_dx, *_dy,
                    thrLowCanny * 256, thrHighCanny * 256,
                    3 | CV_CANNY_L2_GRADIENT,
                    _level, params );
#ifdef CCTAG_EXTRA_LAYER_DEBUG
    _edgesNotThin = _edges->clone();
#endif
  
    thin(*_edges,_temp);
}

#ifdef CCTAG_WITH_CUDA
void Level::setLevel( cctag::TagPipe*         cuda_pipe,
                      const cctag::Parameters& params )
{
    if( ! _cuda_allocates ) {
        std::cerr << "This function makes no sense without CUDA in " << __FUNCTION__ << ":" << __LINE__ << std::endl;
        exit( -__LINE__ );
    }

    // The pipeline owns the pixels; Level owns the cv::Mat headers it returns.
    std::unique_ptr<cv::Mat> source(cuda_pipe->getPlane(_level));
    std::unique_ptr<cv::Mat> derivativeX(cuda_pipe->getDx(_level));
    std::unique_ptr<cv::Mat> derivativeY(cuda_pipe->getDy(_level));
    std::unique_ptr<cv::Mat> magnitude(cuda_pipe->getMag(_level));
    std::unique_ptr<cv::Mat> edges(cuda_pipe->getEdges(_level));

    delete _src;
    delete _dx;
    delete _dy;
    delete _mag;
    delete _edges;
    _src = source.release();
    _dx = derivativeX.release();
    _dy = derivativeY.release();
    _mag = magnitude.release();
    _edges = edges.release();
}
#endif // CCTAG_WITH_CUDA

const cv::Mat & Level::getSrc() const
{
    return *_src;
}

#ifdef CCTAG_EXTRA_LAYER_DEBUG
const cv::Mat & Level::getCannyNotThin() const
{
    return _edgesNotThin;
}
#endif

const cv::Mat & Level::getDx() const
{
    return *_dx;
}

const cv::Mat & Level::getDy() const
{
    return *_dy;
}

const cv::Mat & Level::getMag() const
{
    return *_mag;
}

const cv::Mat & Level::getEdges() const
{
    return *_edges;
}

}
