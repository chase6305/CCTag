/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef _CCTAG_LINE_HPP_
#define _CCTAG_LINE_HPP_

#include <cctag/geometry/Point.hpp>
#include <cctag/Params.hpp>
#include <vector>

namespace cctag {

class ImageCut
{
public:
    ImageCut() = default;

    ImageCut(Point2d<Eigen::Vector3f> start, DirectedPoint2d<Eigen::Vector3f> stop)
      : ImageCut(start, stop, 0.f, 1.f, kDefaultSampleCutLength)
    {}

    ImageCut(Point2d<Eigen::Vector3f> start,
             DirectedPoint2d<Eigen::Vector3f> stop,
             float signalBegin,
             float signalEnd)
      : ImageCut(start, stop, signalBegin, signalEnd, kDefaultSampleCutLength)
    {}

    ImageCut(Point2d<Eigen::Vector3f> start,
             DirectedPoint2d<Eigen::Vector3f> stop,
             std::size_t sampleCount)
      : ImageCut(start, stop, 0.f, 1.f, sampleCount)
    {}

    ImageCut(Point2d<Eigen::Vector3f> start,
             DirectedPoint2d<Eigen::Vector3f> stop,
             float signalBegin,
             float signalEnd,
             std::size_t sampleCount)
      : _start(start), _stop(stop), _imgSignal(sampleCount), _beginSig(signalBegin), _endSig(signalEnd)
    {}

    ImageCut(const ImageCut&) = default;
    ImageCut& operator=(const ImageCut&) = default;
    ImageCut(ImageCut&&) = default;
    ImageCut& operator=(ImageCut&&) = default;
  
    const Point2d<Eigen::Vector3f>& start() const { return _start; }
    Point2d<Eigen::Vector3f>& start() { return _start; }

    const DirectedPoint2d<Eigen::Vector3f>& stop() const { return _stop; }
    DirectedPoint2d<Eigen::Vector3f>& stop() { return _stop; }

    const std::vector<float>& imgSignal() const { return _imgSignal; }
    std::vector<float>& imgSignal() { return _imgSignal; }

    float beginSig() const { return _beginSig; }
    float endSig() const { return _endSig; }

    bool outOfBounds() const { return _outOfBounds; }
    void setOutOfBounds(bool outOfBounds) { _outOfBounds = outOfBounds; }

    virtual ~ImageCut() = default;

private:
    // Initial image center; center optimization does not update this point.
    Point2d<Eigen::Vector3f> _start;

    // Outer edge point and its image gradient.
    DirectedPoint2d<Eigen::Vector3f> _stop;

    // Sampled image signal, rectified when read through the optimized homography.
    std::vector<float> _imgSignal;

    // True when the cut cannot be sampled from the image.
    bool _outOfBounds = false;

    // Sampled interval along the cut, where 0 is the center and 1 is the outer edge.
    float _beginSig = 0.f;
    float _endSig = 1.f;

};

}


#endif
