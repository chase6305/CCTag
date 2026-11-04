/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef _CCTAG_CONDITIONER_HPP_
#define _CCTAG_CONDITIONER_HPP_

#include <cctag/geometry/Point.hpp>
#include <cctag/Statistic.hpp>
#include <cctag/geometry/Ellipse.hpp>

namespace cctag {
namespace numerical {
namespace optimization {

inline Eigen::Matrix3f conditionerFromEllipse(const geometry::Ellipse& ellipse)
{
    Eigen::Matrix3f transform;
    static const float sqrt2 = std::sqrt(2.f);
    const float meanSemiAxis = (ellipse.a() + ellipse.b()) / 2.f;

    // Center this ellipse at the origin and scale its mean semi-axis to sqrt(2).
    transform(0, 0) = sqrt2 / meanSemiAxis;
    transform(0, 1) = 0.f;
    transform(0, 2) = -sqrt2 * ellipse.center().x() / meanSemiAxis;

    transform(1, 0) = 0.f;
    transform(1, 1) = sqrt2 / meanSemiAxis;
    transform(1, 2) = -sqrt2 * ellipse.center().y() / meanSemiAxis;

    transform(2, 0) = 0.f;
    transform(2, 1) = 0.f;
    transform(2, 2) = 1.f;
    return transform;
}


inline void conditionerFromImage( const int c, const int r, const int f,  Eigen::Matrix3f & trans, Eigen::Matrix3f & invTrans)
{
	//using namespace boost::numeric;
	trans(0,0) = 1.f / f  ; trans(0,1) = 0.f       ; trans(0,2) = -c/(2.0f * f);
	trans(1,0) = 0.f      ; trans(1,1) = 1.f / f   ; trans(1,2) = -r/(2.0f * f);
	trans(2,0) = 0.f      ; trans(2,1) = 0.f       ; trans(2,2) = 1.f;

	invTrans(0,0) = f   ; invTrans(0,1) = 0.f   ; invTrans(0,2) = c / 2.0f;
	invTrans(1,0) = 0.f ; invTrans(1,1) = f     ; invTrans(1,2) = r / 2.0f;
	invTrans(2,0) = 0.f ; invTrans(2,1) = 0.f   ; invTrans(2,2) = 1.f;
}

template <class T>
inline void condition(T & point, const Eigen::Matrix3f & mTransformation)
{
  //using namespace boost::numeric;
  const Eigen::Vector3f conditionedPoint = mTransformation*point;
  point.x() = conditionedPoint(0)/conditionedPoint(2);
  point.y() = conditionedPoint(1)/conditionedPoint(2);
}

template <class T>
inline void condition(std::vector<T> & points, const Eigen::Matrix3f & mTransformation)
{
  //using namespace boost::numeric;
  for(auto & point: points)
    condition(point, mTransformation);
}

}
}
}

#endif

