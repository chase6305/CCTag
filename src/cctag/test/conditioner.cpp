/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include <doctest/doctest.h>

#include <cctag/optimization/conditioner.hpp>

#include <array>
#include <cmath>

TEST_SUITE("test_conditioner") {

TEST_CASE("ellipse_normalization_uses_each_ellipses_own_scale")
{
    using Point = cctag::Point2d<Eigen::Vector3f>;
    using Ellipse = cctag::numerical::geometry::Ellipse;
    const std::array<Ellipse, 3> ellipses{{
        Ellipse(Point(20.f, 30.f), 10.f, 6.f, 0.f),
        Ellipse(Point(200.f, -60.f), 100.f, 50.f, .3f),
        Ellipse(Point(-4.f, 9.f), 3.f, 1.f, -.5f)
    }};

    // Revisit both larger and smaller ellipses in opposite orders.
    for(const int index : {0, 1, 2, 2, 1, 0})
    {
        const auto& ellipse = ellipses[index];
        const auto transform = cctag::numerical::optimization::conditionerFromEllipse(ellipse);
        const float meanRadius = (ellipse.a() + ellipse.b()) / 2.f;
        Point center = ellipse.center();
        Point offset(ellipse.center().x() + meanRadius, ellipse.center().y());
        cctag::numerical::optimization::condition(center, transform);
        cctag::numerical::optimization::condition(offset, transform);
        CHECK_EQ(center.x(), doctest::Approx(0.f).epsilon(1e-5));
        CHECK_EQ(center.y(), doctest::Approx(0.f).epsilon(1e-5));
        CHECK_EQ(offset.x(), doctest::Approx(std::sqrt(2.f)).epsilon(1e-5));
        CHECK_EQ(offset.y(), doctest::Approx(0.f).epsilon(1e-5));
    }
}

TEST_CASE("ellipse_normalization_is_invariant_to_uniform_image_scaling")
{
    using Point = cctag::Point2d<Eigen::Vector3f>;
    using Ellipse = cctag::numerical::geometry::Ellipse;
    const Ellipse small(Point(4.f, -2.f), 6.f, 2.f, .4f);
    const Ellipse large(Point(40.f, -20.f), 60.f, 20.f, .4f);
    Point smallPoint(7.f, -1.f), largePoint(70.f, -10.f);
    cctag::numerical::optimization::condition(
        smallPoint, cctag::numerical::optimization::conditionerFromEllipse(small));
    cctag::numerical::optimization::condition(
        largePoint, cctag::numerical::optimization::conditionerFromEllipse(large));
    CHECK_EQ(smallPoint.x(), doctest::Approx(largePoint.x()).epsilon(1e-5));
    CHECK_EQ(smallPoint.y(), doctest::Approx(largePoint.y()).epsilon(1e-5));
}

} // TEST_SUITE
