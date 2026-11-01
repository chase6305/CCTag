

#include <doctest/doctest.h>

#include <cctag/geometry/Point.hpp>
#include <cctag/geometry/Ellipse.hpp>
#include <cctag/geometry/Distance.hpp>
#include <cctag/geometry/EllipseFromPoints.hpp>
#include <Eigen/Geometry>
#include <cctag/utils/Math.hpp>
#include <limits>

using Point3f = cctag::Point2d<Eigen::Vector3f>;

TEST_SUITE("test_ellipse") {

TEST_CASE("test_pixel_distance_avoids_integer_square_overflow")
{
    using Point3i = cctag::Point2d<Eigen::Vector3i>;
    const Point3i a(-32760, 0);
    const Point3i b(32760, 0);
    CHECK_EQ(cctag::numerical::distancePoints2D(a, b), 65520.f);
    CHECK_EQ(cctag::numerical::distancePoints2D(Point3f(0.f, 0.f), Point3f(3.f, 4.f)), 5.f);
}

TEST_CASE("test_nonfinite_ellipse_coordinates_throw_before_integer_conversion")
{
    using namespace cctag::numerical::geometry;
    Ellipse ellipse(Point3f(0.f, 0.f), 5.f, 3.f, 0.f);
    cctag::Point2d<Eigen::Vector3i> a, b, c, d;
    for(float invalid : {std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN()})
    {
        ellipse.center().x() = invalid;
        CHECK_THROWS_AS(computeIntermediatePoints(ellipse, a, b, c, d), std::runtime_error);
    }
}

TEST_CASE("test_ellipse_representation")
{
    const float aGT = 5.0f;
    const float bGT = 3.0f;
    const float cxGT = 16.0f;
    const float cyGT = -4.0f;

    // some points satisfying x^2/aGT^2 + y^2/bGT^2 = 1
    std::vector<Point3f> pts;
    pts.emplace_back( 5.0000f, .0f);
    pts.emplace_back( 4.7553f, 0.9271f);
    pts.emplace_back( 4.0451f, 1.7634f);
    pts.emplace_back( 2.9389f, 2.4271f);
    pts.emplace_back( 1.5451f, 2.8532f);
    pts.emplace_back( 0.0000f, 3.0000f);
    pts.emplace_back(-1.5451f, 2.8532f);
    pts.emplace_back(-2.9389f, 2.4271f);
    pts.emplace_back(-4.0451f, 1.7634f);
    pts.emplace_back(-4.7553f, 0.9271f);
    pts.emplace_back(-5.0000f, .0f);
    pts.emplace_back(-4.7553f, -0.9271f);
    pts.emplace_back(-4.0451f, -1.7634f);
    pts.emplace_back(-2.9389f, -2.4271f);
    pts.emplace_back(-1.5451f, -2.8532f);
    pts.emplace_back(-0.0000f, -3.0000f);
    pts.emplace_back( 1.5451f, -2.8532f);
    pts.emplace_back( 2.9389f, -2.4271f);
    pts.emplace_back( 4.0451f, -1.7634f);
    pts.emplace_back( 4.7553f, -0.9271f);
    pts.emplace_back( 5.0000f, .0f);

    const float pi_f = cctag::numerical::pi<float>();
    const float pi_2_f = (cctag::numerical::pi<float>() / 2.f);
    const std::size_t numTrials = 10;
    const float step = pi_f/numTrials;

    for(std::size_t i = 0; i < numTrials; ++i)
    {
        const float angleGT = i*step;
        using namespace cctag::numerical::geometry;
        cctag::numerical::geometry::Ellipse el(Point3f(cxGT, cyGT), aGT, bGT, angleGT); // avoid conflict with wingdi.h on windows

        // checking getters
        CHECK_UNARY(std::fabs(aGT - el.a()) < 0.00001);
        CHECK_UNARY(std::fabs(bGT - el.b()) < 0.00001);
        CHECK_UNARY(std::fabs(angleGT - el.angle()) < 0.00001);
        CHECK_UNARY(std::fabs(cxGT - el.center()(0)) < 0.00001);
        CHECK_UNARY(std::fabs(cyGT - el.center()(1)) < 0.00001);

        const auto &conic = el.matrix();

        const Eigen::Rotation2D<float> rotation{angleGT};
        Eigen::Matrix3f transform;
        transform.setIdentity();
        transform.topLeftCorner<2,2>() = rotation.toRotationMatrix();
        transform(0, 2) = cxGT;
        transform(1, 2) = cyGT;

        // check the (transformed) points satisfy the equation p'*C*p = 0
        for (const auto &p : pts)
        {
            const float result = p.transpose() * transform.transpose() * conic * transform * p;
            CHECK_UNARY(std::fabs(result) < 0.001);
        }

        // Test building the ellipse form a matrix
        cctag::numerical::geometry::Ellipse fromMatrix(conic);

        // major axis aligned with y-axis choice, so a and b are swapped and the angle is 90deg away
        CHECK_UNARY(std::fabs(fromMatrix.b() - aGT) < 0.001);
        CHECK_UNARY(std::fabs(fromMatrix.a() - bGT) < 0.001);
        CHECK_UNARY(std::fabs(fromMatrix.angle() - angleGT) - pi_2_f < 0.001);

        // test decomposition
        Ellipse::Matrix canonic;
        Ellipse::Matrix primal;
        Ellipse::Matrix dual;
        el.getCanonicForm(canonic, primal, dual);

        // test primal and dual are inverse
        const Ellipse::Matrix identity = primal * dual;
        for(Ellipse::Matrix::Index r = 0; r < identity.rows(); ++r)
        {
            for(Ellipse::Matrix::Index c = 0; c < identity.cols(); ++c)
            {
                if(r == c)
                {
                    CHECK_UNARY(std::fabs(identity(r, c) - 1) < 0.001);
                }
                else
                {
                    CHECK_UNARY(std::fabs(identity(r, c)) < 0.001);
                }
            }
        }

        // test points
        for (const auto &p : pts)
        {
            // since the axis are inverted, invert also the points
            const Point3f point{p.y(), p.x()};
            const float result = point.transpose() * canonic  * point;
            CHECK_UNARY(std::fabs(result) < 0.001);
        }

        // test building from canonical form
        cctag::numerical::geometry::Ellipse fromCanonic{canonic};
        CHECK_UNARY(std::fabs(fromCanonic.b() - aGT) < 0.001);
        CHECK_UNARY(std::fabs(fromCanonic.a() - bGT) < 0.001);
        CHECK_UNARY(std::fabs(fromCanonic.angle()) < 0.0001);
        // in canonical form the center is the origin
        CHECK_UNARY(std::fabs(fromCanonic.center()(0)) < 0.0001);
        CHECK_UNARY(std::fabs(fromCanonic.center()(1)) < 0.0001);
    }
}

} // TEST_SUITE

