#define BOOST_TEST_MODULE testIdentification
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>

#include <cctag/Identification.hpp>

BOOST_AUTO_TEST_SUITE(test_identification)

BOOST_AUTO_TEST_CASE(failed_center_optimization_preserves_input_geometry)
{
    using Point = cctag::Point2d<Eigen::Vector3f>;
    const cv::Mat image(64, 64, CV_8U, cv::Scalar(100));
    cctag::CCTagVisualDebug::instance().initBackgroundImage(image);
    const cctag::numerical::geometry::Ellipse ellipse(Point(32.f, 32.f), 10.f, 8.f, 0.f);
    cctag::Parameters params;
    params._imagedCenterNGridSample = 3;
    Point center(32.f, 32.f);
    Eigen::Matrix3f homography = Eigen::Matrix3f::Identity();
    const cctag::DirectedPoint2d<Eigen::Vector3f> stop(42.f, 32.f, 1.f, 0.f);
    // One cut has no readable cut pair, so no center candidate can be selected.
    std::vector<cctag::ImageCut> cuts{cctag::ImageCut(center, stop, std::size_t(16))};
    float residual = 0.f;

    BOOST_CHECK(!cctag::identification::imageCenterOptimizationGlob(
        homography, cuts, center, residual, 0.1f, image, ellipse, params));
    BOOST_CHECK_EQUAL(center.x(), 32.f);
    BOOST_CHECK_EQUAL(center.y(), 32.f);
    BOOST_CHECK(homography.isIdentity());
    cctag::CCTagVisualDebug::instance().clearSessions();
}

BOOST_AUTO_TEST_SUITE_END()
