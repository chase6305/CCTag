#define BOOST_TEST_MODULE testIdentification
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>

#include <cctag/Identification.hpp>
#include <limits>
#include <cmath>
#include <stdexcept>

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

BOOST_AUTO_TEST_CASE(convolution_clamps_boundaries_and_keeps_first_peak)
{
    cctag::ImageCut cut;
    cut.imgSignal() = {5.f, 0.f};
    auto peak = cctag::identification::convImageCut({1.f, 1.f, 1.f}, cut);
    BOOST_CHECK_EQUAL(peak.first, 10.f);
    BOOST_CHECK_EQUAL(peak.second, 0.f);
    cut.imgSignal() = {-7.f, -2.f, -5.f};
    peak = cctag::identification::convImageCut({1.f}, cut);
    BOOST_CHECK_EQUAL(peak.first, -2.f);
    BOOST_CHECK_EQUAL(peak.second, 1.f);
    cut.imgSignal() = {3.f, 3.f, 2.f};
    peak = cctag::identification::convImageCut({1.f}, cut);
    BOOST_CHECK_EQUAL(peak.first, 3.f);
    BOOST_CHECK_EQUAL(peak.second, 0.f);
}

BOOST_AUTO_TEST_CASE(convolution_handles_empty_singleton_and_nan_signals)
{
    cctag::ImageCut cut;
    BOOST_CHECK_THROW(cctag::identification::convImageCut({1.f}, cut), std::invalid_argument);
    cut.imgSignal() = {-3.f};
    auto peak = cctag::identification::convImageCut({1.f, 1.f, 1.f}, cut);
    BOOST_CHECK_EQUAL(peak.first, -9.f);
    BOOST_CHECK_EQUAL(peak.second, 0.f);
    peak = cctag::identification::convImageCut({}, cut);
    BOOST_CHECK_EQUAL(peak.first, 0.f);
    BOOST_CHECK_EQUAL(peak.second, 0.f);
    cut.imgSignal() = {std::numeric_limits<float>::quiet_NaN(), 2.f};
    peak = cctag::identification::convImageCut({1.f}, cut);
    BOOST_CHECK(std::isnan(peak.first));
    BOOST_CHECK_EQUAL(peak.second, 0.f);
    cut.imgSignal() = {1.f, std::numeric_limits<float>::quiet_NaN(), 2.f};
    peak = cctag::identification::convImageCut({1.f}, cut);
    BOOST_CHECK_EQUAL(peak.first, 2.f);
    BOOST_CHECK_EQUAL(peak.second, 2.f);
}

BOOST_AUTO_TEST_SUITE_END()
