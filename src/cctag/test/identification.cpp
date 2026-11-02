
#include <doctest/doctest.h>

#include <cctag/Identification.hpp>
#include <limits>
#include <cmath>
#include <stdexcept>

TEST_SUITE("test_identification") {

TEST_CASE("failed_center_optimization_preserves_input_geometry")
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

    CHECK_UNARY(!cctag::identification::imageCenterOptimizationGlob(
        homography, cuts, center, residual, 0.1f, image, ellipse, params));
    CHECK_EQ(center.x(), 32.f);
    CHECK_EQ(center.y(), 32.f);
    CHECK_UNARY(homography.isIdentity());
    cctag::CCTagVisualDebug::instance().clearSessions();
}

TEST_CASE("convolution_clamps_boundaries_and_keeps_first_peak")
{
    cctag::ImageCut cut;
    cut.imgSignal() = {5.f, 0.f};
    auto peak = cctag::identification::convImageCut({1.f, 1.f, 1.f}, cut);
    CHECK_EQ(peak.first, 10.f);
    CHECK_EQ(peak.second, 0.f);
    cut.imgSignal() = {-7.f, -2.f, -5.f};
    peak = cctag::identification::convImageCut({1.f}, cut);
    CHECK_EQ(peak.first, -2.f);
    CHECK_EQ(peak.second, 1.f);
    cut.imgSignal() = {3.f, 3.f, 2.f};
    peak = cctag::identification::convImageCut({1.f}, cut);
    CHECK_EQ(peak.first, 3.f);
    CHECK_EQ(peak.second, 0.f);
}

TEST_CASE("convolution_handles_empty_singleton_and_nan_signals")
{
    cctag::ImageCut cut;
    CHECK_THROWS_AS(cctag::identification::convImageCut({1.f}, cut), std::invalid_argument);
    cut.imgSignal() = {-3.f};
    auto peak = cctag::identification::convImageCut({1.f, 1.f, 1.f}, cut);
    CHECK_EQ(peak.first, -9.f);
    CHECK_EQ(peak.second, 0.f);
    peak = cctag::identification::convImageCut({}, cut);
    CHECK_EQ(peak.first, 0.f);
    CHECK_EQ(peak.second, 0.f);
    cut.imgSignal() = {std::numeric_limits<float>::quiet_NaN(), 2.f};
    peak = cctag::identification::convImageCut({1.f}, cut);
    CHECK_UNARY(std::isnan(peak.first));
    CHECK_EQ(peak.second, 0.f);
    cut.imgSignal() = {1.f, std::numeric_limits<float>::quiet_NaN(), 2.f};
    peak = cctag::identification::convImageCut({1.f}, cut);
    CHECK_EQ(peak.first, 2.f);
    CHECK_EQ(peak.second, 2.f);
}

TEST_CASE("profile_votes_keep_last_id_on_ties_and_skip_unreadable_cuts")
{
    using Point = cctag::Point2d<Eigen::Vector3f>;
    using Stop = cctag::DirectedPoint2d<Eigen::Vector3f>;
    const cctag::identification::RadiusRatioBank bank{{2.f, 1.6f, 1.25f}, {2.f, 1.6f, 1.25f}};
    cctag::ImageCut readable(Point(0.f, 0.f), Stop(1.f, 0.f, 1.f, 0.f), 0.f, 1.f, std::size_t(64));
    for(std::size_t i = 0; i < readable.imgSignal().size(); ++i)
        readable.imgSignal()[i] = (i / 16) % 2 ? 0.f : 255.f;
    auto unreadable = readable;
    unreadable.setOutOfBounds(true);
    const std::vector<cctag::ImageCut> cuts{readable, unreadable};
    std::vector<std::list<float>> votes(2);
    votes[0].push_back(0.25f);

    CHECK_UNARY(cctag::identification::orazioDistanceRobust(votes, bank, cuts, 0.5f));
    CHECK_EQ(votes[0].size(), 1);
    CHECK_EQ(votes[0].front(), 0.25f);
    REQUIRE_EQ(votes[1].size(), 1);
    CHECK_UNARY(std::isfinite(votes[1].front()));
    CHECK_GE(votes[1].front(), 0.f);
    CHECK_LE(votes[1].front(), 1.f);
}

} // TEST_SUITE
