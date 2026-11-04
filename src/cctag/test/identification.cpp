#include <doctest/doctest.h>

#include <cctag/Identification.hpp>
#include <limits>
#include <cmath>
#include <stdexcept>

namespace {
using Point = cctag::Point2d<Eigen::Vector3f>;
using Stop = cctag::DirectedPoint2d<Eigen::Vector3f>;

cv::Mat gradientImage()
{
    cv::Mat image(64, 64, CV_8U);
    for(int y = 0; y < image.rows; ++y)
        for(int x = 0; x < image.cols; ++x)
            image.at<unsigned char>(y, x) = 2 * x + y;
    return image;
}
}

TEST_SUITE("test_identification")
{
    TEST_CASE("center_residual_requires_equal_readable_sample_counts")
    {
        const cv::Mat image = gradientImage();
        Point center(32.f, 32.f);
        const cctag::numerical::geometry::Ellipse ellipse(center, 10.f, 8.f, 0.f);
        std::vector<cctag::ImageCut> cuts{cctag::ImageCut(center, Stop(42.f, 32.f, 1.f, 0.f), std::size_t(16)),
                                          cctag::ImageCut(center, Stop(32.f, 42.f, 0.f, 1.f), std::size_t(32))};
        bool readable = true;
        CHECK_EQ(cctag::identification::costFunctionGlob(Eigen::Matrix3f::Identity(), cuts, image, readable),
                 std::numeric_limits<float>::max());
        CHECK_UNARY(!readable);
        cctag::Parameters params;
        params._imagedCenterNeighbourSize = 0.f;
        Eigen::Matrix3f homography = Eigen::Matrix3f::Identity();
        float residual = 0.f;
        CHECK_UNARY(!cctag::identification::refineConicFamilyGlob(
          0, homography, center, cuts, image, nullptr, ellipse, params, nullptr, residual));

        cuts.front().imgSignal().resize(32);
        cuts.insert(cuts.begin(), cctag::ImageCut(center, Stop(-100.f, 32.f, 1.f, 0.f), std::size_t(64)));
        CHECK_UNARY(cctag::identification::refineConicFamilyGlob(
          0, homography, center, cuts, image, nullptr, ellipse, params, nullptr, residual));
    }

    TEST_CASE("skipping_center_search_still_computes_geometry_and_residual")
    {
        const cv::Mat image = gradientImage();
        const Point originalCenter(32.f, 32.f);
        const cctag::numerical::geometry::Ellipse ellipse(originalCenter, 10.f, 8.f, 0.f);
        cctag::Parameters params;
        params._imagedCenterNeighbourSize = 0.f;
        float expectedResidual = 0.f;
        Eigen::Matrix3f expectedHomography;
        cctag::identification::computeHomographyFromEllipseAndImagedCenter(ellipse, originalCenter, expectedHomography);
        for(float initialResidual : {0.f, 100.f, std::numeric_limits<float>::max()})
        {
            Point center = originalCenter;
            Eigen::Matrix3f homography = Eigen::Matrix3f::Constant(initialResidual);
            float residual = initialResidual;
            std::vector<cctag::ImageCut> cuts{cctag::ImageCut(center, Stop(42.f, 32.f, 1.f, 0.f), std::size_t(32)),
                                              cctag::ImageCut(center, Stop(32.f, 42.f, 0.f, 1.f), std::size_t(32))};
            REQUIRE_UNARY(cctag::identification::refineConicFamilyGlob(
              0, homography, center, cuts, image, nullptr, ellipse, params, nullptr, residual));
            CHECK_GT(residual, 0.f);
            CHECK_UNARY(std::isfinite(residual));
            CHECK_UNARY(homography.isApprox(expectedHomography));
            if(initialResidual == 0.f)
                expectedResidual = residual;
            else
                CHECK_EQ(residual, expectedResidual);
        }
        for(float invalidSize : {-1.f, std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN()})
        {
            params._imagedCenterNeighbourSize = invalidSize;
            Point center = originalCenter;
            Eigen::Matrix3f homography = Eigen::Matrix3f::Identity();
            float residual = 0.f;
            std::vector<cctag::ImageCut> cuts{cctag::ImageCut(center, Stop(42.f, 32.f, 1.f, 0.f), std::size_t(32)),
                                              cctag::ImageCut(center, Stop(32.f, 42.f, 0.f, 1.f), std::size_t(32))};
            CHECK_UNARY(!cctag::identification::refineConicFamilyGlob(
              0, homography, center, cuts, image, nullptr, ellipse, params, nullptr, residual));
        }
        cctag::CCTagVisualDebug::instance().clearSessions();
    }

    TEST_CASE("undersampled_cuts_cannot_drive_center_optimization")
    {
        const cv::Mat image(64, 64, CV_8U, cv::Scalar(100));
        const Point originalCenter(32.f, 32.f);
        const cctag::numerical::geometry::Ellipse ellipse(originalCenter, 10.f, 8.f, 0.f);
        const cctag::Parameters params;
        for(std::size_t sampleCount : {0, 1})
        {
            std::vector<cctag::ImageCut> cuts(2,
                                              cctag::ImageCut(originalCenter, Stop(42.f, 32.f, 1.f, 0.f), sampleCount));
            bool readable = true;
            cctag::identification::costFunctionGlob(Eigen::Matrix3f::Identity(), cuts, image, readable);
            CHECK_UNARY(!readable);
            Point center = originalCenter;
            Eigen::Matrix3f homography = Eigen::Matrix3f::Identity();
            float residual = 0.f;
            CHECK_UNARY(!cctag::identification::imageCenterOptimizationGlob(
              homography, cuts, center, residual, 0.2f, image, ellipse, params));
            CHECK_EQ(center.x(), originalCenter.x());
            CHECK_EQ(center.y(), originalCenter.y());
            auto& cut = cuts.front();
            cctag::identification::cutInterpolated(cut, image);
            CHECK_UNARY(cut.outOfBounds());
        }
    }

    TEST_CASE("center_refinement_requires_two_profiles_when_search_is_skipped")
    {
        const cv::Mat image = gradientImage();
        Point center(32.f, 32.f);
        const cctag::numerical::geometry::Ellipse ellipse(center, 10.f, 8.f, 0.f);
        std::vector<cctag::ImageCut> cuts{cctag::ImageCut(center, Stop(42.f, 32.f, 1.f, 0.f), std::size_t(32))};
        cctag::Parameters params;
        params._imagedCenterNeighbourSize = 0.f;
        Eigen::Matrix3f homography = Eigen::Matrix3f::Identity();
        float residual = 0.f;
        CHECK_UNARY(!cctag::identification::refineConicFamilyGlob(
          0, homography, center, cuts, image, nullptr, ellipse, params, nullptr, residual));
        cctag::CCTagVisualDebug::instance().clearSessions();
    }

    TEST_CASE("unreadable_statistics_do_not_contribute_id_votes")
    {
        const cctag::identification::RadiusRatioBank bank{{2.f, 1.5f}, {3.f, 2.f}};
        for(float value :
            {0.f, 0.1f, 50.f, std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN()})
        {
            CAPTURE(value);
            std::vector<cctag::ImageCut> cuts{
              cctag::ImageCut(Point(0.f, 0.f), Stop(1.f, 0.f, 1.f, 0.f), std::size_t(100))};
            cuts.front().imgSignal().assign(100, value);
            std::vector<std::list<float>> votes(bank.size());
            cctag::identification::orazioDistanceRobust(votes, bank, cuts, 0.5f);
            for(const auto& idVotes : votes)
                CHECK_UNARY(idVotes.empty());
        }

        cctag::ImageCut cut(Point(0.f, 0.f), Stop(1.f, 0.f, 1.f, 0.f), std::size_t(100));
        for(std::size_t sample = 0; sample < cut.imgSignal().size(); ++sample)
            cut.imgSignal()[sample] = (sample / 10) % 2 ? 20.f : 80.f;
        cut.imgSignal().front() = std::numeric_limits<float>::quiet_NaN();
        std::vector<std::list<float>> votes(bank.size());
        cctag::identification::orazioDistanceRobust(votes, bank, {cut}, 0.5f);
        for(const auto& idVotes : votes)
            CHECK_UNARY(idVotes.empty());
    }

    TEST_CASE("nearby_grid_requires_at_least_two_samples_per_axis")
    {
        const Point center(32.f, 24.f);
        const cctag::numerical::geometry::Ellipse ellipse(center, 12.f, 8.f, 0.f);
        std::vector<Point> points;
        for(std::size_t sampleCount : {0, 1})
            CHECK_THROWS_AS(cctag::identification::getNearbyPoints(
                              ellipse, center, points, 0.5f, sampleCount, cctag::identification::GRID),
                            std::invalid_argument);
        cctag::identification::getNearbyPoints(ellipse, center, points, 0.5f, 3, cctag::identification::GRID);
        REQUIRE_EQ(points.size(), 9);
        CHECK_EQ(points.front().x(), doctest::Approx(29.f));
        CHECK_EQ(points.front().y(), doctest::Approx(21.f));
        CHECK_EQ(points[4].x(), doctest::Approx(center.x()));
        CHECK_EQ(points[4].y(), doctest::Approx(center.y()));
        CHECK_EQ(points.back().x(), doctest::Approx(35.f));
        CHECK_EQ(points.back().y(), doctest::Approx(27.f));
    }

    TEST_CASE("nearby_grid_rejects_an_overflowing_point_count")
    {
        const Point center(32.f, 24.f);
        const cctag::numerical::geometry::Ellipse ellipse(center, 12.f, 8.f, 0.f);
        std::vector<Point> points;
        CHECK_THROWS_AS(
          cctag::identification::getNearbyPoints(
            ellipse, center, points, 0.5f, std::numeric_limits<std::size_t>::max(), cctag::identification::GRID),
          std::length_error);
    }

    TEST_CASE("center_objective_matches_high_precision_pairwise_distance")
    {
        Eigen::Matrix3f homography = Eigen::Matrix3f::Identity();
        homography(0, 0) = homography(1, 1) = 40.f;
        homography(0, 2) = homography(1, 2) = 64.f;
        for(bool constantImage : {false, true})
        {
            cv::Mat image(128, 128, CV_8U);
            for(int y = 0; y < image.rows; ++y)
                for(int x = 0; x < image.cols; ++x)
                    image.at<unsigned char>(y, x) = constantImage ? 100 : (x * 17 + y * 23 + x * y) % 256;
            for(int cutCount : {2, 3, 8, 22, 50})
            {
                CAPTURE(constantImage);
                CAPTURE(cutCount);
                std::vector<cctag::ImageCut> cuts;
                for(int cutIndex = 0; cutIndex < cutCount; ++cutIndex)
                {
                    const float angle = 2.f * cctag::numerical::pi<float>() * cutIndex / cutCount;
                    cuts.emplace_back(Point(64.f, 64.f),
                                      Stop(64.f + 30.f * std::cos(angle), 64.f + 30.f * std::sin(angle), 1.f, 0.f),
                                      0.25f,
                                      1.f,
                                      std::size_t(100));
                }
                bool readable;
                const float residual = cctag::identification::costFunctionGlob(homography, cuts, image, readable);
                REQUIRE_UNARY(readable);
                long double expected = 0;
                std::size_t pairs = 0;
                for(std::size_t first = 0; first + 1 < cuts.size(); ++first)
                    for(std::size_t second = first + 1; second < cuts.size(); ++second)
                    {
                        for(std::size_t sample = 0; sample < cuts[first].imgSignal().size(); ++sample)
                        {
                            const long double difference = static_cast<long double>(cuts[first].imgSignal()[sample]) -
                                                           cuts[second].imgSignal()[sample];
                            expected += difference * difference;
                        }
                        ++pairs;
                    }
                expected /= pairs;
                const long double scale =
                  std::max(std::abs(expected), static_cast<long double>(std::numeric_limits<float>::min()));
                CHECK_LE(std::abs(static_cast<long double>(residual) - expected),
                         2 * std::numeric_limits<float>::epsilon() * scale);
            }
        }
    }

    TEST_CASE("short_signals_do_not_assign_an_id_without_votes")
    {
        cv::Mat image(64, 64, CV_8U);
        for(int y = 0; y < image.rows; ++y)
            for(int x = 0; x < image.cols; ++x)
                image.at<unsigned char>(y, x) = 2 * x + 10;
        const Point center(32.f, 32.f);
        const cctag::numerical::geometry::Ellipse ellipse(center, 10.f, 8.f, 0.f);
        cctag::CCTag tag;
        tag.setId(7);
        tag.setCenterImg(center);
        tag.setHomography(Eigen::Matrix3f::Identity());
        tag.setRescaledOuterEllipse(ellipse);
        std::vector<cctag::ImageCut> cuts(2, cctag::ImageCut(center, Stop(42.f, 32.f, 1.f, 0.f), std::size_t(16)));
        const cctag::identification::RadiusRatioBank bank{{2.f, 1.6f, 1.25f}};
        const cctag::Parameters params;
        CHECK_EQ(cctag::identification::identify_step_2(0, tag, cuts, bank, image, nullptr, params),
                 cctag::status::id_not_reliable);
        CHECK_EQ(tag.id(), 7);
        CHECK_UNARY(tag.ellipses().empty());
        cctag::CCTagVisualDebug::instance().clearSessions();
    }

    TEST_CASE("identification_without_a_marker_bank_preserves_the_input")
    {
        cctag::CCTag tag;
        tag.setCenterImg(cctag::Point2d<Eigen::Vector3f>(12.f, 18.f));
        tag.setHomography(Eigen::Matrix3f::Identity());
        tag.setQuality(0.5f);
        std::vector<cctag::ImageCut> cuts;
        const cv::Mat image(32, 32, CV_8U, cv::Scalar(100));
        const cctag::Parameters params;
        CHECK_EQ(cctag::identification::identify_step_2(0, tag, cuts, {}, image, nullptr, params),
                 cctag::status::id_not_reliable);
        CHECK_EQ(tag.x(), 12.f);
        CHECK_EQ(tag.y(), 18.f);
        CHECK_UNARY(tag.homography().isIdentity());
        CHECK_EQ(tag.quality(), 0.5f);
    }

    TEST_CASE("center_refinement_rejects_a_profile_without_contrast")
    {
        const cv::Mat image(32, 32, CV_8U, cv::Scalar(0));
        const cctag::numerical::geometry::Ellipse ellipse(Point(16.f, 16.f), 10.f, 8.f, 0.f);
        std::vector<cctag::ImageCut> cuts{
          cctag::ImageCut(Point(0.f, 0.f), Stop(20.f, 16.f, 1.f, 0.f), std::size_t(32)),
          cctag::ImageCut(Point(0.f, 0.f), Stop(16.f, 20.f, 0.f, 1.f), std::size_t(32))};
        cctag::Parameters params;
        params._imagedCenterNeighbourSize = 0.f;
        Point center(16.f, 16.f);
        Eigen::Matrix3f homography = Eigen::Matrix3f::Identity();
        float residual = 0.f;
        CHECK_UNARY(!cctag::identification::refineConicFamilyGlob(
          0, homography, center, cuts, image, nullptr, ellipse, params, nullptr, residual));
        CHECK_UNARY(std::isfinite(residual));
        cctag::CCTagVisualDebug::instance().clearSessions();
    }

    TEST_CASE("resampling_recomputes_bounds_for_each_homography")
    {
        const cv::Mat image(32, 32, CV_8U, cv::Scalar(100));
        std::vector<cctag::ImageCut> cuts{
          cctag::ImageCut(Point(0.f, 0.f), Stop(20.f, 16.f, 1.f, 0.f), std::size_t(16)),
          cctag::ImageCut(Point(0.f, 0.f), Stop(16.f, 20.f, 0.f, 1.f), std::size_t(16))};
        Eigen::Matrix3f shifted = Eigen::Matrix3f::Identity();
        shifted(0, 2) = shifted(1, 2) = 100.f;
        bool readable = true;
        cctag::identification::costFunctionGlob(shifted, cuts, image, readable);
        CHECK_UNARY(!readable);

        const float residual =
          cctag::identification::costFunctionGlob(Eigen::Matrix3f::Identity(), cuts, image, readable);
        CHECK_UNARY(readable);
        CHECK_LT(residual, 1e-8f);
        for(const auto& cut : cuts)
            CHECK_UNARY(!cut.outOfBounds());
    }

    TEST_CASE("affine_resampling_clears_a_previous_bounds_failure")
    {
        const cv::Mat image(32, 32, CV_8U, cv::Scalar(100));
        cctag::ImageCut cut(Point(8.f, 16.f), Stop(-10.f, 16.f, 1.f, 0.f), std::size_t(16));
        cctag::identification::cutInterpolated(cut, image);
        CHECK_UNARY(cut.outOfBounds());
        cut.stop() = Stop(24.f, 16.f, 1.f, 0.f);
        cctag::identification::cutInterpolated(cut, image);
        CHECK_UNARY(!cut.outOfBounds());
        for(float sample : cut.imgSignal())
            CHECK_EQ(sample, 50.f);
    }

    TEST_CASE("legacy_smoothing_handles_empty_and_constant_signals")
    {
        cctag::ImageCut cut;
        CHECK_NOTHROW(cctag::identification::blurImageCut(1.f, cut));
        CHECK_UNARY(cut.imgSignal().empty());
        for(std::size_t sampleCount : {1, 3, 16})
        {
            cut.imgSignal().assign(sampleCount, 100.f);
            cctag::identification::blurImageCut(1.f, cut);
            CHECK_EQ(cut.imgSignal().size(), sampleCount);
            for(float sample : cut.imgSignal())
                CHECK_EQ(sample, doctest::Approx(100.f));
        }
    }

    TEST_CASE("homography_projects_the_marker_origin_to_the_imaged_center")
    {
        const cctag::numerical::geometry::Ellipse ellipse(Point(32.f, 24.f), 12.f, 8.f, 0.3f);
        const Point imagedCenter(33.f, 25.f);
        Eigen::Matrix3f homography;
        cctag::identification::computeHomographyFromEllipseAndImagedCenter(ellipse, imagedCenter, homography);
        float x, y;
        cctag::identification::applyHomography(x, y, homography, 0.f, 0.f);
        CHECK_EQ(x, doctest::Approx(imagedCenter.x()));
        CHECK_EQ(y, doctest::Approx(imagedCenter.y()));
        for(const Point& unitPoint : {Point(1.f, 0.f), Point(0.f, 1.f), Point(-1.f, 0.f), Point(0.f, -1.f)})
        {
            cctag::identification::applyHomography(x, y, homography, unitPoint.x(), unitPoint.y());
            const Eigen::Vector3f imagePoint(x, y, 1.f);
            CHECK_EQ(imagePoint.dot(ellipse.matrix() * imagePoint), doctest::Approx(0.f).epsilon(0.001));
        }
    }

    TEST_CASE("median_keeps_input_and_averages_even_samples")
    {
        const std::vector<float> even{7.f, 1.f, 9.f, 3.f};
        CHECK_EQ(cctag::identification::computeMedian(even), 5.f);
        CHECK_EQ(even.front(), 7.f);
        CHECK_EQ(even.back(), 3.f);
        CHECK_EQ(cctag::identification::computeMedian(std::vector<float>{7.f, 1.f, 3.f}), 3.f);
        CHECK_EQ(cctag::identification::computeMedian(std::vector<float>{2.f}), 2.f);
        CHECK_THROWS_AS(cctag::identification::computeMedian(std::vector<float>{}), std::invalid_argument);
    }

    TEST_CASE("center_refinement_rejects_a_grid_that_cannot_shrink")
    {
        const cv::Mat image(64, 64, CV_8U, cv::Scalar(100));
        const cctag::numerical::geometry::Ellipse ellipse(Point(32.f, 32.f), 10.f, 8.f, 0.f);
        const cctag::DirectedPoint2d<Eigen::Vector3f> stop(42.f, 32.f, 1.f, 0.f);
        cctag::Parameters params;
        Point center(32.f, 32.f);
        Eigen::Matrix3f homography = Eigen::Matrix3f::Identity();
        std::vector<cctag::ImageCut> cuts(2, cctag::ImageCut(center, stop, std::size_t(32)));
        float residual = 0.f;
        for(std::size_t gridSize : {0, 1, 2, 3, 4})
        {
            params._imagedCenterNGridSample = gridSize;
            CHECK_UNARY(!cctag::identification::refineConicFamilyGlob(
              0, homography, center, cuts, image, nullptr, ellipse, params, nullptr, residual));
        }
        CHECK_EQ(center.x(), 32.f);
        CHECK_EQ(center.y(), 32.f);
        CHECK_UNARY(homography.isIdentity());
        cctag::CCTagVisualDebug::instance().clearSessions();
    }

    TEST_CASE("cut_collection_preserves_fractional_start_offset")
    {
        const cv::Mat image(32, 32, CV_8U, cv::Scalar(100));
        std::vector<cctag::ImageCut> cuts;
        cctag::identification::collectCuts(cuts, image, Point(8.f, 16.f), {Stop(24.f, 16.f, 1.f, 0.f)}, 16, 0.25f);

        REQUIRE_EQ(cuts.size(), 1);
        CHECK_EQ(cuts.front().beginSig(), 0.25f);
        CHECK_EQ(cuts.front().endSig(), 1.f);
        CHECK_EQ(cuts.front().imgSignal().size(), 16);
        CHECK_UNARY(!cuts.front().outOfBounds());
    }

    TEST_CASE("profile_voting_handles_empty_inputs_and_short_signals")
    {
        const cctag::identification::RadiusRatioBank bank{{2.f, 1.6f, 1.25f}};
        std::vector<std::list<float>> votes(1);
        std::vector<cctag::ImageCut> cuts;
        CHECK_UNARY(!cctag::identification::orazioDistanceRobust(votes, bank, cuts, 0.5f));
        cuts.emplace_back();
        CHECK_UNARY(!cctag::identification::orazioDistanceRobust(votes, {}, cuts, 0.5f));
        for(std::size_t size : {0, 1, 29, 30})
        {
            cuts.front().imgSignal().resize(size, 100.f);
            CHECK_UNARY(cctag::identification::orazioDistanceRobust(votes, bank, cuts, 0.5f));
            CHECK_UNARY(votes.front().empty());
        }
    }

    TEST_CASE("failed_center_optimization_preserves_input_geometry")
    {
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

    TEST_CASE("cost_function_rejects_missing_readable_pairs")
    {
        const cv::Mat image(32, 32, CV_8U, cv::Scalar(100));
        const Eigen::Matrix3f homography = Eigen::Matrix3f::Identity();
        std::vector<cctag::ImageCut> cuts;
        const auto checkMissingPair = [&] {
            bool readable = true;
            CHECK_EQ(cctag::identification::costFunctionGlob(homography, cuts, image, readable),
                     std::numeric_limits<float>::max());
            CHECK_UNARY(!readable);
        };
        checkMissingPair();
        cuts.emplace_back(Point(0.f, 0.f), Stop(20.f, 16.f, 1.f, 0.f), std::size_t(16));
        checkMissingPair();
        cuts.emplace_back(Point(0.f, 0.f), Stop(-1.f, -1.f, 1.f, 0.f), std::size_t(16));
        checkMissingPair();
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
