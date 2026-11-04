/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include <cctag/cuda/frameparam.h>
#include <cctag/cuda/pinned_counters.h>
#include <cctag/cuda/tag.h>
#include <doctest/doctest.h>
#include <opencv2/imgproc.hpp>

#include <array>
#include <cmath>
#include <future>
#include <limits>
#include <memory>
#include <stdexcept>

namespace cctag::identification {
// Host signatures of the CUDA kernels' launch stubs.
void idBestNearbyPoint31max(NearbyPointGrid*, std::size_t);
void idBestNearbyPoint32plus(NearbyPointGrid*, std::size_t);
}

namespace {

bool hasCudaDevice()
{
    int count = 0;
    const auto status = cudaGetDeviceCount(&count);
    if(status != cudaSuccess)
        cudaGetLastError();
    return status == cudaSuccess && count > 0;
}

cctag::Parameters singleLayerParameters()
{
    cctag::Parameters params;
    params._numberOfMultiresLayers = 1;
    params._numberOfProcessedMultiresLayers = 1;
    return params;
}

cv::Mat ringImage(int side)
{
    cv::Mat image(side, side, CV_8U, cv::Scalar(255));
    int color = 0;
    for(int radius = side / 3; radius > side / 10; radius -= 4)
    {
        cv::circle(image, {side / 2, side / 2}, radius, cv::Scalar(color), cv::FILLED);
        color = 255 - color;
    }
    return image;
}

int exportedPointCount(cctag::TagPipe& pipe, const cv::Mat& image, int frame)
{
    pipe.load(frame, image.data);
    pipe.tagframe();
    cctag::EdgePointCollection points(image.cols, image.rows);
    std::vector<cctag::EdgePoint*> seeds;
    pipe.convertToHost(0, points, seeds, 1000000);
    return points.get_point_count();
}

} // namespace

TEST_SUITE("test_cudaParameters") {

TEST_CASE("device_parameter_snapshots_are_independent")
{
    cctag::Parameters first(3), second(4);
    first._cannyThrLow = 0.01f;
    second._cannyThrLow = 0.2f;
    const cctag::FrameParam firstFrame(first), secondFrame(second);
    CHECK_EQ(firstFrame.nCrowns, 3);
    CHECK_EQ(secondFrame.nCrowns, 4);
    CHECK_EQ(firstFrame.cannyThrLow_x_256, first._cannyThrLow * 256.f);
    CHECK_EQ(secondFrame.cannyThrLow_x_256, second._cannyThrLow * 256.f);
}

TEST_CASE("identification_parameter_limits_are_checked_before_device_work")
{
    const cctag::Parameters defaults;
    CHECK_NOTHROW(cctag::FrameParam{defaults});
    for(const std::size_t cuts : {std::size_t(2), std::size_t(22)})
        for(const std::size_t samples : {std::size_t(2), std::size_t(100)})
        {
            auto params = defaults;
            params._numCutsInIdentStep = cuts;
            params._sampleCutLength = samples;
            params._imagedCenterNeighbourSize = 0.f;
            CHECK_NOTHROW(cctag::FrameParam{params});
        }

    const auto largest = std::numeric_limits<std::size_t>::max();
    for(const auto count : {std::size_t(0), std::size_t(1), std::size_t(2),
                            std::size_t(3), std::size_t(4), std::size_t(6), largest})
    {
        auto params = defaults;
        params._imagedCenterNGridSample = count;
        CHECK_THROWS_AS(cctag::FrameParam{params}, std::invalid_argument);
    }
    for(const auto count : {std::size_t(0), std::size_t(1), std::size_t(23), largest})
    {
        auto params = defaults;
        params._numCutsInIdentStep = count;
        CHECK_THROWS_AS(cctag::FrameParam{params}, std::invalid_argument);
    }
    for(const auto count : {std::size_t(0), std::size_t(1), std::size_t(101), largest})
    {
        auto params = defaults;
        params._sampleCutLength = count;
        CHECK_THROWS_AS(cctag::FrameParam{params}, std::invalid_argument);
    }
    for(const float width : {-1.f, std::numeric_limits<float>::quiet_NaN(),
                             std::numeric_limits<float>::infinity(),
                             -std::numeric_limits<float>::infinity()})
    {
        auto params = defaults;
        params._imagedCenterNeighbourSize = width;
        CHECK_THROWS_AS(cctag::FrameParam{params}, std::invalid_argument);
    }

    cctag::TagPipe pipe(defaults);
    auto invalid = defaults;
    invalid._imagedCenterNGridSample = 4;
    CHECK_THROWS_AS(pipe.configure(32, 32, invalid, nullptr), std::invalid_argument);
    CHECK_THROWS_AS(pipe.checkTagAllocations(1, invalid), std::invalid_argument);
    CHECK_THROWS_AS(pipe.uploadCuts(0, nullptr, invalid), std::invalid_argument);
    const cctag::Point2d<Eigen::Vector3f> center(16.f, 16.f);
    const cctag::numerical::geometry::Ellipse ellipse(center, 8.f, 8.f, 0.f);
    CHECK_THROWS_AS(pipe.imageCenterOptLoop(0, 1, ellipse, center, 2, invalid, nullptr),
                    std::invalid_argument);
}

TEST_CASE("cut_upload_rejects_metadata_that_disagrees_with_the_configuration")
{
    cctag::Parameters params;
    params._numCutsInIdentStep = 2;
    params._sampleCutLength = 2;
    cctag::TagPipe pipe(params);
    const cctag::ImageCut cut({}, {}, std::size_t(2));
    std::vector<cctag::ImageCut> cuts(3, cut);
    CHECK_THROWS_AS(pipe.uploadCuts(1, &cuts, params), std::invalid_argument);
    cuts.resize(2);
    for(const std::size_t length : {std::size_t(0), std::size_t(1), std::size_t(3), std::size_t(101)})
    {
        cuts.front().imgSignal().resize(length);
        CHECK_THROWS_AS(pipe.uploadCuts(1, &cuts, params), std::invalid_argument);
    }
}

TEST_CASE("center_refinement_rejects_invalid_geometry_before_device_work")
{
    cctag::Parameters params;
    cctag::TagPipe pipe(params);
    const cctag::Point2d<Eigen::Vector3f> center(16.f, 16.f);
    const cctag::numerical::geometry::Ellipse ellipse(center, 8.f, 8.f, 0.f);
    CHECK_THROWS_AS(pipe.imageCenterOptLoop(0, 1, {}, center, 2, params, nullptr),
                    std::invalid_argument);
    auto invalidEllipse = ellipse;
    invalidEllipse.matrix()(0, 0) = std::numeric_limits<float>::quiet_NaN();
    CHECK_THROWS_AS(pipe.imageCenterOptLoop(0, 1, invalidEllipse, center, 2, params, nullptr),
                    std::invalid_argument);
    auto invalidCenter = center;
    invalidCenter.x() = std::numeric_limits<float>::infinity();
    CHECK_THROWS_AS(pipe.imageCenterOptLoop(0, 1, ellipse, invalidCenter, 2, params, nullptr),
                    std::invalid_argument);
    params._imagedCenterNeighbourSize = std::numeric_limits<float>::max();
    CHECK_THROWS_AS(pipe.imageCenterOptLoop(0, 1, ellipse, center, 2, params, nullptr),
                    std::invalid_argument);
}

TEST_CASE("empty_first_tag_keeps_later_metadata_and_measures_a_zero_width_center" * doctest::skip(!hasCudaDevice()))
{
    auto params = singleLayerParameters();
    params._numCutsInIdentStep = 3;
    params._sampleCutLength = 2;
    params._imagedCenterNeighbourSize = 0.f;
    cctag::TagPipe pipe(params);
    cv::Mat image(32, 32, CV_8U);
    for(int y = 0; y < image.rows; ++y)
        for(int x = 0; x < image.cols; ++x)
            image.at<unsigned char>(y, x) = 2 * x;
    pipe.initialize(image.cols, image.rows, nullptr);
    pipe.load(0, image.data);
    pipe.tagframe();
    pipe.checkTagAllocations(2, params);

    const cctag::Point2d<Eigen::Vector3f> center(16.f, 16.f);
    const cctag::numerical::geometry::Ellipse ellipse(center, 8.f, 8.f, 0.f);
    const cctag::ImageCut sameSignal(center, {16.f, 24.f, 0.f, 0.f}, 1.f, 1.f, std::size_t(2));
    std::array<std::vector<cctag::ImageCut>, 2> cuts;
    cuts[0].assign(3, sameSignal);
    cuts[1] = cuts[0];
    pipe.uploadCuts(2, cuts.data(), params);
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

    cuts[0].clear();
    cuts[1][0].stop().x() = 8.f;
    cuts[1][0].stop().y() = 16.f;
    cuts[1][2].stop().x() = 24.f;
    cuts[1][2].stop().y() = 16.f;
    pipe.uploadCuts(2, cuts.data(), params);
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

    auto* result = cctag::PinnedCounters::getPointPtr(pipe.getId(), __FILE__, __LINE__);
    for(const float width : {0.f, .001f})
    {
        params._imagedCenterNeighbourSize = width;
        pipe.imageCenterOptLoop(1, 2, ellipse, center, 3, params, result);
        REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
        CHECK(result->readable);
        CHECK_EQ(result->point.x, doctest::Approx(center.x()));
        CHECK_EQ(result->point.y, doctest::Approx(center.y()));
        CHECK_EQ(result->resSize, 3);
        CHECK_EQ(result->result, doctest::Approx(768.f));
        CHECK_EQ(result->residual, doctest::Approx(256.f));
        for(int row = 0; row < 3; ++row)
            for(int col = 0; col < 3; ++col)
                CHECK(std::isfinite(result->mHomography(row, col)));
    }
    for(const int count : {0, 1})
    {
        pipe.imageCenterOptLoop(1, 2, ellipse, center, count, params, result);
        CHECK_FALSE(result->readable);
    }
}

TEST_CASE("nearby_point_reduction_preserves_fractional_residuals" * doctest::skip(!hasCudaDevice()))
{
    cctag::NearbyPointGrid points{};
    for(int y = 0; y < 5; ++y)
        for(int x = 0; x < 5; ++x)
        {
            auto& point = points.getGrid(x, y);
            point.point = make_float2(x, y);
            point.readable = true;
            point.result = .75f;
            point.resSize = 1;
            point.residual = -1.f;
        }
    points.getGrid(2, 3).result = .25f;
    void* allocation = nullptr;
    REQUIRE(cudaMalloc(&allocation, sizeof(points)) == cudaSuccess);
    std::unique_ptr<void, decltype(&cudaFree)> storage(allocation, cudaFree);
    auto* devicePoints = static_cast<cctag::NearbyPointGrid*>(allocation);
    std::size_t gridSamples = 5;
    const std::array<decltype(&cctag::identification::idBestNearbyPoint31max), 2> kernels{{
        cctag::identification::idBestNearbyPoint31max,
        cctag::identification::idBestNearbyPoint32plus
    }};
    for(const auto kernel : kernels)
    {
        REQUIRE(cudaMemcpy(devicePoints, &points, sizeof(points), cudaMemcpyHostToDevice) == cudaSuccess);
        void* arguments[] = {&devicePoints, &gridSamples};
        REQUIRE(cudaLaunchKernel(reinterpret_cast<const void*>(kernel), dim3(1), dim3(32),
                                 arguments) == cudaSuccess);
        REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
        cctag::NearbyPointGrid result;
        REQUIRE(cudaMemcpy(&result, devicePoints, sizeof(result), cudaMemcpyDeviceToHost) == cudaSuccess);
        CHECK_EQ(result.getGrid(0, 0).point.x, 2.f);
        CHECK_EQ(result.getGrid(0, 0).point.y, 3.f);
        CHECK_EQ(result.getGrid(0, 0).residual, .25f);
    }
}

TEST_CASE("pipeline_owns_parameters_and_reconfigures_between_frames" * doctest::skip(!hasCudaDevice()))
{
    std::unique_ptr<cctag::TagPipe> pipe;
    {
        auto params = singleLayerParameters();
        pipe = std::make_unique<cctag::TagPipe>(params);
    }
    const auto image = ringImage(96);
    pipe->initialize(image.cols, image.rows, nullptr);
    const int id = pipe->getId();
    CHECK_GT(exportedPointCount(*pipe, image, 0), 0);

    auto params = singleLayerParameters();
    params._sampleCutLength = 101;
    CHECK_THROWS_AS(pipe->configure(image.cols, image.rows, params, nullptr), std::invalid_argument);
    CHECK_EQ(pipe->getId(), id);
    CHECK_GT(exportedPointCount(*pipe, image, 0), 0);
    params = singleLayerParameters();
    params._cannyThrLow = params._cannyThrHigh = 1000000.f;
    pipe->configure(image.cols, image.rows, params, nullptr);
    CHECK_EQ(exportedPointCount(*pipe, image, 1), 0);

    params = singleLayerParameters();
    pipe->configure(image.cols, image.rows, params, nullptr);
    CHECK_GT(exportedPointCount(*pipe, image, 2), 0);

    const auto largerImage = ringImage(128);
    params._numberOfMultiresLayers = 2;
    params._pinnedCounters += 20;
    params._pinnedNearbyPoints += 20;
    pipe->configure(largerImage.cols, largerImage.rows, params, nullptr);
    CHECK_EQ(pipe->getNumOctaves(), 2);
    CHECK_EQ(pipe->getId(), id);
    CHECK_GT(exportedPointCount(*pipe, largerImage, 3), 0);

    pipe->release();
    pipe->release();
    params = singleLayerParameters();
    pipe->configure(image.cols, image.rows, params, nullptr);
    CHECK_GT(exportedPointCount(*pipe, image, 4), 0);
}

TEST_CASE("parallel_pipelines_keep_conflicting_thresholds_isolated" * doctest::skip(!hasCudaDevice()))
{
    const auto low = singleLayerParameters();
    auto high = low;
    high._cannyThrLow = high._cannyThrHigh = 1000000.f;
    high._pinnedCounters += 10;
    high._pinnedNearbyPoints += 10;
    cctag::TagPipe highPipe(high), lowPipe(low);
    const auto highImage = ringImage(96);
    const auto lowImage = highImage.clone();
    highPipe.initialize(highImage.cols, highImage.rows, nullptr);
    lowPipe.initialize(lowImage.cols, lowImage.rows, nullptr);

    for(int frame = 0; frame < 6; ++frame)
    {
        auto highResult = std::async(std::launch::async, [&] {
            return exportedPointCount(highPipe, highImage, frame);
        });
        auto lowResult = std::async(std::launch::async, [&] {
            return exportedPointCount(lowPipe, lowImage, frame);
        });
        CHECK_EQ(highResult.get(), 0);
        CHECK_GT(lowResult.get(), 0);
    }
}

TEST_CASE("short_lived_pipelines_reuse_slots" * doctest::skip(!hasCudaDevice()))
{
    const auto params = singleLayerParameters();
    const auto image = ringImage(64);
    for(int iteration = 0; iteration < MAX_PIPES * 2 + 1; ++iteration)
    {
        cctag::TagPipe pipe(params);
        CHECK_LT(pipe.getId(), MAX_PIPES);
        pipe.initialize(image.cols, image.rows, nullptr);
        CHECK_GT(exportedPointCount(pipe, image, iteration), 0);
    }
}

TEST_CASE("active_pipeline_limit_is_rejected_before_allocation")
{
    const auto params = singleLayerParameters();
    std::vector<std::unique_ptr<cctag::TagPipe>> pipes;
    for(int index = 0; index < MAX_PIPES; ++index)
        pipes.push_back(std::make_unique<cctag::TagPipe>(params));
    CHECK_THROWS_AS(std::make_unique<cctag::TagPipe>(params), std::runtime_error);
}

} // TEST_SUITE
