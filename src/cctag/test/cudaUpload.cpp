/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include <cctag/Detection.hpp>
#include <cctag/cuda/tag.h>
#include <doctest/doctest.h>
#include <opencv2/imgproc.hpp>
#include <tbb/global_control.h>

#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>

namespace {

bool hasCudaDevice()
{
    int count = 0;
    const auto status = cudaGetDeviceCount(&count);
    if(status != cudaSuccess)
        cudaGetLastError();
    return status == cudaSuccess && count > 0;
}

} // namespace

TEST_SUITE("test_cudaUpload") {

TEST_CASE("upload_respects_host_row_stride" * doctest::skip(!hasCudaDevice()))
{
    cctag::Parameters params;
    params._numberOfMultiresLayers = 1;
    params._numberOfProcessedMultiresLayers = 1;
    cv::Mat backing(128, 192, CV_8U);
    for(int y = 0; y < backing.rows; ++y)
        for(int x = 0; x < backing.cols; ++x)
            backing.at<unsigned char>(y, x) = (7 * x + 13 * y) % 256;
    const cv::Mat region = backing(cv::Rect(17, 19, 96, 80));
    REQUIRE_FALSE(region.isContinuous());

    cctag::TagPipe pipe(params);
    pipe.initialize(region.cols, region.rows, nullptr);
    pipe.load(0, region.data, region.step);
    pipe.tagframe();
    std::unique_ptr<cv::Mat> downloaded(pipe.getPlane(0));
    CHECK_EQ(cv::countNonZero(*downloaded != region), 0);

    const auto packed = region.clone();
    pipe.load(1, packed.data);
    pipe.tagframe();
    downloaded.reset(pipe.getPlane(0));
    CHECK_EQ(cv::countNonZero(*downloaded != region), 0);
    CHECK_THROWS_AS(pipe.load(2, region.data, region.cols - 1), std::invalid_argument);
    CHECK_THROWS_AS(pipe.load(2, region.data, std::numeric_limits<std::size_t>::max()),
                    std::invalid_argument);
}

TEST_CASE("public_detection_accepts_noncontinuous_regions" * doctest::skip(!hasCudaDevice()))
{
    tbb::global_control threads(tbb::global_control::max_allowed_parallelism, 1);
    const cctag::CCTagMarkersBank bank(3);
    cv::Mat backing(720, 1024, CV_8U, cv::Scalar(87));
    cv::Mat region = backing(cv::Rect(137, 113, 640, 480));
    region.setTo(cv::Scalar(255));
    const int markerId = 31;
    const int radius = 160;
    const cv::Point center(320, 240);
    cv::circle(region, center, radius, cv::Scalar(0), cv::FILLED);
    int color = 255;
    const auto& ratios = bank.getMarkers()[markerId];
    for(auto ratio = ratios.rbegin(); ratio != ratios.rend(); ++ratio)
    {
        cv::circle(region, center, cvRound(radius / *ratio), cv::Scalar(color), cv::FILLED);
        color = 255 - color;
    }

    cctag::Parameters params(3);
    params.setUseCuda(true);
    for(const cv::Mat image : {region, region.clone()})
    {
        cctag::CCTag::List tags;
        cctag::cctagDetection(tags, 0, 0, image, params, bank, false, nullptr);
        int correctMarkers = 0;
        for(const auto& tag : tags)
            if(tag.id() == markerId && tag.getStatus() == 1
               && std::hypot(tag.x() - center.x, tag.y() - center.y) < 1.)
                ++correctMarkers;
        CHECK_EQ(correctMarkers, 1);
    }
}

} // TEST_SUITE
