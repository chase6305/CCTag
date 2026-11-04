/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include <cctag/Identification.hpp>
#include <cctag/cuda/frameparam.h>
#include <cctag/cuda/nearby_point.h>
#include <cctag/cuda/tag_cut.h>
#include <doctest/doctest.h>

#include <array>
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>

namespace cctag::identification {
// Host signature of the CUDA kernel's launch stub.
void idGetSignals(cv::cuda::PtrStepSzb, int, const NearbyPointGrid*, const CutStructGrid*,
                  CutSignalGrid*, FrameParam);
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

using DeviceMemory = std::unique_ptr<void, decltype(&cudaFree)>;

DeviceMemory allocateDevice(std::size_t bytes)
{
    void* pointer = nullptr;
    const auto status = cudaMalloc(&pointer, bytes);
    if(status != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(status));
    return DeviceMemory(pointer, cudaFree);
}

class SignalSampler
{
    DeviceMemory image = allocateDevice(32 * 32);
    DeviceMemory points = allocateDevice(sizeof(cctag::NearbyPointGrid));
    DeviceMemory cuts = allocateDevice(sizeof(cctag::CutStructGrid));
    DeviceMemory signals = allocateDevice(sizeof(cctag::CutSignalGrid));

public:
    cv::Mat pixels = cv::Mat(32, 32, CV_8U);

    SignalSampler()
    {
        for(int y = 0; y < pixels.rows; ++y)
            for(int x = 0; x < pixels.cols; ++x)
                pixels.at<unsigned char>(y, x) = x + 3 * y;
        REQUIRE(cudaMemcpy(image.get(), pixels.data, pixels.total(), cudaMemcpyHostToDevice) == cudaSuccess);
    }

    cctag::identification::CutSignals sample(float x, float y, int sampleCount,
                                            float stopX = 0.f, float stopY = 0.f)
    {
        cctag::NearbyPointGrid hostPoints{};
        auto& point = hostPoints.getGrid(0, 0);
        point.mHomography = cctag::geometry::matrix3x3(1, 0, x, 0, 1, y, 0, 0, 1);
        point.mInvHomography = cctag::geometry::matrix3x3(1, 0, 0, 0, 1, 0, 0, 0, 1);
        cctag::CutStructGrid hostCuts{};
        auto& cut = hostCuts.getGrid(0);
        cut.stop = make_float2(stopX, stopY);
        cut.beginSig = 0.f;
        cut.endSig = 1.f;
        cut.sigSize = sampleCount;
        REQUIRE(cudaMemcpy(points.get(), &hostPoints, sizeof(hostPoints), cudaMemcpyHostToDevice) == cudaSuccess);
        REQUIRE(cudaMemcpy(cuts.get(), &hostCuts, sizeof(hostCuts), cudaMemcpyHostToDevice) == cudaSuccess);

        cv::cuda::PtrStepSzb source(32, 32, static_cast<unsigned char*>(image.get()), 32);
        auto* devicePoints = static_cast<cctag::NearbyPointGrid*>(points.get());
        auto* deviceCuts = static_cast<cctag::CutStructGrid*>(cuts.get());
        auto* deviceSignals = static_cast<cctag::CutSignalGrid*>(signals.get());
        int cutCount = 1;
        cctag::FrameParam params(cctag::Parameters{});
        params.gridNSample = 1;
        void* arguments[] = {&source, &cutCount, &devicePoints,
                             &deviceCuts, &deviceSignals, &params};
        REQUIRE(cudaLaunchKernel(reinterpret_cast<const void*>(cctag::identification::idGetSignals),
                                 dim3(1), dim3(32), arguments) == cudaSuccess);
        REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
        cctag::CutSignalGrid result;
        REQUIRE(cudaMemcpy(&result, signals.get(), sizeof(result), cudaMemcpyDeviceToHost) == cudaSuccess);
        return result.getGrid(0, 0, 0);
    }
};

} // namespace

TEST_SUITE("test_cudaSampling") {

TEST_CASE("homography_sampling_requires_a_complete_bilinear_neighborhood" * doctest::skip(!hasCudaDevice()))
{
    SignalSampler sampler;
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float infinity = std::numeric_limits<float>::infinity();
    const std::array<std::array<float, 2>, 12> positions{{
        {0.f, 0.f}, {.25f, .75f}, {30.75f, 30.75f}, {-.25f, 1.f}, {1.f, -.25f},
        {31.f, 1.f}, {1.f, 31.f}, {1.f, 31.25f}, {1.f, 32.f},
        {nan, 1.f}, {infinity, 1.f}, {1.f, -infinity}
    }};
    for(const auto& position : positions)
    {
        CAPTURE(position[0]);
        CAPTURE(position[1]);
        const auto result = sampler.sample(position[0], position[1], 32);
        cctag::ImageCut cpuCut({}, {}, std::size_t(32));
        cpuCut.stop().x() = cpuCut.stop().y() = 0.f;
        Eigen::Matrix3f homography = Eigen::Matrix3f::Identity();
        homography(0, 2) = position[0];
        homography(1, 2) = position[1];
        cctag::identification::extractSignalUsingHomography(cpuCut, sampler.pixels, homography,
                                                           Eigen::Matrix3f::Identity());
        CHECK_EQ(bool(result.outOfBounds), cpuCut.outOfBounds());
        if(!cpuCut.outOfBounds())
            CHECK_EQ(result.sig[0], doctest::Approx((position[0] + 3.f * position[1]) / 2.f));
    }
}

TEST_CASE("partial_warps_report_invalid_samples_and_clear_previous_bounds" * doctest::skip(!hasCudaDevice()))
{
    SignalSampler sampler;
    CHECK_EQ(sampler.sample(1.f, 1.f, 0).outOfBounds, 1);
    CHECK_EQ(sampler.sample(1.f, 1.f, 1).outOfBounds, 1);
    for(int count : {2, 31, 32, 33, 65, 67, 100})
    {
        CAPTURE(count);
        const auto outside = sampler.sample(0.f, 1.f, count, 31.2f);
        CHECK_EQ(outside.outOfBounds, 1);
        const auto inside = sampler.sample(.25f, .75f, count, 20.f, 10.f);
        CHECK_EQ(inside.outOfBounds, 0);
        CHECK_EQ(inside.sig[count - 1], doctest::Approx((20.25f + 3.f * 10.75f) / 2.f));
    }
}

} // TEST_SUITE
