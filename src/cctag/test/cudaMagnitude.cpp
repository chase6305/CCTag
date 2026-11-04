/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include <cctag/cuda/ptrstep.h>
#include <cctag/cuda/tag.h>
#include <doctest/doctest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <memory>
#include <random>
#include <stdexcept>

namespace cctag {
// Host signature of the CUDA kernel's launch stub.
void compute_mag_l2(cv::cuda::PtrStepSz16s, cv::cuda::PtrStepSz16s, cv::cuda::PtrStepSz32u);
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

cv::Mat magnitudes(const cv::Mat& dx, const cv::Mat& dy)
{
    REQUIRE(dx.type() == CV_16SC1);
    REQUIRE(dy.type() == CV_16SC1);
    REQUIRE(dx.size() == dy.size());
    REQUIRE(dx.isContinuous());
    REQUIRE(dy.isContinuous());
    const auto gradientBytes = dx.total() * sizeof(std::int16_t);
    const auto magnitudeBytes = dx.total() * sizeof(std::uint32_t);
    auto deviceX = allocateDevice(gradientBytes);
    auto deviceY = allocateDevice(gradientBytes);
    auto deviceResult = allocateDevice(magnitudeBytes);
    REQUIRE(cudaMemcpy(deviceX.get(), dx.data, gradientBytes, cudaMemcpyHostToDevice) == cudaSuccess);
    REQUIRE(cudaMemcpy(deviceY.get(), dy.data, gradientBytes, cudaMemcpyHostToDevice) == cudaSuccess);

    cv::cuda::PtrStepSz16s x(dx.rows, dx.cols, static_cast<std::int16_t*>(deviceX.get()), dx.step);
    cv::cuda::PtrStepSz16s y(dy.rows, dy.cols, static_cast<std::int16_t*>(deviceY.get()), dy.step);
    cv::cuda::PtrStepSz32u result(dx.rows, dx.cols, static_cast<std::uint32_t*>(deviceResult.get()),
                                dx.cols * sizeof(std::uint32_t));
    void* arguments[] = {&x, &y, &result};
    REQUIRE(cudaLaunchKernel(reinterpret_cast<const void*>(cctag::compute_mag_l2),
                             dim3((dx.cols + 31) / 32, dx.rows), dim3(32), arguments) == cudaSuccess);
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
    cv::Mat output(dx.size(), CV_32SC1);
    REQUIRE(cudaMemcpy(output.data, deviceResult.get(), magnitudeBytes, cudaMemcpyDeviceToHost) == cudaSuccess);
    return output;
}

std::uint64_t squaredMagnitude(std::int16_t x, std::int16_t y)
{
    const std::int64_t wideX = x, wideY = y;
    return wideX * wideX + wideY * wideY;
}

std::uint32_t floatMagnitude(std::int16_t x, std::int16_t y)
{
    return static_cast<std::uint32_t>(std::sqrt(static_cast<float>(squaredMagnitude(x, y))));
}

} // namespace

TEST_SUITE("test_cudaMagnitude") {

TEST_CASE("l2_magnitude_keeps_wide_squares_and_unsigned_sum" * doctest::skip(!hasCudaDevice()))
{
    const std::array<std::array<std::int16_t, 2>, 11> cases{{
        {300, 400}, {200, 200}, {181, 181}, {182, 0}, {185, 0}, {1020, 1020},
        {-1020, 1020}, {-32768, -32768}, {32767, 32767}, {-32768, 32767}, {13860, 13860}
    }};
    cv::Mat dx(1, cases.size(), CV_16SC1), dy(dx.size(), CV_16SC1);
    for(std::size_t index = 0; index < cases.size(); ++index)
    {
        dx.ptr<std::int16_t>()[index] = cases[index][0];
        dy.ptr<std::int16_t>()[index] = cases[index][1];
    }
    const auto actual = magnitudes(dx, dy);
    for(std::size_t index = 0; index < cases.size(); ++index)
    {
        CAPTURE(cases[index][0]);
        CAPTURE(cases[index][1]);
        CHECK_EQ(actual.ptr<std::uint32_t>()[index], floatMagnitude(cases[index][0], cases[index][1]));
    }
    CHECK_EQ(actual.ptr<std::uint32_t>()[0], 500);
    CHECK_EQ(actual.ptr<std::uint32_t>()[1], 282);
    CHECK_EQ(actual.ptr<std::uint32_t>()[7], 46340);
    // Float rounding can cross an integer boundary before truncation.
    CHECK_EQ(actual.ptr<std::uint32_t>()[10], 19601);
    CHECK_EQ(static_cast<std::uint32_t>(std::sqrt(static_cast<long double>(squaredMagnitude(13860, 13860)))), 19600);
}

TEST_CASE("l2_magnitude_matches_the_oracle_across_the_sobel_range" * doctest::skip(!hasCudaDevice()))
{
    constexpr int limit = 1020;
    cv::Mat dx(limit * 2 + 1, limit * 2 + 1, CV_16SC1), dy(dx.size(), CV_16SC1);
    for(int row = 0; row < dx.rows; ++row)
        for(int col = 0; col < dx.cols; ++col)
        {
            dx.at<std::int16_t>(row, col) = col - limit;
            dy.at<std::int16_t>(row, col) = row - limit;
        }
    const auto actual = magnitudes(dx, dy);
    std::size_t mismatches = 0;
    for(int row = 0; row < dx.rows; ++row)
        for(int col = 0; col < dx.cols; ++col)
        {
            const auto square = squaredMagnitude(dx.at<std::int16_t>(row, col), dy.at<std::int16_t>(row, col));
            const auto expected = static_cast<std::uint32_t>(std::sqrt(static_cast<long double>(square)));
            mismatches += actual.ptr<std::uint32_t>(row)[col] != expected;
        }
    CHECK_EQ(mismatches, 0);
}

TEST_CASE("l2_magnitude_preserves_float_rounding_over_the_full_short_range" * doctest::skip(!hasCudaDevice()))
{
    cv::Mat dx(1, 100000, CV_16SC1), dy(dx.size(), CV_16SC1);
    std::mt19937 random(0x471ca9);
    for(int index = 0; index < dx.cols; ++index)
    {
        dx.at<std::int16_t>(0, index) = static_cast<int>(random() % 65536) - 32768;
        dy.at<std::int16_t>(0, index) = static_cast<int>(random() % 65536) - 32768;
    }
    const auto actual = magnitudes(dx, dy);
    std::size_t mismatches = 0;
    std::uint32_t maximumFloorDifference = 0;
    for(int index = 0; index < dx.cols; ++index)
    {
        const auto x = dx.at<std::int16_t>(0, index), y = dy.at<std::int16_t>(0, index);
        const auto value = actual.ptr<std::uint32_t>()[index];
        mismatches += value != floatMagnitude(x, y);
        const auto exactFloor = static_cast<std::uint32_t>(std::sqrt(static_cast<long double>(squaredMagnitude(x, y))));
        const auto floorDifference = value > exactFloor ? value - exactFloor : exactFloor - value;
        maximumFloorDifference = std::max(maximumFloorDifference, floorDifference);
    }
    CHECK_EQ(mismatches, 0);
    CHECK_LE(maximumFloorDifference, 1);
}

TEST_CASE("default_gpu_gradients_produce_consistent_magnitudes" * doctest::skip(!hasCudaDevice()))
{
    cctag::Parameters params;
    params._numberOfMultiresLayers = 1;
    params._numberOfProcessedMultiresLayers = 1;
    cv::Mat image(64, 64, CV_8U, cv::Scalar(0));
    image.colRange(32, 64).setTo(cv::Scalar(255));
    cctag::TagPipe pipe(params);
    pipe.initialize(image.cols, image.rows, nullptr);
    pipe.load(0, image.data);
    pipe.tagframe();
    const std::unique_ptr<cv::Mat> dx(pipe.getDx(0)), dy(pipe.getDy(0)), magnitude(pipe.getMag(0));
    std::size_t mismatches = 0;
    int maximumGradient = 0;
    for(int row = 0; row < image.rows; ++row)
        for(int col = 0; col < image.cols; ++col)
        {
            const auto x = dx->at<std::int16_t>(row, col), y = dy->at<std::int16_t>(row, col);
            maximumGradient = std::max(maximumGradient, std::max(std::abs(int(x)), std::abs(int(y))));
            mismatches += magnitude->ptr<std::uint32_t>(row)[col] != floatMagnitude(x, y);
        }
    CHECK_GT(maximumGradient, 181);
    CHECK_EQ(mismatches, 0);
}

} // TEST_SUITE
