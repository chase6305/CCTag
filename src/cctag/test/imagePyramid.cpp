/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include <cctag/ImagePyramid.hpp>
#include <cctag/Params.hpp>
#include <doctest/doctest.h>

#include <new>
#include <type_traits>

namespace {

class FailingImageAllocator : public cv::MatAllocator
{
public:
    explicit FailingImageAllocator(int failureIndex)
      : _upstream(cv::Mat::getDefaultAllocator()), _failureIndex(failureIndex)
    {
        cv::Mat::setDefaultAllocator(this);
    }

    ~FailingImageAllocator() override { cv::Mat::setDefaultAllocator(_upstream); }

    cv::UMatData* allocate(int dimensions, const int* sizes, int type, void* pixels,
                           std::size_t* steps, cv::AccessFlag access, cv::UMatUsageFlags usage) const override
    {
        if(++attempts == _failureIndex)
            throw std::bad_alloc();
        cv::UMatData* data = _upstream->allocate(dimensions, sizes, type, pixels, steps, access, usage);
        data->currAllocator = this;
        ++outstanding;
        return data;
    }

    bool allocate(cv::UMatData* data, cv::AccessFlag access, cv::UMatUsageFlags usage) const override
    {
        return _upstream->allocate(data, access, usage);
    }

    void deallocate(cv::UMatData* data) const override
    {
        --outstanding;
        _upstream->deallocate(data);
    }

    mutable int attempts = 0;
    mutable int outstanding = 0;

private:
    cv::MatAllocator* _upstream;
    int _failureIndex;
};

static_assert(!std::is_copy_constructible_v<cctag::Level>);
static_assert(!std::is_copy_assignable_v<cctag::Level>);
static_assert(!std::is_copy_constructible_v<cctag::ImagePyramid>);
static_assert(!std::is_copy_assignable_v<cctag::ImagePyramid>);

} // namespace

TEST_SUITE("test_imagePyramid") {

TEST_CASE("empty_pyramid_build_is_a_no_op")
{
    cctag::ImagePyramid pyramid;
    cctag::Parameters params;
    params.setUseCuda(false);
    CHECK_NOTHROW(pyramid.build(cv::Mat(), params._cannyThrLow, params._cannyThrHigh, &params));
    CHECK_EQ(pyramid.getNbLevels(), 0);
}

TEST_CASE("level_constructor_releases_images_when_allocation_fails")
{
    FailingImageAllocator allocator(4);
    CHECK_THROWS_AS(cctag::Level(64, 48, 0), std::bad_alloc);
    CHECK_EQ(allocator.attempts, 4);
    CHECK_EQ(allocator.outstanding, 0);
}

TEST_CASE("pyramid_constructor_releases_earlier_levels_when_allocation_fails")
{
    FailingImageAllocator allocator(10);
    CHECK_THROWS_AS(cctag::ImagePyramid(64, 48, 3, false), std::bad_alloc);
    CHECK_EQ(allocator.attempts, 10);
    CHECK_EQ(allocator.outstanding, 0);
}

TEST_CASE("cuda_levels_do_not_allocate_cpu_images")
{
    FailingImageAllocator allocator(1);
    CHECK_NOTHROW(cctag::ImagePyramid(640, 480, 3, true));
    CHECK_EQ(allocator.attempts, 0);
    CHECK_EQ(allocator.outstanding, 0);
}

TEST_CASE("signed_image_conversion_scales_a_strided_roi")
{
    cv::Mat backing(4, 7, CV_16SC1, cv::Scalar(32000));
    cv::Mat input = backing(cv::Rect(2, 1, 3, 2));
    REQUIRE_UNARY(!input.isContinuous());
    const short values[2][3] = {{-100, -50, 0}, {50, 75, 100}};
    const unsigned char expected[2][3] = {{0, 64, 128}, {191, 223, 255}};
    for(int row = 0; row < input.rows; ++row)
        for(int column = 0; column < input.cols; ++column)
            input.at<short>(row, column) = values[row][column];

    cv::Mat output;
    cctag::sIntToUchar(input, output);
    REQUIRE_EQ(output.type(), CV_8UC1);
    REQUIRE_EQ(output.size(), input.size());
    for(int row = 0; row < input.rows; ++row)
        for(int column = 0; column < input.cols; ++column)
        {
            CHECK_EQ(output.at<unsigned char>(row, column), expected[row][column]);
            CHECK_EQ(input.at<short>(row, column), values[row][column]);
        }
    CHECK_EQ(backing.at<short>(0, 0), 32000);
}

TEST_CASE("signed_image_conversion_maps_constant_images_to_zero")
{
    for(short value : {-100, 0, 100})
    {
        const cv::Mat input(2, 3, CV_16SC1, cv::Scalar(value));
        cv::Mat output;
        cctag::sIntToUchar(input, output);
        CHECK_EQ(output.type(), CV_8UC1);
        CHECK_EQ(output.size(), input.size());
        CHECK_EQ(cv::countNonZero(output), 0);
    }
}

} // TEST_SUITE
