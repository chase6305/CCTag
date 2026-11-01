/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include <doctest/doctest.h>
#include <cctag/filter/thinning.hpp>
#include <opencv2/core.hpp>

TEST_SUITE("test_thinning") {

TEST_CASE("scratch_contents_do_not_change_edges")
{
    cv::RNG random(0xcc7a6);
    for(int iteration = 0; iteration < 100; ++iteration)
    {
        cv::Mat input(7, 9, CV_8UC1);
        random.fill(input, cv::RNG::UNIFORM, 0, 2);
        input *= 255;
        auto a = input.clone();
        auto b = input.clone();
        cv::Mat zero(input.size(), CV_8UC1, cv::Scalar(0));
        cv::Mat filled(input.size(), CV_8UC1, cv::Scalar(255));
        cctag::thin(a, zero);
        cctag::thin(b, filled);
        CHECK_EQ(cv::countNonZero(a != b), 0);
        CHECK_EQ(cv::countNonZero(a.row(0) != input.row(0)), 0);
        CHECK_EQ(cv::countNonZero(a.row(a.rows - 1) != input.row(input.rows - 1)), 0);
        CHECK_EQ(cv::countNonZero(a.col(0) != input.col(0)), 0);
        CHECK_EQ(cv::countNonZero(a.col(a.cols - 1) != input.col(input.cols - 1)), 0);
    }
}

TEST_CASE("reused_scratch_handles_small_images_and_rois")
{
    cv::Mat scratch(11, 13, CV_8UC1, cv::Scalar(123));
    for(const cv::Size size : {cv::Size(1, 1), cv::Size(2, 7), cv::Size(7, 2)})
    {
        cv::Mat input(size, CV_8UC1, cv::Scalar(255));
        cctag::thin(input, scratch);
        CHECK_EQ(cv::countNonZero(input != 255), 0);
    }
    cv::Mat parent(11, 13, CV_8UC1, cv::Scalar(255));
    cv::Mat roi = parent(cv::Rect(2, 2, 9, 7));
    cv::Mat contiguous = roi.clone();
    cv::Mat otherScratch;
    cctag::thin(roi, scratch);
    cctag::thin(contiguous, otherScratch);
    CHECK_EQ(cv::countNonZero(roi != contiguous), 0);
    CHECK_EQ(cv::countNonZero(parent.row(0) != 255), 0);
}

} // TEST_SUITE
