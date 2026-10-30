/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#define BOOST_TEST_MODULE testThinning
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>
#include <cctag/filter/thinning.hpp>
#include <opencv2/core.hpp>

BOOST_AUTO_TEST_SUITE(test_thinning)

BOOST_AUTO_TEST_CASE(scratch_contents_do_not_change_edges)
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
        BOOST_CHECK_EQUAL(cv::countNonZero(a != b), 0);
        BOOST_CHECK_EQUAL(cv::countNonZero(a.row(0) != input.row(0)), 0);
        BOOST_CHECK_EQUAL(cv::countNonZero(a.row(a.rows - 1) != input.row(input.rows - 1)), 0);
        BOOST_CHECK_EQUAL(cv::countNonZero(a.col(0) != input.col(0)), 0);
        BOOST_CHECK_EQUAL(cv::countNonZero(a.col(a.cols - 1) != input.col(input.cols - 1)), 0);
    }
}

BOOST_AUTO_TEST_CASE(reused_scratch_handles_small_images_and_rois)
{
    cv::Mat scratch(11, 13, CV_8UC1, cv::Scalar(123));
    for(const cv::Size size : {cv::Size(1, 1), cv::Size(2, 7), cv::Size(7, 2)})
    {
        cv::Mat input(size, CV_8UC1, cv::Scalar(255));
        cctag::thin(input, scratch);
        BOOST_CHECK_EQUAL(cv::countNonZero(input != 255), 0);
    }
    cv::Mat parent(11, 13, CV_8UC1, cv::Scalar(255));
    cv::Mat roi = parent(cv::Rect(2, 2, 9, 7));
    cv::Mat contiguous = roi.clone();
    cv::Mat otherScratch;
    cctag::thin(roi, scratch);
    cctag::thin(contiguous, otherScratch);
    BOOST_CHECK_EQUAL(cv::countNonZero(roi != contiguous), 0);
    BOOST_CHECK_EQUAL(cv::countNonZero(parent.row(0) != 255), 0);
}

BOOST_AUTO_TEST_SUITE_END()
