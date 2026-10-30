/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#define BOOST_TEST_MODULE statistics
#include <boost/test/unit_test.hpp>
#include <cctag/utils/Statistics.hpp>
#include <cmath>

BOOST_AUTO_TEST_SUITE(test_statistics)

BOOST_AUTO_TEST_CASE(empty_and_single_sample)
{
    cctag::numerical::RunningMean mean;
    cctag::numerical::RunningMeanVariance<> incremental;
    cctag::numerical::RunningMeanVariance<false> summed;
    BOOST_CHECK(std::isnan(mean.mean()));
    BOOST_CHECK(std::isnan(summed.mean()));
    BOOST_CHECK_EQUAL(incremental.mean(), 0.f);
    BOOST_CHECK_EQUAL(incremental.variance(), 0.f);
    BOOST_CHECK_EQUAL(summed.variance(), 0.f);
    mean(42.f);
    incremental(42.f);
    summed(42.f);
    BOOST_CHECK_EQUAL(mean.mean(), 42.f);
    BOOST_CHECK_EQUAL(incremental.mean(), 42.f);
    BOOST_CHECK_EQUAL(summed.mean(), 42.f);
    BOOST_CHECK_EQUAL(incremental.variance(), 0.f);
    BOOST_CHECK_EQUAL(summed.variance(), 0.f);
}

BOOST_AUTO_TEST_CASE(population_variance)
{
    cctag::numerical::RunningMeanVariance<> incremental;
    cctag::numerical::RunningMeanVariance<false> summed;
    for (float value : {2.f, 4.f, 4.f, 4.f, 5.f, 5.f, 7.f, 9.f})
    {
        incremental(value);
        summed(value);
    }
    BOOST_CHECK_EQUAL(incremental.mean(), 5.f);
    BOOST_CHECK_EQUAL(summed.mean(), 5.f);
    BOOST_CHECK_CLOSE(incremental.variance(), 4.f, 0.0001f);
    BOOST_CHECK_CLOSE(summed.variance(), 4.f, 0.0001f);
}

BOOST_AUTO_TEST_CASE(mean_rounding_modes)
{
    cctag::numerical::RunningMean mean;
    cctag::numerical::RunningMeanVariance<> incremental;
    cctag::numerical::RunningMeanVariance<false> summed;
    for (float value : {-0x1.1ec148p+11f, -0x1.05999ap+8f, -0x1.23f0ccp+12f, 0x1.155828p+12f})
    {
        mean(value);
        incremental(value);
        summed(value);
    }
    BOOST_CHECK_EQUAL(mean.mean(), -0x1.5ca5c4p+9f);
    BOOST_CHECK_EQUAL(summed.mean(), mean.mean());
    // Incremental averaging rounds at each step, rather than only at the end.
    BOOST_CHECK_EQUAL(incremental.mean(), -0x1.5ca5c8p+9f);
}

BOOST_AUTO_TEST_SUITE_END()
