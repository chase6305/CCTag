/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include <doctest/doctest.h>
#include <cctag/utils/Statistics.hpp>
#include <cmath>
#include <initializer_list>

TEST_SUITE("test_statistics") {

TEST_CASE("empty_and_single_sample")
{
    cctag::numerical::RunningMean mean;
    cctag::numerical::RunningMeanVariance<> incremental;
    cctag::numerical::RunningMeanVariance<false> summed;
    CHECK_UNARY(std::isnan(mean.mean()));
    CHECK_UNARY(std::isnan(summed.mean()));
    CHECK_EQ(incremental.mean(), 0.f);
    CHECK_EQ(incremental.variance(), 0.f);
    CHECK_EQ(summed.variance(), 0.f);
    mean(42.f);
    incremental(42.f);
    summed(42.f);
    CHECK_EQ(mean.mean(), 42.f);
    CHECK_EQ(incremental.mean(), 42.f);
    CHECK_EQ(summed.mean(), 42.f);
    CHECK_EQ(incremental.variance(), 0.f);
    CHECK_EQ(summed.variance(), 0.f);
}

TEST_CASE("population_variance")
{
    cctag::numerical::RunningMeanVariance<> incremental;
    cctag::numerical::RunningMeanVariance<false> summed;
    for (float value : {2.f, 4.f, 4.f, 4.f, 5.f, 5.f, 7.f, 9.f})
    {
        incremental(value);
        summed(value);
    }
    CHECK_EQ(incremental.mean(), 5.f);
    CHECK_EQ(summed.mean(), 5.f);
    CHECK_UNARY(std::abs(incremental.variance() - 4.f) <= 4e-6f);
    CHECK_UNARY(std::abs(summed.variance() - 4.f) <= 4e-6f);
}

TEST_CASE("mean_rounding_modes")
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
    CHECK_EQ(mean.mean(), -0x1.5ca5c4p+9f);
    CHECK_EQ(summed.mean(), mean.mean());
    // Incremental averaging rounds at each step, rather than only at the end.
    CHECK_EQ(incremental.mean(), -0x1.5ca5c8p+9f);
}

} // TEST_SUITE
