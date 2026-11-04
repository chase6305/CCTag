/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include <doctest/doctest.h>
#include <cctag/Statistic.hpp>
#include <cctag/utils/Statistics.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <future>
#include <initializer_list>
#include <limits>
#include <stdexcept>
#include <vector>

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

TEST_CASE("upper_median_copies_values_while_median_ref_fully_sorts_them")
{
    std::vector<float> values{9.f, 1.f, 7.f, 3.f};
    const auto original = values;
    CHECK_EQ(cctag::numerical::median(values), 7.f);
    CHECK_EQ(values, original);
    CHECK_EQ(cctag::numerical::median(original), 7.f);
    CHECK_EQ(cctag::numerical::median(std::vector<float>{-1.f, -4.f, -2.f}), -2.f);
    CHECK_EQ(cctag::numerical::medianRef(values), 7.f);
    const std::vector<float> sorted{1.f, 3.f, 7.f, 9.f};
    CHECK_EQ(values, sorted);

    const std::array<float, 5> odd{9.f, 1.f, 7.f, 3.f, 5.f};
    CHECK_EQ(cctag::numerical::median(odd), 5.f);
    const std::array<float, 1> singleton{-7.f};
    CHECK_EQ(cctag::numerical::median(singleton), -7.f);
}

TEST_CASE("median_helpers_reject_empty_input")
{
    std::vector<float> values;
    std::array<float, 0> empty{};
    CHECK_THROWS_AS(cctag::numerical::median(values), std::invalid_argument);
    CHECK_THROWS_AS(cctag::numerical::medianRef(values), std::invalid_argument);
    CHECK_THROWS_AS(cctag::numerical::median(empty), std::invalid_argument);
    CHECK_THROWS_AS(cctag::numerical::medianRef(empty), std::invalid_argument);
}

TEST_CASE("five_index_sampling_checks_population_bounds")
{
    constexpr std::size_t maximumPopulation = static_cast<std::size_t>(std::numeric_limits<int>::max()) + 1;
    std::array<int, 5> indices{11, 22, 33, 44, 55};
    const auto original = indices;
    for(std::size_t population : {std::size_t(0), std::size_t(1), std::size_t(2), std::size_t(3),
                                 std::size_t(4), maximumPopulation + 1, std::numeric_limits<std::size_t>::max()})
    {
        CHECK_THROWS_AS(cctag::numerical::rand_5_k(indices, population), std::invalid_argument);
        CHECK_EQ(indices, original);
    }
    for(std::size_t population : {std::size_t(5), std::size_t(6), std::size_t(97), maximumPopulation})
    {
        cctag::numerical::rand_5_k(indices, population);
        for(int index : indices)
        {
            CHECK_GE(index, 0);
            CHECK_LT(static_cast<std::size_t>(index), population);
        }
        std::sort(indices.begin(), indices.end());
        CHECK_UNARY(std::adjacent_find(indices.begin(), indices.end()) == indices.end());
    }
}

TEST_CASE("invalid_sampling_requests_do_not_advance_the_thread_local_generator")
{
    const auto sequence = [](bool rejectInvalid) {
        constexpr std::size_t maximumPopulation = static_cast<std::size_t>(std::numeric_limits<int>::max()) + 1;
        const std::array<std::size_t, 4> populations{5, 97, maximumPopulation, 6};
        std::array<std::array<int, 5>, 4> results{};
        for(std::size_t sample = 0; sample < populations.size(); ++sample)
        {
            if(rejectInvalid)
            {
                for(std::size_t population : {std::size_t(0), std::size_t(4), maximumPopulation + 1})
                {
                    try
                    {
                        cctag::numerical::rand_5_k(results[sample], population);
                    }
                    catch(const std::invalid_argument&)
                    {
                        continue;
                    }
                    throw std::runtime_error("An invalid sampling request was accepted");
                }
            }
            cctag::numerical::rand_5_k(results[sample], populations[sample]);
        }
        return results;
    };
    auto reference = std::async(std::launch::async, sequence, false);
    auto withRejections = std::async(std::launch::async, sequence, true);
    CHECK_EQ(reference.get(), withRejections.get());
}

} // TEST_SUITE
