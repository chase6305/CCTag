/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include "Statistic.hpp"
#include "utils/pcg_random.hpp"

#include <algorithm>
#include <limits>

namespace cctag {
namespace numerical {

void rand_5_k(std::array<int, 5>& indices, std::size_t populationSize)
{
    constexpr std::size_t maximumPopulation =
        static_cast<std::size_t>(std::numeric_limits<int>::max()) + 1;
    if(populationSize < indices.size() || populationSize > maximumPopulation)
        throw std::invalid_argument("Sampling five int indices requires a population in [5, INT_MAX+1]");

    // Preserve the deterministic per-thread sequence used by RANSAC.
    static thread_local pcg32 rng(271828);
    const auto bound = static_cast<pcg32::result_type>(populationSize);
    auto next = indices.begin();
    for(std::size_t sample = 0; sample < indices.size(); ++sample)
    {
        int index;
        do
        {
            index = static_cast<int>(rng(bound));
        }
        while(std::find(indices.begin(), next, index) != next);
        *next++ = index;
    }
}

float median(std::vector<float>& values)
{
    auto copy = values;
    return medianRef(copy);
}

} // namespace numerical
} // namespace cctag
