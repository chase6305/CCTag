/*
 * Copyright 2016, Simula Research Laboratory
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#ifndef _CCTAG_STATISTIC_HPP_
#define _CCTAG_STATISTIC_HPP_

#include <algorithm>
#include <array>
#include <cstddef>
#include <stdexcept>
#include <vector>

namespace cctag {
namespace numerical {

// Draw five distinct indices from [0, populationSize), requiring 5..INT_MAX+1.
void rand_5_k(std::array<int, 5>& indices, std::size_t populationSize);

// Fully sort the input and return its upper middle value; reject empty input.
template<class Values>
inline float medianRef(Values& values)
{
    if(values.size() == 0)
        throw std::invalid_argument("A median requires at least one value");
    std::sort(values.begin(), values.end());
    return values[values.size() / 2];
}

// Return the upper median of a copy, leaving the caller's values unchanged.
template<class Values>
inline float median(Values values)
{
    return medianRef(values);
}

float median(std::vector<float>& values);

} // namespace numerical
} // namespace cctag

#endif
