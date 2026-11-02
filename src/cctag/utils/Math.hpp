/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#pragma once

#include <cmath>
#include <stdexcept>
#include <type_traits>

namespace cctag {
namespace numerical {

template<typename T>
constexpr T pi()
{
    return static_cast<T>(3.141592653589793238462643383279502884L);
}

template<typename T>
constexpr typename std::conditional<std::is_integral<T>::value, double, T>::type square(T value)
{
    // Promote integers before multiplying to avoid overflowing pixel differences.
    using Result = typename std::conditional<std::is_integral<T>::value, double, T>::type;
    return static_cast<Result>(value) * static_cast<Result>(value);
}

template<typename T>
inline int sign(T value)
{
    return value == 0 ? 0 : (std::signbit(value) ? -1 : 1);
}

template<typename T>
inline auto round(T value) -> decltype(std::round(value))
{
    if(!std::isfinite(value))
        throw std::runtime_error("Cannot round a non-finite value");
    return std::round(value);
}

} // namespace numerical
} // namespace cctag
