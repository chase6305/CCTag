/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#pragma once

#include <cstddef>

namespace cctag {
namespace numerical {

class RunningMean
{
public:
    void operator()(float sample)
    {
        _sum += sample;
        ++_count;
    }

    float mean() const { return _sum / _count; }

private:
    float _sum = 0.f;
    std::size_t _count = 0;
};

// Keep the operation order of the original signal statistics: the updated
// mean is used to compute the population variance. Regression reports use a
// sum-based mean instead, selected by IncrementalMean = false.
template<bool IncrementalMean = true>
class RunningMeanVariance
{
public:
    void operator()(float sample)
    {
        ++_count;
        if constexpr (IncrementalMean)
            _mean = (_mean * (_count - 1) + sample) / _count;
        else
        {
            _sum += sample;
            _mean = _sum / _count;
        }

        if (_count > 1)
        {
            const float difference = sample - _mean;
            _variance = (_variance * (_count - 1)) / _count
                      + (difference * difference) / (_count - 1);
        }
    }

    float mean() const
    {
        if constexpr (IncrementalMean)
            return _mean;
        else
            return _sum / _count;
    }
    float variance() const { return _variance; }

private:
    std::size_t _count = 0;
    float _mean = 0.f;
    float _sum = 0.f;
    float _variance = 0.f;
};

} // namespace numerical
} // namespace cctag
