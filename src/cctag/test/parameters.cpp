/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include <cctag/Params.hpp>
#include <cctag/cctag_config.hpp>
#include <doctest/doctest.h>

TEST_SUITE("test_parameters") {

TEST_CASE("cuda_switch_follows_build_configuration")
{
#ifdef CCTAG_WITH_CUDA
    constexpr bool cudaBuild = true;
#else
    constexpr bool cudaBuild = false;
#endif
    cctag::Parameters params;
    CHECK_EQ(cctag::kDefaultUseCuda, cudaBuild);
    CHECK_EQ(params._useCuda, cudaBuild);
    params.setUseCuda(false);
    CHECK_UNARY(!params._useCuda);
#ifdef CCTAG_WITH_CUDA
    params.setUseCuda(true);
    CHECK_UNARY(params._useCuda);
#endif
}

} // TEST_SUITE
