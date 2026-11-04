/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include <cctag/ImageCut.hpp>
#include <doctest/doctest.h>

#include <utility>

TEST_SUITE("test_imageCut") {

TEST_CASE("default_cut_has_an_empty_signal_and_full_sampling_interval")
{
    const cctag::ImageCut cut;
    CHECK_UNARY(cut.imgSignal().empty());
    CHECK_UNARY(!cut.outOfBounds());
    CHECK_EQ(cut.beginSig(), 0.f);
    CHECK_EQ(cut.endSig(), 1.f);
}

TEST_CASE("moving_a_cut_transfers_signal_storage_and_copying_remains_independent")
{
    cctag::ImageCut source({10.f, 20.f}, {30.f, 40.f, 3.f, 4.f}, 0.25f, 0.75f, std::size_t(7));
    source.imgSignal()[0] = 42.f;
    source.setOutOfBounds(true);
    const float* signal = source.imgSignal().data();

    cctag::ImageCut moved(std::move(source));
    CHECK_EQ(moved.imgSignal().data(), signal);
    CHECK_EQ(moved.imgSignal()[0], 42.f);
    CHECK_EQ(moved.start().x(), 10.f);
    CHECK_EQ(moved.stop().dY(), 4.f);
    CHECK_EQ(moved.beginSig(), 0.25f);
    CHECK_EQ(moved.endSig(), 0.75f);
    CHECK_UNARY(moved.outOfBounds());

    cctag::ImageCut copied(moved);
    copied.imgSignal()[0] = 99.f;
    CHECK_NE(copied.imgSignal().data(), signal);
    CHECK_EQ(moved.imgSignal()[0], 42.f);

    cctag::ImageCut assigned;
    assigned = std::move(moved);
    CHECK_EQ(assigned.imgSignal().data(), signal);
    CHECK_EQ(assigned.imgSignal()[0], 42.f);
}

} // TEST_SUITE
