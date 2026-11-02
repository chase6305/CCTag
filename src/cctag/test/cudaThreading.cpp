/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include <cctag/cuda/tag_threads.h>
#include <doctest/doctest.h>

#include <chrono>
#include <future>

TEST_SUITE("test_cudaThreading") {

TEST_CASE("each_worker_observes_a_frame_once")
{
    cctag::TagFrameGate ready;
    ready.post();
    CHECK_EQ(ready.wait(0), 1);
    CHECK_EQ(ready.wait(0), 1);

    std::promise<void> entered;
    auto next = std::async(std::launch::async, [&] {
        entered.set_value();
        return ready.wait(1);
    });
    entered.get_future().wait();
    const auto state = next.wait_for(std::chrono::milliseconds(30));
    ready.post();
    const auto generation = next.get();
    CHECK_EQ(state, std::future_status::timeout);
    CHECK_EQ(generation, 2);
}

} // TEST_SUITE
