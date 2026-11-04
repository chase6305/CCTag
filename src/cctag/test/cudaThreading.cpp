/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include <cctag/cuda/tag_threads.h>
#include <doctest/doctest.h>

#include <array>
#include <atomic>
#include <chrono>
#include <future>
#include <stdexcept>
#include <vector>

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

TEST_CASE("stopping_the_frame_gate_wakes_all_waiters_and_reset_starts_a_new_run")
{
    cctag::TagFrameGate ready;
    auto first = std::async(std::launch::async, [&] { return ready.wait(0); });
    auto second = std::async(std::launch::async, [&] { return ready.wait(0); });
    ready.stop();
    ready.post();
    CHECK_EQ(first.get(), 0);
    CHECK_EQ(second.get(), 0);
    CHECK_EQ(ready.wait(0), 0);
    ready.reset();
    ready.post();
    CHECK_EQ(ready.wait(0), 1);
}

TEST_CASE("idle_workers_stop_and_destruct_without_a_round")
{
    std::atomic<bool> invoked{false};
    {
        cctag::TagThreads workers;
        workers.init(3, [&](int) { invoked = true; });
        workers.stop();
        workers.stop();
        workers.init(2, [&](int) { invoked = true; });
    }
    CHECK_UNARY(!invoked);
}

TEST_CASE("every_round_processes_each_layer_exactly_once")
{
    std::array<int, 3> visits{};
    cctag::TagThreads workers;
    workers.init(visits.size(), [&](int layer) { ++visits[layer]; });
    for(int round = 0; round < 100; ++round)
        workers.oneRound();
    workers.stop();
    for(int count : visits)
        CHECK_EQ(count, 100);
}

TEST_CASE("concurrent_round_callers_are_serialized")
{
    std::array<int, 3> visits{};
    cctag::TagThreads workers;
    workers.init(visits.size(), [&](int layer) { ++visits[layer]; });
    std::vector<std::future<void>> producers;
    for(int producer = 0; producer < 4; ++producer)
        producers.push_back(std::async(std::launch::async, [&] {
            for(int round = 0; round < 10; ++round)
                workers.oneRound();
        }));
    for(auto& producer : producers)
        producer.get();
    workers.stop();
    for(int count : visits)
        CHECK_EQ(count, 40);
}

TEST_CASE("stop_waits_for_the_active_round_before_joining_workers")
{
    std::promise<void> entered;
    std::promise<void> complete;
    auto completion = complete.get_future().share();
    std::array<int, 2> visits{};
    cctag::TagThreads workers;
    workers.init(visits.size(), [&](int layer) {
        if(layer == 0)
        {
            entered.set_value();
            completion.wait();
        }
        ++visits[layer];
    });

    auto round = std::async(std::launch::async, [&] { workers.oneRound(); });
    entered.get_future().wait();
    std::promise<void> stopping;
    auto stopped = std::async(std::launch::async, [&] {
        stopping.set_value();
        workers.stop();
    });
    stopping.get_future().wait();
    const auto state = stopped.wait_for(std::chrono::milliseconds(30));
    complete.set_value();
    round.get();
    stopped.get();
    CHECK_EQ(state, std::future_status::timeout);
    CHECK_EQ(visits[0], 1);
    CHECK_EQ(visits[1], 1);
    CHECK_THROWS_AS(workers.oneRound(), std::logic_error);
}

TEST_CASE("layer_exceptions_complete_the_round_and_are_rethrown_to_the_caller")
{
    std::array<int, 3> visits{};
    bool fail = true;
    cctag::TagThreads workers;
    workers.init(visits.size(), [&](int layer) {
        ++visits[layer];
        if(layer == 1 && fail)
            throw std::runtime_error("layer failed");
    });
    CHECK_THROWS_WITH_AS(workers.oneRound(), "layer failed", std::runtime_error);
    for(int count : visits)
        CHECK_EQ(count, 1);
    fail = false;
    CHECK_NOTHROW(workers.oneRound());
    workers.stop();
    for(int count : visits)
        CHECK_EQ(count, 2);
}

TEST_CASE("initialization_replaces_workers_and_invalid_arguments_preserve_the_current_run")
{
    cctag::TagThreads workers;
    CHECK_THROWS_AS(workers.oneRound(), std::logic_error);
    int visits = 0;
    workers.init(1, [&](int) { ++visits; });
    CHECK_THROWS_AS(workers.init(0, [](int) {}), std::invalid_argument);
    CHECK_THROWS_AS(workers.init(1, {}), std::invalid_argument);
    workers.oneRound();
    CHECK_EQ(visits, 1);

    std::array<int, 2> replacementVisits{};
    workers.init(replacementVisits.size(), [&](int layer) { ++replacementVisits[layer]; });
    workers.oneRound();
    workers.stop();
    workers.init(1, [&](int) { ++visits; });
    workers.oneRound();
    workers.stop();
    CHECK_EQ(visits, 2);
    CHECK_EQ(replacementVisits[0], 1);
    CHECK_EQ(replacementVisits[1], 1);
}

} // TEST_SUITE
