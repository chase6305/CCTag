/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include <doctest/doctest.h>

#include <cctag/Types.hpp>
#include <cctag/EllipseGrowing.hpp>

#include <atomic>
#include <limits>
#include <thread>

TEST_SUITE("test_processedFlags") {

TEST_CASE("adjacent_points_keep_concurrent_flag_updates")
{
    cctag::EdgePointCollection points(2, 1);
    points.add_point(0, 0, 1.f, 0.f);
    points.add_point(1, 0, 1.f, 0.f);
    constexpr int rounds = 2000;
    std::atomic<int> phase{0}, completed{0};
    const auto worker = [&](int index) {
        for(int round = 1; round <= rounds; ++round)
        {
            while(phase.load(std::memory_order_acquire) != round) {}
            points.set_processed_in(points(index), true);
            points.set_processed_aux(points(index), true);
            completed.fetch_add(1, std::memory_order_release);
        }
    };
    std::thread first(worker, 0), second(worker, 1);
    unsigned lostUpdates = 0;
    for(int round = 1; round <= rounds; ++round)
    {
        completed.store(0, std::memory_order_relaxed);
        for(int index : {0, 1})
        {
            points.set_processed_in(points(index), false);
            points.set_processed_aux(points(index), false);
        }
        phase.store(round, std::memory_order_release);
        // Check only after both writes complete, so a missing flag is a lost update.
        while(completed.load(std::memory_order_acquire) != 2) {}
        for(int index : {0, 1})
        {
            lostUpdates += !points.test_processed_in(points(index));
            lostUpdates += !points.test_processed_aux(points(index));
        }
    }
    first.join();
    second.join();
    CHECK_EQ(lostUpdates, 0);
}

TEST_CASE("null_points_are_rejected_by_flag_accessors")
{
    cctag::EdgePointCollection points(1, 1);
    CHECK_THROWS_AS(points.set_processed_in(nullptr, true), std::out_of_range);
    CHECK_THROWS_AS(points.set_processed_aux(nullptr, true), std::out_of_range);
    CHECK_THROWS_AS(points.test_processed_in(nullptr), std::out_of_range);
    CHECK_THROWS_AS(points.test_processed_aux(nullptr), std::out_of_range);
}

TEST_CASE("image_dimensions_fit_the_stored_edge_coordinates")
{
    const int maximumCoordinate = std::numeric_limits<cctag::EdgePoint::Scalar>::max();
    const std::size_t maximumDimension = static_cast<std::size_t>(maximumCoordinate) + 1;
    cctag::EdgePointCollection wide(maximumDimension, 1);
    wide.add_point(maximumCoordinate, 0, 1.f, 0.f);
    CHECK_EQ(wide(0)->x(), maximumCoordinate);
    CHECK_EQ(wide(0)->y(), 0);
    cctag::EdgePointCollection tall(1, maximumDimension);
    tall.add_point(0, maximumCoordinate, 1.f, 0.f);
    CHECK_EQ(tall(0)->x(), 0);
    CHECK_EQ(tall(0)->y(), maximumCoordinate);

    cctag::EdgePointCollection image8k(8192, 1);
    image8k.add_point(8191, 0, 1.f, 0.f);
    CHECK_EQ(image8k(0)->x(), 8191);
    CHECK_THROWS_AS(cctag::EdgePointCollection(maximumDimension + 1, 1), std::length_error);
    CHECK_THROWS_AS(cctag::EdgePointCollection(1, maximumDimension + 1), std::length_error);
    CHECK_THROWS_AS(cctag::EdgePointCollection(40001, 1), std::length_error);
}

TEST_CASE("edge_point_copy_and_assignment_preserve_their_existing_contracts")
{
    cctag::EdgePoint source(7, 8, 3.f, 4.f);
    const auto mask = (std::uint64_t{1} << 63) | 5;
    source._flowLength = 12.f;
    source._processed.store(mask, std::memory_order_relaxed);
    source._isMax = 8;
    source._nSegmentOut = 9;
    const cctag::EdgePoint copy(source);
    CHECK_EQ(copy.x(), 7);
    CHECK_EQ(copy.y(), 8);
    CHECK_EQ(copy.normGradient(), 5.f);
    CHECK_EQ(copy._flowLength, 0.f);
    CHECK_EQ(copy._processed.load(std::memory_order_relaxed), 0);
    CHECK_EQ(copy._isMax, -1);
    CHECK_EQ(copy._nSegmentOut, -1);

    cctag::EdgePoint assigned;
    assigned = source;
    CHECK_EQ(assigned.x(), 7);
    CHECK_EQ(assigned.y(), 8);
    CHECK_EQ(assigned.dX(), 3.f);
    CHECK_EQ(assigned.dY(), 4.f);
    CHECK_EQ(assigned.normGradient(), 5.f);
    CHECK_EQ(assigned._flowLength, 12.f);
    CHECK_EQ(assigned._processed.load(std::memory_order_relaxed), mask);
    CHECK_EQ(assigned._isMax, 8);
    CHECK_EQ(assigned._nSegmentOut, 9);
    assigned = assigned;
    CHECK_EQ(assigned._processed.load(std::memory_order_relaxed), mask);
}

TEST_CASE("run_masks_include_bit_63_and_reject_invalid_indices")
{
    using Point = cctag::Point2d<Eigen::Vector3f>;
    using Ellipse = cctag::numerical::geometry::Ellipse;
    cctag::EdgePointCollection points(3, 3);
    points.add_point(1, 1, 1.f, 0.f);
    Ellipse inner(Point(1.f, 1.f), .5f, .5f, 0.f);
    Ellipse outer(Point(1.f, 1.f), 2.f, 2.f, 0.f);
    std::vector<cctag::EdgePoint*> output;
    cctag::connectedPoint(output, 63, points, inner, outer, 1, 1);
    CHECK_EQ(points(0)->_processed.load(std::memory_order_relaxed), std::uint64_t{1} << 63);
    CHECK_THROWS_AS(cctag::connectedPoint(output, -1, points, inner, outer, 1, 1), std::out_of_range);
    CHECK_THROWS_AS(cctag::connectedPoint(output, 64, points, inner, outer, 1, 1), std::out_of_range);
    CHECK_THROWS_AS(cctag::ellipseHull(points, output, outer, 1.f, std::numeric_limits<std::size_t>::max()),
                    std::out_of_range);
    CHECK_THROWS_AS(cctag::ellipseGrowing2(points, {}, output, outer, 1.f, 64, true), std::out_of_range);
    CHECK_EQ(points(0)->_processed.load(std::memory_order_relaxed), std::uint64_t{1} << 63);
    CHECK_UNARY(output.empty());
}

TEST_CASE("connected_point_rejects_missing_and_out_of_bounds_seeds")
{
    using Point = cctag::Point2d<Eigen::Vector3f>;
    using Ellipse = cctag::numerical::geometry::Ellipse;
    cctag::EdgePointCollection points(3, 3);
    Ellipse inner(Point(1.f, 1.f), .5f, .5f, 0.f);
    Ellipse outer(Point(1.f, 1.f), 2.f, 2.f, 0.f);
    std::vector<cctag::EdgePoint*> output;
    CHECK_THROWS_AS(cctag::connectedPoint(output, 0, points, inner, outer, 1, 1), std::invalid_argument);
    CHECK_THROWS_AS(cctag::connectedPoint(output, 0, points, inner, outer, -1, 1), std::out_of_range);
    CHECK_THROWS_AS(cctag::connectedPoint(output, 0, points, inner, outer, 1, 3), std::out_of_range);
    CHECK_UNARY(output.empty());
}

TEST_CASE("different_runs_keep_each_others_visited_bits")
{
    using Point = cctag::Point2d<Eigen::Vector3f>;
    using Ellipse = cctag::numerical::geometry::Ellipse;
    cctag::EdgePointCollection points(3, 3);
    points.add_point(1, 1, 1.f, 0.f);
    Ellipse inner(Point(1.f, 1.f), .5f, .5f, 0.f);
    Ellipse outer(Point(1.f, 1.f), 2.f, 2.f, 0.f);
    constexpr int rounds = 1000;
    std::atomic<int> phase{0}, completed{0};
    const auto worker = [&](int runId) {
        std::vector<cctag::EdgePoint*> output;
        for(int round = 1; round <= rounds; ++round)
        {
            while(phase.load(std::memory_order_acquire) != round) {}
            cctag::connectedPoint(output, runId, points, inner, outer, 1, 1);
            completed.fetch_add(1, std::memory_order_release);
        }
    };
    std::thread first(worker, 0), second(worker, 63);
    unsigned lostUpdates = 0;
    const auto expected = (std::uint64_t{1} << 63) | 1;
    for(int round = 1; round <= rounds; ++round)
    {
        completed.store(0, std::memory_order_relaxed);
        points(0)->_processed.store(0, std::memory_order_relaxed);
        phase.store(round, std::memory_order_release);
        while(completed.load(std::memory_order_acquire) != 2) {}
        lostUpdates += points(0)->_processed.load(std::memory_order_relaxed) != expected;
    }
    first.join();
    second.join();
    CHECK_EQ(lostUpdates, 0);
}

} // TEST_SUITE
