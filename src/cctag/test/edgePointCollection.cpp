
#include <doctest/doctest.h>
#include <cctag/Types.hpp>
#include <cctag/Vote.hpp>
#include <cctag/Canny.hpp>
#include <opencv2/core.hpp>

#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <memory>
#include <limits>
#include <new>

namespace
{
thread_local bool poisonArrayMemory = false;
thread_local bool trackArrayMemory = false;
thread_local std::size_t arrayBytes = 0;
}

// Exercise initialization with reused, nonzero storage rather than fresh pages.
void* operator new[](std::size_t size)
{
    if(trackArrayMemory)
        arrayBytes += size;
    void* memory = std::malloc(size ? size : 1);
    if(!memory)
        throw std::bad_alloc();
    if(poisonArrayMemory)
        std::memset(memory, 0xff, std::min(size, std::size_t(256)));
    return memory;
}

void operator delete[](void* memory) noexcept { std::free(memory); }
void operator delete[](void* memory, std::size_t) noexcept { std::free(memory); }

TEST_SUITE("test_edgePointCollection") {

TEST_CASE("processed_flags_start_clear_in_reused_storage")
{
    struct PoisonGuard
    {
        PoisonGuard() { poisonArrayMemory = true; }
        ~PoisonGuard() { poisonArrayMemory = false; }
    };

    std::unique_ptr<cctag::EdgePointCollection> points;
    {
        PoisonGuard guard;
        points.reset(new cctag::EdgePointCollection(8, 8));
    }
    for(int y = 0; y < 8; ++y)
        for(int x = 0; x < 8; ++x)
            points->add_point(x, y, 1.f, 1.f);

    for(int i = 0; i < 64; ++i)
    {
        CHECK_UNARY(!points->test_processed_in((*points)(i)));
        CHECK_UNARY(!points->test_processed_aux((*points)(i)));
    }
}

TEST_CASE("processed_flags_are_independent_across_word_boundaries")
{
    cctag::EdgePointCollection points(8, 8);
    for(int y = 0; y < 8; ++y)
        for(int x = 0; x < 8; ++x)
            points.add_point(x, y, 1.f, 1.f);

    for(int i = 0; i < 64; ++i)
    {
        points.set_processed_in(points(i), i % 2 == 0);
        points.set_processed_aux(points(i), i % 3 == 0);
    }
    for(int i = 0; i < 64; ++i)
    {
        CHECK_EQ(points.test_processed_in(points(i)), i % 2 == 0);
        CHECK_EQ(points.test_processed_aux(points(i)), i % 3 == 0);
    }
    for(int i = 0; i < 64; ++i)
    {
        points.set_processed_in(points(i), false);
        points.set_processed_aux(points(i), false);
    }
    for(int i = 0; i < 64; ++i)
    {
        CHECK_UNARY(!points.test_processed_in(points(i)));
        CHECK_UNARY(!points.test_processed_aux(points(i)));
    }
}

TEST_CASE("adding_points_keeps_existing_processed_flags")
{
    cctag::EdgePointCollection points(97, 1);
    for(int i = 0; i < 97; ++i)
    {
        points.add_point(i, 0, 1.f, 1.f);
        CHECK_UNARY(!points.test_processed_in(points(i)));
        CHECK_UNARY(!points.test_processed_aux(points(i)));
        points.set_processed_in(points(i), i % 2 == 0);
        points.set_processed_aux(points(i), i % 3 == 0);
        for(int j = 0; j <= i; ++j)
        {
            CHECK_EQ(points.test_processed_in(points(j)), j % 2 == 0);
            CHECK_EQ(points.test_processed_aux(points(j)), j % 3 == 0);
        }
    }
}

TEST_CASE("edge_extraction_preserves_pixel_order_and_roi_strides")
{
    for(int width : {0, 1, 2, 7, 8, 9, 15, 16, 17, 31, 32, 33, 64})
        for(int height : {0, 1, 5})
        {
            cv::Mat edgeParent(height + 4, width + 8, CV_8UC1, cv::Scalar(0));
            cv::Mat dxParent(height + 4, width + 8, CV_16SC1, cv::Scalar(0));
            cv::Mat dyParent(height + 4, width + 8, CV_16SC1, cv::Scalar(0));
            auto edges = edgeParent(cv::Rect(3, 2, width, height));
            auto dx = dxParent(cv::Rect(3, 2, width, height));
            auto dy = dyParent(cv::Rect(3, 2, width, height));
            std::vector<std::array<int, 4>> expected;
            for(int y = 0; y < height; ++y)
                for(int x = 0; x < width; ++x)
                {
                    const bool edge = x == width - 1 || (x + 7 * y) % 19 == 0;
                    edges.at<uchar>(y, x) = edge ? 255 : (x + 3 * y) % 17 == 0 ? 128 : 0;
                    dx.at<short>(y, x) = 13 * x - 7 * y;
                    dy.at<short>(y, x) = 5 * y - 11 * x;
                    if(edge) expected.push_back({x, y, dx.at<short>(y, x), dy.at<short>(y, x)});
                }
            for(bool contiguous : {false, true})
            {
                cctag::EdgePointCollection points(width, height);
                cctag::edgesPointsFromCanny(points, contiguous ? edges.clone() : edges,
                                          contiguous ? dx.clone() : dx, contiguous ? dy.clone() : dy);
                REQUIRE_EQ(points.get_point_count(), expected.size());
                for(std::size_t i = 0; i < expected.size(); ++i)
                {
                    CHECK_EQ(points(i)->x(), expected[i][0]);
                    CHECK_EQ(points(i)->y(), expected[i][1]);
                    CHECK_EQ(points(i)->dX(), expected[i][2]);
                    CHECK_EQ(points(i)->dY(), expected[i][3]);
                }
            }
        }
}

TEST_CASE("small_images_avoid_maximum_camera_buffers")
{
    struct TrackGuard
    {
        TrackGuard() { arrayBytes = 0; trackArrayMemory = true; }
        ~TrackGuard() { trackArrayMemory = false; }
    };
    std::unique_ptr<cctag::EdgePointCollection> points;
    {
        TrackGuard guard;
        points.reset(new cctag::EdgePointCollection(8, 8));
    }
    CHECK_LT(arrayBytes, 64u * 1024u);
}

TEST_CASE("voter_lists_use_all_points_and_can_be_replaced")
{
    cctag::EdgePointCollection points(8, 8);
    std::vector<std::vector<int>> voters(64);
    for(int i = 0; i < 64; ++i)
    {
        points.add_point(i % 8, i / 8, 1.f, 1.f);
        voters[i] = {i, (i + 1) % 64};
    }
    points.create_voter_lists(voters);
    for(int i = 0; i < 64; ++i)
    {
        const auto list = points.voters(points(i));
        REQUIRE_EQ(points.voters_size(points(i)), 2);
        CHECK_EQ(list.first[0], i);
        CHECK_EQ(list.first[1], (i + 1) % 64);
    }
    points.create_voter_lists(std::vector<std::vector<int>>(64));
    for(int i = 0; i < 64; ++i)
        CHECK_EQ(points.voters_size(points(i)), 0);
}

TEST_CASE("empty_and_overflowing_image_sizes")
{
    cctag::EdgePointCollection empty(0, 0);
    CHECK_EQ(empty.get_point_count(), 0);
    CHECK_NOTHROW(empty.create_voter_lists({}));
    CHECK_THROWS_AS(cctag::EdgePointCollection(std::numeric_limits<std::size_t>::max(), 2), std::length_error);
}

TEST_CASE("flat_voter_lists_match_grouped_lists_and_keep_point_order")
{
    cv::RNG random(0xc57a6);
    for(int count : {0, 1, 2, 7, 31, 32, 33, 255, 256, 257, 4096})
    {
        cctag::EdgePointCollection points(count, 1);
        for(int i = 0; i < count; ++i)
            points.add_point(i, 0, 1.f, 1.f);

        for(int trial = 0; trial < 8; ++trial)
        {
            std::vector<int> targets(count, -1);
            std::vector<std::vector<int>> grouped(count);
            for(int i = 0; i < count; ++i)
            {
                // Exercise empty lists, one crowded list and sparse destinations.
                targets[i] = trial == 0 ? -1 : trial == 1 ? count - 1 : random.uniform(-1, count);
                if(targets[i] >= 0)
                    grouped[targets[i]].push_back(i);
            }
            points.create_voter_lists_from_targets(targets);
            CHECK_EQ(points.get_point_count(), count);
            for(int i = 0; i < count; ++i)
            {
                const auto list = points.voters(points(i));
                REQUIRE_EQ(points.voters_size(points(i)), grouped[i].size());
                CHECK_UNARY(std::equal(list.first, list.second, grouped[i].begin()));
            }
        }
    }
}

TEST_CASE("flat_voter_lists_reject_invalid_destinations_and_sizes")
{
    cctag::EdgePointCollection points(2, 1);
    points.add_point(0, 0, 1.f, 1.f);
    points.add_point(1, 0, 1.f, 1.f);
    CHECK_THROWS_AS(points.create_voter_lists_from_targets({0}), std::length_error);
    CHECK_THROWS_AS(points.create_voter_lists_from_targets({-2, 0}), std::out_of_range);
    CHECK_THROWS_AS(points.create_voter_lists_from_targets({0, 2}), std::out_of_range);
    CHECK_NOTHROW(points.create_voter_lists_from_targets({1, 1}));
    const auto list = points.voters(points(1));
    REQUIRE_EQ(points.voters_size(points(1)), 2);
    CHECK_EQ(list.first[0], 0);
    CHECK_EQ(list.first[1], 1);
}

TEST_CASE("voting_keeps_valid_field_lines_after_rejected_ones")
{
    for(unsigned crowns : {3u, 4u})
    {
        cv::Mat dx(16, 64, CV_16SC1, cv::Scalar(100));
        const cv::Mat dy(16, 64, CV_16SC1, cv::Scalar(0));
        cctag::EdgePointCollection points(64, 16);
        const int count = 2 * crowns;
        for(int row : {4, 8})
            for(int i = 0; i < count; ++i)
            {
                // The first row has an excessive gap; the second has equal gaps.
                const int x = 4 + 4 * i + (row == 4 && i >= 3 ? 8 : 0);
                const short gradient = i % 2 ? 100 : -100;
                dx.at<short>(row, x) = gradient;
                points.add_point(x, row, gradient, 0.f);
            }
        points.add_point(4, 12, 100.f, 0.f);
        cctag::Parameters params;
        params._nCrowns = crowns;
        params._distSearch = 16;
        params._ratioVoting = 1.5f;
        params._minVotesToSelectCandidate = 1;
        std::vector<cctag::EdgePoint*> seeds;
        cctag::vote(points, seeds, dx, dy, params);

        REQUIRE_EQ(seeds.size(), 2);
        CHECK_EQ(points(seeds[0]), 2 * count - 1);
        CHECK_EQ(points(seeds[1]), count);
        for(int i = 0; i < points.get_point_count(); ++i)
        {
            const bool winner = i == count || i == 2 * count - 1;
            CHECK_EQ(points.voters_size(points(i)), winner ? 1 : 0);
            CHECK_EQ(points(i)->_isMax, winner ? 1 : -1);
            CHECK_EQ(points(i)->_flowLength, winner ? 4.f * (count - 1) : 0.f);
            if(winner)
                CHECK_EQ(*points.voters(points(i)).first, i == count ? 2 * count - 1 : count);
        }
    }
}

} // TEST_SUITE
