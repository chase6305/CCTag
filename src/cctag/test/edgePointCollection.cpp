
#include <doctest/doctest.h>
#include <cctag/Types.hpp>
#include <cctag/Vote.hpp>
#include <cctag/Canny.hpp>
#include <cctag/Bresenham.hpp>
#include <cctag/ImagePyramid.hpp>
#include <cctag/Multiresolution.hpp>
#include <cctag/Detection.hpp>
#include <cctag/utils/VisualDebug.hpp>
#include <opencv2/core.hpp>
#include <tbb/global_control.h>

#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <memory>
#include <limits>
#include <new>
#include <stdexcept>

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

TEST_CASE("candidate_ordering_handles_large_scores_and_ties")
{
    tbb::global_control threads(tbb::global_control::max_allowed_parallelism, 1);
    cctag::EdgePointCollection points(400, 200);
    points.add_point(2, 2, 100.f, 0.f);
    points.add_point(4, 2, 100.f, 0.f);
    const int offsets[][2] = {
        {40, 0}, {35, 21}, {19, 36}, {0, 39}, {-21, 34}, {-35, 19},
        {-41, 0}, {-34, -21}, {-20, -35}, {0, -41}, {21, -34}, {36, -19}};
    for(int center : {100, 300})
        for(const auto& point : offsets)
            points.add_point(center + point[0], 100 + point[1], point[0], point[1]);

    int firstVoteCount = 50021;
    int secondVoteCount = 40009;
    int selectedGroup = 0;
    SUBCASE("squaring more than 46340 votes does not overflow") {}
    SUBCASE("later seed wins equal scores")
    {
        secondVoteCount = firstVoteCount;
        selectedGroup = 1;
    }
    SUBCASE("seed without voters cannot become a candidate")
    {
        firstVoteCount = 0;
        selectedGroup = 1;
    }

    std::vector<std::vector<int>> voters(points.get_point_count());
    for(int i = 0; i < firstVoteCount; ++i)
        voters[0].push_back(2 + i % 12);
    for(int i = 0; i < secondVoteCount; ++i)
        voters[1].push_back(14 + i % 12);
    points.create_voter_lists(voters);

    cctag::Parameters params;
    params._averageVoteMin = std::numeric_limits<float>::max();
    params._maximumNbCandidatesLoopTwo = 1;
    const cv::Mat image(200, 400, CV_8UC1, cv::Scalar(0));
    cctag::CCTag::List tags;
    cctag::cctagDetectionFromEdges(tags, points, image, {points(0), points(1)},
                                   0, 0, 1.f, params, nullptr);

    // Outlier removal may discard individual points from the selected component.
    const int selectedBegin = 2 + selectedGroup * 12;
    const auto labeledCount = std::count_if(points(selectedBegin), points(selectedBegin + 12),
        [](const cctag::EdgePoint& point) { return point._nSegmentOut == 0; });
    CHECK_GE(labeledCount, 5);
    const int unselectedBegin = 2 + (1 - selectedGroup) * 12;
    for(int i = unselectedBegin; i < unselectedBegin + 12; ++i)
        CHECK_EQ(points(i)->_nSegmentOut, -1);
    cctag::CCTagVisualDebug::instance().clearSessions();
}

TEST_CASE("candidate_budget_is_limited_to_the_64_bit_processing_mask")
{
    const cv::Mat image(64, 64, CV_8UC1, cv::Scalar(0));
    cctag::Parameters params;
    params.setUseCuda(false);
    const cctag::CCTagMarkersBank bank(3);
    cctag::EdgePointCollection points(image.cols, image.rows);
    cctag::CCTag::List tags;

    for(const std::size_t budget : {std::size_t(0), std::size_t(64)})
    {
        params._maximumNbCandidatesLoopTwo = budget;
        CHECK_NOTHROW(cctag::cctagDetectionFromEdges(
            tags, points, image, {}, 0, 0, 1.f, params, nullptr));
    }
    for(const std::size_t budget : {std::size_t(65), std::numeric_limits<std::size_t>::max()})
    {
        params._maximumNbCandidatesLoopTwo = budget;
        CHECK_THROWS_AS(cctag::cctagDetectionFromEdges(
            tags, points, image, {}, 0, 0, 1.f, params, nullptr), std::invalid_argument);
        CHECK_THROWS_AS(cctag::cctagDetection(
            tags, 0, 0, image, params, bank, false, nullptr), std::invalid_argument);
    }
}

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

TEST_CASE("empty_padding_preserves_adjacent_border_search")
{
    cv::Mat dx(4, 4, CV_16SC1, cv::Scalar(4));
    cv::Mat dy(4, 4, CV_16SC1, cv::Scalar(3));
    cctag::EdgePointCollection original(8, 8), padded(5, 5), unpadded(4, 4);
    for(auto* points : {&original, &padded, &unpadded})
    {
        points->add_point(0, 0, 4.f, 3.f);
        points->add_point(3, 3, -4.f, -3.f);
    }
    // At (4, 3), the search checks its adjacent pixel (3, 3).
    CHECK_EQ(cctag::gradientDirectionDescent(original, *original(0), 1, 30, dx, dy, 2500), original(1));
    CHECK_EQ(cctag::gradientDirectionDescent(padded, *padded(0), 1, 30, dx, dy, 2500), padded(1));
    CHECK_EQ(cctag::gradientDirectionDescent(unpadded, *unpadded(0), 1, 30, dx, dy, 2500), nullptr);
}

#ifndef CCTAG_SERIALIZE
TEST_CASE("pyramid_edge_storage_scales_with_each_level")
{
    cv::Mat input(48, 64, CV_8UC1, cv::Scalar(255));
    cctag::Parameters params;
    params.setUseCuda(false);
    cctag::ImagePyramid pyramid(64, 48, 4, false);
    pyramid.build(input, params._cannyThrLow, params._cannyThrHigh, &params);
    cctag::CCTag::List tags;
    arrayBytes = 0;
    struct TrackGuard
    {
        TrackGuard() { trackArrayMemory = true; }
        ~TrackGuard() { trackArrayMemory = false; }
    };
    {
        TrackGuard guard;
        cctag::cctagMultiresDetection(tags, input, pyramid, 0, nullptr, params, nullptr);
    }
    CHECK_UNARY(tags.empty());
    // Four full-size edge collections request over 700 KiB. The complete
    // pyramid, including its border padding, fits within this smaller budget.
    CHECK_LT(arrayBytes, std::size_t(384 * 1024));
}
#endif

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
