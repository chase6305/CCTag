
#include <doctest/doctest.h>
#include <cctag/Types.hpp>

#include <algorithm>
#include <cstdlib>
#include <cstring>
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

} // TEST_SUITE
