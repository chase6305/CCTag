#define BOOST_TEST_MODULE testEdgePointCollection
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>
#include <cctag/Types.hpp>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <new>

namespace
{
thread_local bool poisonArrayMemory = false;
}

// Exercise initialization with reused, nonzero storage rather than fresh pages.
void* operator new[](std::size_t size)
{
    void* memory = std::malloc(size ? size : 1);
    if(!memory)
        throw std::bad_alloc();
    if(poisonArrayMemory)
        std::memset(memory, 0xff, std::min(size, std::size_t(256)));
    return memory;
}

void operator delete[](void* memory) noexcept { std::free(memory); }
void operator delete[](void* memory, std::size_t) noexcept { std::free(memory); }

BOOST_AUTO_TEST_SUITE(test_edgePointCollection)

BOOST_AUTO_TEST_CASE(processed_flags_start_clear_in_reused_storage)
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
        BOOST_CHECK(!points->test_processed_in((*points)(i)));
        BOOST_CHECK(!points->test_processed_aux((*points)(i)));
    }
}

BOOST_AUTO_TEST_CASE(processed_flags_are_independent_across_word_boundaries)
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
        BOOST_CHECK_EQUAL(points.test_processed_in(points(i)), i % 2 == 0);
        BOOST_CHECK_EQUAL(points.test_processed_aux(points(i)), i % 3 == 0);
    }
    for(int i = 0; i < 64; ++i)
    {
        points.set_processed_in(points(i), false);
        points.set_processed_aux(points(i), false);
    }
    for(int i = 0; i < 64; ++i)
    {
        BOOST_CHECK(!points.test_processed_in(points(i)));
        BOOST_CHECK(!points.test_processed_aux(points(i)));
    }
}

BOOST_AUTO_TEST_SUITE_END()
