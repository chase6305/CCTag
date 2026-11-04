/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include <doctest/doctest.h>

#ifdef CCTAG_WITH_CUDA
#include <cctag/cuda/debug_macros.hpp>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <type_traits>

namespace {

void allocatePoisonedHost(void** pointer, std::size_t bytes)
{
    const auto error = cudaMallocHost(pointer, bytes);
    POP_CUDA_FATAL_TEST(error, "edgeList test host allocation failed");
    std::memset(*pointer, 0xa5, bytes);
}

struct Sample
{
    std::uint64_t value;
};

bool hasCudaDevice()
{
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

} // namespace

// Poison fresh allocations so the test does not depend on CUDA allocator reuse.
#undef POP_CUDA_MALLOC_HOST
#define POP_CUDA_MALLOC_HOST(pointer, bytes) allocatePoisonedHost(pointer, bytes)
#include <cctag/cuda/edge_list.h>

static_assert(!std::is_copy_constructible<cctag::HostEdgeList<Sample>>::value,
              "The pinned host allocation has one owner");
static_assert(!std::is_copy_constructible<cctag::EdgeList<Sample>>::value,
              "The list allocation has one owner");
static_assert(std::is_trivially_copyable<cctag::DevEdgeList<Sample>>::value,
              "Device views must remain safe to pass to kernels by value");

TEST_SUITE("test_edgeList") {

TEST_CASE("allocations_are_initialized_once_and_released_by_their_owner" * doctest::skip(!hasCudaDevice()))
{
    cctag::PinnedCounters::init(0);
    cctag::FrameMetaPtr metadata(0, 0);
    cctag::EdgeList<Sample> first(0, metadata, cctag::List_size_voters);
    cctag::EdgeList<Sample> second(0, metadata, cctag::List_size_inner_points);
    first.alloc(32, cctag::EdgeListBoth);
    second.alloc(32, cctag::EdgeListBoth);
    first.init(0);
    second.init(0);
    const auto allZero = [](const auto& list, std::size_t count) {
        return std::all_of(list.host.ptr, list.host.ptr + count,
                           [](const Sample& sample) { return sample.value == 0; });
    };
    CHECK_UNARY(allZero(first, 32));
    CHECK_UNARY(allZero(second, 32));
    CHECK_EQ(first.host.size, 0);
    CHECK_EQ(second.host.size, 0);

    // A frame reset preserves host storage instead of clearing it again.
    first.host.ptr[0].value = 42;
    first.init(0);
    CHECK_EQ(first.host.ptr[0].value, 42);
    first.initHost();
    CHECK_UNARY(allZero(first, 32));

    first.release();
    first.alloc(8, cctag::EdgeListBoth);
    first.init(0);
    CHECK_UNARY(allZero(first, 8));
    CHECK_EQ(cudaDeviceSynchronize(), cudaSuccess);
    first.release();
    second.release();

    {
        cctag::EdgeList<Sample> automatic(0, metadata, cctag::List_size_voters);
        automatic.alloc(16, cctag::EdgeListBoth);
        automatic.init(0);
        CHECK_UNARY(automatic.dev.ptr != nullptr);
        CHECK_UNARY(automatic.host.ptr != nullptr);
    }
    {
        cctag::EdgeList<Sample> deviceOnly(0, metadata, cctag::List_size_voters);
        deviceOnly.alloc(16, cctag::EdgeListDevOnly);
        deviceOnly.init(0);
        CHECK_UNARY(deviceOnly.dev.ptr != nullptr);
        CHECK_EQ(deviceOnly.host.ptr, nullptr);
    }
    {
        cctag::EdgeList<Sample> unallocated(0, metadata, cctag::List_size_voters);
        CHECK_EQ(unallocated.dev.ptr, nullptr);
        CHECK_EQ(unallocated.host.ptr, nullptr);
    }
    cctag::PinnedCounters::release(0);
}

} // TEST_SUITE
#endif // CCTAG_WITH_CUDA
