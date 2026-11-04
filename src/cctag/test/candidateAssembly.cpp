#include <doctest/doctest.h>

#include <cctag/EllipseGrowing.hpp>

#include <atomic>
#include <thread>

namespace
{
using Groups = std::vector<std::vector<cctag::DirectedPoint2d<Eigen::Vector3f>>>;

struct FlowPoints
{
    cctag::EdgePointCollection edges{41, 41};
    std::vector<cctag::EdgePoint*> outer;
    cctag::numerical::geometry::Ellipse ellipse{cctag::Point2d<Eigen::Vector3f>(20.f, 20.f), 12.f, 12.f, 0.f};

    explicit FlowPoints(int rays)
    {
        for(int ray = 0; ray < rays; ++ray)
        {
            edges.add_point(30, 20 + ray, 1.f, 0.f);
            for(int ring = 1; ring < 6; ++ring)
                edges.add_point(26 - ring, 20 + ray, ring % 2 ? -1.f : 1.f, 0.f);
            for(int ring = 0; ring < 5; ++ring)
            {
                if(ring % 2)
                    edges.set_after(edges(ray * 6 + ring), ray * 6 + ring + 1);
                else
                    edges.set_before(edges(ray * 6 + ring), ray * 6 + ring + 1);
            }
            outer.push_back(edges(ray * 6));
        }
    }

    bool assemble(const std::vector<cctag::EdgePoint*>& children, Groups& groups)
    {
        return cctag::addCandidateFlowtoCCTag(edges, children, outer, ellipse, groups, 6);
    }
};

bool samePoints(const Groups& first, const Groups& second)
{
    if(first.size() != second.size())
        return false;
    for(std::size_t ring = 0; ring < first.size(); ++ring)
    {
        if(first[ring].size() != second[ring].size())
            return false;
        for(std::size_t point = 0; point < first[ring].size(); ++point)
        {
            const auto& left = first[ring][point];
            const auto& right = second[ring][point];
            if(left.x() != right.x() || left.y() != right.y() || left.dX() != right.dX() || left.dY() != right.dY())
                return false;
        }
    }
    return true;
}
}

TEST_SUITE("test_candidateAssembly")
{
    TEST_CASE("duplicates_keep_the_first_visit_order")
    {
        FlowPoints flow(2);
        Groups expected, duplicated;
        REQUIRE_UNARY(flow.assemble({flow.outer[1], flow.outer[0]}, expected));
        REQUIRE_UNARY(flow.assemble({flow.outer[1], flow.outer[0], flow.outer[1]}, duplicated));
        CHECK_UNARY(samePoints(expected, duplicated));
        for(std::size_t ring = 0; ring + 1 < expected.size(); ++ring)
        {
            REQUIRE_EQ(expected[ring].size(), 2);
            CHECK_EQ(expected[ring][0].y(), 21.f);
            CHECK_EQ(expected[ring][1].y(), 20.f);
        }
    }

    TEST_CASE("assembly_preserves_independent_auxiliary_flags")
    {
        FlowPoints flow(1);
        Groups expected, actual;
        REQUIRE_UNARY(flow.assemble(flow.outer, expected));
        for(int point = 0; point < flow.edges.get_point_count(); ++point)
            flow.edges.set_processed_aux(flow.edges(point), true);
        REQUIRE_UNARY(flow.assemble(flow.outer, actual));
        CHECK_UNARY(samePoints(expected, actual));
        for(int point = 0; point < flow.edges.get_point_count(); ++point)
            CHECK_UNARY(flow.edges.test_processed_aux(flow.edges(point)));
    }

    TEST_CASE("concurrent_candidates_keep_all_their_shared_flow_points")
    {
        FlowPoints flow(1);
        Groups expected;
        REQUIRE_UNARY(flow.assemble(flow.outer, expected));
        const std::vector<cctag::EdgePoint*> repeated(20000, flow.outer.front());
        std::atomic<int> failures{0};
        for(int round = 0; round < 8; ++round)
        {
            std::atomic<int> ready{0};
            std::atomic<bool> start{false};
            auto assemble = [&](bool longCandidate) {
                ready.fetch_add(1, std::memory_order_relaxed);
                while(!start.load(std::memory_order_acquire))
                    std::this_thread::yield();
                for(int iteration = 0; iteration < (longCandidate ? 1 : 100); ++iteration)
                {
                    Groups actual;
                    if(!flow.assemble(longCandidate ? repeated : flow.outer, actual) || !samePoints(expected, actual))
                        failures.fetch_add(1, std::memory_order_relaxed);
                }
            };
            std::thread first(assemble, true), second(assemble, false);
            while(ready.load(std::memory_order_relaxed) < 2)
                std::this_thread::yield();
            start.store(true, std::memory_order_release);
            first.join();
            second.join();
        }
        CHECK_EQ(failures.load(std::memory_order_relaxed), 0);
    }

    TEST_CASE("broken_flow_links_keep_the_existing_errors")
    {
        FlowPoints flow(1);
        Groups groups;
        flow.edges.set_before(flow.outer.front(), -1);
        CHECK_THROWS_AS(flow.assemble(flow.outer, groups), std::out_of_range);
        flow.edges.set_before(flow.outer.front(), 10);
        CHECK_THROWS_AS(flow.assemble(flow.outer, groups), std::logic_error);
    }
}
