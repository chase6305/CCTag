/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include <cctag/CCTag.hpp>
#include <doctest/doctest.h>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/serialization/nvp.hpp>

#include <cstring>
#include <memory>
#include <new>
#include <sstream>

namespace {

std::size_t serializedCircleCount(cctag::CCTag& tag)
{
    std::ostringstream encoded;
    {
        boost::archive::text_oarchive archive(encoded);
        tag.serialize(archive, 0);
    }
    std::istringstream input(encoded.str());
    boost::archive::text_iarchive archive(input);
    std::size_t circles = 0;
    archive >> boost::serialization::make_nvp("_nCircles", circles);
    return circles;
}

} // namespace

TEST_SUITE("test_cctag") {

TEST_CASE("default_marker_has_initialized_coordinates_and_transformation")
{
    alignas(cctag::CCTag) unsigned char storage[sizeof(cctag::CCTag)];
    std::memset(storage, 0xa5, sizeof(storage));
    auto* tag = new(storage) cctag::CCTag;
    CHECK_EQ(tag->x(), 0.f);
    CHECK_EQ(tag->y(), 0.f);
    CHECK_EQ(tag->centerImg().w(), 1.f);
    CHECK_EQ(tag->pyramidLevel(), 0);
    CHECK_EQ(tag->scale(), 1.f);
    CHECK_UNARY(tag->homography().isIdentity());
    CHECK_EQ(tag->id(), 0);
    CHECK_EQ(tag->getStatus(), 0);
    CHECK_EQ(tag->quality(), 0.f);
    CHECK_EQ(tag->nCircles(), tag->radiusRatios().size() + 1);
    tag->~CCTag();
}

TEST_CASE("copy_and_polymorphic_clone_preserve_id_scores_and_rescaled_edge_points")
{
    using Point = cctag::Point2d<Eigen::Vector3f>;
    using DirectedPoint = cctag::DirectedPoint2d<Eigen::Vector3f>;
    const cctag::numerical::geometry::Ellipse ellipse(Point(12.f, 18.f), 5.f, 3.f, 0.2f);
    const std::vector<std::vector<DirectedPoint>> points{{{10.f, 20.f, 1.f, 2.f}}};
    Eigen::Matrix3f homography = Eigen::Matrix3f::Identity();
    homography(0, 2) = 12.f;
    homography(1, 2) = 18.f;
    cctag::CCTag source(7, Point(12.f, 18.f), points, ellipse, homography, 2, 4.f, 0.8f);
    const cctag::IdSet scores{{7, 0.9f}, {4, 0.2f}};
    const std::vector<DirectedPoint> outerPoints{{50.f, 70.f, 1.f, 2.f}, {60.f, 80.f, 3.f, 4.f}};
    source.setIdSet(scores);
    source.setRescaledOuterEllipsePoints(outerPoints);
    source.setStatus(cctag::status::id_reliable);

    const cctag::CCTag copied(source);
    const cctag::ICCTag& base = source;
    std::unique_ptr<cctag::ICCTag> cloned(base.clone());
    const auto* clonedMarker = dynamic_cast<const cctag::CCTag*>(cloned.get());
    REQUIRE_NE(clonedMarker, nullptr);

    source.setIdSet({});
    source.setRescaledOuterEllipsePoints({});
    for(const auto* tag : {&copied, clonedMarker})
    {
        CHECK_EQ(tag->idSet(), scores);
        CHECK_EQ(tag->rescaledOuterEllipsePoints().size(), outerPoints.size());
        for(std::size_t index = 0; index < tag->rescaledOuterEllipsePoints().size(); ++index)
        {
            CHECK_UNARY(tag->rescaledOuterEllipsePoints()[index].isApprox(outerPoints[index]));
            CHECK_UNARY(tag->rescaledOuterEllipsePoints()[index].gradient().isApprox(outerPoints[index].gradient()));
        }
        CHECK_EQ(tag->id(), 7);
        CHECK_EQ(tag->getStatus(), cctag::status::id_reliable);
        CHECK_EQ(tag->quality(), 0.8f);
        CHECK_EQ(tag->pyramidLevel(), 2);
        CHECK_EQ(tag->scale(), 4.f);
        CHECK_UNARY(tag->homography().isApprox(homography));
    }
}

TEST_CASE("detected_circle_groups_define_the_count_before_identification")
{
    using Point = cctag::Point2d<Eigen::Vector3f>;
    using DirectedPoint = cctag::DirectedPoint2d<Eigen::Vector3f>;
    for(std::size_t crowns : {3, 4})
    {
        const cctag::Parameters params(crowns);
        const cctag::CCTagMarkersBank bank(crowns);
        const auto& profile = bank.getMarkers().front();
        std::vector<std::vector<DirectedPoint>> circles(params._nCircles);
        for(std::size_t circle = 0; circle < circles.size(); ++circle)
        {
            const float radius = circle < profile.size() ? 10.f / profile[circle] : 10.f;
            circles[circle].emplace_back(20.f + radius, 30.f, 1.f, 0.f);
        }
        const cctag::numerical::geometry::Ellipse ellipse(Point(20.f, 30.f), 10.f, 10.f, 0.f);
        Eigen::Matrix3f homography = Eigen::Matrix3f::Identity();
        homography(0, 0) = homography(1, 1) = 10.f;
        homography(0, 2) = 20.f;
        homography(1, 2) = 30.f;
        cctag::CCTag tag(-1, Point(20.f, 30.f), circles, ellipse, homography, 0, 1.f);
        CHECK_EQ(tag.nCircles(), params._nCircles);
        CHECK_EQ(serializedCircleCount(tag), params._nCircles);
    }
}

TEST_CASE("recognized_profiles_update_circle_metadata_and_serialization")
{
    cctag::CCTag tag;
    for(std::size_t crowns : {4, 3})
    {
        const cctag::Parameters params(crowns);
        const cctag::CCTagMarkersBank bank(crowns);
        const auto& profile = bank.getMarkers().front();
        tag.setRadiusRatios(profile);
        CHECK_EQ(tag.nCircles(), params._nCircles);
        CHECK_EQ(tag.nCircles(), profile.size() + 1);
        CHECK_EQ(serializedCircleCount(tag), params._nCircles);
        std::unique_ptr<cctag::CCTag> cloned(tag.clone());
        CHECK_EQ(cloned->nCircles(), params._nCircles);
    }
}

} // TEST_SUITE
