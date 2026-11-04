/*
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at http://mozilla.org/MPL/2.0/.
 */
#include <cctag/CCTagMarkersBank.hpp>
#include <doctest/doctest.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <utility>

namespace {

class BankFile
{
public:
    explicit BankFile(const std::string& contents)
      : path(std::filesystem::temp_directory_path() /
             ("cctag-bank-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + ".txt"))
    {
        std::ofstream output(path);
        output << contents;
        if(!output)
            throw std::runtime_error("Unable to create a temporary marker bank");
    }

    ~BankFile()
    {
        std::error_code error;
        std::filesystem::remove(path, error);
    }

    std::filesystem::path path;
};

} // namespace

TEST_SUITE("test_markersBank") {

TEST_CASE("built_in_profiles_keep_their_one_based_identifiers")
{
    for(std::size_t crowns : {3, 4})
    {
        const cctag::CCTagMarkersBank bank(crowns);
        const auto& markers = bank.getMarkers();
        REQUIRE_EQ(markers.size(), crowns == 3 ? 32 : 128);
        for(std::size_t markerIndex = 0; markerIndex < markers.size(); ++markerIndex)
        {
            CHECK_EQ(markers[markerIndex].size(), 2 * crowns - 1);
            CHECK_EQ(bank.identify(markers[markerIndex]), markerIndex + 1);
        }
    }
}

TEST_CASE("reading_accepts_decimals_and_fractions_and_appends_in_file_order")
{
    BankFile file("\n  2.5\t 10 / 6 5/4\r\n\t2e0 3/2 1.1");
    cctag::CCTagMarkersBank bank(file.path.string());
    const std::vector<std::vector<float>> expected{{2.5f, 10.f / 6.f, 1.25f}, {2.f, 1.5f, 1.1f}};
    CHECK_EQ(bank.getMarkers(), expected);
    bank.read(file.path.string());
    REQUIRE_EQ(bank.getMarkers().size(), 4);
    CHECK_EQ(bank.getMarkers()[2], expected[0]);
    CHECK_EQ(bank.getMarkers()[3], expected[1]);
}

TEST_CASE("invalid_rows_leave_the_existing_bank_unchanged")
{
    cctag::CCTagMarkersBank bank(3);
    const auto original = bank.getMarkers();
    for(const char* invalid : {"2 1.5 junk", "2.0.5", "1/1.5", "2/0", "0", "-2", "nan", "inf", "1e39"})
    {
        BankFile file(std::string("2 1.5 1.25\n") + invalid + '\n');
        CHECK_THROWS_WITH_AS(bank.read(file.path.string()), doctest::Contains(":2"), std::invalid_argument);
        CHECK_UNARY(bank.getMarkers() == original);
    }
}

TEST_CASE("empty_and_missing_files_are_rejected_without_changing_the_bank")
{
    cctag::CCTagMarkersBank bank(3);
    const auto original = bank.getMarkers();
    for(const char* contents : {"", " \t\n\r\n"})
    {
        BankFile file(contents);
        CHECK_THROWS_AS(bank.read(file.path.string()), std::invalid_argument);
        CHECK_UNARY(bank.getMarkers() == original);
    }
    BankFile missing("");
    std::filesystem::remove(missing.path);
    CHECK_THROWS_AS(bank.read(missing.path.string()), std::ios_base::failure);
    CHECK_UNARY(bank.getMarkers() == original);
}

TEST_CASE("identification_requires_complete_finite_profiles")
{
    BankFile file("2 1.5\n2 1.5 1.25\n");
    const cctag::CCTagMarkersBank bank(file.path.string());
    CHECK_EQ(bank.identify({2.f, 1.5f}), 1);
    CHECK_EQ(bank.identify({2.f, 1.5f, 1.25f}), 2);
    CHECK_THROWS_AS(bank.identify({2.f}), std::runtime_error);
    CHECK_THROWS_AS(bank.identify({}), std::invalid_argument);
    CHECK_THROWS_AS(bank.identify({std::numeric_limits<float>::quiet_NaN(), 1.5f}), std::invalid_argument);
    CHECK_THROWS_AS(bank.identify({std::numeric_limits<float>::infinity(), 1.5f}), std::invalid_argument);
}

TEST_CASE("moving_a_bank_transfers_profiles_and_copying_remains_independent")
{
    cctag::CCTagMarkersBank original(4);
    const auto* profiles = original.getMarkers().data();
    cctag::CCTagMarkersBank moved(std::move(original));
    CHECK_EQ(moved.getMarkers().data(), profiles);
    cctag::CCTagMarkersBank copy(moved);
    CHECK_NE(copy.getMarkers().data(), profiles);
    CHECK_EQ(copy.getMarkers(), moved.getMarkers());
    cctag::CCTagMarkersBank assigned(3);
    assigned = std::move(moved);
    CHECK_EQ(assigned.getMarkers().data(), profiles);
}

} // TEST_SUITE
