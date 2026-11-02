
#include <doctest/doctest.h>

#include <cctag/utils/LogTime.hpp>

#include <chrono>
#include <sstream>
#include <string>

TEST_SUITE("test_logTime") {

TEST_CASE("unused_measurements_produce_no_output")
{
    cctag::logtime::Mgmt::Measurement measurement;
    std::ostringstream output;
    measurement.print(output);
    CHECK_UNARY(!measurement.doPrint());
    CHECK_UNARY(output.str().empty());
}

TEST_CASE("averages_truncated_milliseconds_and_owns_the_first_name")
{
    cctag::logtime::Mgmt::Measurement measurement;
    std::string name = "detect";
    measurement.log(name.c_str(), std::chrono::microseconds(1999));
    name.assign("changed");
    measurement.log("ignored", std::chrono::microseconds(2999));

    std::ostringstream output;
    measurement.print(output);
    CHECK_UNARY(measurement.doPrint());
    CHECK_EQ(output.str(), "detect: 1.5ms \n");

    const auto copy = measurement;
    std::ostringstream copiedOutput;
    copy.print(copiedOutput);
    CHECK_EQ(copiedOutput.str(), output.str());
}

TEST_CASE("reset_reuses_reserved_slots")
{
    static_assert(cctag::logtime::Mgmt::Clock::is_steady, "Elapsed times require a monotonic clock");
    cctag::logtime::Mgmt durations(1);
    durations.log("first");
    CHECK_EQ(durations._idx, 1);
    durations.log("outside reserved slots");
    CHECK_EQ(durations._idx, 1);

    durations.resetStartTime();
    CHECK_EQ(durations._idx, 0);
    durations.log("second frame");
    CHECK_EQ(durations._idx, 1);

    std::ostringstream output;
    durations.print(output);
    CHECK_EQ(output.str().find("(0) first: "), 0);
    CHECK_EQ(output.str().find("\n"), output.str().size() - 1);
}

TEST_CASE("zero_reserved_slots_ignore_probes")
{
    cctag::logtime::Mgmt durations(0);
    durations.log("ignored");
    std::ostringstream output;
    durations.print(output);
    CHECK_EQ(durations._idx, 0);
    CHECK_UNARY(output.str().empty());
}

} // TEST_SUITE
