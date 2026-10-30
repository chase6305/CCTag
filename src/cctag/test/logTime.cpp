#define BOOST_TEST_MODULE testLogTime
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>

#include <cctag/utils/LogTime.hpp>

#include <chrono>
#include <sstream>
#include <string>

BOOST_AUTO_TEST_SUITE(test_logTime)

BOOST_AUTO_TEST_CASE(unused_measurements_produce_no_output)
{
    cctag::logtime::Mgmt::Measurement measurement;
    std::ostringstream output;
    measurement.print(output);
    BOOST_CHECK(!measurement.doPrint());
    BOOST_CHECK(output.str().empty());
}

BOOST_AUTO_TEST_CASE(averages_truncated_milliseconds_and_owns_the_first_name)
{
    cctag::logtime::Mgmt::Measurement measurement;
    std::string name = "detect";
    measurement.log(name.c_str(), std::chrono::microseconds(1999));
    name.assign("changed");
    measurement.log("ignored", std::chrono::microseconds(2999));

    std::ostringstream output;
    measurement.print(output);
    BOOST_CHECK(measurement.doPrint());
    BOOST_CHECK_EQUAL(output.str(), "detect: 1.5ms \n");

    const auto copy = measurement;
    std::ostringstream copiedOutput;
    copy.print(copiedOutput);
    BOOST_CHECK_EQUAL(copiedOutput.str(), output.str());
}

BOOST_AUTO_TEST_CASE(reset_reuses_reserved_slots)
{
    static_assert(cctag::logtime::Mgmt::Clock::is_steady, "Elapsed times require a monotonic clock");
    cctag::logtime::Mgmt durations(1);
    durations.log("first");
    BOOST_CHECK_EQUAL(durations._idx, 1);
    durations.log("outside reserved slots");
    BOOST_CHECK_EQUAL(durations._idx, 1);

    durations.resetStartTime();
    BOOST_CHECK_EQUAL(durations._idx, 0);
    durations.log("second frame");
    BOOST_CHECK_EQUAL(durations._idx, 1);

    std::ostringstream output;
    durations.print(output);
    BOOST_CHECK_EQUAL(output.str().find("(0) first: "), 0);
    BOOST_CHECK_EQUAL(output.str().find("\n"), output.str().size() - 1);
}

BOOST_AUTO_TEST_CASE(zero_reserved_slots_ignore_probes)
{
    cctag::logtime::Mgmt durations(0);
    durations.log("ignored");
    std::ostringstream output;
    durations.print(output);
    BOOST_CHECK_EQUAL(durations._idx, 0);
    BOOST_CHECK(output.str().empty());
}

BOOST_AUTO_TEST_SUITE_END()
