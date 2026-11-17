"""
Tests for SystemClock module.
"""

import unittest
from unittest.mock import patch, MagicMock

from core.system_clock import (
    SystemClock,
    DateTimeInfo,
    SystemClockError,
    get_system_clock,
    get_current_datetime,
    format_datetime_for_prompt,
)


class TestDateTimeInfo(unittest.TestCase):
    def test_format_for_prompt(self):
        info = DateTimeInfo(
            iso_datetime="2026-11-17T07:12:00+07:00",
            date="2026-11-17",
            time="07:12:00",
            timezone="Asia/Ho_Chi_Minh",
            utc_offset="+07:00",
            weekday="Tuesday",
            timestamp=1763350320.0,
        )
        formatted = info.format_for_prompt()
        self.assertIn("CURRENT_DATETIME: 2026-11-17T07:12:00+07:00", formatted)
        self.assertIn("TIMEZONE: Asia/Ho_Chi_Minh", formatted)
        self.assertIn("UTC_OFFSET: +07:00", formatted)

    def test_format_short(self):
        info = DateTimeInfo(
            iso_datetime="2026-11-17T07:12:00+07:00",
            date="2026-11-17",
            time="07:12:00",
            timezone="Asia/Ho_Chi_Minh",
            utc_offset="+07:00",
            weekday="Tuesday",
            timestamp=1763350320.0,
        )
        short = info.format_short()
        self.assertIn("2026-11-17", short)
        self.assertIn("07:12:00", short)
        self.assertIn("Asia/Ho_Chi_Minh", short)


class TestSystemClock(unittest.TestCase):
    def setUp(self):
        self.clock = SystemClock(default_timezone="UTC")

    def test_now_returns_datetime_info(self):
        info = self.clock.now()
        self.assertIsInstance(info, DateTimeInfo)
        self.assertIsInstance(info.iso_datetime, str)
        self.assertIsInstance(info.date, str)
        self.assertIsInstance(info.time, str)
        self.assertIsInstance(info.timezone, str)
        self.assertIsInstance(info.utc_offset, str)
        self.assertIsInstance(info.weekday, str)
        self.assertIsInstance(info.timestamp, float)

    def test_now_date_format(self):
        info = self.clock.now()
        # Date should be YYYY-MM-DD
        parts = info.date.split("-")
        self.assertEqual(len(parts), 3)
        self.assertEqual(len(parts[0]), 4)  # Year
        self.assertEqual(len(parts[1]), 2)  # Month
        self.assertEqual(len(parts[2]), 2)  # Day

    def test_now_time_format(self):
        info = self.clock.now()
        # Time should be HH:MM:SS
        parts = info.time.split(":")
        self.assertEqual(len(parts), 3)
        for part in parts:
            self.assertEqual(len(part), 2)

    def test_now_utc_offset_format(self):
        info = self.clock.now()
        # UTC offset should be ±HH:MM
        self.assertTrue(info.utc_offset.startswith(("+", "-")))
        parts = info.utc_offset[1:].split(":")
        self.assertEqual(len(parts), 2)

    def test_get_timezone_info(self):
        tz_info = self.clock.get_timezone_info()
        self.assertIn("timezone", tz_info)
        self.assertIn("utc_offset", tz_info)
        self.assertIsInstance(tz_info["timezone"], str)
        self.assertIsInstance(tz_info["utc_offset"], str)

    def test_format_for_model(self):
        formatted = self.clock.format_for_model()
        self.assertIn("CURRENT_DATETIME:", formatted)
        self.assertIn("TIMEZONE:", formatted)
        self.assertIn("UTC_OFFSET:", formatted)

    @patch("core.system_clock.datetime")
    def test_now_with_mocked_time(self, mock_datetime):
        mock_now = MagicMock()
        mock_now.strftime.side_effect = lambda fmt: {
            "%Y-%m-%d": "2026-11-17",
            "%H:%M:%S": "07:12:00",
            "%A": "Tuesday",
        }[fmt]
        mock_now.timestamp.return_value = 1763350320.0
        mock_now.replace.return_value = mock_now
        mock_now.astimezone.return_value = mock_now
        mock_datetime.datetime.now.return_value = mock_now

        with patch("core.system_clock.HAS_TZLOCAL", False):
            with patch("core.system_clock.time.tzname", ("UTC", "UTC")):
                info = self.clock.now()
                self.assertEqual(info.date, "2026-11-17")
                self.assertEqual(info.time, "07:12:00")
                self.assertEqual(info.weekday, "Tuesday")


class TestGlobalFunctions(unittest.TestCase):
    def test_get_system_clock_returns_instance(self):
        clock = get_system_clock()
        self.assertIsInstance(clock, SystemClock)

    def test_get_current_datetime(self):
        info = get_current_datetime()
        self.assertIsInstance(info, DateTimeInfo)

    def test_format_datetime_for_prompt(self):
        formatted = format_datetime_for_prompt()
        self.assertIn("CURRENT_DATETIME:", formatted)
        self.assertIn("TIMEZONE:", formatted)
        self.assertIn("UTC_OFFSET:", formatted)


if __name__ == "__main__":
    unittest.main()