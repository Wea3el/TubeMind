"""Unit tests for tubemind/models.py pure helper functions.

All functions here are pure (no I/O, no external deps), so no mocking is
needed. Tests are organized by equivalence class as per Week 7 guidance.
"""

import time
import unittest

from tubemind.models import iso8601_duration_to_seconds, now_ms, seconds_to_label, yt_watch_url


class TestIso8601DurationToSeconds(unittest.TestCase):
    # --- valid inputs ---

    def test_hours_minutes_seconds(self):
        self.assertEqual(iso8601_duration_to_seconds("PT1H2M3S"), 3723)

    def test_minutes_and_seconds(self):
        self.assertEqual(iso8601_duration_to_seconds("PT12M30S"), 750)

    def test_seconds_only(self):
        self.assertEqual(iso8601_duration_to_seconds("PT45S"), 45)

    def test_hours_only(self):
        self.assertEqual(iso8601_duration_to_seconds("PT2H"), 7200)

    def test_minutes_only(self):
        self.assertEqual(iso8601_duration_to_seconds("PT4M"), 240)

    def test_large_duration(self):
        self.assertEqual(iso8601_duration_to_seconds("PT10H59M59S"), 39599)

    # --- boundary / zero ---

    def test_zero_duration_string(self):
        self.assertEqual(iso8601_duration_to_seconds("PT0S"), 0)

    def test_exactly_240_seconds(self):
        # 4 minutes is the default min_seconds threshold
        self.assertEqual(iso8601_duration_to_seconds("PT4M"), 240)

    # --- invalid / edge ---

    def test_empty_string(self):
        self.assertEqual(iso8601_duration_to_seconds(""), 0)

    def test_none(self):
        self.assertEqual(iso8601_duration_to_seconds(None), 0)

    def test_non_string_int(self):
        self.assertEqual(iso8601_duration_to_seconds(300), 0)

    def test_malformed_garbage(self):
        self.assertEqual(iso8601_duration_to_seconds("garbage"), 0)

    def test_missing_pt_prefix(self):
        self.assertEqual(iso8601_duration_to_seconds("1H2M3S"), 0)

    def test_partial_prefix_only(self):
        self.assertEqual(iso8601_duration_to_seconds("PT"), 0)


class TestSecondsToLabel(unittest.TestCase):
    # --- boundary / zero ---

    def test_zero_returns_empty(self):
        self.assertEqual(seconds_to_label(0), "")

    def test_negative_returns_empty(self):
        self.assertEqual(seconds_to_label(-1), "")

    def test_large_negative_returns_empty(self):
        self.assertEqual(seconds_to_label(-3600), "")

    # --- sub-minute equivalence class ---

    def test_under_a_minute(self):
        self.assertEqual(seconds_to_label(45), "0:45")

    def test_one_second(self):
        self.assertEqual(seconds_to_label(1), "0:01")

    def test_59_seconds(self):
        self.assertEqual(seconds_to_label(59), "0:59")

    # --- minute equivalence class ---

    def test_exactly_one_minute(self):
        self.assertEqual(seconds_to_label(60), "1:00")

    def test_minutes_and_seconds(self):
        self.assertEqual(seconds_to_label(93), "1:33")

    def test_59_minutes_59_seconds(self):
        self.assertEqual(seconds_to_label(3599), "59:59")

    # --- hour equivalence class ---

    def test_exactly_one_hour(self):
        self.assertEqual(seconds_to_label(3600), "1:00:00")

    def test_hours_minutes_seconds(self):
        self.assertEqual(seconds_to_label(3661), "1:01:01")

    def test_zero_padding_preserved(self):
        self.assertEqual(seconds_to_label(3600 + 60 + 5), "1:01:05")

    def test_multi_hour(self):
        self.assertEqual(seconds_to_label(7322), "2:02:02")


class TestYtWatchUrl(unittest.TestCase):
    def test_no_offset(self):
        self.assertEqual(yt_watch_url("abc123"), "https://www.youtube.com/watch?v=abc123")

    def test_with_fractional_offset(self):
        url = yt_watch_url("abc123", offset_seconds=90.9)
        self.assertEqual(url, "https://www.youtube.com/watch?v=abc123&t=90s")

    def test_zero_offset(self):
        url = yt_watch_url("abc123", offset_seconds=0)
        self.assertEqual(url, "https://www.youtube.com/watch?v=abc123&t=0s")

    def test_negative_offset_clamped_to_zero(self):
        url = yt_watch_url("abc123", offset_seconds=-30)
        self.assertIn("t=0s", url)

    def test_offset_truncated_not_rounded(self):
        # int() truncates, so 1.9 → 1
        url = yt_watch_url("vid", offset_seconds=1.9)
        self.assertIn("t=1s", url)


class TestNowMs(unittest.TestCase):
    def test_returns_int(self):
        self.assertIsInstance(now_ms(), int)

    def test_positive(self):
        self.assertGreater(now_ms(), 0)

    def test_within_current_second(self):
        before = int(time.time() * 1000)
        result = now_ms()
        after = int(time.time() * 1000)
        self.assertGreaterEqual(result, before)
        self.assertLessEqual(result, after)

    def test_monotonically_increasing(self):
        t1 = now_ms()
        time.sleep(0.01)
        t2 = now_ms()
        self.assertGreater(t2, t1)


if __name__ == "__main__":
    unittest.main()
