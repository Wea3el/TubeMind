"""Unit tests for pure utility methods on TubeMindApp.

_normalize_alignment_text and _parse_seconds_label are instance methods but
reference no instance state. We call them as unbound methods with a MagicMock
as self so the full __init__ (threads, OpenAI client, dirs) never runs.
"""

import unittest
from unittest.mock import MagicMock

from tubemind.services import TubeMindApp


def _normalize(text: str) -> str:
    return TubeMindApp._normalize_alignment_text(MagicMock(spec=TubeMindApp), text)


def _parse(value: str) -> float:
    return TubeMindApp._parse_seconds_label(MagicMock(spec=TubeMindApp), value)


class TestNormalizeAlignmentText(unittest.TestCase):
    # --- normal inputs ---

    def test_lowercase(self):
        self.assertEqual(_normalize("Hello World"), "hello world")

    def test_removes_punctuation(self):
        self.assertEqual(_normalize("it's a test."), "its a test")

    def test_removes_special_chars(self):
        self.assertEqual(_normalize("Hello, World!"), "hello world")

    def test_numbers_preserved(self):
        self.assertEqual(_normalize("step 1 do this"), "step 1 do this")

    def test_mixed_alphanum_and_symbols(self):
        self.assertEqual(_normalize("Python 3.11: new features!"), "python 311 new features")

    # --- whitespace ---

    def test_collapses_internal_whitespace(self):
        self.assertEqual(_normalize("foo   bar"), "foo bar")

    def test_strips_leading_trailing(self):
        self.assertEqual(_normalize("  hello  "), "hello")

    def test_tabs_collapsed(self):
        self.assertEqual(_normalize("a\tb"), "a b")

    def test_newlines_collapsed(self):
        self.assertEqual(_normalize("line1\nline2"), "line1 line2")

    # --- edge ---

    def test_empty_string(self):
        self.assertEqual(_normalize(""), "")

    def test_all_special_chars(self):
        self.assertEqual(_normalize("!@#$%^&*()"), "")

    def test_only_whitespace(self):
        self.assertEqual(_normalize("   "), "")


class TestParseSecondsLabel(unittest.TestCase):
    # --- HH:MM:SS.ms (3-part) ---

    def test_hh_mm_ss_with_millis(self):
        self.assertAlmostEqual(_parse("01:02:03.500"), 3723.5, places=3)

    def test_hh_mm_ss_zero(self):
        self.assertAlmostEqual(_parse("00:00:00.000"), 0.0)

    def test_hh_mm_ss_no_millis(self):
        self.assertAlmostEqual(_parse("00:01:30"), 90.0)

    def test_hh_mm_ss_large(self):
        self.assertAlmostEqual(_parse("02:00:00"), 7200.0)

    # --- MM:SS.ms (2-part) ---

    def test_mm_ss_with_millis(self):
        self.assertAlmostEqual(_parse("02:30.0"), 150.0)

    def test_mm_ss_one_minute(self):
        self.assertAlmostEqual(_parse("01:00.000"), 60.0)

    def test_mm_ss_zero(self):
        self.assertAlmostEqual(_parse("00:00.000"), 0.0)

    # --- bare float (1-part) ---

    def test_bare_float(self):
        self.assertAlmostEqual(_parse("90.5"), 90.5)

    def test_bare_integer_string(self):
        self.assertAlmostEqual(_parse("60"), 60.0)

    # --- comma decimal separator (VTT style) ---

    def test_comma_decimal_3part(self):
        self.assertAlmostEqual(_parse("01:02:03,500"), 3723.5, places=3)

    def test_comma_decimal_2part(self):
        self.assertAlmostEqual(_parse("01:30,000"), 90.0)

    # --- whitespace ---

    def test_leading_whitespace_stripped(self):
        self.assertAlmostEqual(_parse("  01:30.0"), 90.0)

    def test_trailing_whitespace_stripped(self):
        self.assertAlmostEqual(_parse("01:30.0  "), 90.0)


if __name__ == "__main__":
    unittest.main()
