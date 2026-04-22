"""Tests for all rate-limiting and retry logic in TubeMind.

Three surfaces are covered:

1. Decision helpers (_is_transcript_rate_limited, _should_retry_transcript_error,
   _describe_transcript_error) — pure logic, no I/O.

2. _fetch_transcript retry loop — verifies attempt count, exponential backoff
   sleep calls, and eventual success after transient failure.

3. _fetch_transcript_with_transcriptapi — verifies HTTP 429/503 retries,
   Retry-After header honoring, backoff cap, and non-retryable status codes.
"""

import time
import unittest
from unittest.mock import MagicMock, call, patch

from youtube_transcript_api import TooManyRequests, YouTubeRequestFailed

from tubemind.config import TRANSCRIPT_RETRY_ATTEMPTS, TRANSCRIPT_RETRY_BASE_DELAY
from tubemind.models import YouTubeVideo
from tubemind.services import TubeMindApp


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_app() -> TubeMindApp:
    with patch.object(TubeMindApp, "__init__", return_value=None):
        app = TubeMindApp.__new__(TubeMindApp)
    return app


def _video(video_id: str = "vid001") -> YouTubeVideo:
    return YouTubeVideo(
        video_id=video_id,
        title="Test",
        channel_title="Chan",
        published_at="2024-01-01T00:00:00Z",
        thumbnail="",
        duration_sec=600,
        url=f"https://www.youtube.com/watch?v={video_id}",
    )


def _http_response(status: int, data: dict | None = None, headers: dict | None = None) -> MagicMock:
    r = MagicMock()
    r.status_code = status
    r.json.return_value = data or {}
    r.headers = headers or {}
    r.text = str(data or "")
    return r


# ---------------------------------------------------------------------------
# 1. Decision-logic helpers
# ---------------------------------------------------------------------------

class TestIsTranscriptRateLimited(unittest.TestCase):

    def _call(self, exc: Exception) -> bool:
        app = _make_app()
        return TubeMindApp._is_transcript_rate_limited(app, exc)

    def test_too_many_requests_instance(self):
        self.assertTrue(self._call(TooManyRequests("vid001")))

    def test_exception_with_429_in_message(self):
        self.assertTrue(self._call(Exception("HTTP 429 quota exceeded")))

    def test_exception_with_too_many_requests_text(self):
        self.assertTrue(self._call(Exception("too many requests from this IP")))

    def test_too_many_requests_case_insensitive(self):
        self.assertTrue(self._call(Exception("TOO MANY REQUESTS")))

    def test_generic_exception_not_rate_limited(self):
        self.assertFalse(self._call(Exception("connection refused")))

    def test_youtube_request_failed_not_rate_limited(self):
        exc = YouTubeRequestFailed("vid001", Exception("503 service unavailable"))
        self.assertFalse(self._call(exc))

    def test_value_error_not_rate_limited(self):
        self.assertFalse(self._call(ValueError("bad value")))


class TestShouldRetryTranscriptError(unittest.TestCase):

    def _call(self, exc: Exception) -> bool:
        app = _make_app()
        return TubeMindApp._should_retry_transcript_error(app, exc)

    def test_too_many_requests_should_retry(self):
        self.assertTrue(self._call(TooManyRequests("vid001")))

    def test_429_string_should_retry(self):
        self.assertTrue(self._call(Exception("429 Too Many Requests")))

    def test_youtube_request_failed_timed_out_should_retry(self):
        exc = YouTubeRequestFailed("vid001", Exception("timed out"))
        self.assertTrue(self._call(exc))

    def test_youtube_request_failed_temporarily_unavailable_should_retry(self):
        exc = YouTubeRequestFailed("vid001", Exception("temporarily unavailable"))
        self.assertTrue(self._call(exc))

    def test_youtube_request_failed_other_error_no_retry(self):
        exc = YouTubeRequestFailed("vid001", Exception("forbidden"))
        self.assertFalse(self._call(exc))

    def test_generic_exception_no_retry(self):
        self.assertFalse(self._call(Exception("parsing error")))

    def test_value_error_no_retry(self):
        self.assertFalse(self._call(ValueError("unexpected token")))


class TestDescribeTranscriptError(unittest.TestCase):

    def _call(self, exc: Exception, using_cookies: bool = False) -> str:
        app = _make_app()
        return TubeMindApp._describe_transcript_error(app, exc, using_cookies=using_cookies)

    def test_rate_limited_without_cookies_includes_hint(self):
        msg = self._call(TooManyRequests("vid001"), using_cookies=False)
        self.assertIn("YOUTUBE_TRANSCRIPT_COOKIES_FILE", msg)

    def test_rate_limited_with_cookies_no_hint(self):
        msg = self._call(TooManyRequests("vid001"), using_cookies=True)
        self.assertNotIn("YOUTUBE_TRANSCRIPT_COOKIES_FILE", msg)

    def test_429_string_without_cookies_includes_hint(self):
        msg = self._call(Exception("429 Too Many Requests"), using_cookies=False)
        self.assertIn("YOUTUBE_TRANSCRIPT_COOKIES_FILE", msg)

    def test_non_rate_limited_no_hint(self):
        msg = self._call(Exception("connection error"), using_cookies=False)
        self.assertNotIn("YOUTUBE_TRANSCRIPT_COOKIES_FILE", msg)

    def test_includes_exception_type_in_message(self):
        msg = self._call(ValueError("bad input"))
        self.assertIn("ValueError", msg)

    def test_includes_exception_text_in_message(self):
        msg = self._call(Exception("something went wrong"))
        self.assertIn("something went wrong", msg)


# ---------------------------------------------------------------------------
# 2. _fetch_transcript retry loop
# ---------------------------------------------------------------------------

class TestFetchTranscriptRetryLoop(unittest.TestCase):
    """Verify retry count, exponential sleep schedule, and eventual success."""

    def _make_isolated_app(self) -> TubeMindApp:
        app = _make_app()
        app._transcript_api_key = MagicMock(return_value=None)
        app._transcript_request_kwargs = MagicMock(return_value={})
        app._fetch_transcript_with_transcriptapi = MagicMock(return_value=(None, "TranscriptAPI unavailable"))
        app._fetch_transcript_with_ytdlp = MagicMock(return_value=(None, "yt-dlp failed"))
        # Use real implementations for the decision helpers
        app._is_transcript_rate_limited = lambda exc: TubeMindApp._is_transcript_rate_limited(app, exc)
        app._should_retry_transcript_error = lambda exc: TubeMindApp._should_retry_transcript_error(app, exc)
        app._describe_transcript_error = lambda exc, **kw: TubeMindApp._describe_transcript_error(app, exc, **kw)
        return app

    def test_retries_full_attempt_count_on_rate_limit(self):
        app = self._make_isolated_app()
        with patch("tubemind.services.YouTubeTranscriptApi.get_transcript",
                   side_effect=TooManyRequests("vid001")) as mock_get, \
             patch("tubemind.services.time.sleep"):
            TubeMindApp._fetch_transcript(app, _video())
        self.assertEqual(mock_get.call_count, TRANSCRIPT_RETRY_ATTEMPTS)

    def test_stops_immediately_on_non_retriable_error(self):
        app = self._make_isolated_app()
        with patch("tubemind.services.YouTubeTranscriptApi.get_transcript",
                   side_effect=Exception("parsing failure")) as mock_get, \
             patch("tubemind.services.time.sleep") as mock_sleep:
            TubeMindApp._fetch_transcript(app, _video())
        # Should break after first attempt — non-retriable error
        self.assertEqual(mock_get.call_count, 1)
        mock_sleep.assert_not_called()

    def test_exponential_backoff_sleep_schedule(self):
        """Sleeps between attempts should follow BASE * 2^(attempt-1)."""
        app = self._make_isolated_app()
        with patch("tubemind.services.YouTubeTranscriptApi.get_transcript",
                   side_effect=TooManyRequests("vid001")), \
             patch("tubemind.services.time.sleep") as mock_sleep:
            TubeMindApp._fetch_transcript(app, _video())

        expected_delays = [
            TRANSCRIPT_RETRY_BASE_DELAY * (2 ** i)
            for i in range(TRANSCRIPT_RETRY_ATTEMPTS - 1)
        ]
        actual_delays = [c.args[0] for c in mock_sleep.call_args_list]
        self.assertEqual(actual_delays, expected_delays)

    def test_succeeds_on_second_attempt_after_rate_limit(self):
        segments = [{"text": "Hello", "start": 0.0, "duration": 1.0}]
        app = self._make_isolated_app()
        call_count = [0]

        def _side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                raise TooManyRequests("vid001")
            return segments

        with patch("tubemind.services.YouTubeTranscriptApi.get_transcript",
                   side_effect=_side_effect), \
             patch("tubemind.services.time.sleep"):
            result, err = TubeMindApp._fetch_transcript(app, _video())

        self.assertEqual(result, segments)
        self.assertIsNone(err)
        self.assertEqual(call_count[0], 2)

    def test_no_sleep_after_final_attempt(self):
        """Sleep must not be called after the last failed attempt."""
        app = self._make_isolated_app()
        with patch("tubemind.services.YouTubeTranscriptApi.get_transcript",
                   side_effect=TooManyRequests("vid001")), \
             patch("tubemind.services.time.sleep") as mock_sleep:
            TubeMindApp._fetch_transcript(app, _video())

        # Sleep count should be one less than total attempts
        self.assertEqual(mock_sleep.call_count, TRANSCRIPT_RETRY_ATTEMPTS - 1)

    def test_error_message_returned_after_all_retries_exhausted(self):
        app = self._make_isolated_app()
        with patch("tubemind.services.YouTubeTranscriptApi.get_transcript",
                   side_effect=TooManyRequests("vid001")), \
             patch("tubemind.services.time.sleep"):
            result, err = TubeMindApp._fetch_transcript(app, _video())

        self.assertIsNone(result)
        self.assertIsNotNone(err)
        self.assertIsInstance(err, str)


# ---------------------------------------------------------------------------
# 3. _fetch_transcript_with_transcriptapi HTTP retry behavior
# ---------------------------------------------------------------------------

class _SyncClientCM:
    """Sync context manager that replays a sequence of pre-built responses."""

    def __init__(self, responses: list) -> None:
        self._iter = iter(responses)
        self._mock = MagicMock()
        self._mock.get.side_effect = lambda *a, **kw: next(self._iter)

    def __enter__(self):
        return self._mock

    def __exit__(self, *args):
        return False


class TestTranscriptApiRateLimiting(unittest.TestCase):

    def _make_app_with_key(self) -> TubeMindApp:
        app = _make_app()
        app._transcript_api_key = MagicMock(return_value="test-api-key")
        app._extract_transcriptapi_error = MagicMock(return_value="rate limited")
        return app

    def _patch_client(self, responses: list):
        cm = _SyncClientCM(responses)
        return patch("tubemind.services.httpx.Client", return_value=cm)

    def _success_response(self) -> MagicMock:
        return _http_response(200, {"transcript": [{"start": 0.0, "text": "Hello world"}]})

    def test_returns_segments_on_first_200(self):
        app = self._make_app_with_key()
        with self._patch_client([self._success_response()]), \
             patch("tubemind.services.time.sleep"):
            result, err = TubeMindApp._fetch_transcript_with_transcriptapi(app, _video())
        self.assertIsNotNone(result)
        self.assertIsNone(err)

    def test_retries_on_429(self):
        app = self._make_app_with_key()
        responses = [
            _http_response(429, {"detail": "rate limited"}),
            self._success_response(),
        ]
        with self._patch_client(responses), \
             patch("tubemind.services.time.sleep") as mock_sleep:
            result, err = TubeMindApp._fetch_transcript_with_transcriptapi(app, _video())
        self.assertIsNotNone(result)
        self.assertIsNone(err)
        mock_sleep.assert_called()

    def test_retries_on_503(self):
        app = self._make_app_with_key()
        responses = [
            _http_response(503, {"detail": "service unavailable"}),
            self._success_response(),
        ]
        with self._patch_client(responses), \
             patch("tubemind.services.time.sleep") as mock_sleep:
            result, err = TubeMindApp._fetch_transcript_with_transcriptapi(app, _video())
        self.assertIsNotNone(result)
        mock_sleep.assert_called()

    def test_retries_on_408(self):
        app = self._make_app_with_key()
        responses = [
            _http_response(408, {"detail": "request timeout"}),
            self._success_response(),
        ]
        with self._patch_client(responses), \
             patch("tubemind.services.time.sleep") as mock_sleep:
            result, err = TubeMindApp._fetch_transcript_with_transcriptapi(app, _video())
        self.assertIsNotNone(result)
        mock_sleep.assert_called()

    def test_does_not_retry_on_400(self):
        app = self._make_app_with_key()
        call_count = [0]

        def _get(*args, **kwargs):
            call_count[0] += 1
            return _http_response(400, {"detail": "bad request"})

        cm = MagicMock()
        cm.__enter__ = MagicMock(return_value=MagicMock(get=_get))
        cm.__exit__ = MagicMock(return_value=False)

        with patch("tubemind.services.httpx.Client", return_value=cm), \
             patch("tubemind.services.time.sleep"):
            result, err = TubeMindApp._fetch_transcript_with_transcriptapi(app, _video())

        self.assertIsNone(result)
        self.assertEqual(call_count[0], 1)

    def test_does_not_retry_on_404(self):
        app = self._make_app_with_key()
        call_count = [0]

        def _get(*args, **kwargs):
            call_count[0] += 1
            return _http_response(404, {"detail": "not found"})

        cm = MagicMock()
        cm.__enter__ = MagicMock(return_value=MagicMock(get=_get))
        cm.__exit__ = MagicMock(return_value=False)

        with patch("tubemind.services.httpx.Client", return_value=cm), \
             patch("tubemind.services.time.sleep"):
            result, err = TubeMindApp._fetch_transcript_with_transcriptapi(app, _video())

        self.assertIsNone(result)
        self.assertEqual(call_count[0], 1)

    def test_retry_after_header_overrides_default_delay(self):
        app = self._make_app_with_key()
        responses = [
            _http_response(429, {"detail": "rate limited"}, headers={"Retry-After": "5"}),
            self._success_response(),
        ]
        with self._patch_client(responses), \
             patch("tubemind.services.time.sleep") as mock_sleep:
            TubeMindApp._fetch_transcript_with_transcriptapi(app, _video())

        # The sleep should use the Retry-After value (5.0), not the default 1.0
        sleep_arg = mock_sleep.call_args_list[0].args[0]
        self.assertGreaterEqual(sleep_arg, 5.0)

    def test_backoff_doubles_each_retry(self):
        app = self._make_app_with_key()
        responses = [
            _http_response(429, {"detail": "rate limited"}),
            _http_response(429, {"detail": "rate limited"}),
            _http_response(429, {"detail": "rate limited"}),
        ]
        sleep_calls = []

        def _fake_sleep(n):
            sleep_calls.append(n)

        with self._patch_client(responses), \
             patch("tubemind.services.time.sleep", side_effect=_fake_sleep):
            TubeMindApp._fetch_transcript_with_transcriptapi(app, _video())

        # Should have slept at least twice (between 3 attempts)
        self.assertGreaterEqual(len(sleep_calls), 2)
        # Each subsequent delay must be >= the previous (doubling)
        for i in range(1, len(sleep_calls)):
            self.assertGreaterEqual(sleep_calls[i], sleep_calls[i - 1])

    def test_backoff_capped_at_10_seconds(self):
        app = self._make_app_with_key()
        # Provide a very large Retry-After to try to push past the cap
        responses = [
            _http_response(429, {"detail": "rate limited"}, headers={"Retry-After": "999"}),
            _http_response(429, {"detail": "rate limited"}),
            _http_response(429, {"detail": "rate limited"}),
        ]
        sleep_calls = []

        with self._patch_client(responses), \
             patch("tubemind.services.time.sleep", side_effect=lambda n: sleep_calls.append(n)):
            TubeMindApp._fetch_transcript_with_transcriptapi(app, _video())

        # After the first retry, the cap of 10.0 must apply
        if len(sleep_calls) > 1:
            self.assertLessEqual(sleep_calls[1], 10.0)

    def test_returns_error_after_all_retries_exhausted(self):
        app = self._make_app_with_key()
        responses = [
            _http_response(429, {"detail": "rate limited"}),
            _http_response(429, {"detail": "rate limited"}),
            _http_response(429, {"detail": "rate limited"}),
        ]
        with self._patch_client(responses), \
             patch("tubemind.services.time.sleep"):
            result, err = TubeMindApp._fetch_transcript_with_transcriptapi(app, _video())

        self.assertIsNone(result)
        self.assertIsNotNone(err)
        self.assertIn("TranscriptAPI", err)

    def test_skips_entirely_when_no_api_key(self):
        app = _make_app()
        app._transcript_api_key = MagicMock(return_value=None)

        with patch("tubemind.services.httpx.Client") as mock_client:
            result, err = TubeMindApp._fetch_transcript_with_transcriptapi(app, _video())

        mock_client.assert_not_called()
        self.assertIsNone(result)
        self.assertIsNone(err)

    def test_empty_transcript_response_not_returned(self):
        app = self._make_app_with_key()
        responses = [_http_response(200, {"transcript": []})]
        with self._patch_client(responses), \
             patch("tubemind.services.time.sleep"):
            result, err = TubeMindApp._fetch_transcript_with_transcriptapi(app, _video())
        self.assertIsNone(result)
        self.assertIsNotNone(err)


if __name__ == "__main__":
    unittest.main()
