"""Unit tests for TubeMindApp service methods that hit external APIs.

All network I/O is replaced with unittest.mock so tests run offline and fast.
TubeMindApp.__init__ is bypassed via __new__ + manual attribute setting to
avoid spinning up threads, creating OpenAI clients, and touching the FS.
"""

import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from tubemind.models import YouTubeVideo
from tubemind.services import TubeMindApp


def _make_app() -> TubeMindApp:
    """Return a TubeMindApp instance without running __init__ side-effects."""
    with patch.object(TubeMindApp, "__init__", return_value=None):
        app = TubeMindApp.__new__(TubeMindApp)
    return app


def _video(video_id: str = "vid001", duration_sec: int = 600) -> YouTubeVideo:
    return YouTubeVideo(
        video_id=video_id,
        title="Test Video",
        channel_title="Test Channel",
        published_at="2024-01-01T00:00:00Z",
        thumbnail="",
        duration_sec=duration_sec,
        url=f"https://www.youtube.com/watch?v={video_id}",
    )


# ---------------------------------------------------------------------------
# Helpers for mocking httpx.AsyncClient as an async context manager
# ---------------------------------------------------------------------------

def _make_http_response(data: dict, status: int = 200) -> MagicMock:
    r = MagicMock()
    r.status_code = status
    r.json.return_value = data
    return r


class _AsyncClientCM:
    """Async context manager that replays pre-set responses in order."""

    def __init__(self, responses: list) -> None:
        self._responses = iter(responses)
        self._client = AsyncMock()

    async def __aenter__(self):
        self._client.get = AsyncMock(return_value=next(self._responses))
        return self._client

    async def __aexit__(self, *args):
        return False


# ---------------------------------------------------------------------------
# youtube_search
# ---------------------------------------------------------------------------

_SEARCH_ITEMS = {"items": [{"id": {"videoId": "vid001"}}, {"id": {"videoId": "vid002"}}]}

_VIDEOS_ITEMS = {
    "items": [
        {
            "id": "vid001",
            "snippet": {
                "title": "Python Tutorial",
                "channelTitle": "TechChan",
                "publishedAt": "2024-01-01T00:00:00Z",
                "thumbnails": {"medium": {"url": "https://img/001"}},
            },
            "contentDetails": {"duration": "PT10M"},  # 600s — passes 240s filter
        },
        {
            "id": "vid002",
            "snippet": {
                "title": "Short Clip",
                "channelTitle": "TechChan",
                "publishedAt": "2024-01-02T00:00:00Z",
                "thumbnails": {"medium": {"url": "https://img/002"}},
            },
            "contentDetails": {"duration": "PT3M"},  # 180s — fails 240s filter
        },
    ]
}


class TestYoutubeSearch(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.app = _make_app()

    def _patch_client(self, search_resp, videos_resp):
        cm = _AsyncClientCM([search_resp, videos_resp])
        return patch("tubemind.services.httpx.AsyncClient", return_value=cm)

    async def test_returns_videos_above_min_seconds(self):
        with self._patch_client(_make_http_response(_SEARCH_ITEMS), _make_http_response(_VIDEOS_ITEMS)):
            with patch.dict("os.environ", {"YOUTUBE_API_KEY": "fake"}):
                results = await self.app.youtube_search(
                    "python", max_videos=5, min_seconds=240, order="relevance"
                )
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].video_id, "vid001")
        self.assertEqual(results[0].duration_sec, 600)

    async def test_short_videos_filtered_out(self):
        with self._patch_client(_make_http_response(_SEARCH_ITEMS), _make_http_response(_VIDEOS_ITEMS)):
            with patch.dict("os.environ", {"YOUTUBE_API_KEY": "fake"}):
                results = await self.app.youtube_search(
                    "python", max_videos=5, min_seconds=240, order="relevance"
                )
        ids = [v.video_id for v in results]
        self.assertNotIn("vid002", ids)

    async def test_empty_search_returns_empty_list(self):
        empty_search = _make_http_response({"items": []})
        with self._patch_client(empty_search, _make_http_response({})):
            with patch.dict("os.environ", {"YOUTUBE_API_KEY": "fake"}):
                results = await self.app.youtube_search(
                    "xyzzy obscure", max_videos=5, min_seconds=240, order="relevance"
                )
        self.assertEqual(results, [])

    async def test_max_videos_cap_respected(self):
        # Build 5 search items and 5 video detail items all passing the filter
        many_ids = [f"v{i:03d}" for i in range(5)]
        search_data = {"items": [{"id": {"videoId": vid}} for vid in many_ids]}
        videos_data = {
            "items": [
                {
                    "id": vid,
                    "snippet": {
                        "title": f"Video {vid}",
                        "channelTitle": "Chan",
                        "publishedAt": "2024-01-01T00:00:00Z",
                        "thumbnails": {"medium": {"url": ""}},
                    },
                    "contentDetails": {"duration": "PT10M"},
                }
                for vid in many_ids
            ]
        }
        with self._patch_client(_make_http_response(search_data), _make_http_response(videos_data)):
            with patch.dict("os.environ", {"YOUTUBE_API_KEY": "fake"}):
                results = await self.app.youtube_search(
                    "test", max_videos=3, min_seconds=240, order="relevance"
                )
        self.assertLessEqual(len(results), 3)

    async def test_raises_on_search_api_error(self):
        bad = _make_http_response({"error": "quota exceeded"}, status=403)
        cm = _AsyncClientCM([bad])
        with patch("tubemind.services.httpx.AsyncClient", return_value=cm):
            with patch.dict("os.environ", {"YOUTUBE_API_KEY": "fake"}):
                with self.assertRaises(RuntimeError):
                    await self.app.youtube_search(
                        "test", max_videos=5, min_seconds=240, order="relevance"
                    )

    async def test_result_has_correct_watch_url(self):
        with self._patch_client(_make_http_response(_SEARCH_ITEMS), _make_http_response(_VIDEOS_ITEMS)):
            with patch.dict("os.environ", {"YOUTUBE_API_KEY": "fake"}):
                results = await self.app.youtube_search(
                    "python", max_videos=5, min_seconds=240, order="relevance"
                )
        self.assertIn("vid001", results[0].url)
        self.assertIn("youtube.com/watch", results[0].url)


# ---------------------------------------------------------------------------
# _fetch_transcript
# ---------------------------------------------------------------------------

class TestFetchTranscript(unittest.TestCase):
    def setUp(self):
        self.app = _make_app()
        # Isolate: skip TranscriptAPI and yt-dlp unless the test overrides them
        self.app._transcript_api_key = MagicMock(return_value=None)
        self.app._transcript_request_kwargs = MagicMock(return_value={})
        self.app._should_retry_transcript_error = MagicMock(return_value=False)
        self.app._describe_transcript_error = MagicMock(return_value="mocked error")
        self.app._fetch_transcript_with_transcriptapi = MagicMock(return_value=(None, "TranscriptAPI unavailable"))
        self.app._fetch_transcript_with_ytdlp = MagicMock(return_value=(None, "yt-dlp failed"))
        self.video = _video()

    def test_success_returns_segments_and_no_error(self):
        segments = [{"text": "Hello world", "start": 0.0, "duration": 2.0}]
        with patch("tubemind.services.YouTubeTranscriptApi.get_transcript", return_value=segments):
            result, err = TubeMindApp._fetch_transcript(self.app, self.video)
        self.assertEqual(result, segments)
        self.assertIsNone(err)

    def test_no_transcript_found_falls_back_to_any_language(self):
        from youtube_transcript_api import NoTranscriptFound

        generic_segments = [{"text": "Auto-caption", "start": 0.0, "duration": 1.0}]

        def _side_effect(*args, **kwargs):
            if kwargs.get("languages"):
                raise NoTranscriptFound("vid001", ["en"], {})
            return generic_segments

        with patch("tubemind.services.YouTubeTranscriptApi.get_transcript", side_effect=_side_effect):
            result, err = TubeMindApp._fetch_transcript(self.app, self.video)
        self.assertEqual(result, generic_segments)
        self.assertIsNone(err)

    def test_all_providers_fail_returns_none_and_error_string(self):
        with patch("tubemind.services.YouTubeTranscriptApi.get_transcript", side_effect=Exception("blocked")):
            result, err = TubeMindApp._fetch_transcript(self.app, self.video)
        self.assertIsNone(result)
        self.assertIsInstance(err, str)
        self.assertGreater(len(err), 0)

    def test_empty_segment_list_treated_as_failure(self):
        # An empty transcript is not useful — the function should not return it
        with patch("tubemind.services.YouTubeTranscriptApi.get_transcript", return_value=[]):
            result, err = TubeMindApp._fetch_transcript(self.app, self.video)
        # Empty list is falsy so the success branch is skipped
        self.assertIsNone(result)

    def test_transcriptapi_key_present_tries_it_first(self):
        segments = [{"text": "From TranscriptAPI", "start": 0.0, "duration": 1.0}]
        self.app._transcript_api_key = MagicMock(return_value="some-key")
        self.app._fetch_transcript_with_transcriptapi = MagicMock(return_value=(segments, None))

        with patch("tubemind.services.YouTubeTranscriptApi.get_transcript") as mock_yt:
            result, err = TubeMindApp._fetch_transcript(self.app, self.video)

        self.assertEqual(result, segments)
        self.assertIsNone(err)
        mock_yt.assert_not_called()

    def test_error_string_includes_provider_messages(self):
        self.app._fetch_transcript_with_ytdlp = MagicMock(return_value=(None, "yt-dlp: bot check"))
        with patch("tubemind.services.YouTubeTranscriptApi.get_transcript", side_effect=Exception("rate limit")):
            _, err = TubeMindApp._fetch_transcript(self.app, self.video)
        self.assertIn("yt-dlp", err)


if __name__ == "__main__":
    unittest.main()
