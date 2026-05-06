"""Response-time and concurrency tests for multi-board usage.

What is tested (no real APIs needed):
  - Concurrent DB reads: list_boards + get_board_for_user from N threads
  - Concurrent DB writes: create_board_note from N threads on different boards
  - Concurrent youtube_search: asyncio.gather over mocked HTTP verifies that
    parallel calls finish faster than the same calls run sequentially
  - Board isolation under concurrency: writes to board A must never appear
    when reading board B while both run simultaneously

What is NOT tested here:
  - LightRAG indexing latency (requires real filesystem + model)
  - OpenAI response times (requires real API key and network)
  These would just measure mock speed, which is meaningless.

Timing assertions use generous ceilings and relative comparisons
(concurrent < sequential) to avoid flakiness on slow CI machines.
"""

import asyncio
import time
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor, as_completed
from unittest.mock import AsyncMock, MagicMock, patch

from tubemind import auth
from tubemind.models import YouTubeVideo
from tubemind.services import TubeMindApp

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

NUM_BOARDS = 8   # boards created per test
NUM_THREADS = 8  # concurrent workers


def _make_app() -> TubeMindApp:
    with patch.object(TubeMindApp, "__init__", return_value=None):
        app = TubeMindApp.__new__(TubeMindApp)
    return app


def _insert_board(user_id: str, title: str, updated_at: int = 1_000_000) -> int:
    row = auth.boards_table.insert(dict(
        user_id=user_id,
        title=title,
        summary="",
        topic_anchor="",
        status="idle",
        created_at=updated_at,
        updated_at=updated_at,
        last_question_at=0,
    ))
    return int(row["id"] if isinstance(row, dict) else row)


def _delete_boards(board_ids: list[int]) -> None:
    for bid in board_ids:
        auth.boards_table.delete_where("id = ?", [bid])


def _delete_notes_for_boards(board_ids: list[int]) -> None:
    for bid in board_ids:
        auth.board_notes_table.delete_where("board_id = ?", [bid])


# ---------------------------------------------------------------------------
# 1. Concurrent DB reads
# ---------------------------------------------------------------------------

class TestConcurrentBoardReads(unittest.TestCase):
    """N threads reading board data simultaneously must all succeed and be fast."""

    def setUp(self):
        self.user_id = "perf-user-reads"
        self.board_ids = [
            _insert_board(self.user_id, f"Board {i}", updated_at=1_000_000 + i)
            for i in range(NUM_BOARDS)
        ]

    def tearDown(self):
        _delete_boards(self.board_ids)

    def test_concurrent_list_boards_all_succeed(self):
        errors = []

        def _list(_):
            try:
                result = auth.list_boards(self.user_id)
                assert len(result) >= NUM_BOARDS
            except Exception as e:
                errors.append(e)

        with ThreadPoolExecutor(max_workers=NUM_THREADS) as pool:
            list(pool.map(_list, range(NUM_THREADS)))

        self.assertEqual(errors, [], f"Errors during concurrent reads: {errors}")

    def test_concurrent_get_board_for_user_all_succeed(self):
        errors = []

        def _get(board_id):
            try:
                result = auth.get_board_for_user(self.user_id, board_id)
                assert result is not None
                assert result["id"] == board_id
            except Exception as e:
                errors.append(e)

        with ThreadPoolExecutor(max_workers=NUM_THREADS) as pool:
            list(pool.map(_get, self.board_ids))

        self.assertEqual(errors, [], f"Errors during concurrent get_board_for_user: {errors}")

    def test_concurrent_reads_complete_within_time_budget(self):
        """N concurrent reads should finish well under 2 seconds."""
        start = time.perf_counter()

        with ThreadPoolExecutor(max_workers=NUM_THREADS) as pool:
            futures = [pool.submit(auth.list_boards, self.user_id) for _ in range(NUM_THREADS * 4)]
            for f in as_completed(futures):
                f.result()  # raise if any failed

        elapsed = time.perf_counter() - start
        self.assertLess(elapsed, 2.0, f"Concurrent reads took {elapsed:.2f}s — too slow")

    def test_concurrent_reads_return_consistent_data(self):
        """Every thread must see the same boards regardless of execution order."""
        # Note: SQLite in-process reads are sub-millisecond — thread overhead
        # dominates, so a "concurrent faster than sequential" assertion would
        # always fail. We assert correctness instead.
        results: list[list] = []
        lock = threading.Lock()

        def _list(_):
            boards = auth.list_boards(self.user_id)
            with lock:
                results.append([b["id"] for b in boards])

        with ThreadPoolExecutor(max_workers=NUM_THREADS) as pool:
            list(pool.map(_list, range(NUM_THREADS)))

        expected_ids = sorted(self.board_ids)
        for thread_result in results:
            self.assertEqual(sorted(thread_result), expected_ids)


# ---------------------------------------------------------------------------
# 2. Concurrent DB writes
# ---------------------------------------------------------------------------

class TestConcurrentBoardWrites(unittest.TestCase):
    """N threads writing notes to different boards must not corrupt data."""

    def setUp(self):
        self.user_id = "perf-user-writes"
        self.board_ids = [
            _insert_board(self.user_id, f"Write Board {i}")
            for i in range(NUM_BOARDS)
        ]

    def tearDown(self):
        _delete_notes_for_boards(self.board_ids)
        _delete_boards(self.board_ids)

    def test_concurrent_note_writes_all_succeed(self):
        errors = []

        def _write(board_id):
            try:
                auth.create_board_note(
                    board_id,
                    question=f"Q for board {board_id}",
                    answer=f"A for board {board_id}",
                    query_mode="mix",
                )
            except Exception as e:
                errors.append(e)

        with ThreadPoolExecutor(max_workers=NUM_THREADS) as pool:
            list(pool.map(_write, self.board_ids))

        self.assertEqual(errors, [], f"Errors during concurrent writes: {errors}")

    def test_concurrent_writes_produce_correct_note_counts(self):
        notes_per_board = 3

        def _write_notes(board_id):
            for i in range(notes_per_board):
                auth.create_board_note(
                    board_id,
                    question=f"Q{i}",
                    answer=f"A{i}",
                    query_mode="mix",
                )

        with ThreadPoolExecutor(max_workers=NUM_THREADS) as pool:
            list(pool.map(_write_notes, self.board_ids))

        for board_id in self.board_ids:
            notes = auth.list_board_notes(board_id)
            self.assertEqual(
                len(notes), notes_per_board,
                f"Board {board_id} has {len(notes)} notes, expected {notes_per_board}",
            )

    def test_concurrent_writes_complete_within_time_budget(self):
        start = time.perf_counter()

        def _write(board_id):
            auth.create_board_note(board_id, "Q", "A", "mix")

        with ThreadPoolExecutor(max_workers=NUM_THREADS) as pool:
            list(pool.map(_write, self.board_ids))

        elapsed = time.perf_counter() - start
        self.assertLess(elapsed, 3.0, f"Concurrent writes took {elapsed:.2f}s — too slow")


# ---------------------------------------------------------------------------
# 3. Board isolation under concurrency
# ---------------------------------------------------------------------------

class TestBoardIsolationUnderConcurrency(unittest.TestCase):
    """Writes to board A must never appear when reading board B."""

    def setUp(self):
        self.user_id = "perf-user-isolation"
        self.board_a = _insert_board(self.user_id, "Board A")
        self.board_b = _insert_board(self.user_id, "Board B")

    def tearDown(self):
        _delete_notes_for_boards([self.board_a, self.board_b])
        _delete_boards([self.board_a, self.board_b])

    def test_notes_stay_scoped_to_their_board_under_concurrent_writes(self):
        barrier = threading.Barrier(2)

        def _write_a():
            barrier.wait()
            for i in range(10):
                auth.create_board_note(self.board_a, f"A-Q{i}", f"A-A{i}", "mix")

        def _write_b():
            barrier.wait()
            for i in range(10):
                auth.create_board_note(self.board_b, f"B-Q{i}", f"B-A{i}", "mix")

        t1 = threading.Thread(target=_write_a)
        t2 = threading.Thread(target=_write_b)
        t1.start(); t2.start()
        t1.join(); t2.join()

        notes_a = auth.list_board_notes(self.board_a)
        notes_b = auth.list_board_notes(self.board_b)

        # No board B notes must appear in board A's list and vice-versa
        for note in notes_a:
            self.assertEqual(note["board_id"], self.board_a)
        for note in notes_b:
            self.assertEqual(note["board_id"], self.board_b)

        self.assertEqual(len(notes_a), 10)
        self.assertEqual(len(notes_b), 10)

    def test_wrong_user_still_blocked_under_concurrent_access(self):
        """Ownership checks must hold even while another user reads the same board."""
        other_user = "perf-user-intruder"
        results = {}
        barrier = threading.Barrier(2)

        def _owner_read():
            barrier.wait()
            results["owner"] = auth.get_board_for_user(self.user_id, self.board_a)

        def _intruder_read():
            barrier.wait()
            results["intruder"] = auth.get_board_for_user(other_user, self.board_a)

        t1 = threading.Thread(target=_owner_read)
        t2 = threading.Thread(target=_intruder_read)
        t1.start(); t2.start()
        t1.join(); t2.join()

        self.assertIsNotNone(results["owner"])
        self.assertIsNone(results["intruder"])


# ---------------------------------------------------------------------------
# 4. Concurrent youtube_search (async, mocked HTTP)
# ---------------------------------------------------------------------------

# youtube_search makes two async with httpx.AsyncClient blocks:
#   1. GET YOUTUBE_SEARCH_URL (possibly twice if first variant doesn't fill quota)
#   2. GET YOUTUBE_VIDEOS_URL
# Supply enough search items to hit the early-break threshold so only one
# search variant call is made, then route by URL in the mock.
_SEARCH_RESP = {"items": [{"id": {"videoId": f"vid{i:03d}"}} for i in range(12)]}
_VIDEOS_RESP = {
    "items": [
        {
            "id": f"vid{i:03d}",
            "snippet": {
                "title": f"Video {i}",
                "channelTitle": "Chan",
                "publishedAt": "2024-01-01T00:00:00Z",
                "thumbnails": {"medium": {"url": ""}},
            },
            "contentDetails": {"duration": "PT10M"},
        }
        for i in range(12)
    ]
}


def _make_http_response(data: dict, status: int = 200) -> MagicMock:
    r = MagicMock()
    r.status_code = status
    r.json.return_value = data
    return r


class _AsyncClientCM:
    """Route GET calls to the correct canned response based on URL substring."""

    def __init__(self, delay: float = 0.0) -> None:
        self._delay = delay

    async def _get(self, url, **kwargs):
        if self._delay:
            await asyncio.sleep(self._delay)
        if "search" in url:
            return _make_http_response(_SEARCH_RESP)
        return _make_http_response(_VIDEOS_RESP)

    async def __aenter__(self):
        mock = AsyncMock()
        mock.get = self._get
        return mock

    async def __aexit__(self, *args):
        return False


class TestConcurrentYoutubeSearch(unittest.IsolatedAsyncioTestCase):
    """asyncio.gather over N searches must finish faster than awaiting them serially."""

    def _make_patched_client(self, delay: float = 0.05):
        """Return a context-manager that patches httpx.AsyncClient with a URL-routed mock."""
        def _factory(**kwargs):
            return _AsyncClientCM(delay=delay)
        return patch("tubemind.services.httpx.AsyncClient", side_effect=_factory)

    async def _run_search(self, app: TubeMindApp) -> list:
        return await app.youtube_search("test", max_videos=5, min_seconds=60, order="relevance")

    async def test_concurrent_searches_all_return_results(self):
        app = _make_app()
        num_searches = 5

        with self._make_patched_client(), \
             patch.dict("os.environ", {"YOUTUBE_API_KEY": "fake"}):
            results = await asyncio.gather(*[self._run_search(app) for _ in range(num_searches)])

        self.assertEqual(len(results), num_searches)
        for r in results:
            self.assertGreater(len(r), 0)

    async def test_concurrent_searches_faster_than_sequential(self):
        """
        Each mocked HTTP call has a 50 ms artificial delay.
        Sequential: num_searches * 2_calls * 50ms = ~500ms for 5 searches.
        Concurrent: asyncio.gather overlaps them — should finish in ~100ms.
        """
        app = _make_app()
        num_searches = 5
        delay = 0.05  # 50 ms per HTTP call

        with self._make_patched_client(delay=delay), \
             patch.dict("os.environ", {"YOUTUBE_API_KEY": "fake"}):

            # Sequential baseline
            seq_start = time.perf_counter()
            for _ in range(num_searches):
                await self._run_search(app)
            seq_elapsed = time.perf_counter() - seq_start

            # Concurrent
            con_start = time.perf_counter()
            await asyncio.gather(*[self._run_search(app) for _ in range(num_searches)])
            con_elapsed = time.perf_counter() - con_start

        self.assertLess(
            con_elapsed, seq_elapsed * 0.75,
            f"Concurrent ({con_elapsed:.3f}s) was not at least 25% faster than "
            f"sequential ({seq_elapsed:.3f}s) — asyncio parallelism may be broken",
        )

    async def test_concurrent_searches_complete_within_time_budget(self):
        """5 searches with 50 ms HTTP delay each should finish well under 1 second."""
        app = _make_app()

        with self._make_patched_client(delay=0.05), \
             patch.dict("os.environ", {"YOUTUBE_API_KEY": "fake"}):
            start = time.perf_counter()
            await asyncio.gather(*[self._run_search(app) for _ in range(5)])
            elapsed = time.perf_counter() - start

        self.assertLess(elapsed, 1.0, f"5 concurrent searches took {elapsed:.3f}s")

    async def test_one_slow_search_does_not_block_others(self):
        """A single slow board search must not serialize all other searches."""
        app = _make_app()

        call_times: list[float] = []

        async def _timed_search():
            start = time.perf_counter()
            await self._run_search(app)
            call_times.append(time.perf_counter() - start)

        with self._make_patched_client(delay=0.1), \
             patch.dict("os.environ", {"YOUTUBE_API_KEY": "fake"}):
            await asyncio.gather(*[_timed_search() for _ in range(4)])

        # If searches were serialized, last call would take ~4× the first.
        # Concurrent execution keeps all times roughly equal.
        self.assertLess(
            max(call_times) / min(call_times), 3.0,
            f"Search times diverged too much: {[f'{t:.3f}' for t in call_times]}",
        )


if __name__ == "__main__":
    unittest.main()
