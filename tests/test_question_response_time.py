"""Response-time tests for the answer_question pipeline.

answer_question is a multi-stage orchestrator. Its three distinct paths are:

  Path A — First question on an empty board (slowest):
    no topic-fit check → initial LightRAG query (empty) → plan research →
    expand corpus (YouTube + transcripts) → second LightRAG query → save note

  Path B — Follow-up on a board that already has evidence (medium):
    topic-fit check → initial LightRAG query (hits) → plan research (no new
    queries needed) → save note  (skips corpus expansion entirely)

  Path C — Off-topic question (fastest):
    topic-fit check says no → early return, nothing else runs

All external calls (OpenAI, LightRAG, YouTube) are replaced with AsyncMocks
carrying small artificial delays so we can measure the pipeline overhead and
verify that:
  - each path finishes within a sensible wall-clock budget
  - faster paths genuinely beat slower ones
  - two boards questioned concurrently finish together, not sequentially
"""

import asyncio
import threading
import time
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from tubemind import auth
from tubemind.models import BoardWorkspace
from tubemind.services import TubeMindApp

# ---------------------------------------------------------------------------
# Canned data
# ---------------------------------------------------------------------------

_FAKE_CHUNK = {
    "content": "The video explains that machine learning requires large datasets.",
    "video_id": "vid001",
    "title": "ML Tutorial",
    "source_url": "https://www.youtube.com/watch?v=vid001",
    "embed_url": "https://www.youtube.com/embed/vid001",
    "start_seconds": 30.0,
    "start_label": "0:30",
}
_FULL_RESULT  = {"answer": "Machine learning requires quality data.", "chunks": [_FAKE_CHUNK]}
_EMPTY_RESULT = {"answer": "", "chunks": []}

# Artificial delays that roughly model real API latency (ms scale for tests)
_DELAY_LLM     = 0.08   # _assess_topic_fit / _plan_research / _refresh_board_summary
_DELAY_QUERY   = 0.10   # _query_board  (LightRAG → OpenAI internally)
_DELAY_EXPAND  = 0.25   # _expand_board_corpus  (YouTube search + transcripts + index)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_app(user_id: str) -> TubeMindApp:
    """Build a TubeMindApp without running __init__ side-effects."""
    with patch.object(TubeMindApp, "__init__", return_value=None):
        app = TubeMindApp.__new__(TubeMindApp)
    app.user_id = user_id
    app.lock = threading.RLock()
    return app


def _attach_mocks(
    app: TubeMindApp,
    *,
    topic_fit: dict | None = None,
    initial_result: dict | None = None,
    final_result: dict | None = None,
    plan_queries: list | None = None,
) -> None:
    """Wire every heavy async sub-method with configurable canned responses."""

    async def _assess(*_a, **_kw):
        await asyncio.sleep(_DELAY_LLM)
        return topic_fit or {"is_fit": True, "warning": ""}

    async def _query(board_id, runtime, question, mode, *, allow_empty):
        await asyncio.sleep(_DELAY_QUERY)
        return (initial_result or _EMPTY_RESULT) if allow_empty else (final_result or _FULL_RESULT)

    async def _plan(*_a, **_kw):
        await asyncio.sleep(_DELAY_LLM)
        return {"queries": plan_queries if plan_queries is not None else []}

    async def _expand(*_a, **_kw):
        await asyncio.sleep(_DELAY_EXPAND)

    async def _refresh(*_a, **_kw):
        await asyncio.sleep(_DELAY_LLM)

    app._assess_topic_fit      = _assess
    app._get_board_runtime     = AsyncMock(return_value=MagicMock())
    app._query_board           = _query
    app._plan_research         = _plan
    app._expand_board_corpus   = _expand
    app._refresh_board_summary = _refresh
    # build_workspace is a sync DB-only method — run it for real so the
    # returned workspace carries the correct board_id, notes, and warnings.


def _insert_board(user_id: str, title: str = "Test Board") -> dict:
    row = auth.boards_table.insert(dict(
        user_id=user_id,
        title=title,
        summary="",
        topic_anchor="machine learning",
        status="idle",
        created_at=1_000_000,
        updated_at=1_000_000,
        last_question_at=0,
    ))
    board_id = int(row["id"] if isinstance(row, dict) else row)
    return auth.get_board_for_user(user_id, board_id)


def _insert_note(board_id: int) -> dict:
    return auth.create_board_note(board_id, "What is ML?", "ML is ...", "mix")


def _cleanup(user_id: str, board_ids: list[int]) -> None:
    for bid in board_ids:
        auth.board_note_chunks_table.delete_where(
            "note_id IN (SELECT id FROM board_notes WHERE board_id = ?)", [bid]
        )
        auth.board_queries_table.delete_where("board_id = ?", [bid])
        auth.board_notes_table.delete_where("board_id = ?", [bid])
        auth.boards_table.delete_where("id = ?", [bid])
    auth.users_table.delete_where("id = ?", [user_id])


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestQuestionResponseTime(unittest.IsolatedAsyncioTestCase):

    def setUp(self):
        self.user_id = "qa-perf-user"
        self._created_board_ids: list[int] = []
        # set_active_board calls users_table[user_id] and raises NotFoundError
        # if the row is absent, so we need a real user row for every test.
        auth.users_table.upsert(
            {"id": self.user_id, "email": "test@test.com", "name": "Test", "picture": "", "active_board_id": None},
            pk="id",
        )

    def tearDown(self):
        _cleanup(self.user_id, self._created_board_ids)

    def _track(self, board: dict) -> dict:
        self._created_board_ids.append(int(board["id"]))
        return board

    # -----------------------------------------------------------------------
    # Path C — off-topic early exit (fastest)
    # -----------------------------------------------------------------------

    async def test_off_topic_question_returns_quickly(self):
        """An off-topic question must be rejected after one LLM call (~_DELAY_LLM)."""
        board = self._track(_insert_board(self.user_id))
        _insert_note(int(board["id"]))  # needs notes so topic-fit runs

        app = _make_app(self.user_id)
        _attach_mocks(app, topic_fit={"is_fit": False, "warning": "Wrong topic"})

        start = time.perf_counter()
        workspace = await app.answer_question(int(board["id"]), "What is quantum computing?")
        elapsed = time.perf_counter() - start

        # Returns with a warning, no note created
        self.assertIsNotNone(workspace.warning)
        self.assertIn("Wrong topic", workspace.warning)
        # Should take only one LLM round-trip
        self.assertLess(elapsed, _DELAY_LLM * 4,
            f"Off-topic path took {elapsed:.3f}s — expected ~{_DELAY_LLM}s")

    async def test_empty_question_rejected_immediately(self):
        """An empty question string must raise before any I/O."""
        app = _make_app(self.user_id)
        _attach_mocks(app)

        start = time.perf_counter()
        with self.assertRaises(ValueError):
            await app.answer_question(None, "   ")
        elapsed = time.perf_counter() - start

        self.assertLess(elapsed, 0.05,
            f"Empty question took {elapsed:.3f}s — should be instant")

    # -----------------------------------------------------------------------
    # Path B — follow-up on existing board (medium)
    # -----------------------------------------------------------------------

    async def test_follow_up_question_completes_within_budget(self):
        """Follow-up on a board with corpus: topic-fit + query + plan + refresh."""
        board = self._track(_insert_board(self.user_id))
        _insert_note(int(board["id"]))

        app = _make_app(self.user_id)
        _attach_mocks(
            app,
            topic_fit={"is_fit": True, "warning": ""},
            initial_result=_FULL_RESULT,  # corpus already has the answer
            plan_queries=[],              # no new YouTube search needed
        )

        start = time.perf_counter()
        workspace = await app.answer_question(int(board["id"]), "How does ML use data?")
        elapsed = time.perf_counter() - start

        # Expected: assess + query + plan + refresh = 4 × _DELAY_LLM + _DELAY_QUERY
        expected_max = (_DELAY_LLM * 3 + _DELAY_QUERY) * 3
        self.assertLess(elapsed, expected_max,
            f"Follow-up path took {elapsed:.3f}s, budget was {expected_max:.3f}s")

    async def test_follow_up_skips_corpus_expansion(self):
        """When the initial query already returns chunks, _expand_board_corpus must not be called."""
        board = self._track(_insert_board(self.user_id))
        _insert_note(int(board["id"]))

        expand_called = []

        app = _make_app(self.user_id)
        _attach_mocks(
            app,
            initial_result=_FULL_RESULT,
            plan_queries=[],
        )

        async def _no_expand(*a, **kw):
            expand_called.append(True)

        app._expand_board_corpus = _no_expand

        await app.answer_question(int(board["id"]), "How does ML use data?")
        self.assertEqual(expand_called, [],
            "_expand_board_corpus was called when it shouldn't have been")

    # -----------------------------------------------------------------------
    # Path A — first question on empty board (slowest)
    # -----------------------------------------------------------------------

    async def test_first_question_empty_board_completes_within_budget(self):
        """First question: plan → expand corpus → re-query → save note."""
        app = _make_app(self.user_id)
        _attach_mocks(
            app,
            initial_result=_EMPTY_RESULT,
            plan_queries=[{"query": "machine learning data", "reason": "need evidence"}],
            final_result=_FULL_RESULT,
        )

        start = time.perf_counter()
        workspace = await app.answer_question(None, "What is machine learning?")
        elapsed = time.perf_counter() - start

        # Track the auto-created board for cleanup
        if workspace.active_board:
            bid = int(workspace.active_board.get("id", 0) or 0)
            if bid and bid not in self._created_board_ids:
                self._created_board_ids.append(bid)

        # Expected: plan + expand + query(×2) + refresh
        expected_max = (_DELAY_QUERY * 2 + _DELAY_LLM + _DELAY_EXPAND + _DELAY_LLM) * 3
        self.assertLess(elapsed, expected_max,
            f"First-question path took {elapsed:.3f}s, budget was {expected_max:.3f}s")

    async def test_first_question_creates_a_note(self):
        """answer_question must persist exactly one note on success."""
        app = _make_app(self.user_id)
        _attach_mocks(
            app,
            initial_result=_EMPTY_RESULT,
            plan_queries=[{"query": "deep learning intro", "reason": "need data"}],
            final_result=_FULL_RESULT,
        )

        workspace = await app.answer_question(None, "Explain deep learning.")
        if workspace.active_board:
            bid = int(workspace.active_board.get("id", 0) or 0)
            if bid and bid not in self._created_board_ids:
                self._created_board_ids.append(bid)
            notes = auth.list_board_notes(bid)
            self.assertEqual(len(notes), 1)
            self.assertEqual(notes[0]["question"], "Explain deep learning.")

    # -----------------------------------------------------------------------
    # Relative timing: slower paths must beat faster ones
    # -----------------------------------------------------------------------

    async def test_path_c_faster_than_path_b(self):
        """Off-topic rejection must be faster than a successful follow-up."""
        board_b = self._track(_insert_board(self.user_id, "Board B"))
        board_c = self._track(_insert_board(self.user_id, "Board C"))
        _insert_note(int(board_b["id"]))
        _insert_note(int(board_c["id"]))

        # Path B
        app_b = _make_app(self.user_id)
        _attach_mocks(app_b, topic_fit={"is_fit": True, "warning": ""}, initial_result=_FULL_RESULT, plan_queries=[])
        b_start = time.perf_counter()
        await app_b.answer_question(int(board_b["id"]), "How does ML use data?")
        path_b_time = time.perf_counter() - b_start

        # Path C
        app_c = _make_app(self.user_id)
        _attach_mocks(app_c, topic_fit={"is_fit": False, "warning": "Wrong topic"})
        c_start = time.perf_counter()
        await app_c.answer_question(int(board_c["id"]), "What is quantum computing?")
        path_c_time = time.perf_counter() - c_start

        self.assertLess(path_c_time, path_b_time,
            f"Path C ({path_c_time:.3f}s) should be faster than Path B ({path_b_time:.3f}s)")

    async def test_path_b_faster_than_path_a(self):
        """Follow-up (no corpus expansion) must be faster than first question (with expansion)."""
        board = self._track(_insert_board(self.user_id))
        _insert_note(int(board["id"]))

        # Path A on a new board (no board_id → auto-create)
        app_a = _make_app(self.user_id)
        _attach_mocks(
            app_a,
            initial_result=_EMPTY_RESULT,
            plan_queries=[{"query": "ml tutorial", "reason": "need data"}],
            final_result=_FULL_RESULT,
        )
        a_start = time.perf_counter()
        ws_a = await app_a.answer_question(None, "What is machine learning?")
        path_a_time = time.perf_counter() - a_start
        if ws_a.active_board:
            bid = int(ws_a.active_board.get("id", 0) or 0)
            if bid and bid not in self._created_board_ids:
                self._created_board_ids.append(bid)

        # Path B on existing board
        app_b = _make_app(self.user_id)
        _attach_mocks(app_b, initial_result=_FULL_RESULT, plan_queries=[])
        b_start = time.perf_counter()
        await app_b.answer_question(int(board["id"]), "How does ML use data?")
        path_b_time = time.perf_counter() - b_start

        self.assertLess(path_b_time, path_a_time,
            f"Path B ({path_b_time:.3f}s) should be faster than Path A ({path_a_time:.3f}s)")

    # -----------------------------------------------------------------------
    # Concurrency — two boards questioned simultaneously
    # -----------------------------------------------------------------------

    async def test_concurrent_questions_on_different_boards_both_succeed(self):
        """Two answer_question calls on different boards must both persist a note."""
        board1 = self._track(_insert_board(self.user_id, "Concurrent Board 1"))
        board2 = self._track(_insert_board(self.user_id, "Concurrent Board 2"))

        app1 = _make_app(self.user_id)
        _attach_mocks(app1, initial_result=_FULL_RESULT, plan_queries=[])

        app2 = _make_app(self.user_id)
        _attach_mocks(app2, initial_result=_FULL_RESULT, plan_queries=[])

        await asyncio.gather(
            app1.answer_question(int(board1["id"]), "What is supervised learning?"),
            app2.answer_question(int(board2["id"]), "What is unsupervised learning?"),
        )

        notes1 = auth.list_board_notes(int(board1["id"]))
        notes2 = auth.list_board_notes(int(board2["id"]))
        self.assertEqual(len(notes1), 1)
        self.assertEqual(len(notes2), 1)
        self.assertEqual(notes1[0]["question"], "What is supervised learning?")
        self.assertEqual(notes2[0]["question"], "What is unsupervised learning?")

    async def test_concurrent_questions_faster_than_sequential(self):
        """asyncio.gather on two boards must finish faster than running them back-to-back."""
        board1 = self._track(_insert_board(self.user_id, "Seq Board 1"))
        board2 = self._track(_insert_board(self.user_id, "Seq Board 2"))

        def _fresh_app():
            app = _make_app(self.user_id)
            _attach_mocks(
                app,
                initial_result=_EMPTY_RESULT,
                plan_queries=[{"query": "ml basics", "reason": "need data"}],
                final_result=_FULL_RESULT,
            )
            return app

        # Sequential baseline
        seq_start = time.perf_counter()
        await _fresh_app().answer_question(int(board1["id"]), "What is ML?")
        # clean up note before second run so board1 stays "empty board" path
        auth.board_notes_table.delete_where("board_id = ?", [int(board1["id"])])
        await _fresh_app().answer_question(int(board1["id"]), "What is ML?")
        seq_elapsed = time.perf_counter() - seq_start
        auth.board_notes_table.delete_where("board_id = ?", [int(board1["id"])])

        # Concurrent
        con_start = time.perf_counter()
        await asyncio.gather(
            _fresh_app().answer_question(int(board1["id"]), "What is ML?"),
            _fresh_app().answer_question(int(board2["id"]), "What is ML?"),
        )
        con_elapsed = time.perf_counter() - con_start

        self.assertLess(
            con_elapsed, seq_elapsed * 0.75,
            f"Concurrent ({con_elapsed:.3f}s) was not at least 25% faster than "
            f"sequential ({seq_elapsed:.3f}s) — asyncio parallelism may be broken",
        )

    async def test_concurrent_questions_notes_isolated_per_board(self):
        """Notes created concurrently must appear only in their own board."""
        board1 = self._track(_insert_board(self.user_id, "Isolation Board 1"))
        board2 = self._track(_insert_board(self.user_id, "Isolation Board 2"))

        app1 = _make_app(self.user_id)
        _attach_mocks(app1, initial_result=_FULL_RESULT, plan_queries=[])
        app2 = _make_app(self.user_id)
        _attach_mocks(app2, initial_result=_FULL_RESULT, plan_queries=[])

        await asyncio.gather(
            app1.answer_question(int(board1["id"]), "Q for board 1"),
            app2.answer_question(int(board2["id"]), "Q for board 2"),
        )

        for note in auth.list_board_notes(int(board1["id"])):
            self.assertEqual(note["board_id"], int(board1["id"]))
        for note in auth.list_board_notes(int(board2["id"])):
            self.assertEqual(note["board_id"], int(board2["id"]))


if __name__ == "__main__":
    unittest.main()
