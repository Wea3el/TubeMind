"""Unit tests for the multi-conversation session architecture.

Tests the new Board -> Sessions -> Notes hierarchy added in Sprint 6.
Uses a real temp SQLite DB (via conftest.py) — no mocking.
"""

import unittest

from tubemind import auth


class TestCreateSession(unittest.TestCase):
    """create_session must persist a row tied to the given board."""

    def setUp(self):
        self.user_id = "test-user-sessions-create"
        ts = 1_000_000
        row = auth.boards_table.insert(
            dict(user_id=self.user_id, title="Session Board", summary="",
                 topic_anchor="", status="idle", created_at=ts, updated_at=ts, last_question_at=0)
        )
        self.board_id = int(row["id"] if isinstance(row, dict) else row)
        self.created_session_ids: list[int] = []

    def tearDown(self):
        for sid in self.created_session_ids:
            auth.board_sessions_table.delete_where("id = ?", [sid])
        auth.boards_table.delete_where("id = ?", [self.board_id])

    def _create(self) -> dict:
        s = auth.create_session(self.board_id)
        self.created_session_ids.append(int(s["id"]))
        return s

    def test_returns_dict_with_id(self):
        s = self._create()
        self.assertIn("id", s)
        self.assertGreater(int(s["id"]), 0)

    def test_board_id_matches(self):
        s = self._create()
        self.assertEqual(int(s["board_id"]), self.board_id)

    def test_created_at_set(self):
        s = self._create()
        self.assertGreater(int(s["created_at"]), 0)

    def test_multiple_sessions_get_distinct_ids(self):
        s1 = self._create()
        s2 = self._create()
        self.assertNotEqual(int(s1["id"]), int(s2["id"]))


class TestListSessions(unittest.TestCase):
    """list_sessions must return sessions for the board, newest first."""

    def setUp(self):
        self.user_id = "test-user-sessions-list"
        ts = 1_000_000
        row = auth.boards_table.insert(
            dict(user_id=self.user_id, title="List Sessions Board", summary="",
                 topic_anchor="", status="idle", created_at=ts, updated_at=ts, last_question_at=0)
        )
        self.board_id = int(row["id"] if isinstance(row, dict) else row)
        self.created_session_ids: list[int] = []

    def tearDown(self):
        # Delete ALL sessions for this board — guards against stale sessions
        # created outside _insert_session (e.g. get_or_create_latest_session
        # called indirectly by other test paths sharing the same board_id).
        auth.board_sessions_table.delete_where("board_id = ?", [self.board_id])
        auth.boards_table.delete_where("id = ?", [self.board_id])

    def _insert_session(self, created_at: int) -> int:
        row = auth.board_sessions_table.insert(
            dict(board_id=self.board_id, created_at=created_at)
        )
        sid = int(row["id"] if isinstance(row, dict) else row)
        self.created_session_ids.append(sid)
        return sid

    def test_empty_for_board_with_no_sessions(self):
        result = auth.list_sessions(self.board_id)
        self.assertEqual(result, [])

    def test_returns_sessions_for_board(self):
        self._insert_session(1000)
        result = auth.list_sessions(self.board_id)
        self.assertEqual(len(result), 1)

    def test_ordered_newest_first(self):
        self._insert_session(1000)
        self._insert_session(9000)
        result = auth.list_sessions(self.board_id)
        self.assertGreater(int(result[0]["created_at"]), int(result[1]["created_at"]))

    def test_does_not_return_other_board_sessions(self):
        other_row = auth.boards_table.insert(
            dict(user_id=self.user_id, title="Other", summary="", topic_anchor="",
                 status="idle", created_at=1000, updated_at=1000, last_question_at=0)
        )
        other_board_id = int(other_row["id"] if isinstance(other_row, dict) else other_row)
        other_session = auth.board_sessions_table.insert(
            dict(board_id=other_board_id, created_at=5000)
        )
        other_sid = int(other_session["id"] if isinstance(other_session, dict) else other_session)
        try:
            result = auth.list_sessions(self.board_id)
            ids = [int(r["id"]) for r in result]
            self.assertNotIn(other_sid, ids)
        finally:
            auth.board_sessions_table.delete_where("id = ?", [other_sid])
            auth.boards_table.delete_where("id = ?", [other_board_id])


class TestGetOrCreateLatestSession(unittest.TestCase):
    """get_or_create_latest_session must return existing or create new."""

    def setUp(self):
        self.user_id = "test-user-sessions-getorcreate"
        ts = 1_000_000
        row = auth.boards_table.insert(
            dict(user_id=self.user_id, title="GetOrCreate Board", summary="",
                 topic_anchor="", status="idle", created_at=ts, updated_at=ts, last_question_at=0)
        )
        self.board_id = int(row["id"] if isinstance(row, dict) else row)

    def tearDown(self):
        auth.board_sessions_table.delete_where("board_id = ?", [self.board_id])
        auth.boards_table.delete_where("id = ?", [self.board_id])

    def test_creates_session_when_none_exist(self):
        s = auth.get_or_create_latest_session(self.board_id)
        self.assertIn("id", s)
        self.assertEqual(int(s["board_id"]), self.board_id)

    def test_returns_existing_session_without_creating_new(self):
        first = auth.get_or_create_latest_session(self.board_id)
        second = auth.get_or_create_latest_session(self.board_id)
        self.assertEqual(int(first["id"]), int(second["id"]))
        self.assertEqual(len(auth.list_sessions(self.board_id)), 1)


class TestListSessionNotes(unittest.TestCase):
    """list_session_notes must return only notes for the given session."""

    def setUp(self):
        self.user_id = "test-user-session-notes"
        ts = 1_000_000
        board_row = auth.boards_table.insert(
            dict(user_id=self.user_id, title="Session Notes Board", summary="",
                 topic_anchor="", status="idle", created_at=ts, updated_at=ts, last_question_at=0)
        )
        self.board_id = int(board_row["id"] if isinstance(board_row, dict) else board_row)
        s1 = auth.board_sessions_table.insert(dict(board_id=self.board_id, created_at=ts))
        s2 = auth.board_sessions_table.insert(dict(board_id=self.board_id, created_at=ts + 1))
        self.session_a_id = int(s1["id"] if isinstance(s1, dict) else s1)
        self.session_b_id = int(s2["id"] if isinstance(s2, dict) else s2)
        self.note_ids: list[int] = []

    def tearDown(self):
        for nid in self.note_ids:
            auth.board_notes_table.delete_where("id = ?", [nid])
        auth.board_sessions_table.delete_where("board_id = ?", [self.board_id])
        auth.boards_table.delete_where("id = ?", [self.board_id])

    def _insert_note(self, session_id: int, question: str) -> int:
        row = auth.board_notes_table.insert(
            dict(board_id=self.board_id, session_id=session_id,
                 question=question, answer="A", query_mode="mix", created_at=1_000_000)
        )
        nid = int(row["id"] if isinstance(row, dict) else row)
        self.note_ids.append(nid)
        return nid

    def test_returns_only_notes_for_given_session(self):
        self._insert_note(self.session_a_id, "Q for session A")
        self._insert_note(self.session_b_id, "Q for session B")
        result = auth.list_session_notes(self.board_id, self.session_a_id)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["question"], "Q for session A")

    def test_empty_when_session_has_no_notes(self):
        result = auth.list_session_notes(self.board_id, self.session_a_id)
        self.assertEqual(result, [])

    def test_ordered_chronologically(self):
        n1 = auth.board_notes_table.insert(
            dict(board_id=self.board_id, session_id=self.session_a_id,
                 question="First", answer="A", query_mode="mix", created_at=1000)
        )
        n2 = auth.board_notes_table.insert(
            dict(board_id=self.board_id, session_id=self.session_a_id,
                 question="Second", answer="A", query_mode="mix", created_at=9000)
        )
        self.note_ids.extend([
            int(n1["id"] if isinstance(n1, dict) else n1),
            int(n2["id"] if isinstance(n2, dict) else n2),
        ])
        result = auth.list_session_notes(self.board_id, self.session_a_id)
        self.assertEqual(result[0]["question"], "First")
        self.assertEqual(result[1]["question"], "Second")

    def test_session_isolation_across_boards(self):
        """Notes from a different board's session must not appear."""
        other_row = auth.boards_table.insert(
            dict(user_id=self.user_id, title="Other Board", summary="", topic_anchor="",
                 status="idle", created_at=1000, updated_at=1000, last_question_at=0)
        )
        other_board_id = int(other_row["id"] if isinstance(other_row, dict) else other_row)
        other_s = auth.board_sessions_table.insert(dict(board_id=other_board_id, created_at=1000))
        other_sid = int(other_s["id"] if isinstance(other_s, dict) else other_s)
        other_note = auth.board_notes_table.insert(
            dict(board_id=other_board_id, session_id=other_sid,
                 question="Other board Q", answer="A", query_mode="mix", created_at=1000)
        )
        other_nid = int(other_note["id"] if isinstance(other_note, dict) else other_note)
        try:
            result = auth.list_session_notes(self.board_id, other_sid)
            self.assertEqual(result, [])
        finally:
            auth.board_notes_table.delete_where("id = ?", [other_nid])
            auth.board_sessions_table.delete_where("id = ?", [other_sid])
            auth.boards_table.delete_where("id = ?", [other_board_id])


class TestCreateBoardNoteWithSession(unittest.TestCase):
    """create_board_note must persist session_id when provided."""

    def setUp(self):
        self.user_id = "test-user-note-session-field"
        ts = 1_000_000
        board_row = auth.boards_table.insert(
            dict(user_id=self.user_id, title="Note Session Field Board", summary="",
                 topic_anchor="", status="idle", created_at=ts, updated_at=ts, last_question_at=0)
        )
        self.board_id = int(board_row["id"] if isinstance(board_row, dict) else board_row)
        s = auth.board_sessions_table.insert(dict(board_id=self.board_id, created_at=ts))
        self.session_id = int(s["id"] if isinstance(s, dict) else s)
        self.note_ids: list[int] = []

    def tearDown(self):
        for nid in self.note_ids:
            auth.board_notes_table.delete_where("id = ?", [nid])
        auth.board_sessions_table.delete_where("board_id = ?", [self.board_id])
        auth.boards_table.delete_where("id = ?", [self.board_id])

    def test_session_id_persisted(self):
        note = auth.create_board_note(self.board_id, "Q?", "A.", "mix", session_id=self.session_id)
        self.note_ids.append(int(note["id"]))
        self.assertEqual(int(note["session_id"]), self.session_id)

    def test_session_id_none_when_omitted(self):
        note = auth.create_board_note(self.board_id, "Q?", "A.", "mix")
        self.note_ids.append(int(note["id"]))
        self.assertIsNone(note.get("session_id"))

    def test_note_retrievable_via_list_session_notes(self):
        note = auth.create_board_note(self.board_id, "Q?", "A.", "mix", session_id=self.session_id)
        self.note_ids.append(int(note["id"]))
        results = auth.list_session_notes(self.board_id, self.session_id)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["question"], "Q?")


if __name__ == "__main__":
    unittest.main()
