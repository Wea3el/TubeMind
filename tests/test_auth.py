"""Unit tests for security-sensitive auth.py helpers.

Uses a real SQLite database (pointed at a temp dir by conftest.py) rather than
mocking the DB layer — this catches query logic bugs that mocks would hide.
Every test inserts its own rows and deletes them in tearDown so tests are
isolated from each other and from the running app's data.
"""

import unittest

from tubemind import auth


class TestGetBoardForUser(unittest.TestCase):
    """get_board_for_user is the ownership gate for every board URL."""

    def setUp(self):
        self.user_a = "test-user-getboard-a"
        self.user_b = "test-user-getboard-b"
        ts = 1_000_000
        row = auth.boards_table.insert(
            dict(
                user_id=self.user_a,
                title="Owned Board",
                summary="",
                topic_anchor="",
                status="idle",
                created_at=ts,
                updated_at=ts,
                last_question_at=0,
            )
        )
        self.board_id = int(row["id"] if isinstance(row, dict) else row)

    def tearDown(self):
        auth.boards_table.delete_where("id = ?", [self.board_id])

    def test_correct_user_gets_board(self):
        result = auth.get_board_for_user(self.user_a, self.board_id)
        self.assertIsNotNone(result)
        self.assertEqual(result["id"], self.board_id)
        self.assertEqual(result["user_id"], self.user_a)

    def test_wrong_user_gets_none(self):
        # Cross-user access must be blocked — board IDs are integers in URLs
        result = auth.get_board_for_user(self.user_b, self.board_id)
        self.assertIsNone(result)

    def test_none_board_id_returns_none(self):
        result = auth.get_board_for_user(self.user_a, None)
        self.assertIsNone(result)

    def test_zero_board_id_returns_none(self):
        result = auth.get_board_for_user(self.user_a, 0)
        self.assertIsNone(result)

    def test_nonexistent_board_returns_none(self):
        result = auth.get_board_for_user(self.user_a, 999_999_999)
        self.assertIsNone(result)


class TestCreateBoard(unittest.TestCase):
    """create_board must persist a row owned by the requesting user."""

    def setUp(self):
        self.user_id = "test-user-createboard"
        self.created_ids: list[int] = []

    def tearDown(self):
        for bid in self.created_ids:
            auth.boards_table.delete_where("id = ?", [bid])

    def _create(self, title="Test Board", topic_anchor="", summary="") -> dict:
        board = auth.create_board(self.user_id, title, topic_anchor=topic_anchor, summary=summary)
        self.created_ids.append(board["id"])
        return board

    def test_returns_dict_with_expected_fields(self):
        board = self._create("My Board")
        self.assertIn("id", board)
        self.assertIn("user_id", board)
        self.assertIn("title", board)

    def test_user_id_matches_requester(self):
        board = self._create("My Board")
        self.assertEqual(board["user_id"], self.user_id)

    def test_title_persisted(self):
        board = self._create("Machine Learning")
        self.assertEqual(board["title"], "Machine Learning")

    def test_default_status_is_idle(self):
        board = self._create()
        self.assertEqual(board["status"], "idle")

    def test_timestamps_set(self):
        board = self._create()
        self.assertGreater(board["created_at"], 0)
        self.assertGreater(board["updated_at"], 0)

    def test_board_retrievable_after_create(self):
        board = self._create("Retrievable")
        fetched = auth.get_board_for_user(self.user_id, board["id"])
        self.assertIsNotNone(fetched)
        self.assertEqual(fetched["title"], "Retrievable")


class TestListBoards(unittest.TestCase):
    """list_boards must return only the requesting user's boards, newest first."""

    def setUp(self):
        self.user_id = "test-user-listboards"
        self.other_user = "test-user-listboards-other"
        self.created_ids: list[int] = []

    def tearDown(self):
        for bid in self.created_ids:
            auth.boards_table.delete_where("id = ?", [bid])

    def _insert(self, user_id: str, title: str, updated_at: int) -> int:
        row = auth.boards_table.insert(
            dict(
                user_id=user_id,
                title=title,
                summary="",
                topic_anchor="",
                status="idle",
                created_at=updated_at,
                updated_at=updated_at,
                last_question_at=0,
            )
        )
        bid = int(row["id"] if isinstance(row, dict) else row)
        self.created_ids.append(bid)
        return bid

    def test_returns_only_own_boards(self):
        self._insert(self.user_id, "Mine", 2000)
        self._insert(self.other_user, "Not Mine", 3000)
        results = auth.list_boards(self.user_id)
        titles = [r["title"] for r in results]
        self.assertIn("Mine", titles)
        self.assertNotIn("Not Mine", titles)

    def test_ordered_by_updated_at_desc(self):
        self._insert(self.user_id, "Older", 1000)
        self._insert(self.user_id, "Newer", 9000)
        results = auth.list_boards(self.user_id)
        own = [r for r in results if r["title"] in ("Older", "Newer")]
        self.assertEqual(own[0]["title"], "Newer")
        self.assertEqual(own[1]["title"], "Older")

    def test_empty_for_unknown_user(self):
        results = auth.list_boards("no-such-user-xyz")
        self.assertEqual(results, [])


class TestNoteOwnership(unittest.TestCase):
    """get_note_for_user must enforce board ownership before returning a note."""

    def setUp(self):
        self.owner = "test-user-noteowner"
        self.intruder = "test-user-noteintruder"
        ts = 1_000_000
        board_row = auth.boards_table.insert(
            dict(
                user_id=self.owner,
                title="Note Board",
                summary="",
                topic_anchor="",
                status="idle",
                created_at=ts,
                updated_at=ts,
                last_question_at=0,
            )
        )
        self.board_id = int(board_row["id"] if isinstance(board_row, dict) else board_row)
        note_row = auth.board_notes_table.insert(
            dict(
                board_id=self.board_id,
                question="What is RAG?",
                answer="Retrieval-augmented generation.",
                query_mode="mix",
                created_at=ts,
            )
        )
        self.note_id = int(note_row["id"] if isinstance(note_row, dict) else note_row)

    def tearDown(self):
        auth.board_notes_table.delete_where("id = ?", [self.note_id])
        auth.boards_table.delete_where("id = ?", [self.board_id])

    def test_owner_gets_note(self):
        note = auth.get_note_for_user(self.owner, self.note_id)
        self.assertIsNotNone(note)
        self.assertEqual(note["id"], self.note_id)

    def test_intruder_gets_none(self):
        note = auth.get_note_for_user(self.intruder, self.note_id)
        self.assertIsNone(note)

    def test_nonexistent_note_returns_none(self):
        note = auth.get_note_for_user(self.owner, 999_999_999)
        self.assertIsNone(note)


if __name__ == "__main__":
    unittest.main()
