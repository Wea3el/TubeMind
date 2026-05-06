"""Unit tests for the RetrievalQuality structured-output model and fallback logic.

Tests the Pydantic model introduced in Sprint 7 for the retrieval quality gate
in services.py.  All tests run without any network calls or OpenAI API keys.
"""

import unittest

from pydantic import ValidationError

from tubemind.services import RetrievalQuality


# ---------------------------------------------------------------------------
# RetrievalQuality model validation
# ---------------------------------------------------------------------------

class TestRetrievalQualityModel(unittest.TestCase):
    """RetrievalQuality is a Pydantic BaseModel with three required fields."""

    def test_all_true_flags(self):
        q = RetrievalQuality(is_relevant=True, has_enough_info=True, reasoning="good coverage")
        self.assertTrue(q.is_relevant)
        self.assertTrue(q.has_enough_info)
        self.assertEqual(q.reasoning, "good coverage")

    def test_all_false_flags(self):
        q = RetrievalQuality(is_relevant=False, has_enough_info=False, reasoning="no data found")
        self.assertFalse(q.is_relevant)
        self.assertFalse(q.has_enough_info)

    def test_relevant_but_insufficient(self):
        """Content is on-topic but doesn't have enough detail."""
        q = RetrievalQuality(is_relevant=True, has_enough_info=False, reasoning="partial answer only")
        self.assertTrue(q.is_relevant)
        self.assertFalse(q.has_enough_info)

    def test_irrelevant_but_info_present(self):
        """Edge case: chunks are off-topic but the answer field is non-empty."""
        q = RetrievalQuality(is_relevant=False, has_enough_info=True, reasoning="off-topic chunks")
        self.assertFalse(q.is_relevant)
        self.assertTrue(q.has_enough_info)

    def test_reasoning_is_string(self):
        q = RetrievalQuality(is_relevant=True, has_enough_info=True, reasoning="test")
        self.assertIsInstance(q.reasoning, str)

    def test_reasoning_empty_string_allowed(self):
        q = RetrievalQuality(is_relevant=False, has_enough_info=False, reasoning="")
        self.assertEqual(q.reasoning, "")

    def test_missing_is_relevant_raises(self):
        with self.assertRaises(ValidationError):
            RetrievalQuality(has_enough_info=True, reasoning="missing field")  # type: ignore[call-arg]

    def test_missing_has_enough_info_raises(self):
        with self.assertRaises(ValidationError):
            RetrievalQuality(is_relevant=True, reasoning="missing field")  # type: ignore[call-arg]

    def test_missing_reasoning_raises(self):
        with self.assertRaises(ValidationError):
            RetrievalQuality(is_relevant=True, has_enough_info=True)  # type: ignore[call-arg]

    def test_non_bool_is_relevant_coerced(self):
        """Pydantic coerces truthy int to bool."""
        q = RetrievalQuality(is_relevant=1, has_enough_info=0, reasoning="coercion test")  # type: ignore[arg-type]
        self.assertIs(q.is_relevant, True)
        self.assertIs(q.has_enough_info, False)


# ---------------------------------------------------------------------------
# Fallback quality-gate logic (no API key required)
# ---------------------------------------------------------------------------

class TestFallbackQualityLogic(unittest.TestCase):
    """Tests for the heuristic fallback used when the structured-output call fails.

    The fallback in _check_retrieval_quality returns a RetrievalQuality built from:
        sufficient = bool(
            chunks and answer
            and "not explicitly listed" not in answer.lower()
            and "not listed" not in answer.lower()
        )
    We test this logic in isolation via a helper that mirrors it exactly.
    """

    @staticmethod
    def _fallback_sufficient(chunks: list, answer: str) -> bool:
        return bool(
            chunks
            and answer
            and "not explicitly listed" not in answer.lower()
            and "not listed" not in answer.lower()
        )

    def test_good_answer_with_chunks(self):
        self.assertTrue(self._fallback_sufficient([{"content": "data"}], "The best headphones are..."))

    def test_empty_chunks_insufficient(self):
        self.assertFalse(self._fallback_sufficient([], "Great answer here"))

    def test_empty_answer_insufficient(self):
        self.assertFalse(self._fallback_sufficient([{"content": "data"}], ""))

    def test_hedge_phrase_not_explicitly_listed(self):
        answer = "The battery life is not explicitly listed in the reviewed sources."
        self.assertFalse(self._fallback_sufficient([{"content": "data"}], answer))

    def test_hedge_phrase_not_listed(self):
        answer = "Price is not listed for this product."
        self.assertFalse(self._fallback_sufficient([{"content": "data"}], answer))

    def test_hedge_phrase_case_insensitive(self):
        answer = "The release date is NOT EXPLICITLY LISTED anywhere."
        self.assertFalse(self._fallback_sufficient([{"content": "data"}], answer))

    def test_no_hedge_phrase_sufficient(self):
        answer = "Based on the reviews, the Sony WH-1000XM5 offers the best noise cancellation."
        self.assertTrue(self._fallback_sufficient([{"content": "data"}], answer))

    def test_multiple_chunks_sufficient(self):
        chunks = [{"content": "chunk1"}, {"content": "chunk2"}, {"content": "chunk3"}]
        self.assertTrue(self._fallback_sufficient(chunks, "Solid answer with evidence."))

    def test_both_empty_insufficient(self):
        self.assertFalse(self._fallback_sufficient([], ""))


if __name__ == "__main__":
    unittest.main()
