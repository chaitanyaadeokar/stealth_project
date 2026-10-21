"""
test_sentence_aggregation.py
----------------------------
Tests for sentence splitting and per-sentence score aggregation (classify_text).
"""

import os
import sys
import unittest

_whisper_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _whisper_dir not in sys.path:
    sys.path.insert(0, _whisper_dir)

from sentence_classifier import SentenceClassifier, split_sentences


def mock_engine(sentence, categories):
    s = sentence.lower()
    if "?" in s:
        scores = {"fact_claim": 0.05, "opinion_debate": 0.05, "question_query": 0.90}
    elif "think" in s or "terrible" in s:
        scores = {"fact_claim": 0.05, "opinion_debate": 0.90, "question_query": 0.05}
    else:
        scores = {"fact_claim": 0.90, "opinion_debate": 0.05, "question_query": 0.05}
    return {"scores": {c: scores[c] for c in categories}}


class TestSplitSentences(unittest.TestCase):
    def test_basic_split(self):
        out = split_sentences("Inflation rose five percent. I think that is terrible. Why did it happen?")
        self.assertEqual(len(out), 3)

    def test_abbreviation_not_split(self):
        out = split_sentences("Dr. Smith said the U.S. economy grew. It was fast.")
        self.assertEqual(len(out), 2)

    def test_empty(self):
        self.assertEqual(split_sentences("   "), [])


class TestClassifyText(unittest.TestCase):
    def setUp(self):
        self.clf = SentenceClassifier(mock_engine=mock_engine)

    def test_per_sentence_results_and_aggregate(self):
        text = (
            "SPEAKER_00: The unemployment rate rose to five percent last year. "
            "I think that is terrible for families. Why did it happen?"
        )
        res = self.clf.classify_text(text)
        self.assertEqual(len(res["sentences"]), 3)
        self.assertEqual(
            [s["top_category"] for s in res["sentences"]],
            ["fact_claim", "opinion_debate", "question_query"],
        )
        self.assertAlmostEqual(sum(res["scores"].values()), 1.0, places=2)
        # Longest sentence (the fact) dominates the weighted mean
        self.assertEqual(res["top_category"], "fact_claim")
        self.assertEqual(sum(res["track_counts"].values()), 3)

    def test_short_backchannels_excluded(self):
        res = self.clf.classify_text("Yeah. The deficit grew by two billion dollars this year.")
        self.assertFalse(res["sentences"][0]["included"])
        self.assertEqual(res["top_category"], "fact_claim")

    def test_empty(self):
        res = self.clf.classify_text("")
        self.assertEqual(res["sentences"], [])
        self.assertIsNone(res["top_category"])


if __name__ == "__main__":
    unittest.main()
