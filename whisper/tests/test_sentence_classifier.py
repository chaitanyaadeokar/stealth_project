"""
test_sentence_classifier.py
---------------------------
Unit tests for the SentenceClassifier module (ModernBERT-base).
Runs fast and isolated using mock engines and interface verification.
"""

import unittest
import sys
import os

# Ensure whisper/ directory is in sys.path
_current_dir = os.path.dirname(os.path.abspath(__file__))
_whisper_dir = os.path.dirname(_current_dir)
if _whisper_dir not in sys.path:
    sys.path.insert(0, _whisper_dir)

from sentence_classifier import (
    SentenceClassifier,
    classify_sentence,
    set_classifier,
    DEFAULT_CATEGORIES,
)


class TestSentenceClassifier(unittest.TestCase):

    def setUp(self):
        # Deterministic mock engine for fast unit tests
        def mock_engine(sentence: str, categories: list):
            s = sentence.lower()
            scores = {cat: 0.1 for cat in categories}
            if "?" in s or s.startswith("what") or s.startswith("how"):
                scores["question"] = 0.95
            elif s.startswith("search") or s.startswith("find"):
                scores["query"] = 0.90
            elif "will" in s or "best" in s or "better" in s:
                scores["claim"] = 0.88
            else:
                scores["fact"] = 0.85

            top = max(scores, key=scores.get)
            return {"top_category": top, "scores": scores}

        self.mock_classifier = SentenceClassifier(mock_engine=mock_engine)
        set_classifier(self.mock_classifier)

    def test_default_categories(self):
        self.assertEqual(DEFAULT_CATEGORIES, ["fact", "claim", "query", "question"])

    def test_return_structure(self):
        sentence = "what is the name of capital of france"
        res = classify_sentence(sentence)
        self.assertIn("text", res)
        self.assertIn("classifications", res)
        self.assertIn("top_category", res)
        self.assertIn("confidence", res)
        self.assertIn("scores", res)
        self.assertEqual(res["text"], sentence)
        self.assertIsInstance(res["classifications"], list)

    def test_question_classification(self):
        res = classify_sentence("What is the capital of France?")
        self.assertEqual(res["classifications"], ["question"])

    def test_query_classification(self):
        res = classify_sentence("find top 10 python libraries")
        self.assertEqual(res["classifications"], ["query"])

    def test_claim_classification(self):
        res = classify_sentence("Electric vehicles will completely replace petrol cars by 2030.")
        self.assertEqual(res["classifications"], ["claim"])

    def test_fact_classification(self):
        res = classify_sentence("Water freezes at zero degrees Celsius.")
        self.assertEqual(res["classifications"], ["fact"])

    def test_empty_string(self):
        res = classify_sentence("   ")
        self.assertEqual(res["classifications"], [])
        self.assertIsNone(res["top_category"])

    def test_custom_categories(self):
        custom_cats = ["urgent", "normal", "spam"]

        def custom_engine(sentence: str, categories: list):
            return {"top_category": "urgent", "scores": {"urgent": 0.99, "normal": 0.01, "spam": 0.0}}

        classifier = SentenceClassifier(mock_engine=custom_engine)
        res = classifier.classify("Server is down!", categories=custom_cats)
        self.assertEqual(res["classifications"], ["urgent"])

    def test_batch_classification(self):
        sentences = [
            "What time is it?",
            "search for nearest coffee shop",
            "Python was created by Guido van Rossum.",
        ]
        results = self.mock_classifier.classify_batch(sentences)
        self.assertEqual(len(results), 3)


if __name__ == "__main__":
    unittest.main()
