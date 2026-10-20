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

sentence = "what is the name of capital of france"

res = classify_sentence(sentence)
print(res)