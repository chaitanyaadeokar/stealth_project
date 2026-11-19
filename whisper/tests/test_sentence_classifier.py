"""Offline classifier contracts: no model downloads or network access."""
import pytest
import sentence_classifier as module
from sentence_classifier import SentenceClassifier


def engine(text, categories):
    return {"scores": {label: 0.9 if label == "question" else 0.1 for label in categories}}


def test_mock_classification_and_batch():
    classifier = SentenceClassifier(mock_engine=engine)
    assert classifier.is_loaded()
    result = classifier.classify("  Why? ")
    assert result["text"] == "  Why? "
    assert result["top_category"] == "question" and result["confidence"] == 0.9
    assert len(classifier.classify_batch(["Why?", "How?"])) == 2


def test_mock_multi_label_matches_pipeline_contract():
    classifier = SentenceClassifier(mock_engine=lambda *args: {"scores": {"claim": 0.8, "opinion": 0.6}})
    assert classifier.classify("Both", multi_label=True)["classifications"] == ["claim", "opinion"]


def test_empty_never_loads_model(monkeypatch):
    classifier = SentenceClassifier()
    monkeypatch.setattr(classifier, "load_model", lambda: pytest.fail("Unexpected model load"))
    assert classifier.classify("")["classifications"] == []
    assert not classifier.is_loaded()


def test_custom_categories_and_singleton(monkeypatch):
    classifier = SentenceClassifier(mock_engine=lambda text, categories: {"scores": {categories[0]: 0.8}})
    monkeypatch.setattr(module, "_default_classifier", classifier)
    assert module.classify_sentence("An opinion", categories=["opinion"])["classifications"] == ["opinion"]


def test_real_pipeline_contract_without_weights():
    classifier = SentenceClassifier()
    calls = []
    def fake_pipeline(text, **kwargs):
        calls.append(kwargs)
        return {"labels": ["opinion", "claim"], "scores": [0.8, 0.6]}
    classifier._pipeline = fake_pipeline
    result = classifier.classify("I think", categories=["claim", "opinion"], multi_label=True)
    assert result["classifications"] == ["opinion", "claim"]
    assert calls[0]["candidate_labels"] == ["claim", "opinion"]
