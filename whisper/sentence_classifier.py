"""
sentence_classifier.py
----------------------
Local sentence classification module using ModernBERT-base.
Classifies input sentences into categories: [fact, claim, query, question] (or custom labels).

Architecture:
  - Zero-shot classification powered by ModernBERT-base (e.g. MoritzLaurer/ModernBERT-base-zeroshot-v2.0).
  - Lazy loading: Weights are loaded upon first inference or explicit load_model() call.
  - Fully unit-testable: Supports mock engines, custom backends, and deterministic heuristics.
  - Returns output format: { "text": sentence, "classifications": [category] }
"""

from __future__ import annotations

import os
import re
from typing import Any, Callable, Dict, List, Optional, Union

# Default target categories as requested
DEFAULT_CATEGORIES: List[str] = ["fact", "claim", "query", "question"]

# Standard zero-shot model built directly on ModernBERT-base
DEFAULT_MODEL_NAME: str = os.environ.get(
    "MODERNBERT_MODEL_NAME", "MoritzLaurer/ModernBERT-base-zeroshot-v2.0"
)

# Hypothesis template for zero-shot NLI
DEFAULT_HYPOTHESIS_TEMPLATE: str = "This text is a {}."


def _heuristic_rule_fallback(sentence: str, candidate_categories: List[str]) -> Optional[str]:
    """
    Fast rule-based heuristic for obvious question and query patterns.
    Used for unit testing mocks and rapid sanity checks.
    """
    s = sentence.strip().lower()
    if not s:
        return None

    # Question detection
    if "question" in candidate_categories:
        if s.endswith("?") or re.match(
            r"^(what|why|who|where|when|which|how|is|are|can|could|do|does|did|will|would|kya|kyun|kaise)\b",
            s,
        ):
            return "question"

    # Query detection (search directives)
    if "query" in candidate_categories:
        if re.match(r"^(search|find|look up|show me|tell me|get|query)\b", s):
            return "query"

    return None


class SentenceClassifier:
    """
    Local Sentence Classifier utilizing ModernBERT-base for low-latency classification.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = "auto",
        hypothesis_template: str = DEFAULT_HYPOTHESIS_TEMPLATE,
        mock_engine: Optional[Callable[[str, List[str]], Dict[str, Any]]] = None,
    ) -> None:
        """
        Initialize the sentence classifier.

        Parameters
        ----------
        model_name : str, optional
            Hugging Face model checkpoint (defaults to ModernBERT-base zero-shot)
        device : str, optional
            'auto' (picks cuda if available, else cpu), 'cuda', 'cpu', or device ID
        hypothesis_template : str
            Template used for NLI zero-shot entailment scoring
        mock_engine : callable, optional
            Injected mock engine for unit testing without downloading model weights.
            Signature: (sentence: str, categories: List[str]) -> Dict[str, float]
        """
        self.model_name = model_name or DEFAULT_MODEL_NAME
        self.hypothesis_template = hypothesis_template
        self.device_setting = device
        self.mock_engine = mock_engine

        self._pipeline = None
        self._device = None

    def _resolve_device(self) -> int:
        """Resolve device setting to transformers device integer or string."""
        if isinstance(self.device_setting, int):
            return self.device_setting

        dev_str = str(self.device_setting).lower()
        if dev_str in ("cuda", "gpu", "auto"):
            try:
                import torch
                if torch.cuda.is_available():
                    return 0
            except ImportError:
                pass
            return -1  # CPU

        return -1  # Default to CPU

    def load_model(self) -> None:
        """Explicitly load the ModernBERT model pipeline into memory."""
        if self.mock_engine is not None:
            return

        if self._pipeline is None:
            try:
                from transformers import pipeline
            except ImportError as err:
                raise ImportError(
                    "The 'transformers' library is required to load ModernBERT. "
                    "Install it via: pip install transformers"
                ) from err

            device_id = self._resolve_device()
            self._device = device_id

            dtype_setting = None
            if device_id != -1:
                try:
                    import torch
                    if torch.cuda.is_available():
                        dtype_setting = torch.float16
                except ImportError:
                    pass

            kwargs = {"device": device_id}
            if dtype_setting is not None:
                kwargs["dtype"] = dtype_setting

            self._pipeline = pipeline(
                "zero-shot-classification",
                model=self.model_name,
                **kwargs,
            )

    def is_loaded(self) -> bool:
        """Return True if model is loaded and ready."""
        return self._pipeline is not None or self.mock_engine is not None

    def classify(
        self,
        sentence: str,
        categories: Optional[List[str]] = None,
        multi_label: bool = False,
    ) -> Dict[str, Any]:
        """
        Classify a single sentence into the target categories.

        Parameters
        ----------
        sentence : str
            The input sentence / utterance.
        categories : list of str, optional
            Candidate categories (defaults to ['fact', 'claim', 'query', 'question']).
        multi_label : bool, default False
            Whether multiple categories can apply independently.

        Returns
        -------
        dict
            {
                "text": sentence,
                "classifications": [top_category],
                "top_category": top_category,
                "scores": {category: score, ...},
                "confidence": float
            }
        """
        sentence_clean = sentence.strip() if sentence else ""
        target_categories = categories or DEFAULT_CATEGORIES

        if not sentence_clean:
            return {
                "text": sentence,
                "classifications": [],
                "top_category": None,
                "confidence": 0.0,
                "scores": {cat: 0.0 for cat in target_categories},
            }

        # 1. Use mock engine if provided (unit testing hook)
        if self.mock_engine is not None:
            engine_result = self.mock_engine(sentence_clean, target_categories)
            scores = engine_result.get("scores", {})
            top_category = engine_result.get("top_category")
            if not top_category and scores:
                top_category = max(scores, key=scores.get)

            return {
                "text": sentence,
                "classifications": [top_category] if top_category else [],
                "top_category": top_category,
                "confidence": round(scores.get(top_category, 0.0), 4) if top_category else 0.0,
                "scores": scores,
            }

        # 2. Ensure model is loaded
        if self._pipeline is None:
            self.load_model()

        # 3. Execute ModernBERT zero-shot classification (batched hypothesis scoring)
        result = self._pipeline(
            sentence_clean,
            candidate_labels=target_categories,
            hypothesis_template=self.hypothesis_template,
            multi_label=multi_label,
            batch_size=len(target_categories),
        )

        labels = result.get("labels", [])
        scores_list = result.get("scores", [])

        scores_dict = {
            label: round(float(score), 4)
            for label, score in zip(labels, scores_list)
        }

        top_category = labels[0] if labels else None
        confidence = scores_dict.get(top_category, 0.0) if top_category else 0.0

        if multi_label:
            threshold = 0.5
            selected = [lbl for lbl, sc in scores_dict.items() if sc >= threshold]
            classifications = selected if selected else ([top_category] if top_category else [])
        else:
            classifications = [top_category] if top_category else []

        return {
            "text": sentence,
            "classifications": classifications,
            "top_category": top_category,
            "confidence": confidence,
            "scores": scores_dict,
        }

    def classify_batch(
        self,
        sentences: List[str],
        categories: Optional[List[str]] = None,
        multi_label: bool = False,
    ) -> List[Dict[str, Any]]:
        """Classify a list of sentences."""
        return [self.classify(s, categories=categories, multi_label=multi_label) for s in sentences]


# ---------------------------------------------------------------------------
# Module-Level Singleton & Convenience API
# ---------------------------------------------------------------------------
_default_classifier: Optional[SentenceClassifier] = None


def get_classifier() -> SentenceClassifier:
    """Retrieve or initialize the default SentenceClassifier singleton."""
    global _default_classifier
    if _default_classifier is None:
        _default_classifier = SentenceClassifier()
    return _default_classifier


def set_classifier(classifier: SentenceClassifier) -> None:
    """Set the default SentenceClassifier (useful for injecting mocks in tests)."""
    global _default_classifier
    _default_classifier = classifier


def classify_sentence(
    sentence: str,
    categories: Optional[List[str]] = None,
    model_name: Optional[str] = None,
    multi_label: bool = False,
) -> Dict[str, Any]:
    """
    Convenience function to classify a sentence into categories.

    Usage
    -----
    >>> result = classify_sentence("What is the speed of light?")
    >>> print(result["classifications"])
    ['question']

    Parameters
    ----------
    sentence : str
        Input sentence to classify.
    categories : list of str, optional
        Target categories (default: ['fact', 'claim', 'query', 'question']).
    model_name : str, optional
        Custom model name to override default.
    multi_label : bool, default False
        Whether to allow multiple independent labels.

    Returns
    -------
    dict
        {
            "text": sentence,
            "classifications": [selected_category],
            "top_category": selected_category,
            "confidence": float,
            "scores": {category: score, ...}
        }
    """
    if model_name is not None and model_name != DEFAULT_MODEL_NAME:
        classifier = SentenceClassifier(model_name=model_name)
        return classifier.classify(sentence, categories=categories, multi_label=multi_label)

    return get_classifier().classify(sentence, categories=categories, multi_label=multi_label)
