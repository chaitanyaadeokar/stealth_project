"""Intent routing independent of microphone, speech models, or network imports."""
from __future__ import annotations

import math
import os
import time
import logger
from sentence_classifier import SentenceClassifier
from conversation import ConversationContext

ROUTING_CATEGORIES = ["question", "query", "claim", "fact", "opinion", "debate", "statement"]
ROUTES = {"question": "questions", "query": "questions", "claim": "claims",
          "fact": "claims", "opinion": "opinions", "debate": "opinions"}


def question_handler(turn, context):
    return {"status": "unavailable", "text": "Answer retrieval is not implemented yet.", "citations": []}


def opinion_handler(turn, context):
    return {"status": "unavailable", "text": "The opinion/debate panel is not implemented yet.", "citations": []}


class ClaimHandler:
    """Reuse the existing model assessment without representing it as verified evidence."""
    def __init__(self, llm_fn=None):
        self.llm_fn = llm_fn

    def __call__(self, turn, context):
        if self.llm_fn is None:
            return {"status": "unavailable", "text": "Claim assessment is not configured.", "citations": []}
        history = "\n".join(f"[{t['speaker']}]: {t['text']}" for t in context)
        prompt = f"Previous conversation (context only):\n{history}\n\nCurrent statement by [{turn['speaker']}]:\n{turn['text']}"
        response = self.llm_fn(prompt)
        if not isinstance(response, str) or not response.strip():
            raise ValueError("Empty or invalid LLM response")
        return {"status": "unverified", "text": response.strip(), "citations": [],
                "limitation": "Model assessment only; evidence retrieval is not implemented."}


class Orchestrator:
    def __init__(self, store, classifier=None, handlers=None, llm_fn=None, min_confidence=0.5,
                 context_turns=20, context_chars=12000):
        if not 0 <= min_confidence <= 1:
            raise ValueError("min_confidence must be between zero and one")
        self.classifier = classifier if classifier is not None else SentenceClassifier(device=os.environ.get("CLASSIFIER_DEVICE", "cpu"))
        self.context = ConversationContext(store, context_turns, context_chars)
        self.min_confidence = min_confidence
        self.handlers = {"questions": question_handler, "claims": ClaimHandler(llm_fn), "opinions": opinion_handler}
        self.handlers.update(handlers or {})

    def process(self, turn):
        started = time.perf_counter()
        classify_intent = getattr(self.classifier, "classify_intent", None)
        classification = (classify_intent(turn["text"]) if classify_intent else
                          self.classifier.classify(turn["text"], categories=ROUTING_CATEGORIES))
        confidence = float(classification.get("confidence", 0))
        label = classification.get("top_category")
        classification_seconds = time.perf_counter() - started
        route = (ROUTES.get(label) if math.isfinite(confidence) and confidence >= self.min_confidence
                 and not classification.get("ambiguous", False) else None)
        context = self.context.for_turn(turn)
        if label == "other" and confidence >= self.min_confidence and classification.get("source") == "explicit_speech_act":
            output = {"status": "no_action", "text": "Conversational acknowledgment; no handler needed.", "citations": []}
        elif route is None:
            output = {"status": "needs_clarification", "text": "Intent is unsupported or uncertain.", "citations": []}
        else:
            output = self.handlers[route](turn, context)
        if not isinstance(output, dict) or not {"status", "text"} <= output.keys():
            raise ValueError("Handler must return status and text")
        logger.log_event("intent_routed", session_id=turn["session_id"], utterance_id=turn["id"],
                         route=route, label=label, confidence=confidence, scores=classification.get("scores"),
                         ambiguous=classification.get("ambiguous", False), source=classification.get("source"),
                         classification_wall_sec=classification_seconds, status=output["status"])
        return {**output, "session_id": turn["session_id"], "utterance_id": turn["id"],
                "sequence": turn["sequence"], "speaker": turn["speaker"], "route": route,
                "classification": classification, "context_ids": [t["id"] for t in context],
                "classification_wall_sec": classification_seconds}

    def preload(self):
        """Fail before microphone capture if the local classifier cannot initialize."""
        load = getattr(self.classifier, "load_model", None)
        if load:
            load()
        self.classifier.classify("The Earth orbits the Sun.", categories=["claim", "question", "opinion"])
