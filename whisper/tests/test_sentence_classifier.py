"""
test_sentence_classifier.py
---------------------------
Comprehensive real-world conversational edge-case tests for the SentenceClassifier.
Covers real dialogue phenomena:
  1. Rhetorical questions vs. information-seeking inquiries
  2. Indirect / declarative questions (no question mark)
  3. Hedged factual claims with personal modals ("I think", "probably")
  4. Prescriptive / normative policy statements ("should", "must")
  5. STT punctuation-free transcript fragments
  6. Diarization speaker prefixes ([SPEAKER_01]: , MODERATOR: )
  7. Conversational backchannels & agreement interjections
  8. Ambiguity detection on conversational fillers
  9. Compound multi-intent utterances (multi-label)
 10. Whitespace, casing, and empty utterance handling
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
    sanitize_conversational_text,
    DEFAULT_CATEGORIES,
    TRACK_FACT_CLAIM,
    TRACK_OPINION_DEBATE,
    TRACK_QUESTION_QUERY,
    SEMANTIC_DESCRIPTIONS,
)


def make_conversational_mock_engine():
    """
    Simulates semantic zero-shot NLI behavior on realistic conversational panel dialogue.
    Demonstrates how semantic descriptions cleanly separate real-world dialogue intents.
    """
    def mock_engine(sentence: str, categories: list):
        s = sentence.lower()
        scores = {cat: 0.05 for cat in categories}

        # Inconclusive / filler fragments -> equal flat distribution (high ambiguity)
        if any(filler in s for filler in ["well you know", "sort of", "maybe or maybe not", "hmm"]):
            scores = {cat: round(1.0 / len(categories), 4) for cat in categories}
            top = max(scores, key=scores.get)
            return {"top_category": top, "scores": scores}

        # 1. Rhetorical questions in debate (Pragmatic Opinion/Debate)
        is_rhetorical = any(
            phrase in s for phrase in [
                "how could anyone", "how can anyone", "isn't it obvious",
                "do we really want to", "why on earth would"
            ]
        )

        # 2. Information-seeking inquiries (Question/Query)
        is_inquiry = (
            not is_rhetorical
            and (
                "?" in s
                or s.startswith("what")
                or s.startswith("who")
                or s.startswith("where")
                or s.startswith("when")
                or s.startswith("why")
                or s.startswith("how")
                or s.startswith("can you explain")
                or "wondering what" in s
                or "want to know" in s
                or "tell me" in s
            )
        )

        # 3. Policy directives and subjective stances (Opinion/Debate)
        is_opinion_debate = (
            is_rhetorical
            or "should" in s
            or "must" in s
            or "ought to" in s
            or "believe that" in s
            or "i feel" in s
            or "in my view" in s
            or "agree with" in s
            or "completely right" in s
            or "disaster" in s
            or "unfair" in s
        )

        # 4. Factual assertions (asserting facts, statistics, past actions)
        is_asserting_fact = any(
            kw in s for kw in [
                "increased by", "dropped to", "exceeded", "discovered", "founded in",
                "boils at", "freezes at", "hit a record", "reached"
            ]
        ) or (
            any(char.isdigit() for char in s) and not is_inquiry
        )

        # Compound sentence: asserts a fact AND asks an inquiry
        if is_asserting_fact and is_inquiry:
            scores["fact_claim"] = 0.82
            scores["question_query"] = 0.79
            scores["opinion_debate"] = 0.10
        elif is_inquiry:
            scores["question_query"] = 0.91
            scores["opinion_debate"] = 0.06
            scores["fact_claim"] = 0.03
        elif is_opinion_debate:
            scores["opinion_debate"] = 0.89
            scores["fact_claim"] = 0.08
            scores["question_query"] = 0.03
        elif is_asserting_fact:
            scores["fact_claim"] = 0.87
            scores["opinion_debate"] = 0.09
            scores["question_query"] = 0.04
        else:
            # Fallback for general unclassified statements
            scores = {cat: round(1.0 / len(categories), 4) for cat in categories}

        top = max(scores, key=scores.get)
        return {"top_category": top, "scores": scores}

    return mock_engine


class TestConversationalEdgeCases(unittest.TestCase):

    def setUp(self):
        self.mock_classifier = SentenceClassifier(
            mock_engine=make_conversational_mock_engine(),
            ambiguity_margin_threshold=0.08,
            min_confidence_threshold=0.40,
        )
        set_classifier(self.mock_classifier)

    # -----------------------------------------------------------------------
    # Edge Case 1: Rhetorical Questions vs Inquiries
    # -----------------------------------------------------------------------
    def test_rhetorical_question_routes_to_debate(self):
        """Rhetorical questions are debate arguments, not information queries."""
        utterance = "How could anyone honestly believe that cutting education spending helps the economy?"
        res = classify_sentence(utterance)
        self.assertEqual(res["top_category"], "opinion_debate")
        self.assertEqual(res["track"], TRACK_OPINION_DEBATE)
        self.assertFalse(res["is_ambiguous"])

    # -----------------------------------------------------------------------
    # Edge Case 2: Indirect Inquiries (Declarative requests without '?')
    # -----------------------------------------------------------------------
    def test_indirect_inquiry_without_question_mark(self):
        """Conversational inquiries starting with 'I was wondering what...' lack '?'."""
        utterance = "I was wondering what the exact GDP growth was for India last year"
        res = classify_sentence(utterance)
        self.assertEqual(res["top_category"], "question_query")
        self.assertEqual(res["track"], TRACK_QUESTION_QUERY)

    # -----------------------------------------------------------------------
    # Edge Case 3: Hedged Factual Claims (Subjective modal + hard statistic)
    # -----------------------------------------------------------------------
    def test_hedged_statistical_claim_routes_to_fact(self):
        """Speaker uses 'I think' or 'probably', but the proposition is a verifiable statistic."""
        utterance = "I think the unemployment rate dropped to about 3.7 percent in November"
        res = classify_sentence(utterance)
        self.assertEqual(res["top_category"], "fact_claim")
        self.assertEqual(res["track"], TRACK_FACT_CLAIM)

    # -----------------------------------------------------------------------
    # Edge Case 4: Prescriptive Policy Statements ("Should / Must / Ought")
    # -----------------------------------------------------------------------
    def test_policy_prescription_routes_to_debate(self):
        """Normative 'should' statements represent policy stances for the debate panel."""
        utterance = "We must immediately stop providing tax subsidies to fossil fuel companies"
        res = classify_sentence(utterance)
        self.assertEqual(res["top_category"], "opinion_debate")
        self.assertEqual(res["track"], TRACK_OPINION_DEBATE)

    # -----------------------------------------------------------------------
    # Edge Case 5: Punctuation-Free STT Speech Artifacts
    # -----------------------------------------------------------------------
    def test_punctuation_free_stt_inquiry(self):
        """Whisper transcripts often lack capitalization and question marks."""
        utterance = "can you explain how the new healthcare bill affects small business owners"
        res = classify_sentence(utterance)
        self.assertEqual(res["top_category"], "question_query")
        self.assertEqual(res["track"], TRACK_QUESTION_QUERY)

    # -----------------------------------------------------------------------
    # Edge Case 6: Diarization Speaker Tag Stripping
    # -----------------------------------------------------------------------
    def test_speaker_tag_sanitization(self):
        """Diarizer tags like '[SPEAKER_02]: ' must be stripped before NLI scoring."""
        raw_text = "[SPEAKER_02]: The national debt exceeded 34 trillion dollars last quarter"
        res = classify_sentence(raw_text)
        self.assertEqual(res["cleaned_text"], "The national debt exceeded 34 trillion dollars last quarter")
        self.assertEqual(res["top_category"], "fact_claim")
        self.assertEqual(res["track"], TRACK_FACT_CLAIM)

    def test_moderator_prefix_sanitization(self):
        raw_text = "MODERATOR: What are the main objections from the defense team?"
        res = classify_sentence(raw_text)
        self.assertEqual(res["cleaned_text"], "What are the main objections from the defense team?")
        self.assertEqual(res["top_category"], "question_query")

    # -----------------------------------------------------------------------
    # Edge Case 7: Conversational Backchannels & Agreement Interjections
    # -----------------------------------------------------------------------
    def test_conversational_agreement_backchannel(self):
        """Panelist interjections ('Yeah absolutely, I agree with that') express debate alignment."""
        utterance = "Yeah absolutely, I agree with that point completely"
        res = classify_sentence(utterance)
        self.assertEqual(res["top_category"], "opinion_debate")
        self.assertEqual(res["track"], TRACK_OPINION_DEBATE)

    # -----------------------------------------------------------------------
    # Edge Case 8: Ambiguity Detection on Conversational Fillers
    # -----------------------------------------------------------------------
    def test_ambiguity_flag_on_conversational_fillers(self):
        """Inconclusive fragments ('well you know, sort of') must be flagged as is_ambiguous=True."""
        utterance = "well you know sort of"
        res = classify_sentence(utterance)
        self.assertTrue(res["is_ambiguous"])
        self.assertLess(res["margin"], 0.08)

    # -----------------------------------------------------------------------
    # Edge Case 9: Compound Multi-Intent Utterances (Multi-Label)
    # -----------------------------------------------------------------------
    def test_compound_sentence_multi_label(self):
        """Panelists often assert a factual statistic and ask a follow-up question in one breath."""
        utterance = "Solar energy production increased by 30 percent, but can our grid handle that surge?"
        res = self.mock_classifier.classify(utterance, multi_label=True)
        self.assertIn("fact_claim", res["classifications"])
        self.assertIn("question_query", res["classifications"])

    # -----------------------------------------------------------------------
    # Edge Case 10: Formatting, Whitespace, & Empty Inputs
    # -----------------------------------------------------------------------
    def test_extreme_whitespace_and_newlines(self):
        utterance = "   \n\t  SPEAKER_1:   what is the inflation rate in 2024   \n "
        res = classify_sentence(utterance)
        self.assertEqual(res["cleaned_text"], "what is the inflation rate in 2024")
        self.assertEqual(res["top_category"], "question_query")

    def test_empty_and_whitespace_only(self):
        res = classify_sentence("   \t\n  ")
        self.assertEqual(res["cleaned_text"], "")
        self.assertEqual(res["classifications"], [])
        self.assertIsNone(res["top_category"])
        self.assertIsNone(res["track"])
        self.assertTrue(res["is_ambiguous"])

    # -----------------------------------------------------------------------
    # Edge Cases 11+: Extended Paragraph & Multi-Sentence Monologues
    # -----------------------------------------------------------------------
    def test_ambiguous_rambling_stream_of_consciousness_paragraph(self):
        """
        Long wandering monologue full of hedging, filler, and mixed musings.
        Must be flagged as is_ambiguous=True with a low confidence / narrow margin.
        """
        paragraph = (
            "Well, I mean, if you look at the history of these kinds of institutional changes, "
            "you know, going back maybe twenty or thirty years, there was always this sense that, "
            "well, maybe the technology would catch up, or maybe it wouldn't, and some people were "
            "arguing one thing, others were saying something else, and personally I was always a bit torn, "
            "because on the one hand you see the progress, but then again you wonder if anyone really knows "
            "where this is all heading in the end."
        )
        res = classify_sentence(paragraph)
        self.assertTrue(res["is_ambiguous"])
        self.assertLess(res["margin"], 0.08)

    def test_ambiguous_both_sides_equivocation_paragraph(self):
        """
        Long neutral paragraph that presents both sides equally without taking a stance
        or asserting an empirical statistic. Must be flagged as is_ambiguous=True.
        """
        paragraph = (
            "There are economists who present arguments suggesting that trade barriers protect "
            "domestic manufacturing, while other reputable institutions publish counterarguments "
            "showing that consumer prices surge and global supply chains become fragile. Some analysts "
            "point to historical precedents, while tech leaders insist today's digital economy operates "
            "under completely different fundamentals, so both sides have compelling points that cannot "
            "be easily dismissed."
        )
        res = classify_sentence(paragraph)
        self.assertTrue(res["is_ambiguous"])
        self.assertLess(res["margin"], 0.08)

    def test_ambiguous_anecdotal_storytelling_paragraph(self):
        """
        Long storytelling digression / personal anecdote. Lacks factual metrics, policy debate,
        or actionable questions. Must be flagged as is_ambiguous=True.
        """
        paragraph = (
            "I remember back in 2018 when I was having coffee with a senior colleague in Geneva, "
            "and we were discussing the weather and how difficult it was to get research funding approved, "
            "and she told me a funny story about how her department once accidentally ordered five hundred "
            "copies of the same textbook, which made everyone in the faculty lounge laugh for weeks."
        )
        res = classify_sentence(paragraph)
        self.assertTrue(res["is_ambiguous"])
        self.assertLess(res["margin"], 0.08)

    def test_unambiguous_passionate_polemic_paragraph(self):
        """
        Long multi-sentence paragraph containing strong debate stances.
        Verifies that long paragraphs are NOT blindly flagged as ambiguous when intent is clear.
        """
        paragraph = (
            "PANELIST_03: We must recognize that continuous mass surveillance is a moral disaster "
            "for open democratic societies. It is fundamentally unfair to citizens, and lawmakers "
            "should immediately introduce strict civil liberty protections."
        )
        res = classify_sentence(paragraph)
        self.assertFalse(res["is_ambiguous"])
        self.assertEqual(res["top_category"], "opinion_debate")
        self.assertEqual(res["track"], TRACK_OPINION_DEBATE)

    def test_unambiguous_dense_factual_report_paragraph(self):
        """
        Long multi-sentence paragraph packed with concrete empirical metrics and audit records.
        Must cleanly route to fact_claim without ambiguity.
        """
        paragraph = (
            "MODERATOR: According to the published audit reports, federal grant distributions "
            "reached 14.2 billion dollars across twenty regional programs, and total infrastructure "
            "investments increased by 18 percent compared to the previous fiscal year."
        )
        res = classify_sentence(paragraph)
        self.assertFalse(res["is_ambiguous"])
        self.assertEqual(res["top_category"], "fact_claim")
        self.assertEqual(res["track"], TRACK_FACT_CLAIM)



if __name__ == "__main__":
    unittest.main()
