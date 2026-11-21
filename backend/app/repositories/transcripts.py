"""Compatibility writes for the original transcript tables."""

from backend.app.models import TranscriptSegment, ReasoningEntry


def add_legacy(db, event):
    model = TranscriptSegment if event.type == "transcript.final" else ReasoningEntry
    db.add(model(mic_id=event.mic_id, speaker=event.speaker, text=event.text))
