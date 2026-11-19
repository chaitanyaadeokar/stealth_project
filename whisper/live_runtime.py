"""Shared live adapter helpers; heavyweight audio/model imports are deferred."""
import os
import math
import re
import uuid
from pathlib import Path

import logger


def load_environment():
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / ".env")


def validate_environment():
    if not os.getenv("GROQ_API_KEY"):
        raise RuntimeError("Set GROQ_API_KEY in .env before starting live claim assessments")
    timeout = float(os.environ.get("LLM_TIMEOUT_SEC", "30"))
    if not math.isfinite(timeout) or not 0 < timeout <= 300:
        raise ValueError("LLM_TIMEOUT_SEC must be between 0 and 300 seconds")
    import importlib.util
    if importlib.util.find_spec("transformers") is None:
        raise RuntimeError("Intent routing requires transformers; install requirements.txt")


def prepare_segments(buffer, segments, *, language=None, offset=0, chunk_id=None, display_fn=None):
    """Preserve segment/word speaker order and split only explicit sentence boundaries.

    Sentence parts inherit the enclosing segment's timestamps; no word alignment
    is invented. Stable chunk IDs make resubmission idempotent after partial failure.
    """
    chunk_id = chunk_id or uuid.uuid4().hex
    index = 0
    records = []
    from utterance_segmentation import split_utterances
    for segment in sorted(segments, key=lambda item: item.get("start", 0)):
        words = segment.get("words") or []
        if words and re.sub(r"\s+", "", "".join(w.get("word", "") for w in words)) != re.sub(r"\s+", "", segment.get("text", "")):
            # Incomplete word alignment must not silently discard unaligned transcript text.
            logger.log_event("alignment_incomplete", fallback="segment_timestamps")
            words = []
        groups = []
        if words:
            for word in words:
                speaker = word.get("speaker", segment.get("speaker", "UNKNOWN"))
                if (groups and groups[-1]["speaker"] == speaker
                        and word["start"] - groups[-1]["end"] < 0.75):
                    groups[-1]["text"] += word["word"]
                    groups[-1]["end"] = word["end"]
                    groups[-1]["words"].append(word)
                else:
                    groups.append({"speaker": speaker, "text": word["word"],
                                   "start": word["start"], "end": word["end"], "words": [word]})
        else:
            groups = [dict(segment, speaker=segment.get("speaker", "UNKNOWN"))]
        for group in groups:
            for text, left, right in split_utterances(group.get("text", "")):
                if not text:
                    continue
                key = uuid.uuid5(uuid.NAMESPACE_URL, f"{buffer.session_id}/{chunk_id}/{index}").hex
                word_spans, cursor = [], 0
                for word in group.get("words", []):
                    word_end = cursor + len(word["word"])
                    if word_end > left and cursor < right:
                        word_spans.append(word)
                    cursor = word_end
                start = word_spans[0]["start"] if word_spans else group.get("start", 0)
                end = word_spans[-1]["end"] if word_spans else group.get("end", 0)
                records.append(dict(speaker=group["speaker"], text=text, utterance_id=key,
                                    display_text=display_fn(text) if display_fn else text,
                                    language=language, start=offset + start, end=offset + end))
                index += 1
    return records


def submit_segments(buffer, segments, *, language=None, offset=0, chunk_id=None, display_fn=None, audio_commit=None):
    records = prepare_segments(buffer, segments, language=language, offset=offset,
                               chunk_id=chunk_id, display_fn=display_fn)
    result = buffer.append_many(records, audio_commit=audio_commit) if audio_commit is not None else buffer.append_many(records)
    for record in records:
        logger.log_stt_append("finalized chunk", record["speaker"], record["text"])
    return result


def flush_audio(model, audio, audio_queue, buffer, sample_rate, language_mode, task, offset=0):
    """Called after the microphone stops. Finalize tail audio or save it for recovery."""
    import numpy as np
    import queue
    chunks = [audio]
    while True:
        try:
            chunks.append(audio_queue.get_nowait().flatten())
        except queue.Empty:
            break
    pending = np.concatenate(chunks)
    if not len(pending):
        return
    chunk_id = uuid.uuid4().hex
    try:
        segments, info = model.transcribe(pending, language=language_mode if language_mode in ("hi", "en") else None,
                                           task=task, vad_filter=True, condition_on_previous_text=False)
        finalized = [{"text": s.text, "start": s.start, "end": s.end, "speaker": "UNKNOWN"} for s in segments]
        submit_segments(buffer, finalized, language=info.language, offset=offset, chunk_id=chunk_id)
    except Exception as exc:
        safe_session = uuid.uuid5(uuid.NAMESPACE_URL, buffer.session_id).hex
        path = Path(buffer.store.path).parent / f"{safe_session}-tail-{chunk_id}.npz"
        np.savez_compressed(path, audio=pending, sample_rate=sample_rate, offset=offset)
        logger.log_error(f"Final audio not submitted; recovery audio saved to {path}", exc)
        print(f"[Shutdown] Unsubmitted audio saved to {path}")
