"""Continuous capture/replay tests using the prerecorded JFK fixture and controlled models."""
import json
import threading
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import logger
from audio_capture import AudioSpool, AudioSegmenter, MicrophoneRecorder, SpeechPipeline
from capture_runtime import run_capture
from conversation import ConversationStore
from diarization import SpeakerTracker, assign_word_speakers
from llm_pipeline import TranscriptBuffer
from llm_pipeline import start_llm_pipeline
from sentence_classifier import SentenceClassifier
from utterance_segmentation import explicit_intent, split_utterances


class FakeModel:
    def __init__(self, delay=0):
        self.delay, self.calls = delay, 0
    def transcribe(self, audio, **kwargs):
        time.sleep(self.delay)
        self.calls += 1
        duration = len(audio) / 16000
        word = SimpleNamespace(word=" A recorded statement.", start=0, end=duration)
        return iter([SimpleNamespace(text="A recorded statement.", start=0, end=duration, words=[word])]), SimpleNamespace(language="en")


class BrokenClassifier:
    def load_model(self):
        raise RuntimeError("broken weights")
    def classify(self, *args, **kwargs):
        raise AssertionError("Must fail while preloading")


def ready_engine(text, categories):
    return {"scores": {"claim": 0.9}}


@pytest.fixture
def buffer(tmp_path):
    logger.configure(log_path=str(tmp_path / "events.log"))
    store = ConversationStore(tmp_path / "conversation.sqlite3")
    buffer = TranscriptBuffer(store, store.create_session("recording"))
    yield buffer
    store.close()
    logger._custom_log_path = None


def test_prerecorded_roundtrip_and_recovery(buffer, tmp_path):
    from faster_whisper.audio import decode_audio
    audio = decode_audio("tests/jfk.flac", sampling_rate=16000)
    spool = AudioSpool(tmp_path / "recording", session_id=buffer.session_id)
    for block in np.array_split(audio, 44):
        spool.append(block)
    spool.finish()
    assert np.array_equal(spool.read(0, len(audio)), audio)
    model = FakeModel()
    summary = run_capture(model, buffer, recovery=spool.directory, language="en", max_seconds=2)
    assert summary["unprocessed_sec"] == 0
    assert model.calls == 6  # six finalized chunks, never growing-buffer interim retranscriptions
    jobs = buffer.store.jobs(buffer.session_id)
    assert [job["sequence"] for job in jobs] == list(range(1, len(jobs) + 1))
    assert [job["start"] for job in jobs] == sorted(job["start"] for job in jobs)
    count = len(jobs)
    again = run_capture(model, buffer, recovery=spool.directory, language="en", max_seconds=2)
    assert again["chunks_committed"] == 0 and len(buffer.store.jobs(buffer.session_id)) == count


def test_sustained_overload_stops_before_reading_an_unretainable_block(tmp_path):
    spool = AudioSpool(tmp_path / "overload", session_id="s", max_pending_seconds=0.5)
    class Stream:
        reads = 0
        def __init__(self, **kwargs):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def read(self, frames):
            self.reads += 1
            return np.full(frames, self.reads, dtype=np.float32), False
    recorder = MicrophoneRecorder(spool, Stream)
    recorder.start()
    assert recorder.done.wait(3)
    recorder.stop()
    assert isinstance(recorder.error, OverflowError)
    state = spool.snapshot()
    assert state["recorded_samples"] == 8000 and state["processed_samples"] == 0
    assert np.array_equal(spool.read(0, 8000), np.r_[np.ones(4000), np.full(4000, 2)])
    assert "budget" in state["error"]


def test_hardware_loss_is_reported_after_returned_audio_is_retained(tmp_path):
    spool = AudioSpool(tmp_path / "device", session_id="s")
    class Stream:
        def __init__(self, **kwargs): pass
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def read(self, frames): return np.ones(frames, dtype=np.float32), True
    recorder = MicrophoneRecorder(spool, Stream)
    recorder.start()
    assert recorder.done.wait(3)
    assert spool.snapshot()["recorded_samples"] == 4000
    assert "lost input" in str(recorder.error)


def test_cpu_work_does_not_block_continuous_capture(buffer):
    class Stream:
        def __init__(self, **kwargs): self.reads=0
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def read(self, frames):
            time.sleep(0.005)
            self.reads += 1
            if self.reads > 20:
                raise KeyboardInterrupt
            return np.ones(frames, dtype=np.float32) * 0.1, False
    model = FakeModel(delay=0.05)
    summary = run_capture(model, buffer, language="en", stream_factory=Stream,
                          max_seconds=0.5, shutdown_timeout=10, max_pending_seconds=120)
    assert summary["recorded_sec"] == 5
    assert summary["unprocessed_sec"] == 0
    assert summary["chunks_committed"] == model.calls
    assert buffer.store.jobs(buffer.session_id)


def test_speech_stages_overlap_but_results_retire_in_order(tmp_path):
    spool = AudioSpool(tmp_path / "pipeline", session_id="s")
    spool.append(np.ones(32000, dtype=np.float32))
    spool.finish()
    order, entered = [], threading.Event()
    class Model(FakeModel):
        def transcribe(self, audio, **kwargs):
            order.append("asr")
            if self.calls:
                assert entered.wait(1)
            return super().transcribe(audio, **kwargs)
    def diarizer(audio, **kwargs):
        entered.set()
        time.sleep(0.02)
        return [(0, len(audio) / 16000, "local")]
    pipeline = SpeechPipeline(spool, Model(), diarizer, language="en")
    try:
        pipeline.submit((0, 16000, True))
        pipeline.submit((16000, 32000, True))
        with pytest.raises(OverflowError):
            pipeline.submit((32000, 48000, True))
        results = []
        deadline = time.monotonic() + 3
        while pipeline.pending and time.monotonic() < deadline:
            result = pipeline.pop_ready()
            if result: results.append(result)
            time.sleep(0.005)
        assert [r["start"] for r in results] == [0, 16000]
        assert results[0]["segments"][0]["speaker"] == results[1]["segments"][0]["speaker"]
    finally:
        pipeline.close()


def test_nearby_alignment_has_distance_and_ambiguity_limits():
    transcript = {"segments": [{"text": "edge", "start": 1.05, "end": 1.15},
                                {"text": "far", "start": 4, "end": 4.1}]}
    result = assign_word_speakers([(0, 1, "A")], transcript, fill_nearest=True)
    assert [s["speaker"] for s in result["segments"]] == ["A", "UNKNOWN"]
    ambiguous = assign_word_speakers([(0, 1, "A"), (1.2, 2, "B")],
        {"segments": [{"text": "gap", "start": 1.09, "end": 1.11}]}, fill_nearest=True)
    assert ambiguous["segments"][0]["speaker"] == "UNKNOWN"


def test_speaker_tracker_uses_overlap_not_local_numbers():
    tracker = SpeakerTracker()
    first = tracker.update([(0, 1, "local0"), (1, 2, "local1"), (2, 3, "local0")], 0)
    second = tracker.update([(0, 1, "local1"), (1, 2, "local0"), (2, 3, "local1")], 0)
    assert [t[2] for t in first] == [t[2] for t in second] == ["SPEAKER_00", "SPEAKER_01", "SPEAKER_00"]


@pytest.mark.parametrize("text,expected", [
    ("What is this", "question"), ("Tell me why", "question"), ("kya yeh sach hai", "question"),
    ("यह कैसे काम करता है?", "question"), ("क्या यह सही है", "question"),
    ("I think we should discuss this", "opinion"), ("We should improve this", "opinion"),
    ("Mujhe lagta hai yeh behtar hai", "opinion"), ("मेरी राय अलग है", "opinion"), ("Okay.", "other")])
def test_explicit_multilingual_speech_acts(text, expected):
    assert explicit_intent(text) == expected


def test_segmentation_preserves_decimals_abbreviations_and_mixed_questions():
    text = "Dr. Sen measured 3.14 volts.Why is that?I think it is better, why not test it?"
    parts = [part for part, _, _ in split_utterances(text)]
    assert "Dr. Sen measured 3.14 volts." in parts
    assert "Why is that?" in parts
    assert "I think it is better" in parts
    assert "why not test it?" in parts


def test_low_margin_remains_ambiguous_and_no_rule_for_factual_claim():
    classifier = SentenceClassifier()
    labels = []
    def pipeline(text, **kwargs):
        assert kwargs["hypothesis_template"] == "The speaker {}."
        choices = kwargs["candidate_labels"]
        return {"labels": choices, "scores": [0.51, 0.49, 0, 0]}
    classifier._pipeline = pipeline
    assert explicit_intent("The Earth is round.") is None
    assert classifier.classify_intent("The Earth is round.")["ambiguous"]


def test_acknowledgment_atomicity_and_manifest_catchup(buffer, tmp_path):
    spool = AudioSpool(tmp_path / "crash", session_id=buffer.session_id)
    spool.append(np.ones(16000, dtype=np.float32))
    spool.finish()
    buffer.append_many([{"speaker": "A", "text": "Already committed", "utterance_id": "stable"}],
                       audio_commit=(spool.state["id"], 0, 16000))
    assert spool.snapshot()["processed_samples"] == 0
    result = run_capture(FakeModel(), buffer, recovery=spool.directory)
    assert result["chunks_committed"] == 0 and result["unprocessed_sec"] == 0


def test_drain_timeout_preserves_audio_and_returns_promptly(buffer, tmp_path):
    spool = AudioSpool(tmp_path / "slow", session_id=buffer.session_id)
    spool.append(np.ones(16000, dtype=np.float32))
    spool.finish()
    # A recorder stop triggers the bounded drain; a retained-spool replay is allowed
    # to finish normally, so exercise a native-stage failure separately here.
    pipeline = SpeechPipeline(spool, FakeModel(delay=0.2), language="en")
    pipeline.submit((0, 16000, True))
    started = time.perf_counter()
    pipeline.close(wait=False)
    assert time.perf_counter() - started < 0.1
    assert spool.snapshot()["processed_samples"] == 0


def test_actual_capture_shutdown_deadline_retains_unacknowledged_audio(buffer):
    release = threading.Event()
    class Model(FakeModel):
        def transcribe(self, *args, **kwargs):
            release.wait(3)
            return super().transcribe(*args, **kwargs)
    class Stream:
        def __init__(self, **kwargs): self.reads = 0
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def read(self, frames):
            self.reads += 1
            if self.reads > 2: raise KeyboardInterrupt
            return np.ones(frames, dtype=np.float32), False
    started = time.perf_counter()
    try:
        with pytest.raises(TimeoutError, match="recover"):
            run_capture(Model(), buffer, language="en", stream_factory=Stream,
                        max_seconds=0.25, shutdown_timeout=0.1)
        assert time.perf_counter() - started < 2
        manifests = list((Path(buffer.store.path).parent / "audio").glob("*/capture.json"))
        assert len(manifests) == 1
        state = json.loads(manifests[0].read_text())
        assert state["recorded_samples"] == 8000 and state["processed_samples"] == 0
    finally:
        release.set()


def test_sentence_splits_use_word_timestamps_and_incomplete_alignment_keeps_text(buffer):
    from live_runtime import submit_segments
    submit_segments(buffer, [{"text": "First. Second?", "start": 0, "end": 2, "words": [
        {"word": "First.", "start": 0, "end": 0.5, "speaker": "A"},
        {"word": " Second?", "start": 1, "end": 2, "speaker": "A"}]}])
    turns = buffer.store.turns(buffer.session_id)
    assert [(t["start"], t["end"]) for t in turns] == [(0, 0.5), (1, 2)]
    submit_segments(buffer, [{"text": "Keep every word", "start": 3, "end": 5, "words": [
        {"word": "Keep", "start": 3, "end": 3.5}]}])
    assert buffer.store.turns(buffer.session_id)[-1]["text"] == "Keep every word"


def test_model_other_does_not_suppress_a_substantive_claim(buffer):
    from orchestrator import Orchestrator
    classifier = SentenceClassifier()
    classifier._pipeline = lambda text, **kw: {"labels": list(kw["candidate_labels"])[::-1], "scores": [0.8, 0.1, 0.05, 0.05]}
    buffer.append("A", "Vaccines contain microchips.")
    result = Orchestrator(buffer.store, classifier).process(buffer.store.turns(buffer.session_id)[0])
    assert result["status"] != "no_action" and result["classification"]["top_category"] != "other"


def test_speaker_state_is_committed_and_restored(buffer, tmp_path):
    tracker = SpeakerTracker()
    tracker.update([(0, 1, "x"), (1, 2, "y")], 0)
    buffer.append_many([], audio_commit=("spool", 0, 32000, tracker.state()))
    restored = SpeakerTracker()
    restored.restore(buffer.store.audio_speaker_state(buffer.session_id, "spool"))
    assert restored.update([(0, 1, "y"), (1, 2, "x")], 0) == tracker.previous


def test_overlap_words_are_emitted_once(tmp_path):
    spool = AudioSpool(tmp_path / "boundary", session_id="s")
    spool.append(np.ones(32000, dtype=np.float32))
    spool.finish()
    class Model:
        calls = 0
        def transcribe(self, audio, **kwargs):
            self.calls += 1
            words = ([SimpleNamespace(word=" boundary", start=0.8, end=1.1)] if self.calls == 1 else
                     [SimpleNamespace(word=" boundary", start=0.2, end=0.5),
                      SimpleNamespace(word=" fresh", start=0.6, end=0.8)])
            return iter([SimpleNamespace(start=words[0].start, end=words[-1].end,
                text="".join(w.word for w in words), words=words)]), SimpleNamespace(language="en")
    pipeline = SpeechPipeline(spool, Model(), language="en")
    try:
        pipeline.submit((0, 16000, True))
        pipeline.submit((16000, 32000, True))
        results = [future.result(timeout=2) for future in pipeline.pending]
        text = " ".join(s["text"] for result in results for s in result["segments"])
        assert text == "boundary fresh"
    finally:
        pipeline.close()


def test_preload_failure_releases_session_and_success_is_ready_before_return(tmp_path):
    db = tmp_path / "startup.sqlite3"
    with pytest.raises(RuntimeError, match="broken weights"):
        start_llm_pipeline(db_path=db, session_id="startup", classifier=BrokenClassifier(), preload=True, startup_timeout=10)
    buffer, process, _ = start_llm_pipeline(db_path=db, session_id="startup",
        classifier=SentenceClassifier(mock_engine=ready_engine), preload=True, startup_timeout=10)
    assert process.is_alive()
    assert buffer.close(timeout=10)["worker_exitcode"] == 0


def test_cpu_auto_factory_preloads_vad_and_warms_before_return(monkeypatch):
    import live_whisper_diarize as live
    events, options = [], {}
    class Model:
        def transcribe(self, audio, **kwargs):
            events.append("warm")
            assert kwargs["vad_filter"] is False
            return iter([]), SimpleNamespace(language="en")
    def factory(name, **kwargs):
        assert name == "base"
        options.update(kwargs)
        return Model()
    monkeypatch.delenv("WHISPER_CPU_MODEL", raising=False)
    monkeypatch.setattr(live, "_setup_cuda_dlls", lambda: None)
    monkeypatch.setitem(sys.modules, "faster_whisper", SimpleNamespace(WhisperModel=factory))
    monkeypatch.setitem(sys.modules, "faster_whisper.vad", SimpleNamespace(get_vad_model=lambda: events.append("vad")))
    monkeypatch.setitem(sys.modules, "ctranslate2", SimpleNamespace(get_cuda_device_count=lambda: 0))
    live.create_whisper_model("auto", device="cpu")
    assert events == ["vad", "warm"] and options["cpu_threads"] == 4 and options["use_auth_token"] is False


def test_terminal_speaker_turns_follow_word_order(capsys):
    from live_whisper_diarize import _print_diarized_output
    _print_diarized_output([{"speaker": "A", "text": "Hello Yes Again", "words": [
        {"speaker": "A", "word": "Hello"}, {"speaker": "B", "word": " Yes"},
        {"speaker": "A", "word": " Again"}]}], "hindi", "EN")
    import re
    assert re.findall(r"\[(A|B) \| EN\]", capsys.readouterr().out) == ["A", "B", "A"]
