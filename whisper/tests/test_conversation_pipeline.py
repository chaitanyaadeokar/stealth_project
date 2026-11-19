import importlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

import logger
from conversation import ConversationContext, ConversationStore
from diarization import assign_word_speakers
from live_runtime import submit_segments, flush_audio
from llm_pipeline import TranscriptBuffer, process_next, start_llm_pipeline
from orchestrator import Orchestrator
from sentence_classifier import SentenceClassifier


def mock_engine(text, categories):
    label = "question" if text.endswith("?") else "opinion" if text.startswith("I think") else "claim"
    return {"scores": {label: 0.9}}


def fake_llm(text):
    return "UNCERTAIN: mock assessment"


def slow_llm(text):
    time.sleep(30)
    return "late"


@pytest.fixture
def store(tmp_path):
    logger.configure(log_path=str(tmp_path / "test.log"))
    database = ConversationStore(tmp_path / "sessions.sqlite3")
    yield database
    database.close()
    logger._custom_log_path = None


@pytest.fixture
def classifier():
    return SentenceClassifier(mock_engine=mock_engine)


def test_chronology_context_and_restart(store):
    session = store.create_session("first")
    other = store.create_session("other")
    a = store.append(session, "A", "First", utterance_id="first")
    store.append(other, "C", "Hidden")
    store.append(session, "B", "Second")
    store.append(session, "A", "Third")
    turns = store.turns(session)
    assert [t["text"] for t in turns] == ["First", "Second", "Third"]
    assert [t["sequence"] for t in turns] == [1, 2, 3]
    assert store.append(session, "A", "First", utterance_id=a) == a
    with pytest.raises(ValueError):
        store.append(other, "A", "First", utterance_id=a)
    assert [t["text"] for t in ConversationContext(store).for_turn(turns[-1])] == ["First", "Second"]
    assert sum(len(t["text"]) for t in ConversationContext(store, max_chars=3).for_turn(turns[-1])) == 3
    reopened = ConversationStore(store.path)
    try:
        assert len(reopened.turns(session)) == 3
    finally:
        reopened.close()


@pytest.mark.parametrize("text,route,status", [
    ("Why?", "questions", "unavailable"), ("A claim", "claims", "unverified"),
    ("I think this is better", "opinions", "unavailable")])
def test_routes_and_correlated_results(store, classifier, text, route, status):
    session = store.create_session()
    identity = store.append(session, "A", text)
    assert process_next(store, session, "worker", Orchestrator(store, classifier, llm_fn=fake_llm))
    job = store.jobs(session)[0]
    assert job["status"] == "succeeded"
    assert job["result"]["route"] == route
    assert job["result"]["status"] == status
    assert job["result"]["utterance_id"] == identity
    assert not process_next(store, session, "worker", Orchestrator(store, classifier))


@pytest.mark.parametrize("label,confidence", [("question", 0.1), ("statement", 0.9), ("bogus", 1), ("claim", float("nan"))])
def test_uncertain_intent_abstains(store, label, confidence):
    class Classifier:
        def classify(self, text, **kwargs):
            return {"top_category": label, "confidence": confidence}
    session = store.create_session()
    store.append(session, "A", "uncertain")
    result = Orchestrator(store, Classifier()).process(store.turns(session)[0])
    assert result["route"] is None
    assert result["status"] == "needs_clarification"


def test_failure_retry_ack_and_no_reordering(store, classifier):
    session = store.create_session()
    store.append(session, "A", "Claim one")
    store.append(session, "B", "Claim two")
    calls = []
    def flaky(turn, context):
        calls.append(turn["sequence"])
        if len(calls) == 1:
            raise RuntimeError("temporary")
        return {"status": "ok", "text": "done"}
    orch = Orchestrator(store, classifier, handlers={"claims": flaky})
    process_next(store, session, "worker", orch, retry_delay=0)
    first = store.jobs(session)[0]
    assert first["status"] == "pending" and first["error"] == "temporary"
    process_next(store, session, "worker", orch, retry_delay=0)
    process_next(store, session, "worker", orch, retry_delay=0)
    assert calls == [1, 1, 2]
    assert [j["attempts"] for j in store.jobs(session)] == [2, 1]


def test_retry_exhaustion_and_lease_recovery(store, classifier):
    session = store.create_session()
    identity = store.append(session, "A", "Claim")
    turn = store.claim(session, "dead", lease_seconds=-1)
    with pytest.raises(RuntimeError):
        store.complete(identity, "wrong-owner", {})
    def broken(turn, context):
        raise ValueError("permanent")
    orch = Orchestrator(store, classifier, handlers={"claims": broken})
    process_next(store, session, "new", orch, max_attempts=2, retry_delay=0)
    job = store.jobs(session)[0]
    assert job["status"] == "failed" and job["attempts"] == 2
    assert not process_next(store, session, "new", orch)


def test_backlog_and_retry_wait(store):
    session = store.create_session()
    identity = store.append(session, "A", "First", max_pending=1)
    with pytest.raises(OverflowError):
        store.append(session, "B", "Second", max_pending=1)
    store.claim(session, "worker")
    store.fail(identity, "worker", "later", retry_delay=10)
    store.append(session, "B", "Second")
    assert store.claim(session, "worker") is None


def test_segments_preserve_word_speakers_and_original_text(store):
    session = store.create_session()
    buffer = TranscriptBuffer(store, session)
    segments = [{"start": 0, "end": 3, "text": "Hello! Yes. Again?", "words": [
        {"start": 0, "end": 1, "word": "Hello!", "speaker": "A"},
        {"start": 1, "end": 2, "word": " Yes.", "speaker": "B"},
        {"start": 2, "end": 3, "word": " Again?", "speaker": "A"}]}]
    submit_segments(buffer, segments, offset=10, chunk_id="chunk", display_fn=str.upper)
    submit_segments(buffer, segments, offset=10, chunk_id="chunk", display_fn=str.upper)
    turns = store.turns(session)
    assert [t["speaker"] for t in turns] == ["A", "B", "A"]
    assert turns[0]["text"] == "Hello!" and turns[0]["display_text"] == "HELLO!"
    assert [t["start"] for t in turns] == [10, 11, 12]


def test_diarization_overlap_and_unknown():
    transcript = {"segments": [{"text": "Hi", "start": 0, "end": 2,
                                "words": [{"word": "Hi", "start": 0, "end": 0.5}]},
                               {"text": "Gap", "start": 5, "end": 6}]}
    result = assign_word_speakers([(0, 1, "A"), (1, 3, "B")], transcript)
    assert result["segments"][0]["words"][0]["speaker"] == "A"
    assert result["segments"][1]["speaker"] == "UNKNOWN"
    assert "speaker" not in transcript["segments"][0]


def test_shutdown_flush_and_audio_recovery(store):
    import numpy as np
    import queue
    class Model:
        def transcribe(self, audio, **kwargs):
            assert len(audio) == 3
            return iter([SimpleNamespace(text="Tail", start=0, end=0.1)]), SimpleNamespace(language="en")
    session = store.create_session()
    buffer = TranscriptBuffer(store, session)
    audio_queue = queue.Queue()
    audio_queue.put(np.zeros(1, dtype=np.float32))
    flush_audio(Model(), np.zeros(2, dtype=np.float32), audio_queue, buffer, 16000, "en", "transcribe")
    assert store.turns(session)[0]["text"] == "Tail"
    class Broken:
        def transcribe(self, *args, **kwargs):
            raise RuntimeError("unavailable")
    flush_audio(Broken(), np.ones(2), audio_queue, buffer, 16000, "en", "transcribe")
    assert len(list(Path(store.path).parent.glob("*-tail-*.npz"))) == 1


def test_spawn_shutdown_drains_and_resume_does_not_duplicate(tmp_path, classifier):
    db = tmp_path / "spawn.sqlite3"
    logger.configure(log_path=str(tmp_path / "spawn.log"))
    buffer, process, stop = start_llm_pipeline(fake_llm, interval_sec=60, db_path=db,
                                              classifier=classifier, retry_delay=0)
    session = buffer.session_id
    buffer.append("A", "Why?")
    buffer.append("B", "Claim")
    summary = buffer.close(timeout=15)
    assert not process.is_alive() and summary["pending"] == 0 and not summary["forced"]
    assert buffer.close() == summary
    with pytest.raises(RuntimeError):
        buffer.append("A", "closed")
    second, _, _ = start_llm_pipeline(fake_llm, db_path=db, session_id=session, classifier=classifier)
    assert second.close(timeout=15)["pending"] == 0
    store = ConversationStore(db)
    try:
        assert [job["attempts"] for job in store.jobs(session)] == [1, 1]
    finally:
        store.close()
        logger._custom_log_path = None


def test_forced_shutdown_retains_work(tmp_path, classifier):
    logger.configure(log_path=str(tmp_path / "forced.log"))
    buffer, process, _ = start_llm_pipeline(slow_llm, interval_sec=0.01, db_path=tmp_path / "forced.sqlite3",
                                            classifier=classifier)
    session, db = buffer.session_id, buffer.store.path
    buffer.append("A", "Claim")
    deadline = time.monotonic() + 10
    while buffer.store.jobs(session)[0]["status"] != "running" and time.monotonic() < deadline:
        time.sleep(0.05)
    assert buffer.store.jobs(session)[0]["status"] == "running"
    summary = buffer.close(timeout=0.1)
    assert summary["forced"] and summary["pending"] == 1 and not process.is_alive()
    resumed, _, _ = start_llm_pipeline(fake_llm, interval_sec=60, db_path=db, session_id=session, classifier=classifier)
    assert resumed.close(timeout=15)["pending"] == 0
    logger._custom_log_path = None


@pytest.mark.parametrize("module", ["live_whisper", "live_whisper_diarize"])
def test_live_imports_and_help_are_lightweight(module):
    code = f"import {module}; import sys; assert 'torch' not in sys.modules; assert 'transformers' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True, timeout=15)
    result = subprocess.run([sys.executable, f"{module}.py", "--help"], capture_output=True, text=True, timeout=15)
    assert result.returncode == 0 and "--session-id" in result.stdout


def test_default_logger_is_in_repository(monkeypatch):
    monkeypatch.setattr(logger, "_custom_log_path", None)
    monkeypatch.delenv("VAGUE_LOG_FILE", raising=False)
    monkeypatch.delenv("VAGUE_LOG_PATH", raising=False)
    assert Path(logger.get_log_path()).parent == Path(logger.__file__).parent


def test_batch_is_atomic_on_backlog_failure(store):
    session = store.create_session()
    buffer = TranscriptBuffer(store, session, max_pending=1)
    with pytest.raises(OverflowError):
        submit_segments(buffer, [{"text": "First. Second.", "start": 0, "end": 1}])
    assert store.jobs(session) == []


def test_session_lock_rejects_second_worker(tmp_path, classifier):
    buffer, _, _ = start_llm_pipeline(db_path=tmp_path / "lock.sqlite3", classifier=classifier)
    try:
        with pytest.raises(RuntimeError, match="active pipeline"):
            start_llm_pipeline(db_path=buffer.store.path, session_id=buffer.session_id, classifier=classifier)
    finally:
        buffer.close(timeout=10)


def test_cli_replay_and_inspection(tmp_path):
    db = str(tmp_path / "cli.sqlite3")
    command = [sys.executable, "pipeline_cli.py", "--offline", "--db", db, "--session-id", "replay",
               "--input", "tests/fixtures/conversation.jsonl"]
    subprocess.run(command, check=True, capture_output=True, timeout=20)
    subprocess.run(command, check=True, capture_output=True, timeout=20)
    result = subprocess.run([sys.executable, "pipeline_cli.py", "--db", db, "--session-id", "replay", "--inspect"],
                            check=True, capture_output=True, text=True, timeout=15)
    jobs = json.loads(result.stdout)
    assert len(jobs) == 3 and all(job["attempts"] == 1 for job in jobs)
    assert [job["result"]["route"] for job in jobs] == ["questions", "claims", "opinions"]


@pytest.mark.parametrize("module_name", ["live_whisper", "live_whisper_diarize"])
@pytest.mark.parametrize("stream_error", [False, True])
def test_live_cleanup_with_fake_audio_model(store, monkeypatch, module_name, stream_error):
    module = importlib.import_module(module_name)
    entry = importlib.import_module("live_whisper_diarize")
    session = store.create_session()
    events = []
    class Buffer(TranscriptBuffer):
        closed = False
        def close(self, **kwargs):
            self.closed = True
            return {"pending": self.store.pending_count(self.session_id)}
    buffer = Buffer(store, session)
    def start(*args, **kwargs):
        assert kwargs["preload"] is True
        events.append("classifier_ready")
        return buffer, None, None
    def create(*args, **kwargs):
        events.append("speech_ready")
        return object()
    def capture(*args, **kwargs):
        events.append("capture")
        if stream_error:
            raise RuntimeError("microphone unavailable")
        buffer.append("UNKNOWN", "The tail.")
    monkeypatch.setattr(entry, "load_environment", lambda: None)
    monkeypatch.setattr(entry, "validate_environment", lambda: None)
    monkeypatch.setattr(entry, "create_whisper_model", create)
    monkeypatch.setattr(entry, "start_llm_pipeline", start)
    monkeypatch.setattr(entry, "run_capture", capture)
    if stream_error:
        with pytest.raises(RuntimeError, match="microphone unavailable"):
            module.live_transcribe(language_mode="en")
    else:
        module.live_transcribe(language_mode="en")
        assert store.turns(session)[0]["text"] == "The tail."
    assert events == ["classifier_ready", "speech_ready", "capture"]
    assert buffer.closed


def test_llm_client_timeout_no_hidden_retries(monkeypatch):
    from llm_client import fact_check_statement
    options = {}
    class Client:
        def __init__(self, **kwargs):
            options.update(kwargs)
            self.chat = SimpleNamespace(completions=self)
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def create(self, **kwargs):
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=" UNCERTAIN "))])
    monkeypatch.setenv("GROQ_API_KEY", "fake-test-token")
    monkeypatch.setenv("LLM_TIMEOUT_SEC", "7")
    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=Client))
    assert fact_check_statement("claim") == "UNCERTAIN"
    assert options["timeout"] == 7 and options["max_retries"] == 0


@pytest.mark.parametrize("label,route", [("question", "questions"), ("query", "questions"),
    ("fact", "claims"), ("claim", "claims"), ("opinion", "opinions"), ("debate", "opinions")])
def test_all_intent_aliases(store, label, route):
    session = store.create_session()
    store.append(session, "A", "Earlier context")
    store.append(session, "B", "Target")
    classifier = SentenceClassifier(mock_engine=lambda *args: {"scores": {label: 0.95}})
    seen = []
    def handler(turn, context):
        seen.extend(t["text"] for t in context)
        return {"status": "ok", "text": "Handled"}
    result = Orchestrator(store, classifier, handlers={route: handler}).process(store.turns(session)[-1])
    assert result["route"] == route and seen == ["Earlier context"]


def test_logs_keep_multiline_entries_together(tmp_path):
    log = tmp_path / "parallel.log"
    code = "import logger,sys; logger.configure(log_path=sys.argv[1]); [logger.log_info('entry\\n  detail') for _ in range(10)]"
    workers = [subprocess.Popen([sys.executable, "-c", code, str(log)]) for _ in range(2)]
    for worker in workers:
        assert worker.wait(timeout=15) == 0
    lines = log.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 40
    assert all(lines[index + 1] == "  detail" for index in range(0, len(lines), 2))
