import json
import signal
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from audio_capture import AudioSpool
from capture_runtime import run_capture
from conversation import ConversationStore
from diarization import SpeakerTracker
from llm_pipeline import TranscriptBuffer, start_llm_pipeline
from sentence_classifier import SentenceClassifier
from shutdown_signals import capture_interrupts, shield_shutdown
from utterance_segmentation import UtteranceAssembler, is_fragment


def record(identity, text, start, end, speaker="SPEAKER_00"):
    return {"utterance_id":identity,"text":text,"display_text":text,"start":start,"end":end,
            "speaker":speaker,"language":"en"}


def ready_engine(text,categories):
    return {"scores":{"claim":0.95}}


def interrupting_llm(text):
    signal.raise_signal(signal.SIGINT)
    return "UNCERTAIN: signal handled by controller"


def blocked_llm(text):
    time.sleep(30)
    return "late result"


def test_two_speaker_ids_survive_nonoverlap_and_local_label_swap():
    tracker = SpeakerTracker(num_speakers=2)
    first = tracker.update([(0,2,"x"),(2,4,"y")],0,{"x":[1,0],"y":[0,1]})
    second = tracker.update([(10,12,"x"),(12,14,"y")],10,{"x":[0,1],"y":[1,0]})
    assert [row[2] for row in first] == ["SPEAKER_00","SPEAKER_01"]
    assert [row[2] for row in second] == ["SPEAKER_01","SPEAKER_00"]
    restored = SpeakerTracker(num_speakers=2)
    restored.restore(json.loads(json.dumps(tracker.state())))
    assert restored.update([(20,22,"new")],20,{"new":[1,0]})[0][2] == "SPEAKER_00"
    assert restored.next_id == 2


def test_fixed_slots_never_create_third_id_without_embeddings():
    tracker=SpeakerTracker(num_speakers=2)
    identities=set()
    for index in range(8):
        identities.update(speaker for _,_,speaker in tracker.update(
            [(index*10,index*10+1,"local_0"),(index*10+1,index*10+2,"local_1")],index*10))
    assert identities == {"SPEAKER_00","SPEAKER_01"}


def test_word_sequence_reconstructed_across_decoder_periods():
    assembler=UtteranceAssembler()
    assert assembler.feed([record("a","The capital of",0,2)]) == []
    assert assembler.feed([record("b","France is.",2,3)]) == []
    assert assembler.feed([record("c","Paris.",3,4)]) == []
    output=assembler.feed([],final=True)
    assert [row["text"] for row in output] == ["The capital of France is Paris."]
    assert output[0]["start"] == 0 and output[0]["end"] == 4
    assert "source_ids" not in output[0]


def test_fragment_reconstruction_and_speaker_turn_order():
    assembler=UtteranceAssembler()
    rows=assembler.feed([record("a","What is the",0,1),record("b","capital of France?",1,2),
                         record("c","I think this is better.",2,4,"SPEAKER_01"),
                         record("d","Why?",4,5)])
    rows += assembler.feed([],final=True)
    assert [row["text"] for row in rows] == ["What is the capital of France?","I think this is better.","Why?"]
    assert [row["speaker"] for row in rows] == ["SPEAKER_00","SPEAKER_01","SPEAKER_00"]


def test_assembly_survives_checkpoint_crash_and_final_flush(tmp_path):
    store=ConversationStore(tmp_path/"turns.sqlite3")
    session=store.create_session()
    try:
        assembler=UtteranceAssembler()
        assert assembler.feed([record("a","The capital of",0,2)]) == []
        store.append_many(session,[],audio_commit=("spool",0,32000,{"utterance_pending":assembler.state()}))
        assert store.jobs(session)==[]
        state=store.audio_speaker_state(session,"spool")
        restored=UtteranceAssembler(state["utterance_pending"])
        restored.feed([record("b","France is Paris.",2,4)])
        output=restored.feed([],final=True)
        store.append_many(session,output,assembly_update=("spool",{"utterance_pending":None}))
        assert store.jobs(session)[0]["text"]=="The capital of France is Paris."
        assert store.audio_speaker_state(session,"spool")["utterance_pending"] is None
        store.append_many(session,output)
        assert len(store.jobs(session))==1
    finally:
        store.close()


def test_boundary_duplicate_requires_overlapping_timestamps():
    assembler=UtteranceAssembler()
    assembler.feed([record("a","What your country",0,6)])
    assembler.feed([record("b","country can do for you.",5.8,9)])
    assert assembler.feed([],final=True)[0]["text"]=="What your country can do for you."
    other=UtteranceAssembler()
    other.feed([record("c","This is very",0,1),record("d","very important.",1,2)])
    assert "very very" in other.feed([],final=True)[0]["text"]


@pytest.mark.parametrize("text,label",[("भारत की राजधानी नई दिल्ली है।","claim"),
    ("भारत की राजधानी क्या है","question"),("यह एक अच्छा विचार है?","question"),
    ("मुझे लगता है यह सही है","opinion"),("What is the","fragment")])
def test_clear_speech_acts_and_unfinished_inputs(text,label):
    classifier=SentenceClassifier()
    if label=="fragment":
        assert is_fragment(text)
        assert classifier.classify_intent(text)["source"]=="unfinished_syntax"
    else:
        assert classifier.classify_intent(text)["top_category"]==label


def test_worker_survives_sigint_and_exits_zero_after_drain(tmp_path):
    previous=signal.getsignal(signal.SIGINT)
    buffer,worker,_=start_llm_pipeline(interrupting_llm,interval_sec=.01,db_path=tmp_path/"signal.sqlite3",
        classifier=SentenceClassifier(mock_engine=ready_engine),preload=True,startup_timeout=10)
    buffer.append("A","A complete factual claim.")
    deadline=time.monotonic()+10
    while buffer.store.pending_count(buffer.session_id) and time.monotonic()<deadline:
        time.sleep(.02)
    summary=buffer.close(timeout=5)
    assert summary["worker_exitcode"]==0 and not summary["forced"] and summary["pending"]==0
    assert signal.getsignal(signal.SIGINT)==previous


def test_console_interrupt_is_request_not_transaction_exception():
    previous=signal.getsignal(signal.SIGINT)
    requested=threading.Event()
    with capture_interrupts(requested):
        signal.raise_signal(signal.SIGINT)
        assert requested.is_set()
        with shield_shutdown():
            signal.raise_signal(signal.SIGINT)
    assert signal.getsignal(signal.SIGINT)==previous


def test_ctrl_c_during_sqlite_commit_finishes_atomic_capture(tmp_path):
    from types import SimpleNamespace
    store=ConversationStore(tmp_path/"capture.sqlite3")
    buffer=TranscriptBuffer(store,store.create_session())
    original=store._append
    signaled=[]
    def append(*args,**kwargs):
        if not signaled:
            signaled.append(True)
            signal.raise_signal(signal.SIGINT)
        return original(*args,**kwargs)
    store._append=append
    class Stream:
        def __init__(self,**kwargs): self.reads=0
        def __enter__(self): return self
        def __exit__(self,*args): pass
        def read(self,frames):
            time.sleep(.01)
            self.reads+=1
            if self.reads>8: raise KeyboardInterrupt
            return np.full(frames,.1,dtype=np.float32),False
    class Model:
        def transcribe(self,audio,**kwargs):
            duration=len(audio)/16000
            return iter([SimpleNamespace(text="Why?",start=0,end=duration,words=None)]),SimpleNamespace(language="en")
    try:
        summary=run_capture(Model(),buffer,language="en",max_seconds=.5,stream_factory=Stream,shutdown_timeout=5)
        assert summary["interrupted"] and summary["unprocessed_sec"]==0
        assert store.jobs(buffer.session_id)
        assert not store.db.in_transaction
    finally:
        store.close()


def test_adapter_count_is_session_bound_and_keeps_centroids(monkeypatch):
    from types import SimpleNamespace
    from diarization import DiarizationPipeline
    options={}
    class Annotation:
        def labels(self): return ["local"]
        def itertracks(self,**kwargs): return iter([(SimpleNamespace(start=0,end=1),None,"local")])
    class Tensor:
        def float(self): return self
        def unsqueeze(self,*args): return self
    monkeypatch.setitem(sys.modules,"torch",SimpleNamespace(from_numpy=lambda _:Tensor()))
    def pipeline(data,**kwargs):
        options.update(kwargs)
        return SimpleNamespace(speaker_diarization=Annotation(),exclusive_speaker_diarization=Annotation(),
                               speaker_embeddings=np.array([[1.,0.]]))
    adapter=object.__new__(DiarizationPipeline)
    adapter.pipeline=pipeline
    turns=adapter(np.ones(16000,dtype=np.float32),num_speakers=2)
    assert options=={"min_speakers":1,"max_speakers":2}
    assert turns.embeddings=={"local":[1.,0.]}


@pytest.mark.parametrize("text,label",[("Good evening everyone.","other"),("Yeah, I'm done.","other"),
    ("But is it true that AI understands emotions?","question"),
    ("So, would it be better to use AI as an assistant?","question"),
    ("I don't think AI understands emotions.","opinion"),
    ("AI should replace teachers.","opinion")])
def test_recording_speech_acts(text,label):
    assert SentenceClassifier().classify_intent(text)["top_category"]==label


def test_tiny_prefix_boundary_repair_preserves_complete_interjections():
    from diarization import assign_word_speakers
    transcript={"segments":[{"text":"I don't think so.","start":0,"end":2,"words":[
        {"word":"I","start":0,"end":.4},{"word":" don't think so.","start":.4,"end":2}]}]}
    result=assign_word_speakers([(0,.35,"SPEAKER_00"),(.35,2,"SPEAKER_01")],transcript)
    assert [w["speaker"] for w in result["segments"][0]["words"]]==["SPEAKER_01","SPEAKER_01"]
    transcript["segments"][0]["words"][0]["word"]="Yes."
    result=assign_word_speakers([(0,.35,"SPEAKER_00"),(.35,2,"SPEAKER_01")],transcript)
    assert result["segments"][0]["words"][0]["speaker"]=="SPEAKER_00"


def test_auto_language_does_not_override_best_language(tmp_path):
    from types import SimpleNamespace
    from audio_capture import SpeechPipeline
    spool=AudioSpool(tmp_path/"language",session_id="s")
    spool.append(np.ones(64000,dtype=np.float32))
    spool.finish()
    calls=[]
    class Model:
        def transcribe(self,audio,**kwargs):
            calls.append(kwargs)
            return iter([SimpleNamespace(text="The Earth is round.",start=0,end=4,words=None)]),SimpleNamespace(
                language="en",language_probability=.755,all_language_probs=[("en",.755),("hi",.108)])
    pipeline=SpeechPipeline(spool,Model(),language="auto")
    try:
        segments,language,_=pipeline._transcribe((0,64000,True))
        assert len(calls)==1 and calls[0]["language"] is None
        assert language=="en" and segments[0]["text"]=="The Earth is round."
    finally:
        pipeline.close()


def test_decoder_question_mark_does_not_close_incomplete_chunk():
    assembler=UtteranceAssembler()
    assert assembler.feed([record("a","Is it true that AI can completely?",0,6)]) == []
    assembler.feed([record("b","understand social needs?",6,9)])
    assert assembler.feed([],final=True)[0]["text"]=="Is it true that AI can completely understand social needs?"


def test_slow_request_defers_durably_and_worker_still_exits_zero(tmp_path):
    path=tmp_path/"deferred.sqlite3"
    buffer,worker,_=start_llm_pipeline(blocked_llm,interval_sec=.01,db_path=path,
        classifier=SentenceClassifier(mock_engine=ready_engine),preload=True,startup_timeout=10)
    session=buffer.session_id
    buffer.append("A","The Earth is round.")
    deadline=time.monotonic()+5
    while buffer.store.jobs(session)[0]["status"] != "running" and time.monotonic()<deadline:
        time.sleep(.01)
    assert buffer.store.jobs(session)[0]["status"]=="running"
    summary=buffer.close(timeout=3)
    assert summary["worker_exitcode"]==0 and not summary["forced"] and summary["pending"]==1
    store=ConversationStore(path)
    try:
        job=store.jobs(session)[0]
        assert job["status"]=="pending" and job["result"] is None
        assert "acknowledgment" in job["error"]
    finally:
        store.close()


@pytest.mark.parametrize("text,expected",[("AI can answer questions instantly.","claim"),
    ("I can answer questions instantly, generate study material and adapt lessons.","claim"),
    ("Can AI answer questions instantly?","question"),("AI can answer questions?","question"),
    ("I think AI can answer questions instantly.","opinion")])
def test_capability_assertion_is_not_a_question_about_questions(text,expected):
    assert SentenceClassifier().classify_intent(text)["top_category"]==expected


@pytest.mark.parametrize("available",[True,False])
def test_fixed_count_enables_backend_and_fails_before_capture_if_unavailable(monkeypatch,available):
    from types import SimpleNamespace
    import live_whisper_diarize as live
    closed=[]
    buffer=SimpleNamespace(session_id="test",close=lambda **kw:closed.append(True) or {"worker_exitcode":0})
    backend=object() if available else None
    monkeypatch.setattr(live,"load_environment",lambda:None)
    monkeypatch.setattr(live,"validate_environment",lambda:None)
    monkeypatch.setattr(live,"start_llm_pipeline",lambda *a,**k:(buffer,None,None))
    monkeypatch.setattr(live,"create_whisper_model",lambda *a,**k:object())
    monkeypatch.setattr(live,"create_diarization_pipeline",lambda **kw:backend)
    captured=[]
    monkeypatch.setattr(live,"run_capture",lambda *a,**kw:captured.append(kw) or {})
    if available:
        live.live_transcribe(num_speakers=2)
        assert captured[0]["diarizer"] is backend and captured[0]["speaker_options"]["num_speakers"]==2
    else:
        with pytest.raises(RuntimeError,match="capture was not started"):
            live.live_transcribe(num_speakers=2)
        assert not captured
    assert closed==[True]
