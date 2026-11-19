"""Reproducible offline diagnostics. Actual model profiles use cached weights only."""
import argparse
import json
import os
import statistics
import time
from pathlib import Path


SAMPLES = [
    ("Why does the moon shine", "question"), ("Tell me the capital of France", "question"),
    ("Can you explain this result", "question"), ("kya pani 100 degree par ubalta hai", "question"),
    ("यह कैसे काम करता है", "question"), ("What do you think about solar energy?", "question"),
    ("In my opinion public transport is better", "opinion"), ("We should invest in renewable energy", "opinion"),
    ("Mujhe lagta hai yeh behtar hai", "opinion"), ("मुझे लगता है हमें शिक्षा पर निवेश करना चाहिए", "opinion"),
    ("The Earth revolves around the Sun.", "claim"), ("Water boils at 100 degrees Celsius at sea level.", "claim"),
    ("भारत की राजधानी नई दिल्ली है।", "claim"), ("Vaccines contain microchips.", "claim"),
    ("Okay.", "other"), ("Thank you", "other"),
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--intent", action="store_true")
    parser.add_argument("--asr", action="store_true")
    parser.add_argument("--diarization", action="store_true")
    parser.add_argument("--speech-pipeline", action="store_true", help="Profile actual chunked speech stages using cached models")
    parser.add_argument("--cpu-base", action="store_true", help="Use the cached CPU auto-profile model in data/models")
    parser.add_argument("--paced", action="store_true", help="Also feed prerecorded audio at microphone speed with actual speech models")
    parser.add_argument("--output", default="data/phase01_profile.json")
    args = parser.parse_args()
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    import torch
    torch.set_num_threads(2)
    result = {"logical_cpus": os.cpu_count(), "offline": True}
    if args.intent:
        from sentence_classifier import SentenceClassifier
        classifier = SentenceClassifier(device="cpu")
        started = time.perf_counter()
        classifier.load_model()
        result["classifier_load_sec"] = time.perf_counter() - started
        rows = []
        for text, expected in SAMPLES:
            row = {"text": text, "expected": expected}
            for mode in ("old", "new"):
                started, cpu = time.perf_counter(), time.process_time()
                prediction = (classifier.classify(text, categories=["question", "query", "claim", "fact", "opinion", "debate", "statement"])
                              if mode == "old" else classifier.classify_intent(text))
                label = prediction["top_category"]
                label = {"query": "question", "fact": "claim", "debate": "opinion", "statement": "other"}.get(label, label)
                accepted = prediction["confidence"] >= 0.5 and not prediction.get("ambiguous", False)
                row[mode] = {"label": label if accepted else None, "confidence": prediction["confidence"],
                             "scores": prediction["scores"], "wall_sec": time.perf_counter() - started,
                             "cpu_sec": time.process_time() - cpu}
            rows.append(row)
            print(json.dumps(row, ensure_ascii=True), flush=True)
        result["intent_samples"] = rows
        result["intent_summary"] = {mode: {
            "correct": sum(row[mode]["label"] == row["expected"] for row in rows),
            "abstained": sum(row[mode]["label"] is None for row in rows),
            "total": len(rows), "median_sec": statistics.median(row[mode]["wall_sec"] for row in rows),
            "total_sec": sum(row[mode]["wall_sec"] for row in rows)} for mode in ("old", "new")}
    if args.asr or args.diarization or args.speech_pipeline:
        from live_whisper_diarize import _setup_cuda_dlls
        _setup_cuda_dlls()
        from faster_whisper.audio import decode_audio
        audio = decode_audio("tests/jfk.flac", sampling_rate=16000)
    if args.asr or args.speech_pipeline:
        from faster_whisper import WhisperModel
        root = (Path("data/models/models--Systran--faster-whisper-base/snapshots") if args.cpu_base else
                Path.home() / ".cache/whisper/models--mobiuslabsgmbh--faster-whisper-large-v3-turbo/snapshots")
        result["asr_model"] = "base" if args.cpu_base else "large-v3-turbo"
        model = WhisperModel(str(next(root.iterdir())), device="cpu", compute_type="int8", cpu_threads=4, local_files_only=True)
        for beam in ((5, 1) if args.asr else ()):
            started, cpu = time.perf_counter(), time.process_time()
            segments, _ = model.transcribe(audio, beam_size=beam, language="en", vad_filter=True,
                                           word_timestamps=True, condition_on_previous_text=False)
            text = " ".join(s.text for s in segments)
            elapsed = time.perf_counter() - started
            result[f"asr_beam_{beam}"] = {"wall_sec": elapsed, "cpu_sec": time.process_time() - cpu,
                                         "audio_sec": len(audio)/16000, "rtf": elapsed/(len(audio)/16000), "text": text}
    if args.diarization:
        from diarization import DiarizationPipeline
        try:
            started = time.perf_counter()
            root = Path.home() / ".cache/huggingface/hub/models--pyannote--speaker-diarization-community-1/snapshots"
            pipeline = DiarizationPipeline(str(next(root.iterdir())), device="cpu")
            load_sec = time.perf_counter() - started
            started, cpu = time.perf_counter(), time.process_time()
            turns = pipeline(audio)
            result["diarization"] = {"load_sec": load_sec, "wall_sec": time.perf_counter()-started,
                                     "cpu_sec": time.process_time()-cpu, "turns": turns}
        except Exception as exc:
            result["diarization"] = {"error": str(exc)}
            pipeline = None
    if args.speech_pipeline:
        import tempfile
        import logger
        from audio_capture import AudioSpool
        from capture_runtime import run_capture
        from conversation import ConversationStore
        from llm_pipeline import TranscriptBuffer
        Path("data").mkdir(exist_ok=True)
        previous_log = logger._custom_log_path
        with tempfile.TemporaryDirectory(prefix="profile-", dir="data") as directory:
            directory = Path(directory)
            logger.configure(log_path=str(directory / "events.log"))
            store = ConversationStore(directory / "profile.sqlite3")
            session = store.create_session()
            buffer = TranscriptBuffer(store, session)
            spool = AudioSpool(directory / "audio", session_id=session)
            writes = []
            for index in range(0, len(audio), 4000):
                started = time.perf_counter()
                spool.append(audio[index:index + 4000])
                writes.append(time.perf_counter() - started)
            spool.finish()
            result["spool_write"] = {"blocks": len(writes), "median_sec": statistics.median(writes),
                                      "max_sec": max(writes), "total_sec": sum(writes)}
            result["speech_pipeline"] = run_capture(model, buffer, recovery=spool.directory,
                diarizer=pipeline if args.diarization else None, language="en", max_seconds=6)
            result["speech_results"] = [{key: job[key] for key in ("text", "speaker", "start", "end")}
                                        for job in store.jobs(session)]
            if args.paced:
                class PacedStream:
                    def __init__(self, **kwargs): self.position = 0
                    def __enter__(self): return self
                    def __exit__(self, *args): pass
                    def read(self, frames):
                        if self.position >= len(audio):
                            raise KeyboardInterrupt
                        block = audio[self.position:self.position + frames]
                        time.sleep(len(block) / 16000)
                        self.position += len(block)
                        return block, False
                result["paced_capture"] = run_capture(model, buffer, diarizer=pipeline if args.diarization else None,
                    language="en", max_seconds=6, stream_factory=PacedStream, shutdown_timeout=30)
            store.close()
            result["speech_stage_events"] = [json.loads(line.split("[EVENT] ", 1)[1]) for line in
                (directory / "events.log").read_text(encoding="utf-8").splitlines()
                if "[EVENT] " in line and '"event": "speech_chunk"' in line]
        logger._custom_log_path = previous_log
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(path), "summary": result.get("intent_summary")}), flush=True)


if __name__ == "__main__":
    main()
