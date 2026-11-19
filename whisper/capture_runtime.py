"""Shared live/recovery loop. Capture, ASR, diarization and intent work have separate owners."""
from __future__ import annotations

import sqlite3
import time
import uuid
import threading
from pathlib import Path

import logger
from audio_capture import AudioSpool, AudioSegmenter, MicrophoneRecorder, SpeechPipeline
from live_runtime import prepare_segments
from utterance_segmentation import UtteranceAssembler
from shutdown_signals import shield_shutdown, capture_interrupts


def run_capture(model, buffer, *, diarizer=None, language="hi", task="transcribe",
                display_fn=None, pause_seconds=0.8, max_seconds=6, energy_threshold=0.015,
                beam_size=1, speaker_options=None, recovery=None, max_pending_seconds=120,
                shutdown_timeout=20, stream_factory=None):
    directory = Path(recovery) if recovery else Path(buffer.store.path).parent / "audio" / uuid.uuid4().hex
    spool = AudioSpool(directory, session_id=buffer.session_id, timeline_offset=buffer.timeline_offset,
                       resume=bool(recovery), max_pending_seconds=max_pending_seconds)
    # SQLite is authoritative: a crash after its commit but before manifest ack is safe.
    checkpoint = buffer.store.audio_checkpoint(buffer.session_id, spool.state["id"])
    if spool.state["processed_samples"] > checkpoint:
        raise ValueError("Recovery spool checkpoint does not match this SQLite database")
    spool.acknowledge(checkpoint)
    segmenter = AudioSegmenter(checkpoint, pause_seconds=pause_seconds, max_seconds=max_seconds,
                              energy_threshold=energy_threshold)
    speaker_state = buffer.store.audio_speaker_state(buffer.session_id, spool.state["id"])
    if speaker_state is None:
        speakers = buffer.store.db.execute("SELECT DISTINCT speaker FROM utterances WHERE session_id=?", (buffer.session_id,))
        indices = [int(row[0].split("_")[-1]) for row in speakers
                   if row[0].startswith("SPEAKER_") and row[0].split("_")[-1].isdigit()]
        speaker_state = {"previous": [], "next_id": max(indices, default=-1) + 1}
    assembler = UtteranceAssembler(speaker_state.get("utterance_pending"))
    committed_state = dict(speaker_state)
    pipeline = SpeechPipeline(spool, model, diarizer, language=language, task=task,
                              beam_size=beam_size, speaker_options=speaker_options,
                              speaker_state=speaker_state)
    recorder = None if recovery else MicrophoneRecorder(spool, stream_factory=stream_factory)
    started, cpu_started = time.perf_counter(), time.process_time()
    stopping, deadline, failure, chunks = bool(recovery), None, None, 0
    interrupted = False
    shutdown_guard = None
    logger.log_event("capture_ready", session_id=buffer.session_id, spool=str(directory),
                     recovery=bool(recovery), checkpoint=checkpoint, max_pending_seconds=max_pending_seconds)
    print(f"[Capture] durable audio: {directory}", flush=True)
    stop_requested = threading.Event()
    signal_guard = capture_interrupts(stop_requested)
    signal_guard.__enter__()
    try:
        if recorder:
            recorder.start()
        while True:
            try:
                if stop_requested.is_set() and not interrupted:
                    interrupted, stopping = True, True
                    deadline = time.monotonic() + shutdown_timeout
                    if recorder:
                        recorder.stop()
                result = pipeline.pop_ready()
                if result is not None:
                    records = prepare_segments(buffer, result["segments"], language=result["language"],
                        offset=result["offset"], chunk_id=f"{spool.state['id']}:{result['start']}:{result['end']}",
                        display_fn=display_fn)
                    ready_records = assembler.feed(records, watermark=spool.state["timeline_offset"]+result["end"]/16000)
                    for record in ready_records:
                        record["display_text"] = display_fn(record["text"]) if display_fn else record["text"]
                    state = {**result["speaker_state"], "utterance_pending": assembler.state()}
                    for attempt in range(3):
                        try:
                            buffer.append_many(ready_records, audio_commit=(spool.state["id"], result["start"], result["end"], state))
                            break
                        except sqlite3.OperationalError as exc:
                            if "locked" not in str(exc).lower() or attempt == 2:
                                raise
                            time.sleep(0.05 * 2 ** attempt)
                    spool.acknowledge(result["end"])
                    committed_state = state
                    chunks += 1
                    metrics = result["metrics"]
                    logger.log_event("speech_chunk", session_id=buffer.session_id, spool_id=spool.state["id"],
                        start_sample=result["start"], end_sample=result["end"],
                        backlog_sec=(spool.snapshot()["recorded_samples"] - result["end"]) / 16000,
                        **metrics)
                    text = " ".join(segment.get("text", "") for segment in result["segments"])
                    if text:
                        if diarizer:
                            from live_whisper_diarize import _print_diarized_output
                            _print_diarized_output(result["segments"], "hinglish" if display_fn else "hindi",
                                                   str(result["language"]).upper())
                        else:
                            print(f"[TRANSCRIBED {result['language']}] {display_fn(text) if display_fn else text}", flush=True)
                if recorder and recorder.done.is_set() and not stopping:
                    stopping, failure = True, recorder.error
                    deadline = time.monotonic() + shutdown_timeout
                snapshot = spool.snapshot()
                while len(pipeline.pending) < 2 and segmenter.cursor < snapshot["recorded_samples"]:
                    end = min(segmenter.cursor + 4000, segmenter.start + segmenter.maximum, snapshot["recorded_samples"])
                    span = segmenter.feed(spool.read(segmenter.cursor, end))
                    if span:
                        pipeline.submit(span)
                if stopping and segmenter.cursor == snapshot["recorded_samples"] and len(pipeline.pending) < 2:
                    tail = segmenter.flush()
                    if tail:
                        pipeline.submit(tail)
                    elif not pipeline.pending:
                        records = assembler.feed([], final=True)
                        for record in records:
                            record["display_text"] = display_fn(record["text"]) if display_fn else record["text"]
                        state = {**committed_state, "utterance_pending": None}
                        buffer.append_many(records, assembly_update=(spool.state["id"],state))
                        break
                if deadline is not None and time.monotonic() >= deadline:
                    logger.log_event("speech_drain_timeout", spool=str(directory))
                    break
                time.sleep(0.01)
            except KeyboardInterrupt:
                interrupted = True
                if shutdown_guard is None:
                    shutdown_guard = shield_shutdown()
                    shutdown_guard.__enter__()
                if stopping:
                    break
                stopping, deadline = True, time.monotonic() + shutdown_timeout
                if recorder:
                    recorder.stop()
            except Exception as exc:
                failure = exc
                break
    finally:
        try:
            with shield_shutdown():
                if recorder:
                    recorder.stop()
                pipeline.close(wait=not pipeline.pending)
        finally:
            # Running native calls cannot be cancelled by a Python thread. Completed
            # results are committed only by this loop; any abandoned input is replayable.
            if shutdown_guard is not None:
                shutdown_guard.__exit__(None,None,None)
            signal_guard.__exit__(None,None,None)
    state = spool.snapshot()
    interrupted = interrupted or stop_requested.is_set()
    summary = {"spool": str(directory), "recorded_sec": state["recorded_samples"] / 16000,
               "peak_pending_sec": state.get("peak_pending_samples", 0) / 16000,
               "unprocessed_sec": (state["recorded_samples"] - state["processed_samples"]) / 16000,
               "chunks_committed": chunks, "wall_sec": time.perf_counter() - started,
               "interrupted": interrupted,
               "process_cpu_sec": time.process_time() - cpu_started, "error": str(failure) if failure else None}
    logger.log_event("capture_summary", session_id=buffer.session_id, **summary)
    print(f"[Capture summary] {summary}", flush=True)
    if failure and not isinstance(failure, KeyboardInterrupt):
        raise RuntimeError(f"Capture stopped: {failure}. Audio retained at {directory}") from failure
    if summary["unprocessed_sec"] and not interrupted:
        raise TimeoutError(f"Speech drain incomplete; restart and recover {directory}. Native inference may still be finishing.")
    return summary
