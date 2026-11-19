"""Disk-backed capture, bounded segmentation, and two-stage speech processing.

Native inference never runs in the capture thread. A finite microphone cannot be
backpressured indefinitely: recording stops explicitly at backlog/disk limits.
Every successfully journaled sample remains available for recovery.
"""
from __future__ import annotations

import json
import os
import shutil
import threading
import time
import uuid
import queue
from collections import deque
from concurrent.futures import Future
from pathlib import Path

import logger


class SerialExecutor:
    """One bounded daemon owner per model; shutdown does not wait on stuck native code."""
    def __init__(self, name):
        self.queue = queue.Queue(maxsize=2)
        self.closed = False
        self.thread = threading.Thread(target=self._run, name=name, daemon=True)
        self.thread.start()

    def _run(self):
        while True:
            item = self.queue.get()
            if item is None:
                return
            future, function, args = item
            if future.set_running_or_notify_cancel():
                try:
                    future.set_result(function(*args))
                except BaseException as exc:
                    future.set_exception(exc)

    def submit(self, function, *args):
        if self.closed:
            raise RuntimeError("Model executor is closed")
        future = Future()
        self.queue.put_nowait((future, function, args))
        return future

    def shutdown(self, wait=True, cancel_futures=False):
        self.closed = True
        if cancel_futures:
            while True:
                try:
                    item = self.queue.get_nowait()
                except queue.Empty:
                    break
                if item is not None:
                    item[0].cancel()
        self.queue.put_nowait(None)
        if wait:
            self.thread.join()


class AudioSpool:
    sample_rate = 16000

    def __init__(self, directory, *, session_id=None, timeline_offset=0, resume=False,
                 max_pending_seconds=120, max_disk_bytes=512 * 1024 * 1024):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.directory / "capture.json"
        self.audio_path = self.directory / "audio.f32"
        self.lock = threading.Lock()
        self.max_pending = int(max_pending_seconds * self.sample_rate)
        self.max_disk_bytes = max_disk_bytes
        if self.max_pending <= 0 or max_disk_bytes <= 0:
            raise ValueError("Capture budgets must be positive")
        if resume:
            self.state = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            if session_id is not None and self.state["session_id"] != session_id:
                raise ValueError("Recovery spool belongs to a different conversation session")
            size = self.audio_path.stat().st_size
            if size < self.state["recorded_samples"] * 4:
                raise ValueError("Audio journal is shorter than its committed manifest")
        else:
            if self.manifest_path.exists() or self.audio_path.exists():
                raise FileExistsError("Capture spool already exists; use explicit recovery")
            self.state = {"id": uuid.uuid4().hex, "session_id": session_id,
                          "timeline_offset": timeline_offset, "recorded_samples": 0,
                          "processed_samples": 0, "closed": False, "error": None}
            self.audio_path.touch()
            self._save()

    def _save(self):
        temporary = self.directory / "capture.json.tmp"
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(self.state, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, self.manifest_path)

    def snapshot(self):
        with self.lock:
            return dict(self.state)

    def capacity_for(self, samples):
        with self.lock:
            pending = self.state["recorded_samples"] - self.state["processed_samples"]
            return (pending + samples <= self.max_pending
                    and (self.state["recorded_samples"] + samples) * 4 <= self.max_disk_bytes)

    def append(self, data):
        import numpy as np
        raw = data if isinstance(data, bytes) else np.asarray(data, dtype="<f4").tobytes()
        if len(raw) % 4:
            raise ValueError("Capture requires mono float32 samples")
        # Check before the next device read too; this second check protects test/replay callers.
        if not self.capacity_for(len(raw) // 4):
            raise OverflowError("Capture budget reached; recorded audio is retained in the spool")
        with self.lock:
            if self.state["closed"]:
                raise RuntimeError("Capture journal is closed")
            with self.audio_path.open("r+b") as stream:
                stream.seek(self.state["recorded_samples"] * 4)
                stream.write(raw)
                stream.flush()
                os.fsync(stream.fileno())
            self.state["recorded_samples"] += len(raw) // 4
            self.state["peak_pending_samples"] = max(self.state.get("peak_pending_samples", 0),
                self.state["recorded_samples"] - self.state["processed_samples"])
            self._save()

    def read(self, start, end):
        import numpy as np
        with self.lock:
            if not 0 <= start <= end <= self.state["recorded_samples"]:
                raise ValueError("Read extends beyond committed audio")
        with self.audio_path.open("rb") as stream:
            stream.seek(start * 4)
            raw = stream.read((end - start) * 4)
        if len(raw) != (end - start) * 4:
            raise OSError("Incomplete audio journal read")
        return np.frombuffer(raw, dtype="<f4").copy()

    def acknowledge(self, end):
        with self.lock:
            if not self.state["processed_samples"] <= end <= self.state["recorded_samples"]:
                raise ValueError("Invalid audio checkpoint")
            self.state["processed_samples"] = end
            self._save()

    def finish(self, error=None):
        with self.lock:
            self.state["closed"] = True
            self.state["error"] = str(error) if error else None
            self._save()


class MicrophoneRecorder:
    """One blocking-read I/O thread; no callback queue and no model work here."""
    def __init__(self, spool, stream_factory=None, block_samples=4000):
        self.spool, self.block_samples = spool, block_samples
        self.stream_factory = stream_factory
        self.stop_event = threading.Event()
        self.done = threading.Event()
        self.error = None
        self.thread = threading.Thread(target=self._run, name="AudioRecorder", daemon=True)

    def start(self):
        self.thread.start()

    def _run(self):
        try:
            if self.stream_factory is None:
                import sounddevice as sd
                factory = sd.RawInputStream
            else:
                factory = self.stream_factory
            with factory(samplerate=self.spool.sample_rate, channels=1, dtype="float32",
                         blocksize=self.block_samples) as stream:
                while not self.stop_event.is_set():
                    if not self.spool.capacity_for(self.block_samples):
                        raise OverflowError("Recording stopped at backlog/disk budget; recover the audio spool")
                    if shutil.disk_usage(self.spool.directory).free < 32 * 1024 * 1024:
                        raise OSError("Recording stopped: less than 32 MiB free disk space")
                    data, overflowed = stream.read(self.block_samples)
                    raw = data.tobytes() if hasattr(data, "tobytes") else bytes(data)
                    self.spool.append(raw)
                    if overflowed:
                        raise RuntimeError("Audio device reported lost input samples; journal contains all returned samples")
        except BaseException as exc:
            self.error = exc
            logger.log_event("capture_stopped", spool_id=self.spool.state["id"], error=str(exc))
        finally:
            try:
                self.spool.finish(self.error)
            except Exception as exc:
                self.error = self.error or exc
                logger.log_event("capture_manifest_failed", spool=str(self.spool.directory), error=str(exc))
            finally:
                self.done.set()

    def stop(self, timeout=2):
        self.stop_event.set()
        self.thread.join(timeout)
        if self.thread.is_alive():
            raise TimeoutError("Audio device read did not stop; capture journal is retained")


class AudioSegmenter:
    """Use sample durations, not processing wall time, to detect pauses and size caps."""
    def __init__(self, start=0, *, sample_rate=16000, energy_threshold=0.015,
                 pause_seconds=0.8, max_seconds=6, min_seconds=4):
        if pause_seconds <= 0 or max_seconds <= 0 or sample_rate <= 0:
            raise ValueError("Segmentation durations must be positive")
        self.start = self.cursor = start
        self.rate, self.threshold = sample_rate, energy_threshold
        self.pause = int(pause_seconds * sample_rate)
        self.maximum = max(1, int(max_seconds * sample_rate))
        self.minimum = min(self.maximum, max(1, int(min_seconds * sample_rate)))
        self.silence = 0
        self.speech = False

    def feed(self, block):
        import numpy as np
        if len(block) > self.maximum:
            raise ValueError("Feed blocks must be no longer than the segment limit")
        self.cursor += len(block)
        active = bool(len(block) and np.sqrt(np.mean(np.square(block))) > self.threshold)
        self.speech |= active
        self.silence = 0 if active else self.silence + len(block)
        if (self.cursor - self.start >= self.maximum or
                (self.speech and self.silence >= self.pause and self.cursor - self.start >= self.minimum)):
            return self.flush()
        return None

    def flush(self):
        if self.cursor == self.start:
            return None
        result = (self.start, self.cursor, self.speech)
        self.start, self.silence, self.speech = self.cursor, 0, False
        return result


class SpeechPipeline:
    """One ASR executor + one diarization executor, at most two ordered chunks.

    ASR for chunk N+1 may overlap diarization for N; each model is single-owner.
    Native CPU thread counts are limited by model factories before these start.
    """
    def __init__(self, spool, model, diarizer=None, *, language="hi", task="transcribe",
                 beam_size=1, window_seconds=4, speaker_options=None, retries=2, speaker_state=None):
        self.spool, self.model, self.diarizer = spool, model, diarizer
        self.language, self.task, self.beam_size = language, task, beam_size
        self.window_samples = int(window_seconds * spool.sample_rate)
        self.options, self.retries = speaker_options or {}, retries
        self.asr_pool = SerialExecutor("Whisper")
        self.diar_pool = SerialExecutor("Diarization")
        self.pending = deque()
        from diarization import SpeakerTracker
        self.tracker = SpeakerTracker(self.options.get("num_speakers"))
        self.tracker.restore(speaker_state)

    def _transcribe(self, span):
        start, end, speech = span
        before, cpu = time.perf_counter(), time.thread_time()
        segments, language = [], self.language
        if speech:
            for attempt in range(self.retries):
                try:
                    # Small acoustic context avoids cutting words at hard chunk boundaries.
                    # Word-midpoint ownership suppresses repeated context. Different
                    # model alignments across calls can still disagree near a boundary.
                    input_start = max(0, start - 6400)
                    input_end = min(self.spool.snapshot()["recorded_samples"], end + 4000)
                    audio = self.spool.read(input_start, input_end)
                    generator, info = self.model.transcribe(audio, language=self.language if self.language in ("hi", "en") else None,
                        task=self.task, beam_size=self.beam_size, vad_filter=True, word_timestamps=True,
                        condition_on_previous_text=False)
                    logger.log_event("asr_language", start_sample=start, requested=self.language,
                                     detected=info.language, probability=getattr(info,"language_probability",None))
                    # Auto means the model's best language, not a forced second
                    # decode merely because Hindi has a 10% secondary score.
                    from live_whisper_diarize import _build_transcript_dict_shifted
                    segments = _build_transcript_dict_shifted(list(generator), (input_start - start) / 16000)["segments"]
                    selected = []
                    for segment in segments:
                        if segment.get("words"):
                            words = [word for word in segment["words"]
                                     if 0 <= (word["start"] + word["end"]) / 2 < (end - start) / 16000]
                            if words:
                                selected.append(dict(segment, words=words,
                                    text="".join(word["word"] for word in words).strip(),
                                    start=words[0]["start"], end=words[-1]["end"]))
                        elif 0 <= (segment["start"] + segment["end"]) / 2 < (end - start) / 16000:
                            selected.append(segment)
                    segments = selected
                    if not segments:
                        logger.log_event("speech_without_transcript", start_sample=start, end_sample=end,
                                         spool_id=self.spool.state["id"], audio_retained=True)
                    language = info.language
                    break
                except Exception as exc:
                    logger.log_event("asr_retry", start_sample=start, attempt=attempt + 1, error=str(exc))
                    if attempt + 1 == self.retries:
                        raise
        return segments, language, {"asr_wall_sec": time.perf_counter() - before,
                                    "asr_thread_cpu_sec": time.thread_time() - cpu}

    def _diarize(self, asr, span):
        from diarization import assign_word_speakers
        start, end, speech = span
        segments, language, metrics = asr.result()
        before = time.perf_counter()
        if self.diarizer and segments:
            left = max(0, start - self.window_samples)
            try:
                turns = self.diarizer(self.spool.read(left, end), **self.options)
                absolute = [(a + left / 16000, b + left / 16000, speaker) for a, b, speaker in turns]
                stable = self.tracker.update(absolute, left / 16000, getattr(turns, "embeddings", None))
                shifted = []
                for segment in segments:
                    item = dict(segment, start=segment["start"] + start / 16000, end=segment["end"] + start / 16000)
                    item["words"] = [dict(word, start=word["start"] + start / 16000, end=word["end"] + start / 16000)
                                     for word in segment.get("words", [])]
                    shifted.append(item)
                segments = assign_word_speakers(stable, {"segments": shifted}, fill_nearest=True, max_gap=0.25)["segments"]
                offset = self.spool.state["timeline_offset"]
            except Exception as exc:
                logger.log_event("diarization_failed", start_sample=start, error=str(exc), fallback="UNKNOWN")
                offset = self.spool.state["timeline_offset"] + start / 16000
        else:
            offset = self.spool.state["timeline_offset"] + start / 16000
        metrics.update(diarization_wall_sec=time.perf_counter() - before,
                       audio_sec=(end - start) / 16000)
        return {"segments": segments, "language": language, "offset": offset,
                "start": start, "end": end, "metrics": metrics, "speaker_state": self.tracker.state()}

    def submit(self, span):
        if len(self.pending) >= 2:
            raise OverflowError("Speech pipeline already has two chunks in flight")
        self.pending.append(self.diar_pool.submit(self._diarize, self.asr_pool.submit(self._transcribe, span), span))

    def pop_ready(self):
        if self.pending and self.pending[0].done():
            return self.pending.popleft().result()
        return None

    def close(self, wait=True):
        self.asr_pool.shutdown(wait=wait, cancel_futures=not wait)
        self.diar_pool.shutdown(wait=wait, cancel_futures=not wait)
