"""Durable chronological dispatch in a bounded, Windows-spawn-compatible worker.

The public factory still returns (buffer, process, stop_event). append() commits
an utterance and job together. Audio callbacks never perform database operations.
"""
from __future__ import annotations

import json
import math
import multiprocessing
import os
import time
import uuid

import logger
from conversation import ConversationStore
from orchestrator import Orchestrator
from session_lock import SessionLock
from shutdown_signals import ignore_worker_interrupts, shield_shutdown
from audio_capture import SerialExecutor


class TranscriptBuffer:
    def __init__(self, store, session_id, max_pending=1000):
        self.store, self.session_id, self.max_pending = store, session_id, max_pending
        self._closed = False
        self.timeline_offset = self.store.db.execute(
            "SELECT coalesce(max(end),0) FROM utterances WHERE session_id=?", (session_id,)).fetchone()[0]

    def append(self, speaker, text, **metadata):
        if self._closed:
            raise RuntimeError("Transcript buffer is closed")
        if hasattr(self, "_process") and not self._process.is_alive():
            raise RuntimeError("Intent worker exited; audio must be retained for recovery")
        return self.store.append(self.session_id, speaker, text, max_pending=self.max_pending, **metadata)

    def speaker_stats(self):
        stats = {}
        for job in self.store.jobs(self.session_id):
            stat = stats.setdefault(job["speaker"], {"total": 0, "pending": 0})
            stat["total"] += 1
            stat["pending"] += job["status"] in ("pending", "running")
        return stats

    def append_many(self, records, *, audio_commit=None, assembly_update=None):
        if self._closed:
            raise RuntimeError("Transcript buffer is closed")
        if hasattr(self, "_process") and not self._process.is_alive():
            raise RuntimeError("Intent worker exited; audio must be retained for recovery")
        result = self.store.append_many(self.session_id, records, max_pending=self.max_pending,
                                       audio_commit=audio_commit, assembly_update=assembly_update)
        if hasattr(self, "_wake"):
            self._wake.set()
        return result

    def close(self, timeout=20):
        with shield_shutdown():
            return self._close(timeout)

    def _close(self, timeout):
        """Drain within one deadline; unfinished work stays durable for resumption."""
        if self._closed:
            return self._shutdown_summary
        # Leave a small reaping margin; a daemon job can be deferred durably
        # without making normal Ctrl+C look like a crashed worker.
        self._deadline.value = time.monotonic()+max(0,timeout)-min(.5,max(0,timeout)/2)
        self._stop.set()
        self._wake.set()
        self._process.join(timeout=max(0, timeout))
        forced = self._process.is_alive()
        if forced:
            self._process.terminate()
            self._process.join(timeout=5)
        if self._process.is_alive():
            raise RuntimeError("Could not stop pipeline worker; database ownership was retained")
        self.store.release(self._owner)
        summary = {"session_id": self.session_id, "forced": forced,
                   "pending": self.store.pending_count(self.session_id),
                   "failed": self.store.job_counts(self.session_id).get("failed", 0),
                   "worker_exitcode": self._process.exitcode}
        self._shutdown_summary, self._closed = summary, True
        self.store.close()
        self._session_lock.close()
        logger.log_info(f"Pipeline shutdown: {json.dumps(summary)}")
        return summary


def process_next(store, session_id, owner, orchestrator, *, max_attempts=3, retry_delay=1,
                 lease_seconds=120):
    """Process one job. A result and its acknowledgment are committed atomically."""
    turn = store.claim(session_id, owner, max_attempts=max_attempts, lease_seconds=lease_seconds)
    if turn is None:
        return False
    started = time.monotonic()
    try:
        logger.log_info(f"Dispatch session={session_id} utterance={turn['id']} attempt={turn['attempts']}")
        result = orchestrator.process(turn)
        store.complete(turn["id"], owner, result)
    except Exception as exc:
        store.fail(turn["id"], owner, exc, max_attempts=max_attempts,
                   retry_delay=retry_delay * 2 ** (turn["attempts"] - 1))
        logger.log_error(f"Dispatch failed session={session_id} utterance={turn['id']}", exc)
    else:
        logger.log_llm_response(json.dumps(result, ensure_ascii=False), time.monotonic() - started,
                                speaker_block=f"session={session_id} utterance={turn['id']}")
        try:
            print(f"\n[Pipeline {session_id}/{turn['sequence']} | {result['route']} | {result['status']}]\n"
                  f"{result['text']}", flush=True)
        except (OSError, UnicodeError):
            pass
    return True


def _llm_worker(db_path, session_id, owner, stop_event, interval_sec, llm_fn,
                max_attempts, retry_delay, classifier, handlers, min_confidence, log_path,
                wake, ready, preload, shutdown_deadline):
    ignore_worker_interrupts()
    logger.configure(log_path=log_path)
    store = ConversationStore(db_path)
    orchestrator = Orchestrator(store, classifier=classifier, handlers=handlers,
                                llm_fn=llm_fn, min_confidence=min_confidence)
    executor = None
    active = None
    def dispatch_one():
        # SQLite connections have one thread owner. A blocked external request
        # never prevents the child controller from observing its shutdown deadline.
        job_store = ConversationStore(db_path)
        try:
            job_orchestrator = Orchestrator(job_store, classifier=orchestrator.classifier,
                handlers=orchestrator.handlers, min_confidence=min_confidence)
            return process_next(job_store,session_id,owner,job_orchestrator,
                max_attempts=max_attempts,retry_delay=retry_delay,
                lease_seconds=max(120,float(os.environ.get("LLM_TIMEOUT_SEC","30"))+30))
        finally:
            job_store.close()
    try:
        if preload:
            if classifier is None:
                import torch
                torch.set_num_threads(max(1, int(os.environ.get("CLASSIFIER_CPU_THREADS", "2"))))
            orchestrator.preload()
        if preload:
            ready.send((True, None))
        executor = SerialExecutor("IntentJob")
        while True:
            if stop_event.is_set() and not shutdown_deadline.value:
                shutdown_deadline.value = time.monotonic()+20
            if active is not None:
                if active.done():
                    worked = active.result()
                    active = None
                    if not worked:
                        time.sleep(.05)
                elif stop_event.is_set() and time.monotonic() >= shutdown_deadline.value:
                    logger.log_event("intent_job_deferred",session_id=session_id,reason="shutdown_deadline")
                    break
                else:
                    time.sleep(.01)
                    continue
            if stop_event.is_set() and time.monotonic() >= shutdown_deadline.value:
                break
            if store.pending_count(session_id):
                active = executor.submit(dispatch_one)
                time.sleep(.01)
                continue
            if stop_event.is_set():
                break
            if not stop_event.is_set():
                wake.wait(interval_sec)
            wake.clear()
    except BaseException as exc:
        try:
            ready.send((False, str(exc)))
        except (OSError, BrokenPipeError):
            pass
        logger.log_event("worker_failed", session_id=session_id, error=str(exc))
        raise
    finally:
        if executor is not None:
            executor.shutdown(wait=active is None,cancel_futures=active is not None)
        ready.close()
        store.close()


def start_llm_pipeline(llm_fn=None, interval_sec=20.0, *, db_path=None, session_id=None,
                       max_attempts=3, retry_delay=1.0, max_pending=1000,
                       classifier=None, handlers=None, min_confidence=0.5,
                       preload=False, startup_timeout=180):
    """Start one sequential worker; injected callables/classifiers must be picklable.

    Resume with the same database and session_id. Succeeded and exhausted jobs are
    not replayed. Delivery is at least once if death occurs between an external
    response and its commit. Call buffer.close() for bounded shutdown. Model
    loading occurs only in the worker on first use.
    """
    if (not math.isfinite(interval_sec) or interval_sec <= 0
            or not isinstance(max_attempts, int) or max_attempts < 1
            or not math.isfinite(retry_delay) or retry_delay < 0
            or not isinstance(max_pending, int) or max_pending < 1):
        raise ValueError("Invalid pipeline interval, retry, or backlog setting")
    if not 0 <= min_confidence <= 1:
        raise ValueError("min_confidence must be between zero and one")
    request_timeout = float(os.environ.get("LLM_TIMEOUT_SEC", "30"))
    if not math.isfinite(request_timeout) or not 0 < request_timeout <= 300:
        raise ValueError("LLM_TIMEOUT_SEC must be between zero and 300 seconds")
    store = ConversationStore(db_path)
    session_id = store.create_session(session_id)
    try:
        session_lock = SessionLock(store.path, session_id)
    except BaseException:
        store.close()
        raise
    owner = uuid.uuid4().hex
    ctx = multiprocessing.get_context("spawn")
    stop = ctx.Event()
    wake = ctx.Event()
    shutdown_deadline = ctx.Value("d",0)
    receiver, sender = ctx.Pipe(duplex=False)
    process = ctx.Process(target=_llm_worker, args=(
        store.path, session_id, owner, stop, interval_sec, llm_fn, max_attempts,
        retry_delay, classifier, handlers, min_confidence, logger.get_log_path(), wake, sender, preload, shutdown_deadline),
        name="LLMPipelineWorker", daemon=True)
    buffer = TranscriptBuffer(store, session_id, max_pending)
    buffer._process, buffer._stop, buffer._owner = process, stop, owner
    buffer._session_lock = session_lock
    buffer._wake = wake
    buffer._deadline = shutdown_deadline
    try:
        process.start()
        sender.close()
        if preload:
            if not receiver.poll(startup_timeout):
                raise TimeoutError("Classifier initialization timed out before capture")
            success, error = receiver.recv()
            if not success:
                raise RuntimeError(f"Classifier initialization failed before capture: {error}")
    except BaseException:
        if process.is_alive():
            process.terminate()
            process.join(5)
        store.close()
        session_lock.close()
        raise
    finally:
        receiver.close()
    print(f"[Pipeline] session={session_id} database={store.path}", flush=True)
    return buffer, process, stop
