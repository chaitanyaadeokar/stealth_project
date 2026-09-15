"""
llm_pipeline.py
---------------
Speaker-keyed rolling transcript buffer -> LLM pipeline using multiprocessing.

Windows 'spawn' compatibility
==============================
On Windows multiprocessing uses 'spawn': every arg passed to Process() must be
picklable.  The ONLY objects passed to the worker are:

    data        : Manager.dict proxy  -- picklable (holds manager address/authkey)
    ptrs        : Manager.dict proxy  -- picklable
    stop_event  : multiprocessing.Event -- picklable
    interval_sec: float               -- picklable
    llm_fn      : module-level func   -- picklable

NO Lock is passed.  Manager dict proxies are internally serialised by the
Manager server process, so concurrent access from main + worker is safe
without an additional explicit lock.

Buffer layout (per speaker)
============================
  data : { speaker_id -> [utt0, utt1, utt2, ...] }
  ptrs : { speaker_id -> int }    <- index of next unsent utterance

Every 20 s the worker reads utterances[ptr:] for each speaker, formats a
speaker-labelled block, sends it to the LLM in a daemon Thread, then advances
each speaker's pointer.

Non-blocking guarantees
=======================
- append()       : proxy dict call, no local I/O, returns immediately.
- get_new_spkrs(): proxy dict calls + pointer advance, no local I/O.
- LLM call       : daemon Thread inside worker -- never blocks the 20-s loop.
- Shutdown       : stop_event.wait(timeout) -- wakes instantly on Ctrl-C.
"""

from __future__ import annotations

import sys
import time
import threading
import multiprocessing
from typing import Any, Callable, Dict, List, Tuple


# ---------------------------------------------------------------------------
# Speaker-keyed transcript buffer  (main-process wrapper -- NEVER pickled)
# ---------------------------------------------------------------------------
class TranscriptBuffer:
    """
    Main-process-only wrapper around Manager dict proxies.

    This object is NEVER passed to the worker Process.  The worker receives
    the raw proxy objects (data, ptrs) which ARE picklable.

    Concurrency
    -----------
    Manager proxy method calls are serialised by the Manager server internally.
    No additional Lock is required.
    """

    def __init__(self, data: Any, ptrs: Any) -> None:
        self._data = data   # Manager.dict: { speaker -> [utt, ...] }
        self._ptrs = ptrs   # Manager.dict: { speaker -> int }

    # ------------------------------------------------------------------
    # Main-process API
    # ------------------------------------------------------------------
    def append(self, speaker: str, text: str) -> None:
        """
        Record one utterance for *speaker*.

        Parameters
        ----------
        speaker : str   e.g. "SPEAKER_00", "SPEAKER_01", "UNKNOWN"
        text    : str   the transcribed utterance
        """
        text = text.strip()
        if not text:
            return

        # Initialise a new speaker atomically
        if speaker not in self._data:
            self._data[speaker] = []
            self._ptrs[speaker] = 0

        # Read-modify-write (required for Manager dict proxy correctness)
        lst = list(self._data[speaker])
        lst.append(text)
        self._data[speaker] = lst

    # ------------------------------------------------------------------
    # Read-only diagnostics (main process)
    # ------------------------------------------------------------------
    def speaker_stats(self) -> Dict[str, Dict[str, int]]:
        """Return {speaker: {total, pending}} for logging."""
        stats: Dict[str, Dict[str, int]] = {}
        for sp in list(self._data.keys()):
            total   = len(self._data.get(sp, []))
            pointer = self._ptrs.get(sp, 0)
            stats[sp] = {"total": total, "pending": total - pointer}
        return stats

    def __repr__(self) -> str:
        try:
            stats = self.speaker_stats()
            parts = [f"{sp}(total={v['total']}, pending={v['pending']})"
                     for sp, v in stats.items()]
            return f"TranscriptBuffer([{', '.join(parts)}])"
        except Exception:
            return "TranscriptBuffer(<unavailable>)"


# ---------------------------------------------------------------------------
# Helpers used inside the worker process
# ---------------------------------------------------------------------------
def _get_new_speakers(
    data: Any,
    ptrs: Any,
) -> Dict[str, List[str]]:
    """
    Read new utterances (after each speaker's pointer) from the Manager proxies
    and advance each pointer.  Returns {} when nothing new is available.

    Called from the worker process with raw proxy objects.
    """
    result: Dict[str, List[str]] = {}
    for speaker in list(data.keys()):
        ptr        = ptrs.get(speaker, 0)
        utterances = list(data.get(speaker, []))
        new_utts   = utterances[ptr:]
        if new_utts:
            result[speaker] = new_utts
            ptrs[speaker]   = len(utterances)   # advance pointer
    return result


def _format_speaker_block(new_speakers: Dict[str, List[str]]) -> str:
    """
    Format per-speaker new utterances into a readable LLM prompt block.

    Example:
        [SPEAKER_00]: Hello, how are you doing today?
        [SPEAKER_01]: I am fine, thanks for asking.
    """
    lines = []
    for speaker, utts in new_speakers.items():
        lines.append(f"[{speaker}]: {' '.join(utts)}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# LLM worker target  (runs in its own Process)
# ---------------------------------------------------------------------------
def _llm_worker(
    data: Any,
    ptrs: Any,
    stop_event: multiprocessing.Event,
    interval_sec: float,
    llm_fn: Callable[[str], str],
) -> None:
    """
    Worker function executed in a dedicated Process.

    Receives ONLY picklable objects:
        data, ptrs   : Manager.dict proxies
        stop_event   : multiprocessing.Event
        interval_sec : float
        llm_fn       : module-level callable

    Non-blocking design
    -------------------
    1. stop_event.wait(timeout) replaces sleep() -- wakes instantly on Ctrl-C.
    2. Each LLM call runs in a daemon Thread -- the 20-s loop never stalls.
    3. In-flight threads joined (max 15 s) on graceful shutdown.
    """
    active_threads: List[threading.Thread] = []

    def _dispatch_llm(speaker_block: str, timestamp: str) -> None:
        """Daemon thread: calls LLM and prints result."""
        try:
            result = llm_fn(speaker_block)
            bar = "-" * 64
            print(
                f"\n{bar}\n"
                f"[LLM Pipeline | {timestamp}]\n"
                f"--- Transcript (speaker-wise) ---\n"
                f"{speaker_block}\n"
                f"--- LLM Response ---\n"
                f"{result}\n"
                f"{bar}",
                flush=True,
            )
        except Exception as exc:
            print(
                f"\n[LLM Pipeline] LLM call failed: {exc}",
                file=sys.stderr,
                flush=True,
            )

    print(
        f"\n[LLM Pipeline] Worker ready  (interval={interval_sec}s)",
        flush=True,
    )

    while not stop_event.is_set():
        # Non-blocking wait: wakes on timeout OR stop signal
        stop_event.wait(timeout=interval_sec)

        new_speakers = _get_new_speakers(data, ptrs)

        if new_speakers:
            ts            = time.strftime("%H:%M:%S")
            speaker_block = _format_speaker_block(new_speakers)

            # Log pointer positions
            ptr_summary = " | ".join(
                f"{sp} ptr={ptrs.get(sp, 0)}" for sp in new_speakers
            )
            print(f"[LLM Pipeline | {ts}] Dispatching -- {ptr_summary}", flush=True)

            t = threading.Thread(
                target=_dispatch_llm,
                args=(speaker_block, ts),
                daemon=True,
            )
            t.start()
            active_threads.append(t)

        # Prune finished threads
        active_threads = [t for t in active_threads if t.is_alive()]

    # ------------------------------------------------------------------
    # Graceful shutdown: drain in-flight LLM calls
    # ------------------------------------------------------------------
    if active_threads:
        print(
            f"\n[LLM Pipeline] Draining {len(active_threads)} in-flight call(s)...",
            flush=True,
        )
        for t in active_threads:
            t.join(timeout=15)

    print("\n[LLM Pipeline] Worker stopped cleanly.", flush=True)


# ---------------------------------------------------------------------------
# Public factory
# ---------------------------------------------------------------------------
def start_llm_pipeline(
    llm_fn: Callable[[str], str],
    interval_sec: float = 20.0,
) -> Tuple[TranscriptBuffer, multiprocessing.Process, multiprocessing.Event]:
    """
    Create the speaker-keyed buffer and start the LLM worker process.

    The Manager is created here and its proxy dicts are passed to both
    TranscriptBuffer (main process wrapper) AND the worker Process args.
    The Manager object itself is stored only on the returned buffer, so it
    lives as long as the buffer lives and is never pickled.

    Usage
    -----
        buffer, worker, stop_ev = start_llm_pipeline(fact_check_statement)

        # After diarization:
        buffer.append("SPEAKER_00", "Hello there.")
        buffer.append("SPEAKER_01", "Hi, how are you?")

        # Fallback (no diarization):
        buffer.append("UNKNOWN", raw_text)

        # On shutdown (KeyboardInterrupt):
        stop_ev.set()
        worker.join(timeout=20)

    Parameters
    ----------
    llm_fn       : module-level Callable[[str], str]  (must be picklable)
    interval_sec : float  seconds between LLM dispatches (default 20)

    Returns
    -------
    buffer     : TranscriptBuffer         -- call buffer.append() from main loop
    process    : multiprocessing.Process  -- daemon worker (already started)
    stop_event : multiprocessing.Event   -- set() to initiate shutdown
    """
    # Manager lives in main process; proxy objects are picklable connection refs
    manager = multiprocessing.Manager()
    data    = manager.dict()    # { speaker_id: [utt0, utt1, ...] }
    ptrs    = manager.dict()    # { speaker_id: int }
    # Manager.Event IS picklable (proxy object over socket).
    # multiprocessing.Event is NOT picklable on Windows 'spawn' mode.
    stop_event = manager.Event()

    # Thin main-process wrapper -- never passed to the worker
    buffer          = TranscriptBuffer(data, ptrs)
    buffer._manager = manager   # keep Manager server alive as long as buffer lives

    # Worker receives ONLY picklable args: proxy dicts + Manager.Event + scalars + fn
    process = multiprocessing.Process(
        target=_llm_worker,
        args=(data, ptrs, stop_event, interval_sec, llm_fn),
        name="LLMPipelineWorker",
        daemon=True,
    )
    process.start()

    print(
        f"\n[LLM Pipeline] Worker started  (PID={process.pid}, interval={interval_sec}s)",
        flush=True,
    )

    return buffer, process, stop_event
