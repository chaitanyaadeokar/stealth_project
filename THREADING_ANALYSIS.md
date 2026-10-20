# Threading & Concurrency Architecture Report

## 1. Executive Summary

This project implements a multi-tier concurrent architecture designed for real-time speech processing and asynchronous Large Language Model (LLM) fact-checking. To ensure the microphone audio capture and live transcription never experience audio stuttering, dropped frames, or network stalls, the codebase separates concerns using:

1. **Native OS Audio Threads** (`sounddevice` / PortAudio C-thread) for real-time microphone ingestion.
2. **Thread-Safe Queues** (`queue.Queue`) for lockless inter-thread audio transport.
3. **The Main Thread** for energy-based Voice Activity Detection (VAD), faster-whisper inference, pyannote diarization, and buffer feeding.
4. **Isolated Multiprocessing** (`multiprocessing.Process` & `multiprocessing.Manager`) to detach the periodic LLM pipeline from the Python Global Interpreter Lock (GIL) and STT execution.
5. **Dynamic Daemon Worker Threads** (`threading.Thread`) inside the LLM worker process to execute network I/O calls asynchronously without stalling the 20-second dispatch loop.
6. **Thread-Safe Log Synchronization** (`threading.Lock`) to serialize concurrent file writes to `vague.log`.

---

## 2. Architecture & Data Flow Diagram

```mermaid
flowchart TD
    subgraph AudioCapture["Microphone Ingestion (PortAudio)"]
        Mic["Microphone Hardware"] -->|16kHz PCM Audio| AudioThread["PortAudio C-Thread (sounddevice)"]
        AudioThread -->|audio_callback() indata| AudioQ[("queue.Queue (audio_queue)")]
    end

    subgraph MainProcess["Main Process (STT & Diarization)"]
        AudioQ -->|Drain non-blocking| MainThread["Main Application Thread"]
        MainThread -->|Energy VAD calculation| VAD["Speech Activity Detection"]
        MainThread -->|Accumulated Chunks| WhisperEngine["CTranslate2 Whisper (OpenMP/CUDA)"]
        MainThread -->|Rolling Audio Context| PyannoteEngine["Pyannote Diarization (PyTorch)"]
        MainThread -->|Append Speaker Utterance| SharedBuffer[("multiprocessing.Manager().dict()")]
        MainThread -->|Log Events| VagueLog[("vague.log")]
    end

    subgraph LLMProcess["LLM Pipeline Worker Process (multiprocessing.Process)"]
        SharedBuffer -->|Read every 20s| WorkerLoop["Worker Process Loop (stop_event.wait)"]
        WorkerLoop -->|Spawn on new data| DaemonThread1["Daemon Thread: _dispatch_llm()"]
        WorkerLoop -->|Spawn on next interval| DaemonThread2["Daemon Thread: _dispatch_llm()"]
        DaemonThread1 -->|Groq API Call| GroqLLM["Groq LLM (gpt-oss-120b)"]
        DaemonThread2 -->|Groq API Call| GroqLLM
        DaemonThread1 -->|Log Response & Latency| VagueLog
        DaemonThread2 -->|Log Response & Latency| VagueLog
    end
```

---

## 3. Concurrency Component Breakdown

### 3.1 Audio Ingestion Thread (`sounddevice.InputStream`)
* **File Reference**: `whisper/live_whisper_diarize.py` & `whisper/live_whisper.py`
* **Thread Type**: Dedicated native thread spawned by PortAudio (C-level).
* **Execution Model**:
  The stream operates in callback mode at a 16 kHz sample rate with a block size of 100 ms (`0.1 * 16000 = 1600` samples).
  Whenever the operating system audio buffer has 100 ms of fresh PCM audio, PortAudio calls `audio_callback()` in its own high-priority thread.
* **Synchronization Mechanism**:
  ```python
  def audio_callback(indata, frames, time_info, status):
      if status:
          print(status, file=sys.stderr)
      audio_queue.put(indata.copy())
  ```
  `audio_queue` is a Python standard library `queue.Queue`. It provides thread-safe, mutex-protected FIFO queuing so that the audio callback can return in sub-millisecond time without blocking the soundcard.

---

### 3.2 Main Processing Thread (STT, Diarization, and VAD)
* **File Reference**: `whisper/live_whisper_diarize.py`
* **Responsibilities**:
  1. **Audio Drain Loop**: Repeatedly pulls incoming chunks from `audio_queue` without blocking.
  2. **Energy Thresholding & Silence Detection**: Computes RMS energy of chunks. If RMS exceeds `energy_threshold`, it marks `speech_started = True` and tracks `last_speech_time`.
  3. **STT Finalization Triggers**:
     * **Condition 1 (Pause)**: If speech was active and silence exceeds `pause_threshold` (0.8s), it finalizes.
     * **Condition 2 (30-Second Limit)**: If continuous audio reaches `max_speech_duration` (30.0s), it finalizes.
  4. **Speaker Diarization Alignment**: Runs `_assign_speakers_to_segments` against `rolling_audio` to match Whisper word timestamps with speaker labels.
  5. **Buffer Feeding**: Appends finalized utterances into `llm_buffer` (`TranscriptBuffer`).

---

### 3.3 The LLM Worker Process (`multiprocessing.Process`)
* **File Reference**: `whisper/llm_pipeline.py`
* **Why Multiprocessing Instead of Threading?**
  * On Windows, Python's Global Interpreter Lock (GIL) and `ctranslate2` / PyTorch computation can cause micro-stalls if heavy JSON serialization, HTTP requests, or timeout handling run in Python threads within the same process.
  * Spawning a dedicated process (`LLMPipelineWorker`) completely isolates the LLM pipeline from the real-time audio pipeline.
* **IPC (Inter-Process Communication)**:
  * Uses `multiprocessing.Manager().dict()` for `data` (transcripts keyed by speaker) and `ptrs` (speaker utterance pointers).
  * Uses `manager.Event()` for `stop_event` (which works cleanly across Windows `spawn` boundaries).
  * The worker loop uses non-blocking synchronization:
    ```python
    stop_event.wait(timeout=interval_sec)
    ```
    This eliminates busy-waiting loops and wakes up instantly on shutdown without delay.

---

### 3.4 Daemon Dispatch Threads (`threading.Thread`)
* **File Reference**: `whisper/llm_pipeline.py`
* **Thread Type**: Short-lived Python daemon threads created inside the `LLMPipelineWorker` process.
* **Why Daemon Threads Here?**
  * Calling the Groq LLM API (`client.chat.completions.create`) is a network I/O operation that takes 1–3 seconds (or longer during network jitter).
  * If the worker process called the LLM synchronously, the 20-second interval timer would stall, drifting the schedule and missing updates.
  * Instead, every 20 seconds, the worker launches:
    ```python
    t = threading.Thread(
        target=_dispatch_llm,
        args=(speaker_block, ts),
        daemon=True,
    )
    t.start()
    active_threads.append(t)
    ```
* **Thread Lifecycle Management**:
  * **Pruning**: `active_threads = [t for t in active_threads if t.is_alive()]` dynamically removes completed threads.
  * **Graceful Drain on Exit**: On shutdown, remaining in-flight threads are given up to 15 seconds to finish their API responses:
    ```python
    for t in active_threads:
        t.join(timeout=15)
    ```

---

### 3.5 Thread-Safe File Logging Lock (`whisper/logger.py`)
* **File Reference**: `whisper/logger.py`
* **Mechanism**:
  ```python
  _process_lock = threading.Lock()
  ```
* **Purpose**:
  Because both the main STT thread and concurrent LLM daemon threads write to `vague.log`, `_process_lock` prevents line interleaving and race conditions during file write operations.
  Each write opens in append mode (`"a"`), flushes immediately, and closes, which also maintains safe concurrency across Windows processes.

---

### 3.6 Lower-Level C++/CUDA Compute Thread Pools
* **Libraries**: `ctranslate2`, `torch`, `pyannote.audio`
* **Under the Hood**:
  * `faster-whisper` uses CTranslate2, which spawns an internal thread pool (via OpenMP / native C++ threads) sized to CPU cores (e.g., 4 to 8 threads) to execute transformer matrix multiplications.
  * When CUDA is active, asynchronous CUDA compute streams execute kernel dispatches without blocking CPU execution.

---

## 4. Summary Table of Concurrency Roles

| Thread / Process Component | Context / Parent | Technology | Purpose | Blocking Characteristics |
| :--- | :--- | :--- | :--- | :--- |
| **PortAudio Audio Thread** | OS / Hardware | PortAudio C-thread | Ingests 16kHz microphone audio frames | Hard real-time; never blocks |
| **Audio Queue** | Shared (Mic $\to$ Main) | `queue.Queue` | Safely transfers audio chunks between threads | Thread-safe, lock-protected FIFO |
| **Main STT Thread** | Main Process | Python Main Thread | Runs VAD, Whisper transcription, Diarization, and Buffer append | Handles compute loop; drains audio queue |
| **LLM Worker Process** | Subprocess | `multiprocessing.Process` | Isolated process for periodic LLM pipeline | Decoupled from STT process; independent GIL |
| **Manager Proxy Server** | Background Thread | `multiprocessing.Manager` | Serializes read/writes to speaker dicts and pointers | Socket-based RPC between main and worker processes |
| **LLM Dispatch Daemon Threads** | Inside LLM Worker | `threading.Thread` | Makes remote Groq API network calls and logs results | Non-blocking to 20s interval loop; joined on exit |
| **Logger Synchronization Lock** | Process-local | `threading.Lock` | Ensures clean atomic writes to `vague.log` | Short, millisecond-level mutex lock |
| **Inference Compute Threads** | Background (CT2/Torch) | OpenMP / MKL / BLAS | Performs tensor math across CPU cores | Native compute threads managed by C++ runtimes |

---

## 5. Non-Blocking Guarantees & Fault Tolerance

1. **Audio Drops Prevented**:
   The PortAudio callback performs zero heavy operations—it solely calls `audio_queue.put(indata.copy())`. Even if Whisper transcription takes 500 ms on a chunk, incoming audio accumulates safely in the queue.
2. **STT Loop Never Waits on LLM**:
   The main thread merely appends text via `llm_buffer.append(speaker, text)`. It never waits for network handshakes, JSON decoding, or LLM tokens.
3. **LLM Loop Never Misses Ticks**:
   The LLM worker process executes each prompt in a daemon thread. If an LLM call takes 4 seconds, the next 20-second timer continues running on schedule without delay.
4. **Clean Windows Shutdown**:
   When `Ctrl+C` (`KeyboardInterrupt`) is received, the main thread sets `llm_stop_event`. The worker immediately wakes from `stop_event.wait()` (avoiding `sleep()` stalls), drains active worker threads, and exits cleanly.
