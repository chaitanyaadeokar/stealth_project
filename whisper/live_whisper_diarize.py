import os
import sys
import time
import queue
import threading
import argparse
import re
import warnings
import numpy as np
import sounddevice as sd

# Suppress noisy numpy RuntimeWarnings from pyannote.audio internals
warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy")
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# Module-level reusable Groq client — initialized once in live_transcribe(), reused by _llm_worker.
# Avoids creating a new HTTP connection pool on every LLM call.
_groq_client = None

# Ensure Windows terminal handles UTF-8 cleanly
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# --- Windows CUDA DLL Setup ---
def _setup_cuda_dlls():
    try:
        import site
        candidate_paths = []
        for sp in site.getsitepackages() + [p for p in sys.path if "site-packages" in p]:
            nvidia_base = os.path.join(sp, "nvidia")
            if os.path.isdir(nvidia_base):
                for pkg in ["cublas", "cudnn", "cuda_nvrtc"]:
                    bin_dir = os.path.join(nvidia_base, pkg, "bin")
                    if os.path.isdir(bin_dir):
                        candidate_paths.append(bin_dir)

        unique_paths = list(dict.fromkeys(candidate_paths))
        if unique_paths:
            for p in unique_paths:
                if hasattr(os, "add_dll_directory"):
                    try:
                        os.add_dll_directory(p)
                    except OSError:
                        pass
            os.environ["PATH"] = ";".join(unique_paths) + ";" + os.environ.get("PATH", "")
    except Exception:
        pass

_setup_cuda_dlls()

import ctranslate2
from faster_whisper import WhisperModel

# --- Inject local whisperx into sys.path so we can import DiarizationPipeline ---
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_WHISPERX_ROOT = os.path.normpath(os.path.join(_SCRIPT_DIR, "..", "whisperx"))
if _WHISPERX_ROOT not in sys.path:
    sys.path.insert(0, _WHISPERX_ROOT)

# --- Speaker Diarization imports (optional, graceful fallback) ---
_DIARIZATION_AVAILABLE = False
DiarizationPipeline = None
_assign_word_speakers = None

try:
    from whisperx.diarize import DiarizationPipeline, assign_word_speakers as _assign_word_speakers
    _DIARIZATION_AVAILABLE = True
except ImportError as _e:
    print(f"⚠️  Diarization not available (missing dependency): {_e}", file=sys.stderr)
    print("    Install with: pip install pyannote.audio torch", file=sys.stderr)


# ---------------------------------------------------------------------------
# ANSI color codes for speaker labels
# ---------------------------------------------------------------------------
_SPEAKER_COLORS = [
    "\033[96m",   # Cyan
    "\033[93m",   # Yellow
    "\033[92m",   # Green
    "\033[95m",   # Magenta
    "\033[91m",   # Red
    "\033[94m",   # Blue
]
_RESET = "\033[0m"
_BOLD  = "\033[1m"


def _speaker_color(speaker_id: str) -> str:
    """Map a speaker label like SPEAKER_00 to a stable ANSI color."""
    try:
        idx = int(speaker_id.split("_")[-1])
    except (ValueError, IndexError):
        idx = hash(speaker_id)
    return _SPEAKER_COLORS[idx % len(_SPEAKER_COLORS)]


# ---------------------------------------------------------------------------
# Fact-check via Groq
# ---------------------------------------------------------------------------
FACT_CHECK_SYSTEM_PROMPT = (
    "You are a fact-checking assistant. Analyze the statement provided by the "
    "speaker. Determine whether it is TRUE, FALSE, or UNCERTAIN. Give a short "
    "explanation for your decision."
)


def fact_check_statement(statement: str) -> str:
    """Send the exact Whisper transcription to Groq for fact-checking."""
    global _groq_client
    if _groq_client is None:
        # Fallback: construct client if not pre-initialized (e.g. called outside live_transcribe)
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY was not found in the .env file.")
        _groq_client = OpenAI(
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1",
        )
    response = _groq_client.chat.completions.create(
        model="openai/gpt-oss-120b",
        messages=[
            {"role": "system", "content": FACT_CHECK_SYSTEM_PROMPT},
            {"role": "user", "content": statement},
        ],
    )
    return response.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# Devanagari → Hinglish transliteration (unchanged from original)
# ---------------------------------------------------------------------------
VOWELS = {
    'अ': 'a', 'आ': 'aa', 'इ': 'i', 'ई': 'ee', 'उ': 'u', 'ऊ': 'oo',
    'ए': 'e', 'ऐ': 'ai', 'ओ': 'o', 'औ': 'au', 'ऋ': 'ri'
}
MATRAS = {
    'ा': 'a', 'ि': 'i', 'ी': 'ee', 'ु': 'u', 'ू': 'oo',
    'े': 'e', 'ै': 'ai', 'ो': 'o', 'ौ': 'au', 'ृ': 'ri',
    'ं': 'n', 'ँ': 'n', 'ः': 'h'
}
NUKTA = {
    'क़': 'q', 'ख़': 'kh', 'ग़': 'gh', 'ज़': 'z', 'ड़': 'r', 'ढ़': 'rh', 'फ़': 'f'
}
CONSONANTS = {
    'क': 'k', 'ख': 'kh', 'ग': 'g', 'घ': 'gh', 'ङ': 'ng',
    'च': 'ch', 'छ': 'chh', 'ज': 'j', 'झ': 'jh', 'ञ': 'ny',
    'ट': 't', 'ठ': 'th', 'ड': 'd', 'ढ': 'dh', 'ण': 'n',
    'त': 't', 'थ': 'th', 'द': 'd', 'ध': 'dh', 'न': 'n',
    'प': 'p', 'फ': 'f', 'ब': 'b', 'भ': 'bh', 'म': 'm',
    'य': 'y', 'र': 'r', 'ल': 'l', 'व': 'v', 'श': 'sh', 'ष': 'sh', 'स': 's', 'ह': 'h'
}


def devanagari_to_hinglish(text: str) -> str:
    """
    Converts Hindi Devanagari words to natural conversational Romanized Hinglish,
    leaving English words and punctuation completely untouched.
    """
    if not re.search(r'[\u0900-\u097F]', text):
        return text

    for k, v in NUKTA.items():
        text = text.replace(k, v)

    words = text.split(' ')
    out_words = []

    for word in words:
        if not re.search(r'[\u0900-\u097F]', word):
            out_words.append(word)
            continue

        rom = ''
        chars = list(word)
        n = len(chars)
        i = 0
        while i < n:
            ch = chars[i]
            if ch in VOWELS:
                rom += VOWELS[ch]
                i += 1
            elif ch in CONSONANTS:
                base = CONSONANTS[ch]
                next_ch = chars[i+1] if (i + 1) < n else ''

                if next_ch == '्':  # Virama / half letter
                    rom += base
                    i += 2
                elif next_ch in MATRAS:
                    rom += base + MATRAS[next_ch]
                    i += 2
                elif next_ch == '' or next_ch in ' .,?!:;"\'':
                    rom += base
                    i += 1
                elif next_ch in CONSONANTS:
                    lookahead2 = chars[i+2] if (i + 2) < n else ''
                    if lookahead2 in MATRAS and i >= 1:
                        rom += base
                    else:
                        rom += base + 'a'
                    i += 1
                else:
                    rom += base + 'a'
                    i += 1
            elif ch == '।':
                rom += '.'
                i += 1
            elif ch in MATRAS or ch == '्':
                i += 1
            else:
                rom += ch
                i += 1

        # Conversational cleanup
        rom = re.sub(r'aee\b', 'ai', rom)
        rom = re.sub(r'ee\b', 'i', rom)
        out_words.append(rom)

    return ' '.join(out_words)


# ---------------------------------------------------------------------------
# Model initialization
# ---------------------------------------------------------------------------
def create_whisper_model(
    model_size: str = "large-v3-turbo",
    device: str = None,
    compute_type: str = None
):
    """Initializes and returns the optimized Faster-Whisper model."""
    if device is None:
        device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"

    if compute_type is None:
        compute_type = "int8_float16" if device == "cuda" else "int8"

    print(f"Loading Whisper model '{model_size}' on {device.upper()} ({compute_type})...")
    model = WhisperModel(
        model_size,
        device=device,
        compute_type=compute_type,
        download_root=os.path.join(os.path.expanduser("~"), ".cache", "whisper")
    )
    print("Model loaded successfully!")
    return model


def create_diarization_pipeline(device: str = None):
    """
    Initialize the pyannote speaker diarization pipeline.

    Uses pyannote/speaker-diarization-community-1 (no HF license required).
    Falls back gracefully if unavailable, returning None.

    Args:
        device: 'cuda' or 'cpu'. Auto-detected if None.

    Returns:
        DiarizationPipeline instance, or None if unavailable.
    """
    if not _DIARIZATION_AVAILABLE:
        return None

    hf_token = os.getenv("HF_TOKEN")

    if device is None:
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"

    model_name = "pyannote/speaker-diarization-community-1"
    print(f"Loading diarization model '{model_name}' on {device.upper()}...")
    try:
        pipeline = DiarizationPipeline(
            model_name=model_name,
            token=hf_token,
            device=device,
        )
        print("Diarization model loaded successfully!")
        return pipeline
    except Exception as e:
        print(f"⚠️  Could not load diarization model: {e}", file=sys.stderr)
        print("    Continuing without speaker diarization.", file=sys.stderr)
        return None


# ---------------------------------------------------------------------------
# Speaker assignment helpers
# ---------------------------------------------------------------------------
def _build_transcript_dict_shifted(segments_list: list, offset_sec: float = 0.0) -> dict:
    """
    Build a WhisperX-compatible transcript dict from faster-whisper segments,
    shifting all timestamps forward by `offset_sec`.

    This is needed for rolling-buffer diarization: Whisper reports timestamps
    relative to the current chunk, but diarization runs on the full rolling
    buffer, so we shift to make them line up.
    """
    seg_dicts = []
    for seg in segments_list:
        d = {
            "start": seg.start + offset_sec,
            "end":   seg.end  + offset_sec,
            "text":  seg.text.strip(),
        }
        if hasattr(seg, "words") and seg.words:
            d["words"] = [
                {
                    "word":  w.word,
                    "start": w.start + offset_sec,
                    "end":   (w.end or w.start) + offset_sec,
                }
                for w in seg.words
            ]
        seg_dicts.append(d)
    return {"segments": seg_dicts}


def _assign_speakers_to_segments(
    diarize_pipeline,
    rolling_audio: np.ndarray,
    segments_list: list,
    chunk_offset_sec: float = 0.0,
    num_speakers=None,
    min_speakers=None,
    max_speakers=None,
) -> list:
    """
    Run diarization on the *rolling* audio buffer and assign speaker labels.

    Key insight: diarization is run on `rolling_audio` (the full session history
    so far) instead of just the current chunk. This gives pyannote cross-segment
    context so it can consistently label SPEAKER_00 vs SPEAKER_01 across time.

    The Whisper segment timestamps are shifted by `chunk_offset_sec` so they
    align with the rolling buffer's timeline before calling assign_word_speakers.

    Args:
        rolling_audio:    Full session audio accumulated so far.
        segments_list:    faster-whisper segments (timestamps relative to current chunk).
        chunk_offset_sec: Start time of current chunk within rolling_audio (seconds).

    Returns:
        List of segment dicts with 'speaker' field assigned.
    """
    # Shift Whisper timestamps to match rolling buffer timeline
    transcript_dict = _build_transcript_dict_shifted(segments_list, offset_sec=chunk_offset_sec)

    try:
        diarize_df = diarize_pipeline(
            rolling_audio,
            num_speakers=num_speakers,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
        )
        result = _assign_word_speakers(diarize_df, transcript_dict, fill_nearest=True)
        return result.get("segments", [])
    except Exception as e:
        print(f"\n⚠️  Diarization failed: {e}", file=sys.stderr)
        return transcript_dict.get("segments", [])


def _print_diarized_output(diarized_segments: list, output_script: str, detected_lang: str):
    """
    Print transcription with speaker labels and colors.
    Groups consecutive segments from the same speaker.
    """
    if not diarized_segments:
        return

    # Group consecutive same-speaker segments
    groups = []
    current_speaker = None
    current_texts = []

    for seg in diarized_segments:
        speaker = seg.get("speaker", "UNKNOWN")
        text = seg.get("text", "").strip()
        if not text:
            continue
        if speaker == current_speaker:
            current_texts.append(text)
        else:
            if current_texts:
                groups.append((current_speaker, " ".join(current_texts)))
            current_speaker = speaker
            current_texts = [text]

    if current_texts:
        groups.append((current_speaker, " ".join(current_texts)))

    print()  # newline after inline preview
    for speaker, text in groups:
        final_text = devanagari_to_hinglish(text) if output_script == "hinglish" else text
        color = _speaker_color(speaker)
        print(f"{color}{_BOLD}🗣  [{speaker} | {detected_lang}]:{_RESET} {final_text}")


# ---------------------------------------------------------------------------
# Pipeline worker threads
# ---------------------------------------------------------------------------

def _whisper_worker(transcription_queue, diarization_queue, model,
                    diarize_pipeline, language_mode, task, output_script):
    """
    Worker thread: pulls audio jobs from transcription_queue, runs Faster-Whisper,
    and pushes results to diarization_queue.

    All model.transcribe() parameters are identical to the original main-loop code.
    """
    while True:
        job = transcription_queue.get()
        try:
            audio        = job["audio"]           # numpy copy of accumulated_audio
            target_lang  = job["target_lang"]     # str or None
            has_diarize  = job["has_diarize"]      # bool

            # --- Transcription (parameters unchanged from original) ---
            segments_gen, info = model.transcribe(
                audio,
                beam_size=5,
                language=target_lang,
                task=task,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500),
                condition_on_previous_text=False,
                word_timestamps=has_diarize,
                no_speech_threshold=0.6,
                log_prob_threshold=-1.0,
            )

            # Auto-mode: re-run in Hindi if Hindi probability is significant
            if language_mode == "auto" and info and info.all_language_probs:
                hi_prob = next((p for l, p in info.all_language_probs if l == "hi"), 0.0)
                if hi_prob >= 0.10 and info.language != "hi":
                    segments_gen, info = model.transcribe(
                        audio,
                        beam_size=5,
                        language="hi",
                        task=task,
                        vad_filter=True,
                        vad_parameters=dict(min_silence_duration_ms=300),
                        condition_on_previous_text=False,
                        word_timestamps=has_diarize,
                    )

            # Materialise the generator (needed for diarization + re-use)
            segments_list = list(segments_gen)
            text_parts    = [s.text.strip() for s in segments_list if s.text.strip()]
            raw_text      = " ".join(text_parts).strip()

            if raw_text:
                detected_lang = info.language.upper() if info and info.language else "UNKNOWN"
                diarization_queue.put({
                    "segments_list":    segments_list,
                    "raw_text":         raw_text,
                    "detected_lang":    detected_lang,
                    "rolling_audio":    job["rolling_audio"],
                    "chunk_offset_sec": job["chunk_offset_sec"],
                    "has_diarize":      has_diarize,
                    "output_script":    output_script,
                })
        except Exception as e:
            print(f"\n❌ [WhisperWorker] Error: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
        finally:
            transcription_queue.task_done()


def _diarization_worker(diarization_queue, llm_queue,
                        diarize_pipeline, output_script,
                        num_speakers, min_speakers, max_speakers):
    """
    Worker thread: pulls transcription results from diarization_queue,
    runs pyannote speaker assignment (unchanged), prints diarized output,
    then buffers text at the SEGMENT level — one entry per Whisper/pyannote
    segment — before forwarding completed speaker turns to llm_queue.

    Terminal output (speaker labels, transcript) remains immediate per-fragment.
    The LLM only receives a genuinely completed speaker turn.

    _assign_speakers_to_segments() and _print_diarized_output() are called
    identically to the original main-loop code.
    """
    # ---------------------------------------------------------------------------
    # Speaker-turn buffer state (local to this worker — no shared state, no locks)
    # ---------------------------------------------------------------------------
    # Maps speaker label → list of individual segment texts buffered for that turn.
    speaker_turn_buffer: dict[str, list[str]] = {}

    # Maps speaker label → rolling-buffer timeline end-time of their last segment.
    # Used to calculate per-speaker temporal gaps independent of other speakers.
    last_end_per_speaker: dict[str, float] = {}

    # A gap between a speaker's own consecutive segments larger than this value
    # (seconds) signals a completed turn, not a breath-group pause.
    TURN_GAP_THRESHOLD: float = 1.5

    # ------------------------------------------------------------------
    # Inline helper: flush one speaker's buffer to llm_queue
    # ------------------------------------------------------------------
    def _send_turn(speaker: str, reason: str) -> None:
        fragments = speaker_turn_buffer.pop(speaker, [])
        if not fragments:
            return
        combined = " ".join(f.strip() for f in fragments if f.strip())
        combined = re.sub(r"  +", " ", combined).strip()
        if not combined:
            return
        attributed = f"[{speaker}]: {combined}"
        print(f"[TURN DEBUG] speaker={speaker} action=FLUSH "
              f'text="{combined[:80]}{"..." if len(combined) > 80 else ""}" '
              f'reason="{reason}"')
        try:
            llm_queue.put_nowait(attributed)
        except queue.Full:
            print("⚠️  [LLMWorker] Queue full — skipping fact-check for this turn.",
                  file=sys.stderr)

    while True:
        job = diarization_queue.get()
        try:
            segments_list    = job["segments_list"]
            raw_text         = job["raw_text"]
            detected_lang    = job["detected_lang"]
            rolling_audio    = job["rolling_audio"]    # numpy copy owned by this job
            chunk_offset_sec = job["chunk_offset_sec"]
            has_diarize      = job["has_diarize"]
            out_script       = job["output_script"]

            # ----------------------------------------------------------------
            # STEP 1 — Run diarization and print (unchanged from original)
            # ----------------------------------------------------------------
            if has_diarize and diarize_pipeline and segments_list:
                print(f"\r\033[K📝 TRANSCRIBED ({detected_lang}) "
                      f"[rolling: {len(rolling_audio)/16000:.1f}s]:")
                # --- Speaker assignment (unchanged call) ---
                diarized = _assign_speakers_to_segments(
                    diarize_pipeline,
                    rolling_audio,
                    segments_list,
                    chunk_offset_sec=chunk_offset_sec,
                    num_speakers=num_speakers,
                    min_speakers=min_speakers,
                    max_speakers=max_speakers,
                )
                # --- Print with speaker labels (unchanged call — immediate display) ---
                _print_diarized_output(diarized, out_script, detected_lang)
            else:
                # No diarization — original behavior
                diarized = []
                print(f"\r\033[KTRANSCRIBED TEXT:\n{raw_text}")

            # ----------------------------------------------------------------
            # STEP 2 — Process each segment independently (segment-level buffering)
            # ----------------------------------------------------------------
            # Track which speakers appeared in this job for the absent-speaker sweep.
            speakers_in_job: set[str] = set()

            for seg in diarized:
                speaker   = seg.get("speaker", "UNKNOWN")
                text      = seg.get("text",    "").strip()
                seg_start = seg.get("start",   0.0)
                seg_end   = seg.get("end",     0.0)

                if not text:
                    continue

                speakers_in_job.add(speaker)

                # Gap between this speaker's current segment and their last segment.
                # Each speaker's gap is tracked independently — SPEAKER_01 talking
                # does NOT affect SPEAKER_00's gap clock.
                gap = seg_start - last_end_per_speaker.get(speaker, seg_start)

                # If the same speaker has been silent long enough, their previous
                # turn is complete — flush it BEFORE buffering the new segment.
                if gap > TURN_GAP_THRESHOLD and speaker_turn_buffer.get(speaker):
                    _send_turn(speaker, f"temporal_gap ({gap:.2f}s > {TURN_GAP_THRESHOLD}s)")

                # Deduplicate: skip only if this text is EXACTLY the immediately
                # previous fragment for this speaker (no fuzzy matching).
                buf = speaker_turn_buffer.setdefault(speaker, [])
                if buf and buf[-1] == text:
                    print(f"[TURN DEBUG] speaker={speaker} action=SKIP_DUPLICATE "
                          f'text="{text[:60]}"')
                else:
                    buf.append(text)
                    print(f"[TURN DEBUG] speaker={speaker} action=BUFFER "
                          f'text="{text[:60]}{"..." if len(text) > 60 else ""}" '
                          f'gap={gap:.2f}s')

                # Advance this speaker's end-time clock.
                last_end_per_speaker[speaker] = seg_end

            # ----------------------------------------------------------------
            # STEP 3 — Absent-speaker flush sweep
            # ----------------------------------------------------------------
            # After processing all segments in this job, check speakers that did
            # NOT appear. If a speaker has been absent for > TURN_GAP_THRESHOLD
            # relative to the latest timestamp in this job, their turn is done.
            if diarized:
                job_latest_end = max(seg.get("end", 0.0) for seg in diarized)
            else:
                job_latest_end = 0.0

            for speaker, fragments in list(speaker_turn_buffer.items()):
                if speaker in speakers_in_job:
                    # Speaker still active in this job — keep accumulating.
                    print(f"[TURN DEBUG] speaker={speaker} action=HOLD "
                          f'reason="speaker still active" '
                          f'gap={job_latest_end - last_end_per_speaker.get(speaker, job_latest_end):.2f}s')
                    continue

                absence = job_latest_end - last_end_per_speaker.get(speaker, job_latest_end)
                if absence > TURN_GAP_THRESHOLD and fragments:
                    _send_turn(speaker, f"speaker_absent ({absence:.2f}s > {TURN_GAP_THRESHOLD}s)")
                elif fragments:
                    print(f"[TURN DEBUG] speaker={speaker} action=HOLD "
                          f'reason="speaker still active" '
                          f'gap={absence:.2f}s')

        except Exception as e:
            print(f"\n❌ [DiarizeWorker] Error: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
        finally:
            diarization_queue.task_done()


def _llm_worker(llm_queue):
    """
    Worker thread: pulls raw_text strings from llm_queue and calls
    fact_check_statement() (unchanged logic, reused _groq_client).
    Prints the LLM response in the same format as the original code.
    """
    while True:
        raw_text = llm_queue.get()
        try:
            # --- Fact-check (unchanged call) ---
            llm_response = fact_check_statement(raw_text)
            print(f"\nLLM RESPONSE:\n{llm_response}")
        except Exception as e:
            print(f"\nLLM ERROR: {e}")
        finally:
            llm_queue.task_done()


# ---------------------------------------------------------------------------
# Main transcription loop
# ---------------------------------------------------------------------------
def live_transcribe(
    model_name: str = "large-v3-turbo",
    language_mode: str = "hi",
    output_script: str = "hinglish",
    task: str = "transcribe",
    energy_threshold: float = 0.015,
    pause_threshold: float = 0.8,
    max_speech_duration: float = 30.0,
    enable_diarization: bool = False,
    num_speakers: int = None,
    min_speakers: int = None,
    max_speakers: int = None,
):
    """
    Real-time microphone transcription with optional speaker diarization.

    Speaker diarization uses pyannote/speaker-diarization-community-1 via the
    local whisperx package. Enable with --diarize flag.
    """
    model = create_whisper_model(model_name)
    sample_rate = 16000
    audio_queue = queue.Queue()

    # --- Initialize diarization pipeline (optional) ---
    diarize_pipeline = None
    if enable_diarization:
        if not _DIARIZATION_AVAILABLE:
            print("⚠️  Diarization requested but pyannote.audio is not installed.")
            print("    Run: pip install pyannote.audio torch")
            print("    Continuing without diarization.\n")
        else:
            diarize_pipeline = create_diarization_pipeline()

    # --- Initialize shared Groq client once (reused by _llm_worker via fact_check_statement) ---
    global _groq_client
    api_key = os.getenv("GROQ_API_KEY")
    if api_key:
        _groq_client = OpenAI(
            api_key=api_key,
            base_url="https://api.groq.com/openai/v1",
        )
    else:
        print("⚠️  GROQ_API_KEY not found — LLM fact-checking will be unavailable.", file=sys.stderr)

    # --- Pipeline queues ---
    transcription_queue = queue.Queue()            # main → WhisperWorker (unbounded)
    diarization_queue   = queue.Queue()            # WhisperWorker → DiarizeWorker (unbounded)
    llm_queue           = queue.Queue(maxsize=2)   # DiarizeWorker → LLMWorker (bounded — drop if backed up)

    # --- Start worker threads (daemon=True: exit automatically when main thread exits) ---
    threading.Thread(
        target=_whisper_worker,
        args=(transcription_queue, diarization_queue, model,
              diarize_pipeline, language_mode, task, output_script),
        daemon=True,
        name="WhisperWorker",
    ).start()
    threading.Thread(
        target=_diarization_worker,
        args=(diarization_queue, llm_queue,
              diarize_pipeline, output_script,
              num_speakers, min_speakers, max_speakers),
        daemon=True,
        name="DiarizeWorker",
    ).start()
    threading.Thread(
        target=_llm_worker,
        args=(llm_queue,),
        daemon=True,
        name="LLMWorker",
    ).start()

    def audio_callback(indata, frames, time_info, status):
        if status:
            print(status, file=sys.stderr)
        audio_queue.put(indata.copy())

    diarize_status = "✅ ON" if diarize_pipeline else ("❌ OFF (unavailable)" if enable_diarization else "⬜ OFF")

    print("\n" + "=" * 65)
    print(f"🎙️  LISTENING (Microphone active @ 16kHz)")
    print(f"🌐  Language Mode      : {language_mode.upper()} ('hi' for Hindi/Hinglish, 'en' for English, 'auto' for router)")
    print(f"🔤  Output Format      : {output_script.upper()} (Romanized English/Hindi)")
    print(f"⚡  Model             : {model_name} on {'CUDA' if ctranslate2.get_cuda_device_count() > 0 else 'CPU'}")
    print(f"👥  Speaker Diarization: {diarize_status}")
    if diarize_pipeline and (num_speakers or min_speakers or max_speakers):
        print(f"    Speakers: num={num_speakers}, min={min_speakers}, max={max_speakers}")
    print("    Press Ctrl+C to stop.")
    print("=" * 65 + "\n")

    try:
        with sd.InputStream(
            samplerate=sample_rate,
            channels=1,
            dtype="float32",
            blocksize=int(sample_rate * 0.1),
            callback=audio_callback
        ):
            accumulated_audio = np.array([], dtype=np.float32)
            last_speech_time = time.time()
            speech_started = False
            last_transcribed_time = 0.0

            # Rolling buffer: accumulates full session audio for cross-segment diarization.
            # We cap it at MAX_ROLLING_SECS to avoid unbounded memory growth.
            # Owned exclusively by the main thread — passed as a copy into worker jobs.
            rolling_audio = np.array([], dtype=np.float32)
            MAX_ROLLING_SECS = 60.0   # keep up to 60 seconds of context for diarization
            MAX_ROLLING_SAMPLES = int(sample_rate * MAX_ROLLING_SECS)

            while True:
                # --- Drain microphone queue into accumulated_audio ---
                while not audio_queue.empty():
                    chunk = audio_queue.get().flatten()
                    accumulated_audio = np.concatenate((accumulated_audio, chunk))

                    rms = np.sqrt(np.mean(chunk**2)) if len(chunk) > 0 else 0
                    if rms > energy_threshold:
                        last_speech_time = time.time()
                        speech_started = True

                current_time = time.time()
                audio_len_sec = len(accumulated_audio) / sample_rate

                is_pause      = speech_started and (current_time - last_speech_time >= pause_threshold)
                is_max_length = audio_len_sec >= max_speech_duration
                is_interim    = speech_started and (current_time - last_transcribed_time >= 3.0) and audio_len_sec >= 3.0

                if audio_len_sec >= 0.5 and (is_pause or is_max_length or is_interim):
                    target_lang = language_mode if language_mode in ["hi", "en"] else None
                    last_transcribed_time = current_time

                    if is_pause or is_max_length:
                        # --- Finalized utterance ---
                        # Rolling buffer management stays in main thread (sole owner of rolling_audio).
                        chunk_offset_sec = len(rolling_audio) / sample_rate
                        rolling_audio = np.concatenate((rolling_audio, accumulated_audio))
                        if len(rolling_audio) > MAX_ROLLING_SAMPLES:
                            excess = len(rolling_audio) - MAX_ROLLING_SAMPLES
                            rolling_audio = rolling_audio[excess:]
                            chunk_offset_sec = max(0.0, chunk_offset_sec - excess / sample_rate)

                        # Queue a copy of all data needed by workers — main thread resets immediately
                        transcription_queue.put({
                            "audio":            accumulated_audio.copy(),
                            "rolling_audio":    rolling_audio.copy(),
                            "chunk_offset_sec": chunk_offset_sec,
                            "target_lang":      target_lang,
                            "has_diarize":      bool(diarize_pipeline),
                            "output_script":    output_script,
                        })

                        # Reset immediately — main thread is unblocked; workers handle the rest
                        accumulated_audio = np.array([], dtype=np.float32)
                        speech_started = False

                    else:
                        # --- Interim preview: run Whisper inline for low-latency display ---
                        # (No diarization or LLM for interim — keep it fast)
                        segments_gen, info = model.transcribe(
                            accumulated_audio,
                            beam_size=5,
                            language=target_lang,
                            task=task,
                            vad_filter=True,
                            vad_parameters=dict(min_silence_duration_ms=500),
                            condition_on_previous_text=False,
                            word_timestamps=False,
                            no_speech_threshold=0.6,
                            log_prob_threshold=-1.0,
                        )
                        text_parts = [s.text.strip() for s in segments_gen if s.text.strip()]
                        raw_text   = " ".join(text_parts).strip()
                        if raw_text:
                            detected_lang = info.language.upper() if info and info.language else "UNKNOWN"
                            final_text = devanagari_to_hinglish(raw_text) if output_script == "hinglish" else raw_text
                            print(f"\r\033[K🎙️ [{detected_lang}]: {final_text}...", end="", flush=True)

                elif not speech_started and audio_len_sec > 2.0:
                    accumulated_audio = np.array([], dtype=np.float32)

                time.sleep(0.05)

    except KeyboardInterrupt:
        print("\n\n🛑 Microphone stopped.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Real-time Whisper transcription with optional Speaker Diarization and Hinglish output."
    )
    parser.add_argument(
        "--model", type=str, default="large-v3-turbo",
        help="Model size: small, medium, large-v3-turbo (default: large-v3-turbo)"
    )
    parser.add_argument(
        "--language", type=str, default="hi", choices=["hi", "en", "auto"],
        help="'hi' (default, best for Hindi & Hinglish), 'en' (pure English), 'auto' (dynamic router)"
    )
    parser.add_argument(
        "--output", type=str, default="hinglish", choices=["hinglish", "hindi"],
        help="'hinglish' (default, Roman Latin script), 'hindi' (pure Devanagari)"
    )
    parser.add_argument(
        "--task", type=str, default="transcribe", choices=["transcribe", "translate"],
        help="'transcribe' for native/hinglish, or 'translate' to translate Hindi to English"
    )

    # --- Speaker Diarization args ---
    parser.add_argument(
        "--diarize", action="store_true", default=False,
        help="Enable speaker diarization (requires pyannote.audio + torch)"
    )
    parser.add_argument(
        "--num-speakers", type=int, default=None,
        help="Exact number of speakers (if known). Overrides --min-speakers and --max-speakers."
    )
    parser.add_argument(
        "--min-speakers", type=int, default=None,
        help="Minimum number of speakers to detect (used when --num-speakers is not set)."
    )
    parser.add_argument(
        "--max-speakers", type=int, default=None,
        help="Maximum number of speakers to detect (used when --num-speakers is not set)."
    )

    args = parser.parse_args()

    live_transcribe(
        model_name=args.model,
        language_mode=args.language,
        output_script=args.output,
        task=args.task,
        enable_diarization=args.diarize,
        num_speakers=args.num_speakers,
        min_speakers=args.min_speakers,
        max_speakers=args.max_speakers,
    )
