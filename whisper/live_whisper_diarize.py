import os
import sys
import time
import queue
import argparse
import re
import warnings
import multiprocessing
import numpy as np
import sounddevice as sd

# Suppress noisy numpy RuntimeWarnings from pyannote.audio internals
warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy")
from dotenv import load_dotenv
from openai import OpenAI
from llm_pipeline import start_llm_pipeline

load_dotenv()

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
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("GROQ_API_KEY was not found in the .env file.")

    client = OpenAI(
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1",
    )
    response = client.chat.completions.create(
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
    llm_interval_sec: float = 20.0,
):
    """
    Real-time microphone transcription with optional speaker diarization.

    Speaker diarization uses pyannote/speaker-diarization-community-1 via the
    local whisperx package. Enable with --diarize flag.
    """
    model = create_whisper_model(model_name)
    sample_rate = 16000
    audio_queue = queue.Queue()

    # --- Start the non-blocking rolling-buffer LLM pipeline ---
    # fact_check_statement is a module-level function so it is picklable on Windows.
    llm_buffer, llm_worker, llm_stop_event = start_llm_pipeline(
        llm_fn=fact_check_statement,
        interval_sec=llm_interval_sec,
    )

    # --- Initialize diarization pipeline (optional) ---
    diarize_pipeline = None
    if enable_diarization:
        if not _DIARIZATION_AVAILABLE:
            print("⚠️  Diarization requested but pyannote.audio is not installed.")
            print("    Run: pip install pyannote.audio torch")
            print("    Continuing without diarization.\n")
        else:
            diarize_pipeline = create_diarization_pipeline()

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
            rolling_audio = np.array([], dtype=np.float32)
            MAX_ROLLING_SECS = 30.0  # keep up to 5 minutes of context
            MAX_ROLLING_SAMPLES = int(sample_rate * MAX_ROLLING_SECS)

            while True:
                # Drain queue
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
                is_interim    = speech_started and (current_time - last_transcribed_time >= 3.0) and audio_len_sec >= 30.0

                if audio_len_sec >= 30.0 and (is_pause or is_max_length or is_interim):
                    target_lang = language_mode if language_mode in ["hi", "en"] else None

                    # --- Transcription ---
                    segments_gen, info = model.transcribe(
                        accumulated_audio,
                        beam_size=5,
                        language=target_lang,
                        task=task,
                        vad_filter=True,
                        vad_parameters=dict(min_silence_duration_ms=500),
                        condition_on_previous_text=False,
                        word_timestamps=bool(diarize_pipeline),
                        no_speech_threshold=0.6,   # discard segments Whisper isn't confident about
                        log_prob_threshold=-1.0,    # drop low-probability hallucinations
                    )

                    # Auto-mode: re-run in Hindi if Hindi probability is significant
                    if language_mode == "auto" and info and info.all_language_probs:
                        hi_prob = next((p for l, p in info.all_language_probs if l == "hi"), 0.0)
                        if hi_prob >= 0.10 and info.language != "hi":
                            segments_gen, info = model.transcribe(
                                accumulated_audio,
                                beam_size=5,
                                language="hi",
                                task=task,
                                vad_filter=True,
                                vad_parameters=dict(min_silence_duration_ms=300),
                                condition_on_previous_text=False,
                                word_timestamps=bool(diarize_pipeline),
                            )

                    # Materialise the generator (needed for diarization + re-use)
                    segments_list = list(segments_gen)
                    text_parts = [s.text.strip() for s in segments_list if s.text.strip()]
                    raw_text = " ".join(text_parts).strip()
                    last_transcribed_time = current_time

                    if raw_text:
                        detected_lang = info.language.upper() if info and info.language else "UNKNOWN"

                        if is_pause or is_max_length:
                            # --- Finalized segment: run diarization if enabled ---
                            if diarize_pipeline and segments_list:
                                # --- Rolling buffer management ---
                                # Record where the current chunk starts in the rolling buffer
                                chunk_offset_sec = len(rolling_audio) / sample_rate

                                # Append current chunk to rolling buffer
                                rolling_audio = np.concatenate((rolling_audio, accumulated_audio))

                                # Trim rolling buffer if it exceeds max duration
                                if len(rolling_audio) > MAX_ROLLING_SAMPLES:
                                    excess = len(rolling_audio) - MAX_ROLLING_SAMPLES
                                    rolling_audio = rolling_audio[excess:]
                                    chunk_offset_sec = max(0.0, chunk_offset_sec - excess / sample_rate)

                                print(f"\r\033[K📝 TRANSCRIBED ({detected_lang}) "
                                      f"[rolling: {len(rolling_audio)/sample_rate:.1f}s]:")
                                diarized = _assign_speakers_to_segments(
                                    diarize_pipeline,
                                    rolling_audio,
                                    segments_list,
                                    chunk_offset_sec=chunk_offset_sec,
                                    num_speakers=num_speakers,
                                    min_speakers=min_speakers,
                                    max_speakers=max_speakers,
                                )
                                _print_diarized_output(diarized, output_script, detected_lang)
                            else:
                                # No diarization — original behavior
                                print(f"\r\033[KTRANSCRIBED TEXT:\n{raw_text}")

                            # Feed the finalized transcript into the speaker-keyed LLM buffer.
                            # The worker process picks it up every llm_interval_sec seconds
                            # in a non-blocking daemon thread -- main loop is never stalled.
                            if diarize_pipeline and diarized:
                                # Group consecutive text per speaker and append individually
                                # so each speaker's pointer advances independently.
                                _speaker_accum: dict = {}
                                for _seg in diarized:
                                    _sp  = _seg.get("speaker", "UNKNOWN")
                                    _txt = _seg.get("text", "").strip()
                                    if _txt:
                                        _out = devanagari_to_hinglish(_txt) if output_script == "hinglish" else _txt
                                        _speaker_accum.setdefault(_sp, []).append(_out)
                                for _sp, _txts in _speaker_accum.items():
                                    llm_buffer.append(_sp, " ".join(_txts))
                            else:
                                # No diarization -- use a single default speaker key
                                llm_buffer.append("UNKNOWN", raw_text)

                            accumulated_audio = np.array([], dtype=np.float32)
                            speech_started = False
                        else:
                            # Interim preview (no diarization overhead)
                            final_text = devanagari_to_hinglish(raw_text) if output_script == "hinglish" else raw_text
                            print(f"\r\033[K🎙️ [{detected_lang}]: {final_text}...", end="", flush=True)
                    else:
                        if is_pause or is_max_length:
                            accumulated_audio = np.array([], dtype=np.float32)
                            speech_started = False

                elif not speech_started and audio_len_sec > 2.0:
                    accumulated_audio = np.array([], dtype=np.float32)

                time.sleep(0.05)

    except KeyboardInterrupt:
        print("\n\n🛑 Microphone stopped.")
        # Signal and wait for the LLM pipeline worker to drain in-flight calls
        llm_stop_event.set()
        llm_worker.join(timeout=20)
        print("🛑 LLM pipeline stopped.")
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
