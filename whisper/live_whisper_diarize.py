from __future__ import annotations

import os
import sys
import time
import queue
import argparse
import re
import logger
from llm_client import fact_check_statement
from llm_pipeline import start_llm_pipeline
from live_runtime import load_environment, validate_environment
from capture_runtime import run_capture
import multiprocessing

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


from diarization import DiarizationPipeline, assign_word_speakers as _assign_word_speakers

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
    model_size: str = "auto",
    device: str = None,
    compute_type: str = None,
    cpu_threads: int = 4,
):
    """Initializes and returns the optimized Faster-Whisper model."""
    _setup_cuda_dlls()
    import ctranslate2
    from faster_whisper import WhisperModel
    if device is None:
        device = "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
    automatic = model_size == "auto"
    if automatic:
        model_size = "large-v3-turbo" if device == "cuda" else os.environ.get("WHISPER_CPU_MODEL", "base")

    if compute_type is None:
        compute_type = "int8_float16" if device == "cuda" else "int8"

    print(f"Loading Whisper model '{model_size}' on {device.upper()} ({compute_type})...")
    model = WhisperModel(
        model_size,
        device=device,
        compute_type=compute_type,
        cpu_threads=cpu_threads,
        num_workers=1,
        use_auth_token=False if automatic else None,
        download_root=(os.path.join(os.path.dirname(__file__), "data", "models") if automatic and device == "cpu"
                       else os.path.join(os.path.expanduser("~"), ".cache", "whisper"))
    )
    from faster_whisper.vad import get_vad_model
    import numpy as np
    get_vad_model()
    warm_started = time.perf_counter()
    warm_segments, _ = model.transcribe(np.zeros(16000, dtype=np.float32), language="en",
        beam_size=1, vad_filter=False, condition_on_previous_text=False, word_timestamps=True)
    list(warm_segments)
    logger.log_event("asr_warmed", model=model_size, device=device, wall_sec=time.perf_counter() - warm_started)
    print("Model and VAD loaded; speech inference warmed successfully!")
    return model


def create_diarization_pipeline(device: str = None):
    """
    Initialize the pyannote speaker diarization pipeline.

    Uses pyannote/speaker-diarization-community-1 (requires model access).
    Falls back gracefully if unavailable, returning None.

    Args:
        device: 'cuda' or 'cpu'. Auto-detected if None.

    Returns:
        DiarizationPipeline instance, or None if unavailable.
    """
    import torch
    torch.set_num_threads(2)
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
    Build a timestamped transcript dict from faster-whisper segments,
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
                    "start": (w.start if w.start is not None else seg.start) + offset_sec,
                    "end":   (w.end if w.end is not None else (w.start if w.start is not None else seg.end)) + offset_sec,
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

    Diarization uses a bounded rolling window for cross-segment context.
    Speaker labels remain provisional as older audio leaves the window.

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
        result = _assign_word_speakers(diarize_df, transcript_dict, fill_nearest=False)
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

    units = []
    for segment in diarized_segments:
        words = segment.get("words") or []
        if words and re.sub(r"\s+", "", "".join(w.get("word", "") for w in words)) == re.sub(r"\s+", "", segment.get("text", "")):
            units.extend({"speaker": word.get("speaker", segment.get("speaker", "UNKNOWN")), "text": word.get("word", "")}
                         for word in words)
        else:
            units.append(segment)
    for seg in units:
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
    model_name="auto", language_mode="hi", output_script="hinglish", task="transcribe",
    energy_threshold=0.015, pause_threshold=0.8, max_speech_duration=6.0,
    enable_diarization=False, num_speakers=None, min_speakers=None, max_speakers=None,
    llm_interval_sec=20.0, db_path=None, session_id=None, beam_size=1, cpu_threads=4,
    device=None, recovery=None, max_pending_seconds=120, shutdown_timeout=20,
):
    """Preload models, then run journaled capture with bounded speech-stage concurrency."""
    load_environment()
    validate_environment()
    if cpu_threads < 1 or beam_size < 1 or max_speech_duration <= 0:
        raise ValueError("CPU threads, beam size and speech duration must be positive")
    if num_speakers is not None:
        if num_speakers < 1:
            raise ValueError("num_speakers must be positive")
        enable_diarization = True
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("MKL_NUM_THREADS", "2")
    if recovery:
        import json
        from pathlib import Path
        manifest = json.loads((Path(recovery) / "capture.json").read_text(encoding="utf-8"))
        if session_id is not None and session_id != manifest["session_id"]:
            raise ValueError("Recovery session does not match --session-id")
        session_id = manifest["session_id"]
    buffer = None
    try:
        # Classifier weights and its first inference finish before capture/model contention.
        buffer, _, _ = start_llm_pipeline(fact_check_statement, interval_sec=llm_interval_sec,
            db_path=db_path, session_id=session_id, preload=True)
        model = create_whisper_model(model_name, device=device, cpu_threads=cpu_threads)
        diarizer = create_diarization_pipeline(device=device) if enable_diarization else None
        if num_speakers is not None and diarizer is None:
            raise RuntimeError("A fixed speaker count requires a working diarization model; capture was not started")
        logger.log_event("models_ready", session_id=buffer.session_id, model=model_name,
                         diarization=bool(diarizer), cpu_threads=cpu_threads, beam_size=beam_size)
        return run_capture(model, buffer, diarizer=diarizer, language=language_mode, task=task,
            display_fn=devanagari_to_hinglish if output_script == "hinglish" else None,
            energy_threshold=energy_threshold, pause_seconds=pause_threshold,
            max_seconds=max_speech_duration, beam_size=beam_size, recovery=recovery,
            max_pending_seconds=max_pending_seconds, shutdown_timeout=shutdown_timeout,
            speaker_options={"num_speakers":num_speakers,"min_speakers":min_speakers,"max_speakers":max_speakers})
    except KeyboardInterrupt:
        logger.log_event("startup_interrupted", session_id=buffer.session_id if buffer else session_id)
        return {"interrupted": True}
    finally:
        if buffer is not None:
            print(f"[Pipeline shutdown] {buffer.close(timeout=shutdown_timeout)}", flush=True)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Real-time Whisper transcription with optional Speaker Diarization and Hinglish output."
    )
    parser.add_argument(
        "--model", type=str, default="auto",
        help="auto: base on CPU, large-v3-turbo on CUDA; or an explicit model/path"
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

    parser.add_argument("--db", default=None, help="SQLite session database (or CONVERSATION_DB)")
    parser.add_argument("--session-id", default=None, help="Resume an existing session")
    parser.add_argument("--llm-interval", type=float, default=20.0, help="Dispatch interval in seconds")
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--beam-size", type=int, default=1)
    parser.add_argument("--max-speech-duration", type=float, default=6)
    parser.add_argument("--backlog-seconds", type=float, default=120)
    parser.add_argument("--shutdown-timeout", type=float, default=20)
    parser.add_argument("--recover-audio", default=None, help="Replay a retained audio spool before starting another capture")
    args = parser.parse_args()

    live_transcribe(
        model_name=args.model,
        device=args.device, cpu_threads=args.cpu_threads, beam_size=args.beam_size,
        max_speech_duration=args.max_speech_duration, max_pending_seconds=args.backlog_seconds,
        shutdown_timeout=args.shutdown_timeout, recovery=args.recover_audio,
        language_mode=args.language,
        output_script=args.output,
        db_path=args.db,
        session_id=args.session_id,
        llm_interval_sec=args.llm_interval,
        task=args.task,
        enable_diarization=args.diarize,
        num_speakers=args.num_speakers,
        min_speakers=args.min_speakers,
        max_speakers=args.max_speakers,
    )
