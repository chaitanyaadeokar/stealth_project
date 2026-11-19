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
from live_runtime import load_environment, validate_environment, submit_segments, flush_audio
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



# --- Natural Devanagari to Hinglish (Romanized Hindi) Transliteration ---
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
                
                if next_ch == '्': # Virama / half letter
                    rom += base
                    i += 2
                elif next_ch in MATRAS:
                    rom += base + MATRAS[next_ch]
                    i += 2
                elif next_ch == '' or next_ch in ' .,?!:;\"\'':
                    # End of word consonant -> schwa deleted
                    rom += base
                    i += 1
                elif next_ch in CONSONANTS:
                    # Hindi schwa deletion in multi-syllable words
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
                
        # Conversational cleanup (e.g. bhaee -> bhai, sahee -> sahi)
        rom = re.sub(r'aee\b', 'ai', rom)
        rom = re.sub(r'ee\b', 'i', rom)
        out_words.append(rom)
        
    return ' '.join(out_words)


def create_whisper_model(
    model_size: str = "large-v3-turbo",
    device: str = None,
    compute_type: str = None
):
    """Initializes and returns the optimized Faster-Whisper model."""
    _setup_cuda_dlls()
    import ctranslate2
    from faster_whisper import WhisperModel
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


def live_transcribe(model_name="auto", language_mode="hi", output_script="hinglish",
                    task="transcribe", energy_threshold=0.015, pause_threshold=0.8,
                    max_speech_duration=6, llm_interval_sec=20, db_path=None, session_id=None, **options):
    """Use the same reliable capture engine, without speaker diarization."""
    from live_whisper_diarize import live_transcribe as run
    return run(model_name=model_name, language_mode=language_mode, output_script=output_script,
               task=task, energy_threshold=energy_threshold, pause_threshold=pause_threshold,
               max_speech_duration=max_speech_duration, llm_interval_sec=llm_interval_sec,
               db_path=db_path, session_id=session_id, enable_diarization=False, **options)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-time Whisper transcription with Hinglish output.")
    parser.add_argument("--model", type=str, default="auto",
                        help="auto: base on CPU, large-v3-turbo on CUDA; or explicit model/path")
    parser.add_argument("--language", type=str, default="hi", choices=["hi", "en", "auto"],
                        help="'hi' (default, best for Hindi & Hinglish), 'en' (pure English), 'auto' (dynamic router)")
    parser.add_argument("--output", type=str, default="hinglish", choices=["hinglish", "hindi"],
                        help="'hinglish' (default, Roman Latin script), 'hindi' (pure Devanagari)")
    parser.add_argument("--task", type=str, default="transcribe", choices=["transcribe", "translate"],
                        help="'transcribe' for native/hinglish, or 'translate' to translate Hindi to English")
    parser.add_argument("--db", default=None, help="SQLite session database (or CONVERSATION_DB)")
    parser.add_argument("--session-id", default=None, help="Resume an existing session")
    parser.add_argument("--llm-interval", type=float, default=20.0, help="Dispatch interval in seconds")
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--beam-size", type=int, default=1)
    parser.add_argument("--max-speech-duration", type=float, default=6)
    parser.add_argument("--backlog-seconds", type=float, default=120)
    parser.add_argument("--shutdown-timeout", type=float, default=20)
    parser.add_argument("--recover-audio", default=None)
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
        task=args.task
    )
