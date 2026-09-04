import os
import sys
import time
import queue
import argparse
import re
import numpy as np
import sounddevice as sd

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


def live_transcribe(
    model_name: str = "large-v3-turbo",
    language_mode: str = "hi",
    output_script: str = "hinglish",
    task: str = "transcribe",
    energy_threshold: float = 0.015,
    pause_threshold: float = 0.8,
    max_speech_duration: float = 10.0,
):
    """
    Real-time microphone transcription optimized for English and Hinglish (Roman Hindi).
    """
    model = create_whisper_model(model_name)
    sample_rate = 16000
    audio_queue = queue.Queue()

    def audio_callback(indata, frames, time_info, status):
        if status:
            print(status, file=sys.stderr)
        audio_queue.put(indata.copy())

    print("\n" + "=" * 65)
    print(f"🎙️  LISTENING (Microphone active @ 16kHz)")
    print(f"🌐  Language Mode : {language_mode.upper()} ('hi' for Hindi/Hinglish, 'en' for English, 'auto' for router)")
    print(f"🔤  Output Format : {output_script.upper()} (Romanized English/Hindi)")
    print(f"⚡  Model         : {model_name} on RTX 2050 (int8_float16)")
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

                is_pause = speech_started and (current_time - last_speech_time >= pause_threshold)
                is_max_length = audio_len_sec >= max_speech_duration
                is_interim = speech_started and (current_time - last_transcribed_time >= 1.5) and audio_len_sec >= 1.0

                if audio_len_sec >= 0.5 and (is_pause or is_max_length or is_interim):
                    target_lang = language_mode if language_mode in ["hi", "en"] else None

                    segments, info = model.transcribe(
                        accumulated_audio,
                        beam_size=5,
                        language=target_lang,
                        task=task,
                        vad_filter=True,
                        vad_parameters=dict(min_silence_duration_ms=300),
                        condition_on_previous_text=False,
                    )

                    # In auto mode, route to Hindi if Hindi probability is significant
                    if language_mode == "auto" and info and info.all_language_probs:
                        hi_prob = next((p for l, p in info.all_language_probs if l == "hi"), 0.0)
                        if hi_prob >= 0.10 and info.language != "hi":
                            segments, info = model.transcribe(
                                accumulated_audio,
                                beam_size=5,
                                language="hi",
                                task=task,
                                vad_filter=True,
                                vad_parameters=dict(min_silence_duration_ms=300),
                                condition_on_previous_text=False,
                            )

                    text_parts = [s.text.strip() for s in segments if s.text.strip()]
                    raw_text = " ".join(text_parts).strip()
                    last_transcribed_time = current_time

                    if raw_text:
                        # Convert to Hinglish (Romanized Hindi) if enabled
                        final_text = devanagari_to_hinglish(raw_text) if output_script == "hinglish" else raw_text
                        detected_lang = info.language.upper() if info and info.language else "UNKNOWN"
                        
                        if is_pause or is_max_length:
                            print(f"\r\033[K[{detected_lang}]: {final_text}")
                            accumulated_audio = np.array([], dtype=np.float32)
                            speech_started = False
                        else:
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
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-time Whisper transcription with Hinglish output.")
    parser.add_argument("--model", type=str, default="large-v3-turbo", 
                        help="Model size: small, medium, large-v3-turbo (default: large-v3-turbo)")
    parser.add_argument("--language", type=str, default="hi", choices=["hi", "en", "auto"],
                        help="'hi' (default, best for Hindi & Hinglish), 'en' (pure English), 'auto' (dynamic router)")
    parser.add_argument("--output", type=str, default="hinglish", choices=["hinglish", "hindi"],
                        help="'hinglish' (default, Roman Latin script), 'hindi' (pure Devanagari)")
    parser.add_argument("--task", type=str, default="transcribe", choices=["transcribe", "translate"],
                        help="'transcribe' for native/hinglish, or 'translate' to translate Hindi to English")
    args = parser.parse_args()

    live_transcribe(
        model_name=args.model,
        language_mode=args.language,
        output_script=args.output,
        task=args.task
    )
