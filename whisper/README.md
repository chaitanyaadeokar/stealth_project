# Live conversation pipeline (Phase 0 and Phase 1)

This repository includes the Whisper speech library below and a live conversation
application. Both live scripts now submit finalized speech to the same SQLite-backed,
chronological intent router. The original transcription models, language options,
model selection and Hinglish conversion remain available. Finalized chunk previews
replace repeated growing-buffer inference in the default live path.

## Application setup

Use a virtual environment. The offline application tests were run on the existing
Python 3.14 environment; this does not certify all speech/GPU dependencies on that
version. For a fresh speech environment, choose a Python version supported by the
binary wheels for your hardware and the declared dependencies (3.11/3.12 are useful
starting points).

```powershell
python -m venv venv
.\venv\Scripts\python.exe -m pip install -r requirements-dev.txt
# Optional diarization (uses pyannote directly; no sibling WhisperX checkout):
.\venv\Scripts\python.exe -m pip install -r requirements-diarization.txt
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
```

Copy the example only when creating a new `.env`; preserve existing credentials.
Set `GROQ_API_KEY` for live model claim assessments. Optional diarization requires
access to `pyannote/speaker-diarization-community-1` and an appropriate `HF_TOKEN`.
The first speech/classification/diarization model load may download weights.
`ffmpeg` on PATH is needed for the original Whisper file-decoding CLI/tests.
CUDA acceleration additionally requires compatible NVIDIA drivers and libraries.

```powershell
.\venv\Scripts\python.exe live_whisper.py --language en
.\venv\Scripts\python.exe live_whisper_diarize.py --language hi --diarize --num-speakers 2
# Resume durable pending jobs and continue the same session:
.\venv\Scripts\python.exe live_whisper.py --db data/sessions.sqlite3 --session-id SESSION_ID
```

Both scripts support `--help`, `--db`, `--session-id`, and `--llm-interval` without
loading speech models. Startup checks required configuration before capture.
Classifier initialization and an ASR/VAD warm-up complete before capture. `--model auto`
(the default) uses multilingual `base` on CPU and `large-v3-turbo` on CUDA. This is
an explicit speed/accuracy tradeoff: select `--model large-v3-turbo` for the previous
model, or set `WHISPER_CPU_MODEL`. CPU auto weights live in `data/models`.
Diarization initialization failure falls back to `UNKNOWN` speakers with diagnostics.
`--num-speakers 2` reserves session identities `SPEAKER_00` and `SPEAKER_01`.
It enables diarization and fails before capture if the backend is unavailable.
Matching uses shared timestamps and persisted speaker embeddings, so local label
swaps and a speaker returning after a gap do not allocate a third ID. Short windows
may contain only one person; the backend uses one-to-two local clusters rather than
inventing a second voice in every window. Ambiguous matches are logged, and actual
speaker-attribution accuracy still needs a labeled two-voice recording.

The microphone now uses a dedicated blocking-read thread and a durable float32 audio
journal, rather than a callback queue. One ASR owner and one diarization owner can
work concurrently, with at most two chunks in flight. The defaults use four ASR
CPU threads, two diarization threads, and two classifier threads. Silence and chunk
boundaries use audio sample time; chunks normally span four to six seconds.

```powershell
# Keep the previous quality-oriented model explicitly (CPU may fall behind):
.\venv\Scripts\python.exe live_whisper_diarize.py --model large-v3-turbo --diarize
# Recover a retained spool without opening the microphone; use its original database:
.\venv\Scripts\python.exe live_whisper_diarize.py --db data/sessions.sqlite3 --recover-audio data/audio/SPOOL_DIRECTORY --diarize
```

`--backlog-seconds` controls a disk-backed unprocessed-audio budget (default 120
seconds). Recording stops explicitly before the next read when that budget, the
512 MiB recording limit, or the free-disk guard is reached. Retained audio is not
discarded. Device-reported input overflow is an error: already lost hardware samples
cannot be reconstructed. `--cpu-threads`, `--beam-size`, `--max-speech-duration`, and
`--shutdown-timeout` expose the performance controls; increasing capacity is not a
substitute for a model that can keep up.

## Text replay and session inspection

No microphone, weights, or API key are needed for an offline replay:

```powershell
.\venv\Scripts\python.exe pipeline_cli.py --offline --input tests/fixtures/conversation.jsonl --session-id demo
.\venv\Scripts\python.exe pipeline_cli.py --inspect --session-id demo
```

Offline mode uses explicit demonstration rules, not ModernBERT. Omit `--offline`
to use the existing ModernBERT classifier and the configured claim-assessment LLM.
Input JSONL accepts `speaker`, `text`, optional stable `id`, `language`, `start`, and
`end`. Stable IDs allow the same input to be replayed into the same session without
duplicate jobs. An existing ID with different speaker/text/session is rejected.
Results and error messages are available through `--inspect` even after restart.

## Current routing and reliability

- Questions/queries route to the question handler, which explicitly reports that
  retrieval is unavailable until Phase 2.
- Facts/claims route to the existing model assessment, marked `unverified` with
  empty citations. Classification as a fact does not establish truth.
- Opinions/debates route to the opinion handler, which reports that the panel is
  unavailable. Handlers can be supplied to `Orchestrator` or `start_llm_pipeline`.
- Explicit greetings/acknowledgments return `no_action`. Uncertain or ambiguous
  model outputs return `needs_clarification`; a model-predicted neutral label never
  discards a substantive statement.
- Intent routing uses explicit English/Hindi/Hinglish speech-act rules and three
  routed NLI categories; acknowledgments are handled by explicit rules. Confidence and the top-two margin are both checked;
  the model's relative scores are not calibrated truth probabilities.

SQLite stores session/utterance IDs, chronological sequence, speaker, original and
display text, available audio times, jobs, attempts, errors, and structured results.
Prior context is bounded to 20 turns/12,000 text characters and excludes future
turns and other sessions. Segmentation preserves titles/decimals, splits mixed
question/opinion clauses, and respects speaker turns and word pauses. Sentence
parts use matched word timestamps when available; incomplete word alignment falls
back to the complete segment text rather than omitting unaligned words.
Acoustic chunks are not automatically intent requests: a bounded utterance assembler
joins unfinished clauses across them, removes overlapping boundary-word repetition,
and closes on speaker turns, real pauses or independent sentences. Its pending text
is committed with the audio checkpoint and restored after a restart. Decoder-added
punctuation alone does not prove a chunk contains a complete sentence.

Each session has one sequential worker and a local OS lock. Input chunks commit
atomically, job claims are transactional, and results are acknowledged only after
successful storage. Failed jobs retry up to three total attempts with exponential
delay; exhausted failures remain stored. SDK retries are disabled to avoid nested
retry loops; HTTP timeout defaults to 30 seconds. SQLite busy timeout is five seconds.
An abrupt worker crash leaves a lease that becomes recoverable after at least 120
seconds. Explicit shutdown releases its worker's unfinished jobs immediately after
the worker has stopped. External LLM execution is at least once across crashes; the
database cannot guarantee exactly-once external requests.

`auto` language mode now trusts the model's best language; a secondary 10% Hindi
score no longer forces a second Hindi decode of English speech. Explicit `hi` and
`en` modes remain available.

Ctrl+C is handled by the controller as a stop request, so it cannot interrupt a
SQLite commit midway. The intent worker ignores console SIGINT/SIGBREAK; its job
owner can defer an unfinished model/API call at the shutdown deadline, preserve the
pending job, and exit normally with code 0. Actual crashes and forced termination
remain visible; exit codes are not rewritten to hide failures.

Ctrl+C stops the recorder and drains recorded speech for up to `--shutdown-timeout`
seconds, then drains the intent worker within its own timeout. The audio spool and
its JSON manifest remain recoverable. SQLite atomically stores transcript jobs and
audio checkpoints/speaker-tracker state; a crash between the SQLite commit and
manifest acknowledgment does not replay completed audio. Native speech calls cannot
be cancelled in place; a speech timeout fails explicitly and requires restarting
the process before recovery. Bounded daemon model owners allow process exit rather
than waiting indefinitely on those calls. Recovery is replay-only, not simultaneous
new microphone capture.

Structured `[EVENT]` JSON log entries record stage duration, CPU counters, backlog,
route/confidence/source, errors, and audio checkpoints. `asr_thread_cpu_sec` measures
the owner thread, not all native backend threads; capture-wide process CPU counters
include ASR/diarization, but exclude the separate intent process.

Defaults write `data/sessions.sqlite3` and repository-local `vague.log`; paths can
be configured in `.env`. Logs coordinate complete entries across processes, but
they are diagnostic logs, not tamper-evident forensic storage.

## Application checks

```powershell
.\venv\Scripts\python.exe -m pytest -q
.\venv\Scripts\python.exe -m pip check
# Deliberate model-download/transcription checks:
.\venv\Scripts\python.exe -m pytest -m integration tests/test_transcribe.py
# Deliberate GPU-only checks:
.\venv\Scripts\python.exe -m pytest -m requires_cuda tests/test_timing.py
```

Default tests exclude downloads and GPU-only checks. The original audio-decoding
test skips explicitly when `ffmpeg` is absent. See
[the latest correction report](PHASE_0_1_CORRECTION_REPORT.md) for same-recording comparisons,
acceptance tests, and the remaining limits. Reproduce cached-model profiles without
network requests or LLM calls:

```powershell
.\venv\Scripts\python.exe profile_phase01.py --intent
.\venv\Scripts\python.exe profile_phase01.py --speech-pipeline --diarization --cpu-base --paced --output data/cpu-profile.json
```

These profiles require cached weights. Model quality, hardware loss, long-duration
capture, and multi-speaker Hindi/Hinglish still need local acceptance testing before
calling the system production-stable.

---

# Whisper

[[Blog]](https://openai.com/blog/whisper)
[[Paper]](https://arxiv.org/abs/2212.04356)
[[Model card]](https://github.com/openai/whisper/blob/main/model-card.md)
[[Colab example]](https://colab.research.google.com/github/openai/whisper/blob/master/notebooks/LibriSpeech.ipynb)

Whisper is a general-purpose speech recognition model. It is trained on a large dataset of diverse audio and is also a multitasking model that can perform multilingual speech recognition, speech translation, and language identification.


## Approach

![Approach](https://raw.githubusercontent.com/openai/whisper/main/approach.png)

A Transformer sequence-to-sequence model is trained on various speech processing tasks, including multilingual speech recognition, speech translation, spoken language identification, and voice activity detection. These tasks are jointly represented as a sequence of tokens to be predicted by the decoder, allowing a single model to replace many stages of a traditional speech-processing pipeline. The multitask training format uses a set of special tokens that serve as task specifiers or classification targets.


## Setup

We used Python 3.9.9 and [PyTorch](https://pytorch.org/) 1.10.1 to train and test our models, but the codebase is expected to be compatible with Python 3.8-3.11 and recent PyTorch versions. The codebase also depends on a few Python packages, most notably [OpenAI's tiktoken](https://github.com/openai/tiktoken) for their fast tokenizer implementation. You can download and install (or update to) the latest release of Whisper with the following command:

    pip install -U openai-whisper

Alternatively, the following command will pull and install the latest commit from this repository, along with its Python dependencies:

    pip install git+https://github.com/openai/whisper.git 

To update the package to the latest version of this repository, please run:

    pip install --upgrade --no-deps --force-reinstall git+https://github.com/openai/whisper.git

It also requires the command-line tool [`ffmpeg`](https://ffmpeg.org/) to be installed on your system, which is available from most package managers:

```bash
# on Ubuntu or Debian
sudo apt update && sudo apt install ffmpeg

# on Arch Linux
sudo pacman -S ffmpeg

# on MacOS using Homebrew (https://brew.sh/)
brew install ffmpeg

# on Windows using Chocolatey (https://chocolatey.org/)
choco install ffmpeg

# on Windows using Scoop (https://scoop.sh/)
scoop install ffmpeg
```

You may need [`rust`](http://rust-lang.org) installed as well, in case [tiktoken](https://github.com/openai/tiktoken) does not provide a pre-built wheel for your platform. If you see installation errors during the `pip install` command above, please follow the [Getting started page](https://www.rust-lang.org/learn/get-started) to install Rust development environment. Additionally, you may need to configure the `PATH` environment variable, e.g. `export PATH="$HOME/.cargo/bin:$PATH"`. If the installation fails with `No module named 'setuptools_rust'`, you need to install `setuptools_rust`, e.g. by running:

```bash
pip install setuptools-rust
```


## Available models and languages

There are six model sizes, four with English-only versions, offering speed and accuracy tradeoffs.
Below are the names of the available models and their approximate memory requirements and inference speed relative to the large model.
The relative speeds below are measured by transcribing English speech on a A100, and the real-world speed may vary significantly depending on many factors including the language, the speaking speed, and the available hardware.

|  Size  | Parameters | English-only model | Multilingual model | Required VRAM | Relative speed |
|:------:|:----------:|:------------------:|:------------------:|:-------------:|:--------------:|
|  tiny  |    39 M    |     `tiny.en`      |       `tiny`       |     ~1 GB     |      ~10x      |
|  base  |    74 M    |     `base.en`      |       `base`       |     ~1 GB     |      ~7x       |
| small  |   244 M    |     `small.en`     |      `small`       |     ~2 GB     |      ~4x       |
| medium |   769 M    |    `medium.en`     |      `medium`      |     ~5 GB     |      ~2x       |
| large  |   1550 M   |        N/A         |      `large`       |    ~10 GB     |       1x       |
| turbo  |   809 M    |        N/A         |      `turbo`       |     ~6 GB     |      ~8x       |

The `.en` models for English-only applications tend to perform better, especially for the `tiny.en` and `base.en` models. We observed that the difference becomes less significant for the `small.en` and `medium.en` models.
Additionally, the `turbo` model is an optimized version of `large-v3` that offers faster transcription speed with a minimal degradation in accuracy.

Whisper's performance varies widely depending on the language. The figure below shows a performance breakdown of `large-v3` and `large-v2` models by language, using WERs (word error rates) or CER (character error rates, shown in *Italic*) evaluated on the Common Voice 15 and Fleurs datasets. Additional WER/CER metrics corresponding to the other models and datasets can be found in Appendix D.1, D.2, and D.4 of [the paper](https://arxiv.org/abs/2212.04356), as well as the BLEU (Bilingual Evaluation Understudy) scores for translation in Appendix D.3.

![WER breakdown by language](https://github.com/openai/whisper/assets/266841/f4619d66-1058-4005-8f67-a9d811b77c62)

## Command-line usage

The following command will transcribe speech in audio files, using the `turbo` model:

```bash
whisper audio.flac audio.mp3 audio.wav --model turbo
```

The default setting (which selects the `turbo` model) works well for transcribing English. However, **the `turbo` model is not trained for translation tasks**. If you need to **translate non-English speech into English**, use one of the **multilingual models** (`tiny`, `base`, `small`, `medium`, `large`) instead of `turbo`. 

For example, to transcribe an audio file containing non-English speech, you can specify the language:

```bash
whisper japanese.wav --language Japanese
```

To **translate** speech into English, use:

```bash
whisper japanese.wav --model medium --language Japanese --task translate
```

> **Note:** The `turbo` model will return the original language even if `--task translate` is specified. Use `medium` or `large` for the best translation results.

Run the following to view all available options:

```bash
whisper --help
```

See [tokenizer.py](https://github.com/openai/whisper/blob/main/whisper/tokenizer.py) for the list of all available languages.


## Python usage

Transcription can also be performed within Python: 

```python
import whisper

model = whisper.load_model("turbo")
result = model.transcribe("audio.mp3")
print(result["text"])
```

Internally, the `transcribe()` method reads the entire file and processes the audio with a sliding 30-second window, performing autoregressive sequence-to-sequence predictions on each window.

Below is an example usage of `whisper.detect_language()` and `whisper.decode()` which provide lower-level access to the model.

```python
import whisper

model = whisper.load_model("turbo")

# load audio and pad/trim it to fit 30 seconds
audio = whisper.load_audio("audio.mp3")
audio = whisper.pad_or_trim(audio)

# make log-Mel spectrogram and move to the same device as the model
mel = whisper.log_mel_spectrogram(audio, n_mels=model.dims.n_mels).to(model.device)

# detect the spoken language
_, probs = model.detect_language(mel)
print(f"Detected language: {max(probs, key=probs.get)}")

# decode the audio
options = whisper.DecodingOptions()
result = whisper.decode(model, mel, options)

# print the recognized text
print(result.text)
```

## More examples

Please use the [🙌 Show and tell](https://github.com/openai/whisper/discussions/categories/show-and-tell) category in Discussions for sharing more example usages of Whisper and third-party extensions such as web demos, integrations with other tools, ports for different platforms, etc.


## License

Whisper's code and model weights are released under the MIT License. See [LICENSE](https://github.com/openai/whisper/blob/main/LICENSE) for further details.
