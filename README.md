# Whisper

## Podcast backend

The podcast app is managed by this repository's root `uv` project. From this directory, run:

```powershell
uv run uvicorn backend.app.main:app --reload
```

Copy `.env.example` to `.env` and set `DATABASE_URL` (and optionally `GROQ_API_KEY`) before starting it.

### Phase 1 discussion persistence

Install dependencies and apply migrations **before** starting the app:

```powershell
uv sync --extra dev
uv run alembic upgrade head
uv run uvicorn backend.app.main:app --reload
```

Use your existing PostgreSQL container. The new `docker-compose.yml` is an
alternative for a fresh installation, not a replacement for your existing volume.
Do not start it alongside another PostgreSQL container using the same port.
For a fresh installation set `POSTGRES_PASSWORD` in your shell, then run
`docker compose up -d postgres`. Set `DATABASE_URL` to match. The app never
starts Docker, Uvicorn or ngrok automatically.
Compose defaults to the existing local development password when none is supplied,
so `docker compose up -d memcached` also works with your existing PostgreSQL container.

Migration `0001` adopts existing `podcast_sessions`, `transcript_segments` and
`reasoning_entries` tables without changing their rows; it creates them only on a
fresh database. Migration `0002` adds the six Phase 1 tables. Back up the database
before migrating. No schema creation occurs at app startup. `alembic downgrade
0001` removes Phase 1 data; use only on disposable databases. The baseline
downgrade intentionally preserves legacy data.

The live flow is finalized Whisper text → Conversation Context Manager →
Orchestration Engine → Intent Classifier → Question/Query, Fact/Claim or
Opinion/Debate adapter. Set `GROQ_API_KEY` to enable agents. `INTENT_MODEL`
defaults to `GROQ_MODEL`; classification uses validated JSON from Groq, not the
unused ModernBERT setting. The Fact/Claim adapter calls the existing fact-check
function. It does not gain web search or verified-source retrieval in Phase 1.
Question/Query answers questions; Opinion/Debate responds to subjective discussion
instead of assigning truth verdicts to preferences. The live conversation uses
one shared session UUID per app process, with a participant per microphone.
Restarting creates a new live session. API clients supply `session_id` to group
their own turns. Phase 2 adds a bounded PostgreSQL-backed context window cached
in Memcached; a process-local window remains as an outage fallback. Use one Uvicorn worker
for the shared microphone feed.

```powershell
$turn = @{
  session_id = [guid]::NewGuid().ToString()
  mic_id = 1
  display_name = 'Speaker 1'
  raw_text = 'Does AI replace teachers?'
} | ConvertTo-Json
$result = Invoke-RestMethod -Method Post http://127.0.0.1:8000/classify -ContentType application/json -Body $turn
$result
Invoke-RestMethod "http://127.0.0.1:8000/sessions/$($result.session_id)/utterances"
Invoke-RestMethod "http://127.0.0.1:8000/utterances/$($result.utterance_id)/classification"
```

`POST /classify` runs the classifier only, returning the session/utterance UUIDs,
classification and `persistence: queued|unavailable`. Its default utterance UUID
is new per request. Reuse an utterance UUID only for the same turn; conflicting
IDs are rejected by persistence. Reads may briefly return 404/empty until the
queue commits; retry them. GET utterances supports `limit` (1–500) and `offset`,
ordered by speech start time. Invalid input returns 422, unavailable classifier
or storage returns 503. No synthetic classification is saved on model failure.

Raw utterances are queued when context receives them. Classification and its
audit entry commit atomically; completed/failed pipeline results are saved
separately. Retry writes use stable IDs to avoid duplicate classifications,
results and audit entries. Audit entries form a globally serialized chain:
`sha256(prev_hash + canonical_payload)`, with a 64-zero genesis hash. Canonical
payload means UTF-8 JSON with sorted keys, compact separators and no NaN. A
transaction advisory lock serializes concurrent appends; PostgreSQL triggers
reject UPDATE, DELETE and TRUNCATE. Database owners can still change triggers;
this is tamper evidence, not protection against a database administrator.

Persistence uses a bounded background queue with exponential backoff and per-write
timeouts. Live STT and WebSocket delivery never wait for DB I/O. Queue overflow,
exhausted retries and shutdown timeouts are logged. This process-local queue is
best-effort: queued writes can be lost on crash or sustained outage; `queued`
does not mean committed. Monitor errors and choose queue/retry settings appropriate
for deployment. PostgreSQL remains the only durable source of truth.

Run Phase 1 tests against a **separate disposable database** (name must end `_test`):

```powershell
docker exec some-postgres createdb -U postgres whisper_phase1_test
$env:TEST_DATABASE_URL = 'postgresql+asyncpg://postgres:mysecretpassword@127.0.0.1:5432/whisper_phase1_test'
uv run --extra dev pytest tests/test_phase1.py -q
uv run --extra dev pytest tests/test_streaming_pipeline.py -q
node --test tests/test_mic_client.cjs tests/test_live_client.cjs tests/test_pcm_capture.cjs
```

The integration fixture migrates the test database, uses unique session IDs and
never clears application tables. Without `TEST_DATABASE_URL`, database tests are
skipped. Model calls are mocked; tests do not require a real Groq token or load
Whisper models. Phase 4–5 analytics are not implemented yet.

### Phase 2 short-term memory

For testing, open **View both microphones** and use the **System memory · testing
view** section. It refreshes every five seconds and has a manual refresh button.
It displays the active live session UUID, committed turns, topic/speaker/claims
state, cache hit/miss/error status, data source, configured TTL and queue depths.
The long-term section displays Phase 3 PostgreSQL evidence, decisions and nullable ethics records.
`GET /api/memory` inspects the live session; `?session_id=<UUID>` inspects another
session. Inspection is read-only and does not warm the cache or extend TTL.
If Memcached is unavailable it displays a labeled PostgreSQL fallback; if both
are unavailable it shows an error rather than an empty-memory success.

Start only Memcached alongside your existing PostgreSQL container, then start the app yourself:

```powershell
uv sync --extra dev
docker compose up -d memcached
uv run uvicorn backend.app.main:app --reload
```

Defaults are `MEMCACHED_HOST=127.0.0.1`, `MEMCACHED_PORT=11211` and
`SESSION_TTL_SECONDS=3600`. The Compose port binds only to localhost. No PostgreSQL
schema change or new migration is required: explicit topic and pending-claim
state is stored under `sessions.metadata.stm_state` in the existing JSONB column.
The app does not start Docker services automatically. Stop the cache with
`docker compose stop memcached`; transcription and classification continue using
PostgreSQL and bounded local fallback when necessary.

Keys are `stm:{session_id}:window` (one JSON array of the last N committed turns)
and `stm:{session_id}:state` (a JSON object with `current_topic`, `active_speaker`,
and `pending_claims`). State includes internal ordering/revision fields so stale
writers cannot move the active speaker or topic backwards. The active speaker
comes from the most recent persisted turn. Topic defaults to null and pending
claims to an empty list; agents can explicitly update them through
`app.state.short_term_memory.update_state(session_id, changes)`. Those updates
commit to PostgreSQL before populating the cache; Phase 2 does not add an extra
LLM call to guess topics or extract claims.

Successful persistence transactions enqueue cache writes. Cache writes run in a
separate bounded queue with retries/backoff, so Memcached outages cannot roll
back database commits or hold up the persistence worker. Cache hits serve the
window directly; cache misses, expiry, malformed blobs and connection errors
rebuild from PostgreSQL and repopulate. If both systems are unavailable, the
context manager uses its bounded local window and logs the failure. Classifier
context excludes the current turn and later turns. Queued but uncommitted turns
are never written to Memcached, so context can briefly lag while persistence is
pending. All reads occur outside the live STT task.

Window writes merge IDs using `gets` + `cas`, or `add` for an absent key. Rebuilds
also merge, so an old database snapshot cannot overwrite a newer cached turn.
`CONTEXT_WINDOW_TURNS` caps the turn count (default 20), and `MEMORY_MAX_BYTES`
caps each encoded UTF-8 blob (default 900000, below Memcached's 1 MB limit).
Oldest turns are removed to fit; raw text remains intact in PostgreSQL. Session
TTL is refreshed on successful writes, expires both blobs independently, and is
limited to 30 days so Memcached treats it as a relative TTL. Reads and CAS retries
have configurable timeouts; exhausted cache retries are logged and can recover
through PostgreSQL rebuild or later committed writes.

Run Phase 2 tests using the same dedicated PostgreSQL test database. Real
Memcached testing uses a separate port; normal cache data is never flushed:

```powershell
docker run --name whisper-memcached-test -p 127.0.0.1:11212:11211 -d memcached:1.6-alpine memcached -m 16 -I 1m
# If that test container already exists: docker start whisper-memcached-test
$env:TEST_DATABASE_URL = 'postgresql+asyncpg://postgres:mysecretpassword@127.0.0.1:5432/whisper_phase1_test'
$env:TEST_MEMCACHED_PORT = '11212'
uv run --extra dev pytest tests/test_phase2.py tests/test_phase1.py tests/test_streaming_pipeline.py -q
docker stop whisper-memcached-test
```

Tests cover hits, misses, PostgreSQL rebuilds, real and simulated CAS concurrency,
TTL expiry, byte limits, corrupt entries, state revision ordering, and cache
outages after a successful PostgreSQL commit. Without `TEST_MEMCACHED_PORT`, only
the real-Memcached test is skipped; PostgreSQL tests need `TEST_DATABASE_URL`.

### Phase 3 long-term memory

Apply the new migration before starting your application:

```powershell
uv run alembic upgrade head
uv run uvicorn backend.app.main:app --reload
```

Revision `0003` adds `ltm_evidence`, `ltm_decisions` and `ltm_ethics` with foreign
keys, indexes and provenance links to committed pipeline results. Unique
`source_result_id` constraints make evidence/decision promotion idempotent;
ethics placeholders are unique per utterance. `ethical_score` and `dimensions`
remain SQL NULL because the Ethical Evaluation Engine is not implemented.

The `MemoryPromotionService` queues work only after successful result commits.
Promotion reads the durable records backing short-term memory from PostgreSQL,
so cache eviction, TTL expiry or Memcached failure does not lose promotion inputs.
It copies important items; it does not delete conversation history or cache entries.
The independent bounded queue retries with backoff and a job timeout. Failed or
overflowed promotion jobs are logged and can be recovered by replaying a session.
Repeated promotion and concurrent requests do not duplicate records.

Defaults: `LTM_PROMOTE_ON_RESULT=true`, `LTM_MIN_CONFIDENCE=0.75`,
`LTM_MIN_RESPONSE_CHARS=80`, `LTM_QUEUE_SIZE=500`, `LTM_JOB_TIMEOUT_SEC=30`.
All completed Fact/Claim results are retained, including uncertain assessments.
Other completed results are promoted if marked `important: true`, contain an
explicit `decision` string, or meet both confidence and response-length thresholds.
Failed pipeline runs and short greetings are not promoted. Decisions record the
agent's explicit decision, or `response_generated`, with the original output as
rationale; these are agent responses, not panel consensus.

The existing Fact/Claim adapter still calls the original fact-check function.
Its new structured evidence hook explicitly labels prose-only assessments as
`uncertain` with no sources. It does not fabricate citations or infer verified
facts from words such as TRUE in model prose. An upstream verification adapter
can supply `output.evidence = {claim, fact_status, sources}`; a true/false status
requires at least one HTTP(S) source object with `verified: true`. Such flags
must come from a trusted verification adapter. Phase 3 stores that provenance;
it does not fetch URLs or implement a new web verification engine. The live
prose-only adapter therefore produces uncertain evidence until verification is added.

Session replay and session-end triggers are available without disrupting STT:

```powershell
# Use the session UUID shown in the memory testing view.
$sessionId = '<SESSION_UUID>'
Invoke-RestMethod -Method Post "http://127.0.0.1:8000/sessions/$sessionId/memory/promote"
Invoke-RestMethod -Method Post "http://127.0.0.1:8000/sessions/$sessionId/end"
Invoke-RestMethod "http://127.0.0.1:8000/api/memory?session_id=$sessionId"
```

Both POST endpoints return 202 when queued, 404 for a missing/uncommitted session,
or 503 if storage is unavailable or the queue is full. End is idempotent: it sets
`sessions.ended_at` once and promotes important committed results. Ending the
current live session rotates its UUID for subsequent finalized utterances; flush
or stop the microphone before calling end if you need an exact speech boundary.
Late results for an ended session are promoted as they commit, even when
`LTM_PROMOTE_ON_RESULT=false`. Graceful application shutdown also requests end
promotion after orchestration and persistence drain. A crash cannot run that
hook; replay the old session after restarting. Long sessions may require a larger
job timeout; replay remains safe after a partially completed attempt.

The memory testing view refreshes long-term records from PostgreSQL, shows
evidence/decision/ethics categories, and reports read failures distinctly from
an empty session. It displays up to 100 records per category. The repository
offers `by_session`, and `evidence(session_id, claim_text, fact_status, limit,
offset)` with case-insensitive literal claim substring matching.

```powershell
$env:TEST_DATABASE_URL = 'postgresql+asyncpg://postgres:mysecretpassword@127.0.0.1:5432/whisper_phase1_test'
uv run --extra dev pytest tests/test_phase3.py tests/test_phase1.py tests/test_memory_inspection.py -q
node --test tests/test_memory_client.cjs
```

Tests use isolated session IDs and mocked agent outputs. They cover idempotent
and concurrent promotion, source attribution, query filters, null ethics,
importance filtering, post-commit ordering, session end and late results, replay,
outages/retries, endpoints and inspection rendering. Analytics remains deferred
to Phases 4 and 5.

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
