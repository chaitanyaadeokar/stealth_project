# Phase 0/1: speaker, utterance, routing and Ctrl+C corrections

Date: 18 November 2026. This supersedes the relevant behavior in the earlier
[reliability report](PHASE_0_1_RELIABILITY_REPORT.md).

## Changes

### Persistent two-speaker identity tracking

The previous tracker only used temporal overlap. When a speaker returned outside
the rolling context it could allocate another ID. The adapter also discarded
pyannote's speaker embeddings.

- Preserve pyannote's centroids alongside the existing list-compatible turn output.
- Match local labels to persistent, normalized session prototypes using overlap and
  cosine similarity; assign simultaneous returning voices jointly rather than
  greedily by local label order.
- With `--num-speakers 2`, assignments are restricted to `SPEAKER_00` and
  `SPEAKER_01`. Persist prototypes, mapping and counter in the existing SQLite audio
  checkpoint. Recovery restores the same identity set.
- A fixed count also enables diarization. If its model cannot load, capture fails
  before opening the microphone instead of silently running without the requested IDs.
- Treat two speakers as a session constraint, not a demand that every short window
  contain two voices. Per-window inference permits one or two local clusters.
- Log uncertain fallback assignments rather than claim unsupported identity certainty.
- Repair a very short, unfinished function-word prefix only when both the segment
  and following word support the other speaker. Complete short interjections remain
  separate. Tests cover `I don't think…` boundaries and preserve `Yes.` turns.

The two IDs are available identities, not fabricated speech. A window containing
one detected voice can emit only one of them. Without a speaker-labeled reference,
the recording's true attribution accuracy cannot be established. The cosine
thresholds are conservative implementation heuristics, not calibrated guarantees.

### Meaningful utterances before classification

The existing chunk/VAD/capture mechanism remains. `UtteranceAssembler` operates
between its decoded records and intent jobs:

- Hold an unfinished final clause across chunks, including artificial decoder periods
  and question marks at a cutoff.
- Join continuations with original chronology and source timing. Deduplicate matching
  boundary words only when their timestamps overlap, preserving genuine repetition.
- Close on speaker changes, real pauses or an independent following sentence. Retain
  bounded length/time limits for uninterrupted monologues.
- Persist pending text and source IDs atomically with the audio checkpoint. A crash
  after audio acknowledgment does not lose a half-built sentence.
- Flush the final pending utterance atomically; source-derived IDs make repeated
  flushes idempotent. If shutdown leaves genuinely incomplete speech, it is retained
  and reported as unfinished rather than completed with invented words.

### Transcription and routing

A concrete language bug was reproduced on the discussion's 44-second region:
English scored 0.755 and Hindi 0.108. The old `auto` path nevertheless forced Hindi
because the secondary score exceeded 0.10. That override has been removed; `auto`
uses the best detected language. Explicit Hindi and English options are preserved.

Routing now compares three substantive speech acts. Neutral acknowledgments use
explicit rules and no longer compete with factual claims as a model category.
Confidence and ambiguity checks remain. Clear Hindi declaratives, internal Hindi
question words, discourse-prefixed questions, negated opinions and grammatical
capability assertions have explicit handling.

The recorded statement `AI can answer questions…` exposed a remaining model error:
the NLI model selected question because of the topical word “questions.” A guarded
subject-plus-capability rule now routes that assertion as a claim; `Can AI answer
questions?` stays a question, and `I think AI can…` stays an opinion.

Question retrieval and debate-panel handlers remain explicitly unavailable. Existing
claim model assessments remain unverified. This change adds no evidence retrieval,
citations, debate agents or other Phase 2+ components.

### Ctrl+C and worker exit status

- The capture controller turns SIGINT/SIGBREAK into a stop request rather than an
  asynchronous exception that can tear an utterance/checkpoint transaction in half.
- The intent worker ignores console interrupts. Repeated console interrupts are
  shielded while cleanup commits and closes resources; previous handlers are restored.
- One existing bounded serial owner executes intent jobs with its own SQLite
  connection. The child controller can observe its shutdown deadline even if a
  model/API call is blocked.
- At the deadline, unfinished work remains pending and the child exits normally.
  No successful result is acknowledged for a deferred request. The parent releases
  ownership only after process exit.
- Actual worker errors and hard termination still retain their real exit status.
  Exit code 1 is not relabeled as success. The normal SIGINT and slow-request cases
  were tested to return code 0.

## Frozen-audio comparison

The retained source was read without changing its original audio, manifest or
session database. Comparison runs use isolated databases and spools under `data/`.

- Source: `data/audio/03815af0ec524c34a21a21e1a9fbb9e5/audio.f32`.
- Length: 4,316,000 mono float32 samples at 16 kHz, or **269.75 seconds**.
- SHA256: `4121b84ef55dcb7e659df7e5e049b0048595e93dc8c8bbac6340d6fb01d14bac`.
- Both models use CPU/int8, four ASR threads, beam size 1 and identical frozen audio.
- Both corrected streaming replays use the same 44–68 second passage, the same
  sample-driven boundaries, a 12-second maximum acoustic chunk, auto language,
  diarizer configuration, assembler and storage path.

| Measurement | Whisper base | Whisper large-v3-turbo |
| --- | ---: | ---: |
| Labeled JFK fixture WER | 0% | 0% |
| JFK fixture wall time after model warm-up | 3.21 s | 42.69 s |
| Whole 269.75 s recording decode wall time | 11.06 s | 492.56 s |
| Corrected 24 s ASR/diarization/assembly replay | 42.08 s | 300.23 s |
| Streaming chunks committed | 3 | 3 |
| Assembled utterances | 2 | 2 |
| Unprocessed audio at completion | 0 s | 0 s |

These are observed single runs on this CPU environment, not controlled statistical
benchmarks. Whole-record times were retained from the initial run, which experienced
concurrent CPU/memory load; their provenance is recorded in the JSON. Model load and
warm-up are separate from the quoted fixture timings. No CUDA device was available.
Neither corrected streaming result is real-time on this CPU run. No speed ratio is
claimed as an isolated model benchmark.

The full-discussion transcripts have **1.83% normalized word disagreement**. That
is not WER: no corrected reference transcript was supplied for the discussion. Both
models still disagree on words in the streamed passage, so this test does not prove
that either model is generally more accurate. The two models' 0% WER results apply
only to the short labeled JFK fixture.

The corrected 24-second passage produced only `SPEAKER_00` from the backend, despite
two configured identity slots. A longer 64-second default-chunk replay of the same
source with base observed both `SPEAKER_00` and `SPEAKER_01`, without allocating further IDs.
Known two-voice embedding fixtures additionally test returning voices after a gap,
swapped local labels and persistence through restart. None of these substitutes
for a human-labeled timeline of the discussion or establishes a diarization error rate.

Artifacts:

- [Corrected same-passage comparison](data/recording_comparison_fixed_auto/comparison.json).
- [Whole-record measurement provenance](data/recording_comparison_20261118/whole_measurements.json).
- [Longer default-chunk replay](data/recording_comparison_final/comparison.json).
- [Intent evaluation](data/intent_reconstruction_final.json).

The earlier partial replay directories record development runs and should not be
confused with the completed corrected comparison. Runtime artifacts are ignored by Git.

## Regression verification

Final default suite: **116 passed, 1 skipped, 22 deselected** in 56.08 seconds.
The skip is the existing `ffmpeg`-dependent decoder test; the deselected cases are
the explicit model-download/GPU tests. Cached native CPU models were tested separately
in the comparisons above. All 43 Python files parse, project TOML/module paths
validate, `pip check` passes, and the original audio SHA256 is unchanged.
No automated regressions were found in the exercised capture, persistence, routing
and shutdown paths; this is not a claim of unmeasured live or speaker accuracy.

The real hybrid intent diagnostic correctly classified **20/20 cases with zero
abstentions**: the existing 16-item development set plus the two decoded passage
utterances from each speech model. The first run caught two capability statements
misrouted as questions (18/20); the guarded grammatical rule fixed both on rerun.
These labels describe the decoded text. They are not an ASR reference, a blind
held-out test, or a guarantee of general classifier accuracy.

Tests cover the original capture journal and overload behavior plus:

- Two fixed IDs, non-overlapping returning speakers, local label swaps, centroid
  extraction, session-level counts, uncertain fallbacks and state restoration.
- Cross-chunk syntax, punctuation at cutoffs, overlapping boundary words, genuine
  repetition, speaker order and atomic pending-utterance recovery/finalization.
- Best-language auto selection, unfinished inputs, Hindi assertions/questions,
  discourse prefixes, capability statements and the actual decoded passage's intents.
- Real spawned workers receiving SIGINT, repeated interrupts, a signal during a
  SQLite commit, slow-request deferral, pending-job persistence and clean exit code 0.

A preexisting continuous-capture test used a wall timer to assume a minimum number
of frames, which failed under benchmark CPU contention. It now uses an exact
20-block source and verifies all five seconds are retained, rather than weakening
the sample-retention assertion or skipping the test.

## Remaining limits

- The discussion lacks a corrected transcript and speaker timeline. Its ASR and
  speaker-attribution accuracy remain unmeasured; do not substitute model agreement
  or a bounded ID count for accuracy.
- Ambiguous acoustics, similar voices, overlapping speech and unknown voices can
  still produce uncertain identity assignments. Two configured slots do not justify
  inventing a second active speaker when only one is detected.
- Word recognition errors remain. Utterance assembly joins decoded words; it cannot
  recover words the recognizer misheard or omitted.
- CPU streaming, particularly the large model, remains slow in these measurements.
  Durable capture/backpressure is preserved, but this report does not certify
  continuous real-time operation or a long live panel soak test.
- A hard kill or crash between a remote LLM response and its database commit may
  repeat that external request on retry. The normal shutdown path preserves work
  and exits cleanly, without claiming exactly-once external execution.

Reproduce comparisons with `compare_recording.py` and the semantic diagnostic with
`evaluate_intent_reconstruction.py`. Both use cached models and no external LLM calls.
For an actual English-only recording, `--language en` avoids per-chunk language
uncertainty; use `--language hi` explicitly for Hindi/Hinglish evaluation.

```powershell
.\venv\Scripts\python.exe compare_recording.py --spool data/audio/03815af0ec524c34a21a21e1a9fbb9e5 --output data/new-comparison --replay-start 44 --replay-seconds 24 --chunk-seconds 12
.\venv\Scripts\python.exe evaluate_intent_reconstruction.py --comparison data/recording_comparison_fixed_auto/comparison.json
.\venv\Scripts\python.exe -m pytest -q
```
