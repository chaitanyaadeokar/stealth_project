# Phase 0/1 live reliability review

> Speaker identity, cross-chunk assembly, auto-language selection, routing and
> shutdown behavior have since been corrected. See the latest
> [correction report](PHASE_0_1_CORRECTION_REPORT.md).

Reviewed and repaired: 18 November 2026. Scope remains Phase 0 and Phase 1.

## Stability verdict

The tested capture, overload, recovery, persistence and routing paths are materially
more reliable. The CPU auto profile kept up with the supplied short recording.
**Phase 0/1 are not yet signed off as production-stable for live panels.** A real
microphone soak test, GPU run, multi-speaker Hindi/Hinglish evaluation and a larger
held-out intent set are still needed. Two substantive claims remained unresolved in
the small classifier check, and the smaller speech model made word errors near
chunk boundaries. No Phase 2+ handler was implemented or disguised as available.

## Root causes and repairs

| Issue | Finding | Repair |
| --- | --- | --- |
| Capture overflow | The consumption loop repeatedly transcribed growing audio, then ran diarization synchronously; queue draining depended on inference completing. | A dedicated blocking-read recorder journals float32 audio independently. One ASR owner and one diarization owner form a bounded pipeline; at most two chunks are in flight. There is no enlarged callback queue. |
| CPU overload | Large-v3-turbo's fixed per-window encoder cost dominated even short chunks. More concurrency alone did not make it real-time. | CPU auto selects multilingual base/int8; CUDA auto retains large-v3-turbo. Explicit model overrides are preserved. Native CPU thread counts are limited. |
| Backpressure | A microphone cannot wait indefinitely for a slower model. The previous callback rejected frames on overflow. | Stop recording explicitly before the next device read at an unprocessed-audio/disk budget. Retain all successfully journaled samples and checkpoints. Hardware-reported lost input is an explicit error, not hidden or repaired with fabricated audio. |
| Unnecessary inference | Repeated interim decoding reprocessed the same samples; many small chunks multiplied fixed costs and fragmented speech acts. | Decode finalized chunks once, using sample-time pauses and a default four-to-six-second window. Bounded acoustic context and word-midpoint ownership suppress repeated context words. Finalized previews replace repeated interim inference. |
| Intent ambiguity | Fact/claim, question/query and opinion/debate competed as seven overlapping labels; generic hypothesis text was weak for speech acts. | Preserve the existing classifier API, but route through four descriptive predicates with an appropriate hypothesis template. Add explicit English/Hindi/Hinglish question, opinion and acknowledgment rules. Keep confidence/margin checks and abstention. |
| Neutral false positives | A substantive claim could be classified as neutral and ignored. | Only explicit acknowledgment rules may produce `no_action`. Model-predicted neutral/ambiguous results require clarification and remain inspectable. |
| Speaker alignment | Exact overlap left small timing gaps unknown; unbounded nearest assignment would invent identities. | Use overlap first and nearest-speaker fallback only within 250 ms with an ambiguity guard. Preserve word/speaker order and complete text if word alignment is incomplete. |
| Speaker consistency | Local diarization label numbers can swap between rolling windows. | Relate labels by shared timestamps with one-to-one matching. Use four seconds of prior context rather than repeatedly diarizing a full 30-second window. Commit tracker state with audio checkpoints for recovery. Without overlap evidence, allocate a new identity rather than assume an old one. |
| Cold startup | Classification initialized on the first live job. | A readiness handshake loads and warms the default classifier before capture; failure releases session resources. ASR, VAD and its first decode initialize before the recorder starts. |
| Persistence/shutdown | An audio checkpoint separate from transcript insertion could acknowledge unsubmitted work or replay committed input. | SQLite commits jobs, audio range and speaker state together with WAL/FULL synchronization. Manifest acknowledgment follows. Recovery trusts the database checkpoint and refuses a mismatched spool/database. Bounded speech drain preserves unfinished raw audio; the intent worker retains its retry/acknowledgment policy. |

Both live scripts now use the same capture engine. Model selection, Hindi/English/
auto language mode, optional diarization, original text, Hinglish display and text
output remain available. CPU auto changes the default model to improve throughput;
it does not claim to preserve the large model's accuracy. Select an explicit model
or `WHISPER_CPU_MODEL` when evaluating a different quality/performance tradeoff.

## Measurements

Single-run measurements on the available Windows/Python 3.14.5 environment, with
12 reported logical CPUs, ASR on CPU/int8 with four threads and pyannote on CPU with
two threads. Model weights were cached locally; no Groq requests or microphone
recording were made. Timings vary with cache warmth and competing CPU work; these
are diagnostic measurements, not controlled statistical benchmarks.

| Workload | Result |
| --- | --- |
| Large-v3-turbo, entire 11 s JFK fixture, beam 5 | 16.71 s wall; 58.91 CPU s; real-time factor 1.52 |
| Same large model, beam 1 | 15.96 s wall; 58.98 CPU s; real-time factor 1.45 |
| Earlier short-chunk large-model pipeline diagnostic | 48.51 s for 11 s audio; demonstrates why short chunks plus the large CPU model are unsuitable here |
| Base, entire fixture, beam 1 | 0.83 s wall; 3.25 CPU s; real-time factor 0.075 |
| Final CPU auto pipeline, actual ASR + diarization, prerecorded replay | 3.76 s for 11 s audio; 12.92 CPU s; two committed chunks; zero unprocessed audio |
| Same final profile, input paced at microphone speed | 13.55 s wall for nominal 11 s input; 12.84 CPU s; two chunks; zero unprocessed audio; about 2.55 s beyond nominal input duration, including journal I/O and final draining |
| Paced profile peak unacknowledged audio | 8.0 s (includes the current four-to-six-second chunk), below the configured backlog budget |
| Final profile journal writes, 44 × 250 ms blocks | Median 3.10 ms, maximum 5.96 ms; 147.72 ms total |
| Final replay ASR stage | 0.77 s for the 6 s chunk; 0.88 s for the 5 s chunk |
| Final replay diarization stage | 1.41 s and 1.56 s respectively |

The CPU model change accounts for much of the speedup. Removing repeated inference,
shortening repeated diarization history, and separating capture from processing
address the scheduling problem; they do not make the large CPU model intrinsically
real-time. The final short-recording check identified one speaker throughout;
multi-speaker identity accuracy has not been measured on real labeled audio.

Speech-profile timings exclude model startup, the separate intent worker and remote
LLM calls. The 16-item intent diagnostic separately uses actual cached ModernBERT inference plus the new
explicit rules. It is a small, manually labeled diagnostic set with rule-covered
phrases, not a blinded held-out accuracy benchmark:

| Routing diagnostic | Previous path | Revised path |
| --- | --- | --- |
| Correct accepted canonical labels, including acknowledgments | 5/16 | 14/16 |
| Abstentions | 9/16 | 2/16 |
| Incorrect accepted labels | 2/16 | 0/16 |
| Total classification wall time | 45.93 s | 7.75 s |

The remaining cases were a Hindi factual assertion and a false factual assertion
misidentified by the NLI model as neutral. Both now remain `needs_clarification`;
neither is silently ignored or treated as verified. Explicit speech-act rules use
deterministic rule scores; NLI scores are relative label scores, not calibrated
truth probabilities. General classification accuracy remains unproven.
For label comparison, old aliases are collapsed (`query` to question, `fact` to claim,
`debate` to opinion, `statement` to other). This gives the old classifier credit for
an acknowledgment label even though its previous orchestrator did not implement a
neutral/no-action route; it is a semantic-label comparison, not an exact replay of
every old delivery behavior.

Raw local profile artifacts:

- [Large-model baseline](data/phase01_asr_baseline.json).
- [CPU model comparison](data/phase01_cpu_auto_profile.json).
- [Final replay and paced capture](data/phase01_cpu_acceptance_profile.json).
- [Final intent diagnostic](data/phase01_intent_final.json).

Profiles under `data/` are ignored runtime artifacts. `profile_phase01.py` provides
the reproducible workload and fixed labeled examples; the results above are also
recorded here so the assessment survives cleanup of local artifacts.

## Verification

Final default suite: **84 passed, 1 skipped, 22 deselected** in 12.47 s.
All 39 Python source files parse, project TOML validates, and `pip check` reports
no broken requirements. The targeted application regression suite has 68 passes.

The final suite includes the original Whisper CPU tests and application regressions.
New coverage exercises:

- Prerecorded JFK decoding through the bundled audio decoder; exact sample roundtrip
  through the journal; chronological replay and duplicate-free checkpoint recovery.
- Continuous simulated capture while inference runs more slowly, finite disk-backed
  overload, explicit device-loss signaling, and retention of every returned sample.
- Bounded concurrent stages and ordered retirement; decoder context-word ownership;
  word-level sentence spans; preservation of incomplete-alignment text.
- Timestamp-gap limits and ambiguity guards; swapped local speaker labels; persisted
  speaker-state restoration.
- English/Hindi/Hinglish speech-act rules, mixed clauses, missing punctuation,
  decimal/title preservation, low-margin ambiguity, and substantive-neutral handling.
- Classifier readiness/failure before capture, session-lock release, real spawned
  intent workers, retries, shutdown deadlines and retained recovery input.
- Existing SQLite atomicity, isolation, idempotence, worker acknowledgment, retry
  exhaustion, handler routing and cross-process diagnostic logging.

The old file-decoder test still requires `ffmpeg`; it skips explicitly when absent.
Default tests exclude the model-download and CUDA-only test cases. Real cached CPU
models were exercised separately by the diagnostic profiles above.

## Operational behavior and limits

- New capture spools live beside the selected database under `audio/RUN_ID/`, with
  `audio.f32` and an atomic `capture.json` manifest. Default unprocessed budget is
  120 s; total recording cap is 512 MiB; recording also stops when free disk falls
  below 32 MiB. A storage failure cannot guarantee retention of the failing write;
  it is reported explicitly, and the prior committed prefix remains recoverable.
- Raw audio is retained even after processing. Disk/log retention is not automatic;
  review and archive/remove finished recordings deliberately. Do not delete a spool
  with outstanding audio you intend to recover.
- Ctrl+C stops recording, then attempts bounded speech draining and bounded intent
  draining. Native model calls cannot be interrupted safely in place. A speech-drain
  timeout leaves bounded daemon owners finishing or stuck, reports recovery input,
  and requires process restart before reuse. It is not an invitation to restart a
  second pipeline on the same model object.
- Recovery does not open a microphone. Use the original database and the retained
  spool directory. SQLite/manifest reconciliation prevents repeating committed
  audio; external LLM requests remain at-least-once across response/commit crashes.
- Small-model and chunk-boundary recognition errors were observed, including repeated
  or altered words. A raw sample-retention success does not prove transcript accuracy.
  Acoustic context and midpoint filtering reduce context repetition but cannot force
  independent model alignments to agree. Use explicit quality settings and evaluate
  on real Hindi/Hinglish panel recordings before choosing a production profile.
- Overlap-based identity mapping cannot recognize a returning speaker without shared
  audio evidence. New recording runs allocate fresh speaker IDs in the session;
  identities remain provisional when overlap is absent or ambiguous.
- The paced run exercises real CPU speech models with prerecorded input, not the
  physical microphone driver. GPU, long recordings, heavy disk contention, and live
  Groq behavior were not benchmarked. Structured logs expose backlog and failure
  conditions for the next local test.
- Phase 2+ remains unchanged: questions and opinions/debates explicitly return
  `unavailable`; configured claim model assessments remain `unverified`, without
  fabricated evidence or citations.

## Run and review

```powershell
.\venv\Scripts\python.exe live_whisper_diarize.py --device cpu --model auto --diarize --language hi
.\venv\Scripts\python.exe live_whisper_diarize.py --db data/sessions.sqlite3 --recover-audio data/audio/RUN_ID --diarize
.\venv\Scripts\python.exe pipeline_cli.py --db data/sessions.sqlite3 --session-id SESSION_ID --inspect
.\venv\Scripts\python.exe -m pytest -q
.\venv\Scripts\python.exe profile_phase01.py --intent
.\venv\Scripts\python.exe profile_phase01.py --speech-pipeline --diarization --cpu-base --paced --output data/profile.json
```

Next acceptance work stays within Phase 0/1: run a 20–30 minute two-speaker recording
in English and Hindi/Hinglish, inspect stage/backlog diagnostics and physical input
loss, score speaker attribution and transcription against a labeled reference, and
expand the intent set. Do not proceed to new architecture components on the strength
of this short fixture alone.
