# Plan: Sentence Classification, Routing & Parallel Retrieval

Status: draft, to be refined in further discussion.
Context sources: `architecture-Diagram.png`, `Intelligent Agent Panel Discussion Architecture.docx`, `whisper/sentence_classifier.py`.

---

## 1. Why we need the sentence classifier

- The architecture routes every utterance to one of three pipelines via an **Intent Classifier**:
  - **Question/Query**: answer retrieval, knowledge base/RAG, evidence fetching, top relevant context.
  - **Fact/Claim**: fact verification, evidence engine, source attribution, verdict (True / False / Uncertain).
  - **Opinion/Debate**: multi-agent panel discussion, AI panelists, reasoning, panel responses.
- Those pipelines are expensive (web/archive search, RAG, multi-agent LLM). A small local classifier decides in milliseconds which one to run, so live interjections stay fast and cheap.
- The ethics evaluation engine scores every statement regardless of track. The classifier only decides the extra work (evidence lookup, retrieval, panel debate).
- A wrong or missing label has visible effects on stage: a missed claim goes unchallenged, a wrongly flagged opinion causes a pointless interruption.

## 2. Current state

- `sentence_classifier.py`: zero-shot ModernBERT-base (`MoritzLaurer/ModernBERT-base-zeroshot-v2.0`), 3 categories, speaker-tag sanitization, margin and ambiguity flags, multi-label option.
- Added: `split_sentences()` (pysbd with regex fallback) and `SentenceClassifier.classify_text()` (per-sentence classification plus word-count-weighted aggregate, short backchannels excluded).
- Tests: `whisper/tests/test_sentence_aggregation.py` (new, passing). `test_ambiguous_anecdotal_storytelling_paragraph` in `test_sentence_classifier.py` fails; believed pre-existing, not yet confirmed.
- Not yet integrated: nothing in `live_whisper_diarize.py` or `llm_pipeline.py` calls the classifier.

## 3. Proposal: split, then parallel retrieval per claim

1. Take the transcribed paragraph and split it into sentences (a turn may contain a claim, an opinion and a question).
2. Classify each sentence individually. Route per sentence, not per paragraph. The aggregate score is for logging/context only.
3. For each sentence that is a verifiable claim or an information-seeking question, run retrieval/search in parallel.
4. Trade tokens and internet usage for higher-quality evidence for the orchestrator agent.
5. Known downside: larger context for the orchestrator (to be handled later).

## 4. Which sentence types need retrieval or search

**Database / knowledge base (RAG)**
- Information-seeking questions about known internal content (policies, past sessions, documents).
- Questions about what was said earlier or by a specific speaker.
- Claims about previously recorded statements or historical records in the archive.
- Requests for definitions, background or context on the topic.
- Claims that can be checked against stored transcripts, videos or gazettes.
- Questions about the agenda, panelists or prior decisions.

**Internet / external search**
- Statistical claims (numbers, percentages, rates, budgets).
- Recent events, news or current-status claims.
- Claims about laws, policies, court rulings, official announcements.
- Scientific or technical claims with a verifiable answer.
- Claims about named people, companies or organizations.
- Historical claims (dates, timelines).
- Comparative or ranking claims ("largest", "first", "higher than").
- Causal claims backed by research.
- Questions needing up-to-date facts outside the knowledge base.

**Neither**
- Pure opinions and value judgments, rhetorical questions, personal anecdotes, backchannels/greetings/filler, trivial or unverifiable statements, hypotheticals with no checkable basis.

**Mixed sentences** (opinion with an embedded factual claim): retrieve only for the factual part.

## 5. Resource constraints

- Production may be **CPU-only (no GPU guaranteed)**.
- Memory budget: **4 GB fast working memory** total.
  - Whisper: ~1.5 GB
  - Classifier encoder: ~1 GB
  - Remaining ~1.5 GB must cover: Python/torch runtime, diarization model (not yet budgeted), embedding model and vector index for RAG, audio buffers, in-flight search results.
- Search/retrieval concurrency: **2 threads** (I/O-bound, low CPU/memory). Can be raised later; real limits are API rate limits and latency.
- Open questions: is the 4 GB RAM or VRAM? Is the orchestrator LLM local or API-based (a local LLM would not fit)?

## 6. Design decisions

- **One worker per model** (Whisper, classifier). No parallel inference on the same model; threads only for network-bound search/retrieval.
- **Explicit CPU core assignment** (`torch.set_num_threads` per model) so Whisper, classifier, diarization and search do not contend.
- **Reduce classifier cost on CPU**: zero-shot NLI runs one pass per label (3 per sentence). Prefer a fine-tuned single-pass head and/or ONNX int8 export. Add a rule-based pre-filter (question mark / wh-word) to skip the model for obvious cases. Skip filler and very short sentences.
- **Whisper on CPU**: consider `faster-whisper` with int8 quantization or a smaller model; measure real-time factor early.
- **Be selective about what is searched**: only verifiable claims and information-seeking questions. Add a check-worthiness score on top of fact/opinion. Cache by normalized claim.
- **Query rewriting**: resolve references ("that policy") using the Context Manager before searching.
- **Retrieval order**: database first (local, fast), web only if the database result is insufficient.
- **Per-claim fan-out**: cap claims per paragraph (about 3-5), ranked by check-worthiness; 2-worker pool; per-search timeouts (about 2-3 s) with partial results; per-source rate limits; fall back to the database if web search fails.
- **Tagging**: every result carries sentence ID, speaker and timestamp; results merged into one evidence packet.
- **Evidence packet for the orchestrator**: claim, 1-3 snippets, source, date, confidence. Deduplicate and rank before sending; never send raw pages.
- **Speaker handling**: sentence buffering must not merge text across speakers; AI agents' own utterances must not re-trigger claim handling.

## 7. Failure handling (classifier is a single point of failure)

| Misclassification | Impact |
|---|---|
| Claim labelled opinion | No fact-check; misstatement stands (silent, worst case) |
| Opinion labelled claim | Verdict on a value judgment, or unwarranted interruption |
| Question labelled claim | Fact-checked but never answered |
| Claim/opinion labelled question | Goes to retrieval with no useful answer, or lost |
| Paragraph given one label | Parts of a mixed turn go to the wrong track |

Mitigations:
- No hard switch: route to the top-2 tracks when the margin is small.
- Bias toward recall for claims (wasted compute is cheaper than a missed error).
- Split before classifying (per-sentence routing).
- Ambiguous cases (`is_ambiguous`): LLM second opinion or moderated-interjection queue.
- Downstream stages may reject input ("not a verifiable claim", "no answer found") and trigger re-routing.
- Gate spontaneous interruptions on high confidence plus a second signal.
- Fail-open: if the classifier crashes or times out, send to all tracks or an LLM; never drop the utterance.
- Log labels as predictions (scores, margin, track chosen), correctable later; use corrections as training data.
- Define what the FeedBack Agent loop corrects and when.

Proposed flow:
```
sentence -> classifier
   |- confident      -> single track
   |- close margin   -> top-2 tracks
   |- ambiguous      -> LLM check or moderator queue
   '- classifier failure -> all tracks (fail-open)
downstream stages can reject -> re-route
```

## 8. Target pipeline

```
Whisper -> diarization -> sentence split -> classifier (1 worker)
        -> worth searching? -> query rewrite (with context)
        -> ThreadPool(2): DB lookup, then web search if needed, with timeouts
        -> dedupe, rank, compress -> evidence packet -> orchestrator
```

## 9. Next steps (not started)

1. Benchmark the classifier on a CPU-only machine matching production, with and without ONNX int8 (`whisper/tests/benchmark_latency.py`).
2. Implement `route()` returning per-sentence `(sentence, speaker, track(s), scores, is_ambiguous, check_worthiness)`; replace the aggregate with per-turn `has_claim` / `has_question` flags.
3. Implement the retrieval skeleton: 2-worker pool, claim cap, DB-then-web order, timeouts, cache, evidence-packet format.
4. Implement the streaming sentence buffer (flush on speaker change or silence timeout).
5. Log every routing decision to the audit trail via `whisper/logger.py`.
6. Run a small evaluation: label 50-100 real transcript sentences and compare zero-shot, rules plus model, and an LLM labeller on accuracy and latency. Decide keep/replace based on results.
7. Read `live_whisper_diarize.py` to confirm how speaker-labelled segments are produced.

## 10. Open questions

- Is the 4 GB limit RAM or VRAM? Is the orchestrator LLM local or API?
- Production CPU core count and RAM?
- Should "statement" and "claim" be separate labels (the diagram lists both; the code aliases `statement` to `fact_claim`)?
- Add an `OTHER` class for backchannels and filler?
- Keep or drop the weighted aggregate score once routing is per sentence?
- Orchestrator context-size strategy for evidence packets.
