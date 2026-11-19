"""Replay a JSONL conversation or inspect durable session/job results."""
import argparse
import json
from pathlib import Path

from conversation import ConversationStore
from llm_client import fact_check_statement
from llm_pipeline import start_llm_pipeline
from sentence_classifier import SentenceClassifier, _heuristic_rule_fallback


def demo_engine(text, categories):
    """Explicit offline demo rules, not a replacement for ModernBERT evaluation."""
    label = _heuristic_rule_fallback(text, categories)
    if label is None:
        label = "opinion" if text.lower().startswith(("i think", "in my opinion", "let's debate")) else "claim"
    return {"scores": {label: 1.0}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help='JSONL records: {"speaker":"A", "text":"Why?", "id":"optional-stable-id"}')
    parser.add_argument("--db", default=None)
    parser.add_argument("--session-id", default=None, help="Session to resume or inspect")
    parser.add_argument("--inspect", action="store_true", help="Print stored jobs/results without model loading")
    parser.add_argument("--offline", action="store_true", help="Demo rules only; no model loading or API calls")
    parser.add_argument("--shutdown-timeout", type=float, default=60)
    args = parser.parse_args(argv)
    if args.shutdown_timeout < 0:
        parser.error("--shutdown-timeout must not be negative")
    if args.inspect:
        if not args.session_id:
            parser.error("--inspect requires --session-id")
        store = ConversationStore(args.db)
        try:
            print(json.dumps(store.jobs(args.session_id), ensure_ascii=False, indent=2))
        finally:
            store.close()
        return 0
    if not args.offline:
        from live_runtime import load_environment, validate_environment
        load_environment()
        validate_environment()
    classifier = SentenceClassifier(mock_engine=demo_engine) if args.offline else None
    buffer, _, _ = start_llm_pipeline(None if args.offline else fact_check_statement,
                                      db_path=args.db, session_id=args.session_id, classifier=classifier)
    try:
        if args.input:
            for number, line in enumerate(args.input.read_text(encoding="utf-8").splitlines(), 1):
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                    buffer.append(item.get("speaker", "UNKNOWN"), item["text"],
                                  utterance_id=item.get("id"), language=item.get("language"),
                                  start=item.get("start"), end=item.get("end"))
                except (KeyError, ValueError, TypeError) as exc:
                    raise ValueError(f"Invalid input record at line {number}: {exc}") from exc
    finally:
        summary = buffer.close(timeout=args.shutdown_timeout)
        print(json.dumps(summary))
    return 1 if summary["pending"] or summary["failed"] or summary["worker_exitcode"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
