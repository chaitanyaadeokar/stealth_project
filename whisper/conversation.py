"""Chronological, durable conversation and dispatch state (standard library only)."""
from __future__ import annotations

import json
import os
import sqlite3
import time
import uuid
from pathlib import Path


def default_db_path():
    return os.environ.get("CONVERSATION_DB", str(Path(__file__).parent / "data" / "sessions.sqlite3"))


class ConversationStore:
    """One connection per owner/thread; transactions serialize writers across processes."""

    def __init__(self, path=None):
        self.path = str(Path(path or default_db_path()).resolve())
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(self.path, timeout=5)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY, created_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS utterances (
                id TEXT PRIMARY KEY, session_id TEXT NOT NULL REFERENCES sessions(id),
                sequence INTEGER NOT NULL, speaker TEXT NOT NULL, text TEXT NOT NULL,
                display_text TEXT, language TEXT, start REAL, end REAL,
                created_at REAL NOT NULL, UNIQUE(session_id, sequence)
            );
            CREATE TABLE IF NOT EXISTS jobs (
                utterance_id TEXT PRIMARY KEY REFERENCES utterances(id),
                status TEXT NOT NULL DEFAULT 'pending', attempts INTEGER NOT NULL DEFAULT 0,
                available_at REAL NOT NULL DEFAULT 0, owner TEXT, lease_until REAL,
                error TEXT, result TEXT
            );
            CREATE INDEX IF NOT EXISTS utterance_order ON utterances(session_id, sequence);
            CREATE TABLE IF NOT EXISTS audio_commits (
                spool_id TEXT NOT NULL, start_sample INTEGER NOT NULL, end_sample INTEGER NOT NULL,
                session_id TEXT NOT NULL REFERENCES sessions(id),
                speaker_state TEXT,
                PRIMARY KEY(spool_id, start_sample)
            );
        """)
        with self.db:
            self.db.execute("BEGIN IMMEDIATE")
            if "speaker_state" not in {row[1] for row in self.db.execute("PRAGMA table_info(audio_commits)")}:
                self.db.execute("ALTER TABLE audio_commits ADD COLUMN speaker_state TEXT")

    def close(self):
        self.db.close()

    def create_session(self, session_id=None):
        session_id = session_id or uuid.uuid4().hex
        with self.db:
            self.db.execute("INSERT OR IGNORE INTO sessions VALUES (?, ?)", (session_id, time.time()))
        return session_id

    def append(self, session_id, speaker, text, *, utterance_id=None, display_text=None,
               language=None, start=None, end=None, max_pending=1000):
        return self.append_many(session_id, [dict(speaker=speaker, text=text,
            utterance_id=utterance_id, display_text=display_text, language=language,
            start=start, end=end)], max_pending=max_pending)[0]

    def append_many(self, session_id, records, *, max_pending=1000, audio_commit=None, assembly_update=None):
        """Commit a finalized chunk atomically so partial input cannot be replayed twice."""
        with self.db:
            self.db.execute("BEGIN IMMEDIATE")
            result = [self._append(session_id, max_pending=max_pending, **record) for record in records]
            if audio_commit is not None:
                spool_id, start, end, *state = audio_commit
                self.db.execute("""INSERT OR IGNORE INTO audio_commits
                    (spool_id,start_sample,end_sample,session_id,speaker_state) VALUES (?,?,?,?,?)""",
                    (spool_id, start, end, session_id, json.dumps(state[0]) if state else None))
            if assembly_update is not None:
                spool_id, state = assembly_update
                self.db.execute("""UPDATE audio_commits SET speaker_state=? WHERE spool_id=? AND session_id=?
                    AND end_sample=(SELECT max(end_sample) FROM audio_commits WHERE spool_id=? AND session_id=?)""",
                    (json.dumps(state),spool_id,session_id,spool_id,session_id))
            return result

    def audio_checkpoint(self, session_id, spool_id):
        return self.db.execute("SELECT coalesce(max(end_sample),0) FROM audio_commits WHERE session_id=? AND spool_id=?",
                               (session_id, spool_id)).fetchone()[0]

    def audio_speaker_state(self, session_id, spool_id):
        row = self.db.execute("""SELECT speaker_state FROM audio_commits WHERE session_id=? AND spool_id=?
            ORDER BY end_sample DESC LIMIT 1""", (session_id, spool_id)).fetchone()
        return json.loads(row[0]) if row and row[0] else None

    def _append(self, session_id, speaker, text, *, utterance_id=None, display_text=None,
                language=None, start=None, end=None, max_pending=1000):
        text = text.strip()
        if not text:
            return None
        if (start is not None and start < 0) or (end is not None and start is not None and end < start):
            raise ValueError("Invalid utterance timestamps")
        utterance_id = utterance_id or uuid.uuid4().hex
        existing = self.db.execute("SELECT * FROM utterances WHERE id=?", (utterance_id,)).fetchone()
        if existing:
            if existing["session_id"] != session_id or existing["text"] != text or existing["speaker"] != speaker:
                raise ValueError("Utterance ID already belongs to different input")
            return utterance_id
        pending = self.db.execute("""SELECT count(*) FROM jobs j JOIN utterances u
            ON u.id=j.utterance_id WHERE u.session_id=? AND j.status IN ('pending','running')""",
            (session_id,)).fetchone()[0]
        if pending >= max_pending:
            raise OverflowError("Conversation backlog is full; input was not accepted")
        sequence = self.db.execute("SELECT coalesce(max(sequence),0)+1 FROM utterances WHERE session_id=?",
                                   (session_id,)).fetchone()[0]
        self.db.execute("INSERT INTO utterances VALUES (?,?,?,?,?,?,?,?,?,?)",
                        (utterance_id, session_id, sequence, speaker, text, display_text,
                         language, start, end, time.time()))
        self.db.execute("INSERT INTO jobs(utterance_id) VALUES (?)", (utterance_id,))
        return utterance_id

    def turns(self, session_id, *, before=None, limit=20):
        if limit < 1:
            return []
        rows = self.db.execute("""SELECT * FROM utterances WHERE session_id=?
            AND (? IS NULL OR sequence < ?) ORDER BY sequence DESC LIMIT ?""",
            (session_id, before, before, limit)).fetchall()
        return [dict(row) for row in reversed(rows)]

    def claim(self, session_id, owner, *, max_attempts=3, lease_seconds=120):
        """Claim oldest unfinished turn; delayed retries block later turns to preserve order."""
        now = time.time()
        with self.db:
            self.db.execute("BEGIN IMMEDIATE")
            self.db.execute("""UPDATE jobs SET status='failed', error=coalesce(error,'Attempt limit reached')
                WHERE status='pending' AND attempts>=? AND utterance_id IN
                (SELECT id FROM utterances WHERE session_id=?)""", (max_attempts, session_id))
            self.db.execute("""UPDATE jobs SET status=CASE WHEN attempts>=? THEN 'failed' ELSE 'pending' END,
                owner=NULL, lease_until=NULL, error='Worker lease expired' WHERE status='running'
                AND lease_until<=? AND utterance_id IN (SELECT id FROM utterances WHERE session_id=?)""",
                (max_attempts, now, session_id))
            row = self.db.execute("""SELECT u.*, j.status, j.attempts, j.available_at FROM utterances u
                JOIN jobs j ON j.utterance_id=u.id WHERE u.session_id=?
                AND j.status IN ('pending','running') ORDER BY u.sequence LIMIT 1""", (session_id,)).fetchone()
            if not row or row["status"] != "pending" or row["available_at"] > now:
                return None
            self.db.execute("""UPDATE jobs SET status='running', attempts=attempts+1,
                owner=?, lease_until=? WHERE utterance_id=?""", (owner, now + lease_seconds, row["id"]))
            result = dict(row)
            result["attempts"] += 1
            return result

    def complete(self, utterance_id, owner, result):
        with self.db:
            cursor = self.db.execute("""UPDATE jobs SET status='succeeded', result=?, error=NULL,
                owner=NULL, lease_until=NULL WHERE utterance_id=? AND owner=? AND status='running'""",
                (json.dumps(result, ensure_ascii=False), utterance_id, owner))
            if cursor.rowcount != 1:
                raise RuntimeError("Job ownership lost before acknowledgment")

    def fail(self, utterance_id, owner, error, *, max_attempts=3, retry_delay=1):
        with self.db:
            self.db.execute("""UPDATE jobs SET status=CASE WHEN attempts>=? THEN 'failed' ELSE 'pending' END,
                error=?, available_at=?, owner=NULL, lease_until=NULL
                WHERE utterance_id=? AND owner=? AND status='running'""",
                (max_attempts, str(error), time.time() + retry_delay, utterance_id, owner))

    def release(self, owner):
        """Only call after the owning worker has stopped; preserve consumed attempt counts."""
        with self.db:
            self.db.execute("""UPDATE jobs SET status='pending', owner=NULL, lease_until=NULL,
                available_at=0, error=coalesce(error,'Worker stopped before acknowledgment; retry retained')
                WHERE owner=? AND status='running'""", (owner,))

    def jobs(self, session_id):
        rows = self.db.execute("""SELECT u.*, j.status, j.attempts, j.error, j.result FROM utterances u
            JOIN jobs j ON j.utterance_id=u.id WHERE u.session_id=? ORDER BY u.sequence""", (session_id,))
        return [dict(row, result=json.loads(row["result"]) if row["result"] else None) for row in rows]

    def pending_count(self, session_id):
        return self.db.execute("""SELECT count(*) FROM jobs j JOIN utterances u ON u.id=j.utterance_id
            WHERE u.session_id=? AND j.status IN ('pending','running')""", (session_id,)).fetchone()[0]

    def job_counts(self, session_id):
        return dict(self.db.execute("""SELECT j.status, count(*) FROM jobs j
            JOIN utterances u ON u.id=j.utterance_id WHERE u.session_id=? GROUP BY j.status""", (session_id,)))


class ConversationContext:
    def __init__(self, store, max_turns=20, max_chars=12000):
        if max_turns < 1 or max_chars < 1:
            raise ValueError("Context limits must be positive")
        self.store, self.max_turns, self.max_chars = store, max_turns, max_chars

    def for_turn(self, turn):
        turns = self.store.turns(turn["session_id"], before=turn["sequence"], limit=self.max_turns)
        selected, remaining = [], self.max_chars
        for previous in reversed(turns):
            text = previous["text"][-remaining:]
            selected.append(dict(previous, text=text))
            remaining -= len(text)
            if remaining <= 0:
                break
        return list(reversed(selected))
