"""Persistent shared memory for experiment history and verified findings."""

from __future__ import annotations

import fcntl
import hashlib
import json
import math
import os
import sqlite3
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PLAYBOOK_SEED = """## 1. Camera Geometry & Back-Projection Equations
- p_world = cam_pos + cam_rot @ [(u-cx)/fx*d, (v-cy)/fy*d, d], where d = depth[v,u] metres; K has fx=fy≈579.4, cx=320, cy=240.
## 2. Verified 3D Vision Filters (Dust/Glare/Occlusion Invariant)
- Depth contains NaN/0 dropouts and 0.25/1.80 m spikes; dust plume occurs during frames 40..70.
## 3. Verified Excavation Ordering & Force-Guarded Control Laws
- The core is buried under rubble_a, and rubble_b leans on its right edge; stop contact at the 18 N fracture threshold.
## 4. Graveyard of Rejected Hypotheses (DO NOT REPEAT)
- No rejected hypotheses are recorded yet.
"""

HEADERS = (
    "## 1. Camera Geometry & Back-Projection Equations",
    "## 2. Verified 3D Vision Filters (Dust/Glare/Occlusion Invariant)",
    "## 3. Verified Excavation Ordering & Force-Guarded Control Laws",
    "## 4. Graveyard of Rejected Hypotheses (DO NOT REPEAT)",
)

MAX_TRANSACTIONS = 2000


def estimate_tokens(text: str) -> int:
    return math.ceil(len(text) / 4)


class SharedBrain:
    """Store the shared playbook, evaluation ledger, and transaction history."""

    def __init__(self, root: str | Path = "shared_brain") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.db_path = self.root / "brain.db"
        self.ledger_path = self.root / "ledger.json"
        self.playbook_path = self.root / "playbook.md"
        self._init_db()
        if not self.ledger_path.exists():
            self._atomic_json(self.ledger_path, [])
        if not self.playbook_path.exists():
            self._atomic_text(self.playbook_path, PLAYBOOK_SEED)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.execute("PRAGMA busy_timeout=30000")
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_db(self) -> None:
        with self._connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS transactions (
                    id INTEGER PRIMARY KEY, ts TEXT NOT NULL, agent TEXT NOT NULL,
                    op TEXT NOT NULL CHECK(op IN ('read','write')), target TEXT NOT NULL,
                    tokens INTEGER NOT NULL, detail_json TEXT
                );
                CREATE TABLE IF NOT EXISTS versions (
                    version INTEGER PRIMARY KEY, ts TEXT NOT NULL, source TEXT NOT NULL,
                    sha256 TEXT NOT NULL, diff_text TEXT NOT NULL, opt_json TEXT,
                    holdout_json TEXT, accepted INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS seed_metrics (
                    version INTEGER NOT NULL, seed INTEGER NOT NULL,
                    split TEXT NOT NULL CHECK(split IN ('opt','holdout')), metrics_json TEXT NOT NULL,
                    PRIMARY KEY(version, seed, split)
                );
                CREATE TABLE IF NOT EXISTS mission_anchors (
                    id INTEGER PRIMARY KEY CHECK(id = 1),
                    digest TEXT NOT NULL,
                    document_json TEXT NOT NULL,
                    source TEXT NOT NULL,
                    recorded_at TEXT NOT NULL,
                    legacy_ledger_sha256 TEXT,
                    legacy_entry_count INTEGER,
                    legacy_max_version INTEGER
                );
                CREATE TABLE IF NOT EXISTS counters (
                    name TEXT PRIMARY KEY, value INTEGER NOT NULL
                );
            """)
            # Seed lifetime counters once, before older transaction rows are pruned.
            for name, query in (
                ("context_reads", "SELECT count(*) FROM transactions WHERE op='read' AND target='playbook.md'"),
                ("context_read_tokens", "SELECT coalesce(sum(tokens),0) FROM transactions WHERE op='read' AND target='playbook.md'"),
                ("context_write_tokens", "SELECT coalesce(sum(tokens),0) FROM transactions WHERE op='write' AND target='playbook.md'"),
                ("conflicts_prevented", "SELECT count(*) FROM transactions WHERE target='conflict_merge'"),
                ("transactions_pruned", "SELECT 0"),
            ):
                db.execute("INSERT OR IGNORE INTO counters(name,value) VALUES(?,?)", (name, db.execute(query).fetchone()[0]))
            self._prune_transactions(db)

    @staticmethod
    def _increment(db, name: str, amount: int) -> None:
        db.execute("INSERT INTO counters(name,value) VALUES(?,?) ON CONFLICT(name) DO UPDATE SET value=value+excluded.value",
                   (name, amount))

    def _prune_transactions(self, db) -> None:
        """Bound ephemeral operation history; version/seed evidence is never pruned."""
        removed = db.execute("DELETE FROM transactions WHERE id IN (SELECT id FROM transactions ORDER BY id DESC LIMIT -1 OFFSET ?)",
                             (MAX_TRANSACTIONS,)).rowcount
        if removed:
            self._increment(db, "transactions_pruned", removed)

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def _locked(self, target: Path):
        lock = open(str(target) + ".lock", "a+")
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        return lock

    @contextmanager
    def campaign_lock(self):
        """Serialize campaign operations that allocate versions and bind mission evidence."""
        lock = self._locked(self.root / "campaign")
        try:
            yield
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
            lock.close()

    def _atomic_text(self, path: Path, text: str) -> None:
        lock = self._locked(path)
        try:
            fd, name = tempfile.mkstemp(prefix=path.name, dir=path.parent)
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    f.write(text)
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(name, path)
            finally:
                if os.path.exists(name):
                    os.unlink(name)
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
            lock.close()

    def _atomic_json(self, path: Path, value: Any) -> None:
        self._atomic_text(path, json.dumps(value, indent=2, ensure_ascii=False) + "\n")

    def log_transaction(self, agent: str, op: str, target: str, tokens: int = 0,
                         detail: dict | None = None) -> None:
        if op not in ("read", "write"):
            raise ValueError("op must be 'read' or 'write'")
        with self._connect() as db:
            db.execute("INSERT INTO transactions(ts,agent,op,target,tokens,detail_json) VALUES(?,?,?,?,?,?)",
                       (self._now(), agent, op, target, tokens,
                        json.dumps(detail, ensure_ascii=False) if detail is not None else None))
            if target == "playbook.md":
                self._increment(db, "context_read_tokens" if op == "read" else "context_write_tokens", tokens)
                if op == "read":
                    self._increment(db, "context_reads", 1)
            if target == "conflict_merge":
                self._increment(db, "conflicts_prevented", 1)
            self._prune_transactions(db)

    def record_version(self, entry: dict, diff_text: str, per_seed_opt: list,
                       per_seed_holdout: list) -> None:
        version = int(entry["version"])
        with self._connect() as db:
            db.execute("INSERT INTO versions VALUES(?,?,?,?,?,?,?,?)", (
                version, entry.get("timestamp", self._now()), entry["source"], entry["sha256"],
                diff_text, json.dumps(entry.get("opt")), json.dumps(entry.get("holdout")),
                int(bool(entry["accepted"]))))
            for split, metrics in (("opt", per_seed_opt), ("holdout", per_seed_holdout)):
                for row in metrics:
                    seed = int(row["seed"])
                    db.execute("INSERT INTO seed_metrics VALUES(?,?,?,?)",
                               (version, seed, split, json.dumps(row, ensure_ascii=False)))

    def record_mission_anchor(self, digest: str, document: dict, *, source: str,
                              legacy_ledger_sha256: str | None = None,
                              legacy_entry_count: int | None = None,
                              legacy_max_version: int | None = None) -> dict:
        """Append the immutable mission identity for this campaign.

        Existing history is never rewritten. A legacy campaign is anchored with
        explicit ledger hash/count metadata so its migration is auditable.
        """
        if source not in {"new_campaign", "existing_no_ledger", "legacy_migration"}:
            raise ValueError("invalid mission anchor source")
        document_json = json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False)
        actual_digest = hashlib.sha256(document_json.encode("utf-8")).hexdigest()
        if digest != actual_digest:
            raise ValueError("mission digest does not match its canonical document")
        if source == "legacy_migration":
            if (not legacy_ledger_sha256 or legacy_entry_count is None or legacy_entry_count <= 0
                    or legacy_max_version is None):
                raise ValueError("legacy migration requires a ledger hash, entry count, and max version")
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT digest,document_json,source,recorded_at,legacy_ledger_sha256,"
                             "legacy_entry_count,legacy_max_version FROM mission_anchors WHERE id=1").fetchone()
            if row is None:
                db.execute("INSERT INTO mission_anchors VALUES(1,?,?,?,?,?,?,?)", (
                    digest, document_json, source, self._now(), legacy_ledger_sha256,
                    legacy_entry_count, legacy_max_version))
            elif row[0] != digest or row[1] != document_json:
                raise RuntimeError("mission contract differs from the persistent campaign anchor")
        return self.mission_anchor()

    def mission_anchor(self) -> dict | None:
        """Load and integrity-check the persistent campaign mission anchor."""
        uri = f"file:{self.db_path}?mode=ro"
        with sqlite3.connect(uri, uri=True) as db:
            row = db.execute("SELECT digest,document_json,source,recorded_at,legacy_ledger_sha256,"
                             "legacy_entry_count,legacy_max_version FROM mission_anchors WHERE id=1").fetchone()
        if row is None:
            return None
        digest, document_json, source, recorded_at, ledger_hash, entry_count, max_version = row
        actual = hashlib.sha256(document_json.encode("utf-8")).hexdigest()
        if actual != digest:
            raise RuntimeError("persistent mission anchor document hash mismatch")
        return {"sha256": digest, "document": json.loads(document_json), "source": source,
                "recorded_at": recorded_at, "legacy_ledger_sha256": ledger_hash,
                "legacy_entry_count": entry_count, "legacy_max_version": max_version}

    def require_mission_anchor(self, digest: str) -> dict:
        anchor = self.mission_anchor()
        if anchor is None:
            raise RuntimeError("persistent campaign mission anchor is missing")
        if anchor["sha256"] != digest:
            raise RuntimeError("mission contract does not match the persistent campaign anchor")
        return anchor

    def validate_ledger_evidence(self, ledger: list[dict]) -> list[dict]:
        """Cross-check JSON ledger decisions and aggregate metrics against SQLite evidence."""
        expected: dict[int, dict] = {}
        for entry in ledger:
            version = entry.get("version")
            if isinstance(version, bool) or not isinstance(version, int) or version in expected:
                raise RuntimeError("ledger has an invalid or duplicate version number")
            expected[version] = entry
        uri = f"file:{self.db_path}?mode=ro"
        with sqlite3.connect(uri, uri=True) as db:
            rows = db.execute("SELECT version,source,sha256,opt_json,holdout_json,accepted FROM versions").fetchall()
        actual = {int(row[0]): row for row in rows}
        if len(actual) != len(rows) or set(actual) != set(expected):
            raise RuntimeError("JSON ledger versions do not match SQLite version evidence")
        checks = []
        for version, entry in sorted(expected.items()):
            _, source, digest, opt_json, holdout_json, accepted = actual[version]
            try:
                opt = json.loads(opt_json) if opt_json is not None else None
                holdout = json.loads(holdout_json) if holdout_json is not None else None
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"SQLite metrics are invalid JSON for version {version}") from exc
            matches = (source == entry.get("source") and digest == entry.get("sha256")
                       and bool(accepted) == bool(entry.get("accepted"))
                       and opt == entry.get("opt") and holdout == entry.get("holdout"))
            checks.append({"version": version, "matches": matches})
        if not all(row["matches"] for row in checks):
            raise RuntimeError("JSON ledger conflicts with SQLite version evidence")
        return checks

    # ------------------------------------------------------------------ Track 2: per-agent context API
    def read_context(self, agent_id: str, max_tokens: int = 1500) -> str:
        """Read the lean shared context for one agent BEFORE it calls Crusoe. Logged as a read transaction."""
        budget = min(max(int(max_tokens), 0), 1499)
        source = self.playbook_path.read_text(encoding="utf-8")
        text = source[:budget * 4]
        self.log_transaction(agent_id, "read", "playbook.md", tokens=estimate_tokens(text),
                             detail={"budget": max_tokens, "effective_budget": budget,
                                     "source_tokens": estimate_tokens(source),
                                     "returned_tokens": estimate_tokens(text), "truncated": len(text) < len(source)})
        return text

    def atomic_update(self, agent_id: str, delta: dict) -> dict:
        """Apply one agent's delta under the file lock AFTER it calls Crusoe.

        delta = {"section": 1..4, "bullets": [..]} for the playbook (optional when only evolving a prompt),
        "evolved_prompt": {...} appends a versioned prompt to evolved_agent_prompts.json. Lock contention is
        counted in the `conflicts_prevented` metric: every time a second agent had to wait for the lock,
        a merge was needed instead of an overwrite."""
        waited = self._contended()
        if delta.get("section") is not None:
            self.update_playbook(agent_id, int(delta["section"]), list(delta.get("bullets", [])))
        if delta.get("evolved_prompt") is not None:
            self._merge_prompt(agent_id, delta["evolved_prompt"])
        if waited:
            self._bump_conflicts(waited)
        self.log_transaction(agent_id, "write", "agent_operation", 0,
                             {"merged": True, "contended": waited, "delta_keys": sorted(delta)})
        return {"merged": True, "contended": waited}

    def _contended(self) -> int:
        """Non-blocking probe of the playbook lock. 1 means another agent holds it right now, so this update
        must merge with that agent's write instead of overwriting it."""
        lock_path = Path(str(self.playbook_path) + ".lock")  # same file _locked() uses
        lock_path.touch(exist_ok=True)
        with open(lock_path, "a+") as fh:
            try:
                fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return 1
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
        return 0

    def _merge_prompt(self, agent_id: str, prompt: dict) -> dict:
        """Append one versioned prompt for `agent_id`; versions are 1-based per agent."""
        path = self.root / "evolved_agent_prompts.json"
        with self._locked(path):
            data = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
            versions = data.setdefault(agent_id, [])
            entry = dict(prompt)
            entry["version"] = len(versions) + 1
            entry["updated"] = self._now()
            versions.append(entry)
            self._write_locked(path, data)
        self.log_transaction(agent_id, "write", "evolved_agent_prompts.json",
                             estimate_tokens(json.dumps(entry, ensure_ascii=False)),
                             {"agent_prompt_version": entry["version"]})
        return entry

    def _bump_conflicts(self, n: int) -> None:
        self.log_transaction("shared_brain", "write", "conflict_merge", 0,
                             {"conflicts_prevented": 1, "waited": n})

    def conflicts_prevented(self) -> int:
        with self._connect() as db:
            return db.execute("SELECT value FROM counters WHERE name='conflicts_prevented'").fetchone()[0]

    def context_efficiency_gain(self) -> int:
        """Tokens the shared brain saved, measured from the transaction log (never a constant): everything
        deposited into the playbook (writes, including graveyard rejections) is what a no-memory agent would
        have to re-derive per generation, while playbook reads are what agents actually consumed."""
        with self._connect() as db:
            totals = dict(db.execute("SELECT name,value FROM counters"))
        deposited = totals["context_write_tokens"]
        actually_read = totals["context_read_tokens"]
        return max(deposited - actually_read, 0)

    def read_playbook(self, agent: str) -> str:
        text = self.playbook_path.read_text(encoding="utf-8")
        self.log_transaction(agent, "read", "playbook.md", estimate_tokens(text))
        return text

    def update_playbook(self, agent: str, section: int, bullets: list[str]) -> None:
        if section not in range(1, 5):
            raise ValueError("section must be 1..4")
        lock = self._locked(self.playbook_path)
        try:
            current = self.playbook_path.read_text(encoding="utf-8")
            parts = self._parse_playbook(current)
            for bullet in bullets:
                clean = bullet.strip()
                if clean and not clean.startswith("-"):
                    clean = "- " + clean
                if clean and clean not in parts[section - 1]:
                    parts[section - 1].append(clean)
            # Preserve at least one graveyard bullet; evict oldest evidence from 1-3 first.
            while estimate_tokens(self._render(parts)) >= 1500:
                evicted = False
                for index in range(3):
                    if parts[index]:
                        parts[index].pop(0)
                        evicted = True
                        break
                if not evicted and len(parts[3]) > 1:
                    parts[3].pop(0)
                    evicted = True
                if not evicted:
                    raise ValueError("playbook headers and graveyard cannot fit token budget")
            output = self._render(parts)
            fd, name = tempfile.mkstemp(prefix=self.playbook_path.name, dir=self.root)
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    f.write(output)
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(name, self.playbook_path)
            finally:
                if os.path.exists(name):
                    os.unlink(name)
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
            lock.close()
        self.log_transaction(agent, "write", "playbook.md", estimate_tokens(output),
                             {"section": section, "bullets": len(bullets)})

    @staticmethod
    def _parse_playbook(text: str) -> list[list[str]]:
        parts: list[list[str]] = [[] for _ in HEADERS]
        active = -1
        for line in text.splitlines():
            if line in HEADERS:
                active = HEADERS.index(line)
            elif active >= 0 and line.startswith("- "):
                parts[active].append(line)
        return parts

    @staticmethod
    def _render(parts: list[list[str]]) -> str:
        return "\n".join("\n".join((header, *bullets)) for header, bullets in zip(HEADERS, parts)) + "\n"

    def load_ledger(self) -> list:
        with self._locked(self.ledger_path):
            return json.loads(self.ledger_path.read_text(encoding="utf-8"))

    def append_ledger(self, entry: dict) -> None:
        lock = self._locked(self.ledger_path)
        try:
            entries = json.loads(self.ledger_path.read_text(encoding="utf-8"))
            entries.append(entry)
            self._write_locked(self.ledger_path, entries)
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
            lock.close()

    @staticmethod
    def _write_locked(path: Path, value: Any) -> None:
        fd, name = tempfile.mkstemp(prefix=path.name, dir=path.parent)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(value, f, indent=2, ensure_ascii=False)
                f.write("\n")
                f.flush()
                os.fsync(f.fileno())
            os.replace(name, path)
        finally:
            if os.path.exists(name):
                os.unlink(name)

    def best_accepted(self) -> dict | None:
        accepted = [e for e in self.load_ledger() if e.get("accepted")]
        return max(accepted, key=lambda e: e["version"]) if accepted else None

    def next_version(self) -> int:
        entries = self.load_ledger()
        return max((int(e["version"]) for e in entries), default=0) + 1

    def reset(self, keep_playbook_seed: bool = True) -> None:
        with self._connect() as db:
            db.execute("DELETE FROM transactions")
            db.execute("DELETE FROM versions")
            db.execute("DELETE FROM seed_metrics")
            db.execute("DELETE FROM mission_anchors")
            db.execute("UPDATE counters SET value=0")
        self._atomic_json(self.ledger_path, [])
        self._atomic_text(self.playbook_path, PLAYBOOK_SEED if keep_playbook_seed else self._render([[], [], [], []]))

    def stats(self) -> dict:
        with self._connect() as db:
            n_transactions = db.execute("SELECT count(*) FROM transactions").fetchone()[0]
            n_versions = db.execute("SELECT count(*) FROM versions").fetchone()[0]
            counters = dict(db.execute("SELECT name,value FROM counters"))
        return {"n_transactions": n_transactions, "n_versions": n_versions,
                "playbook_tokens": estimate_tokens(self.playbook_path.read_text(encoding="utf-8")), **counters}
