"""Persistent shared memory: playbook, ledger, and SQLite audit trail."""

from __future__ import annotations

import fcntl
import json
import math
import os
import sqlite3
import tempfile
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

TOKEN_BUDGET = 1500


def estimate_tokens(text: str) -> int:
    """INTERFACE token estimate: ceil(characters / 4)."""
    return math.ceil(len(text) / 4)


class SharedBrain:
    """Playbook + ledger on disk, version history and transactions in SQLite."""

    def __init__(self, root: str | Path = "shared_brain") -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.db_path = self.root / "brain.db"
        self.ledger_path = self.root / "ledger.json"
        self.playbook_path = self.root / "playbook.md"
        self._ensure_schema()
        if not self.ledger_path.exists():
            self._atomic_write_json(self.ledger_path, [])
        if not self.playbook_path.exists():
            self._atomic_write_text(self.playbook_path, PLAYBOOK_SEED)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30.0, isolation_level=None)
        conn.execute("PRAGMA busy_timeout=30000")
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _ensure_schema(self) -> None:
        with self._connect() as db:
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS transactions (
                    id INTEGER PRIMARY KEY,
                    ts TEXT NOT NULL,
                    agent TEXT NOT NULL,
                    op TEXT NOT NULL CHECK(op IN ('read', 'write')),
                    target TEXT NOT NULL,
                    tokens INTEGER NOT NULL,
                    detail_json TEXT
                );
                CREATE TABLE IF NOT EXISTS versions (
                    version INTEGER PRIMARY KEY,
                    ts TEXT NOT NULL,
                    source TEXT NOT NULL,
                    sha256 TEXT NOT NULL,
                    diff_text TEXT NOT NULL,
                    opt_json TEXT,
                    holdout_json TEXT,
                    accepted INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS seed_metrics (
                    version INTEGER NOT NULL,
                    seed INTEGER NOT NULL,
                    split TEXT NOT NULL CHECK(split IN ('opt', 'holdout')),
                    metrics_json TEXT NOT NULL,
                    PRIMARY KEY (version, seed, split)
                );
                """
            )

    @staticmethod
    def _utc_now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def _lock_path(self, path: Path):
        lock_file = open(f"{path}.lock", "a+", encoding="utf-8")
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        return lock_file

    def _atomic_write_text(self, path: Path, text: str) -> None:
        lock = self._lock_path(path)
        try:
            fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    handle.write(text)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(tmp, path)
            finally:
                if os.path.exists(tmp):
                    os.unlink(tmp)
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
            lock.close()

    def _atomic_write_json(self, path: Path, payload: Any) -> None:
        body = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
        self._atomic_write_text(path, body)

    def log_transaction(
        self,
        agent: str,
        op: str,
        target: str,
        tokens: int = 0,
        detail: dict | None = None,
    ) -> None:
        if op not in ("read", "write"):
            raise ValueError("op must be 'read' or 'write'")
        detail_json = json.dumps(detail, ensure_ascii=False) if detail is not None else None
        with self._connect() as db:
            db.execute(
                "INSERT INTO transactions(ts, agent, op, target, tokens, detail_json) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (self._utc_now(), agent, op, target, tokens, detail_json),
            )

    def record_version(
        self,
        entry: dict,
        diff_text: str,
        per_seed_opt: list,
        per_seed_holdout: list,
    ) -> None:
        version = int(entry["version"])
        ts = entry.get("timestamp", self._utc_now())
        with self._connect() as db:
            db.execute(
                "INSERT OR REPLACE INTO versions "
                "(version, ts, source, sha256, diff_text, opt_json, holdout_json, accepted) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    version,
                    ts,
                    entry["source"],
                    entry["sha256"],
                    diff_text,
                    json.dumps(entry.get("opt")),
                    json.dumps(entry.get("holdout")),
                    int(bool(entry["accepted"])),
                ),
            )
            for split, rows in (("opt", per_seed_opt), ("holdout", per_seed_holdout)):
                for row in rows:
                    db.execute(
                        "INSERT OR REPLACE INTO seed_metrics(version, seed, split, metrics_json) "
                        "VALUES (?, ?, ?, ?)",
                        (version, int(row["seed"]), split, json.dumps(row, ensure_ascii=False)),
                    )

    def read_playbook(self, agent: str) -> str:
        text = self.playbook_path.read_text(encoding="utf-8")
        self.log_transaction(agent, "read", "playbook.md", estimate_tokens(text))
        return text

    @staticmethod
    def _parse_playbook(text: str) -> list[list[str]]:
        sections: list[list[str]] = [[] for _ in HEADERS]
        active = -1
        for line in text.splitlines():
            if line in HEADERS:
                active = HEADERS.index(line)
            elif active >= 0 and line.startswith("- "):
                sections[active].append(line)
        return sections

    @staticmethod
    def _render_playbook(sections: list[list[str]]) -> str:
        blocks = [header + "\n" + "\n".join(bullets) for header, bullets in zip(HEADERS, sections)]
        return "\n".join(blocks) + "\n"

    def _enforce_budget(self, sections: list[list[str]]) -> None:
        while estimate_tokens(self._render_playbook(sections)) >= TOKEN_BUDGET:
            if self._evict_oldest(sections):
                continue
            raise ValueError("playbook headers and graveyard cannot fit token budget")

    @staticmethod
    def _evict_oldest(sections: list[list[str]]) -> bool:
        for idx in range(3):
            if sections[idx]:
                sections[idx].pop(0)
                return True
        if len(sections[3]) > 1:
            sections[3].pop(0)
            return True
        return False

    def update_playbook(self, agent: str, section: int, bullets: list[str]) -> None:
        if section not in (1, 2, 3, 4):
            raise ValueError("section must be 1..4")
        lock = self._lock_path(self.playbook_path)
        try:
            sections = self._parse_playbook(self.playbook_path.read_text(encoding="utf-8"))
            bucket = sections[section - 1]
            for raw in bullets:
                line = raw.strip()
                if not line:
                    continue
                if not line.startswith("-"):
                    line = f"- {line}"
                if line not in bucket:
                    bucket.append(line)
            self._enforce_budget(sections)
            output = self._render_playbook(sections)
            fd, tmp = tempfile.mkstemp(prefix=f".{self.playbook_path.name}.", dir=self.root)
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    handle.write(output)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(tmp, self.playbook_path)
            finally:
                if os.path.exists(tmp):
                    os.unlink(tmp)
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
            lock.close()
        self.log_transaction(
            agent,
            "write",
            "playbook.md",
            estimate_tokens(output),
            {"section": section, "bullets": len(bullets)},
        )

    def load_ledger(self) -> list:
        lock = self._lock_path(self.ledger_path)
        try:
            return json.loads(self.ledger_path.read_text(encoding="utf-8"))
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
            lock.close()

    def append_ledger(self, entry: dict) -> None:
        lock = self._lock_path(self.ledger_path)
        try:
            ledger = json.loads(self.ledger_path.read_text(encoding="utf-8"))
            ledger.append(entry)
            fd, tmp = tempfile.mkstemp(prefix=f".{self.ledger_path.name}.", dir=self.ledger_path.parent)
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    json.dump(ledger, handle, indent=2, ensure_ascii=False)
                    handle.write("\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(tmp, self.ledger_path)
            finally:
                if os.path.exists(tmp):
                    os.unlink(tmp)
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
            lock.close()

    def best_accepted(self) -> dict | None:
        accepted = [row for row in self.load_ledger() if row.get("accepted")]
        if not accepted:
            return None
        return max(accepted, key=lambda row: int(row["version"]))

    def next_version(self) -> int:
        ledger = self.load_ledger()
        if not ledger:
            return 1
        return max(int(row["version"]) for row in ledger) + 1

    def reset(self, keep_playbook_seed: bool = True) -> None:
        with self._connect() as db:
            db.execute("DELETE FROM transactions")
            db.execute("DELETE FROM versions")
            db.execute("DELETE FROM seed_metrics")
        self._atomic_write_json(self.ledger_path, [])
        if keep_playbook_seed:
            self._atomic_write_text(self.playbook_path, PLAYBOOK_SEED)
        else:
            self._atomic_write_text(self.playbook_path, self._render_playbook([[], [], [], []]))

    def stats(self) -> dict:
        with self._connect() as db:
            n_transactions = int(db.execute("SELECT COUNT(*) FROM transactions").fetchone()[0])
            n_versions = int(db.execute("SELECT COUNT(*) FROM versions").fetchone()[0])
        playbook = self.playbook_path.read_text(encoding="utf-8")
        return {
            "n_transactions": n_transactions,
            "n_versions": n_versions,
            "playbook_tokens": estimate_tokens(playbook),
        }
