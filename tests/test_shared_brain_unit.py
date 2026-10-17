"""Unit tests for shared_brain.memory_store.SharedBrain (isolated tmp_path roots)."""
from __future__ import annotations

import json
import multiprocessing as mp
import sqlite3
from pathlib import Path

import pytest

from shared_brain.memory_store import HEADERS, PLAYBOOK_SEED, SharedBrain, estimate_tokens

TOKEN_BUDGET = 1500


@pytest.fixture
def brain(tmp_path: Path) -> SharedBrain:
    return SharedBrain(root=tmp_path)


def _entry(version: int, score: float, accepted: bool, source: str = "baseline") -> dict:
    agg = {"composite_score": score, "vision_3d_rmse_mm": 4.0, "rubble_cleared_count": 1.0,
           "peak_core_contact_force_n": 5.0, "core_shattered": False, "core_extracted": False,
           "extraction_rate": 0.0, "vision_latency_p99_ms": 3.0, "fps": 200.0}
    return {"version": version, "source": source, "contract_id": "c0", "opt": agg, "holdout": agg,
            "accepted": accepted, "reasons": [], "timestamp": "2026-10-17T00:00:00+00:00",
            "sha256": "0" * 64, "path": f"workspace/versions/v{version}.py"}


def _section_bullets(text: str) -> list[list[str]]:
    out: list[list[str]] = []
    for line in text.splitlines():
        if line in HEADERS:
            out.append([])
        elif out and line.startswith("- "):
            out[-1].append(line[2:])
    return out


def _rows(brain: SharedBrain, sql: str) -> list[tuple]:
    with sqlite3.connect(brain.db_path) as con:
        return con.execute(sql).fetchall()


def test_init_seeds_files(brain: SharedBrain) -> None:
    assert brain.load_ledger() == []
    text = brain.playbook_path.read_text()
    assert text == PLAYBOOK_SEED
    assert [h for h in HEADERS if h in text] == list(HEADERS)
    assert "p_world = cam_pos + cam_rot @ [(u-cx)/fx*d, (v-cy)/fy*d, d]" in text
    assert all(len(s) == 1 for s in _section_bullets(text))
    assert estimate_tokens(text) < TOKEN_BUDGET


def test_update_playbook_dedup_and_section(brain: SharedBrain) -> None:
    brain.update_playbook("agent0", 2, ["Median-filter depth 5x5", "Reject d<0.4", "Median-filter depth 5x5"])
    brain.update_playbook("agent0", 2, ["Reject d<0.4"])
    secs = _section_bullets(brain.playbook_path.read_text())
    assert secs[1][1:] == ["Median-filter depth 5x5", "Reject d<0.4"]
    assert [len(s) for s in (secs[0], secs[2], secs[3])] == [1, 1, 1]
    with pytest.raises(ValueError):
        brain.update_playbook("agent0", 5, ["x"])


def test_budget_enforced_under_heavy_appends(brain: SharedBrain) -> None:
    for k in range(300):
        brain.update_playbook(f"agent{k % 3}", 1 + k % 4, [f"hypothesis {k}: " + "lorem ipsum dolor " * (1 + k % 6)])
        assert estimate_tokens(brain.playbook_path.read_text()) < TOKEN_BUDGET
    text = brain.playbook_path.read_text()
    assert [h for h in HEADERS if h in text] == list(HEADERS)
    secs = _section_bullets(text)
    assert len(secs[3]) >= 1                                       # graveyard never emptied
    assert any("hypothesis 299" in b for s in secs for b in s)     # newest survives
    assert not any("hypothesis 0:" in b for s in secs for b in s)  # oldest evicted


def test_graveyard_evicted_only_after_sections_1_to_3(brain: SharedBrain) -> None:
    for k in range(120):
        brain.update_playbook("a", 4, [f"rejected idea {k} " + "x" * 60])
    secs = _section_bullets(brain.playbook_path.read_text())
    assert secs[0] == secs[1] == secs[2] == []
    assert len(secs[3]) >= 1 and secs[3][-1].startswith("rejected idea 119")
    assert estimate_tokens(brain.playbook_path.read_text()) < TOKEN_BUDGET


def test_ledger_append_best_next_version(brain: SharedBrain) -> None:
    assert brain.best_accepted() is None and brain.next_version() == 1
    brain.append_ledger(_entry(1, 0.30, True))
    brain.append_ledger(_entry(2, 0.90, False))
    assert brain.best_accepted()["version"] == 1 and brain.next_version() == 3
    brain.append_ledger(_entry(3, 0.55, True))
    assert brain.best_accepted()["version"] == 3 and brain.next_version() == 4
    ledger = json.loads(brain.ledger_path.read_text())
    assert [e["version"] for e in ledger] == [1, 2, 3] and ledger == brain.load_ledger()


def _worker(root: str, wid: int) -> None:
    b = SharedBrain(root=root)
    for k in range(25):
        b.append_ledger(_entry(wid * 25 + k + 1, wid + k / 100, k % 2 == 0, source=f"llm:w{wid}"))
        b.log_transaction(f"w{wid}", "write", "ledger.json", 1, {"k": k})


def _version_worker(root: str, count: int) -> None:
    b = SharedBrain(root=root)
    for _ in range(count):
        with b.campaign_lock():
            version = b.next_version()
            entry = _entry(version, float(version), version % 2 == 1, source="serialized")
            b.append_ledger(entry)
            b.record_version(entry, "", [], [])


def test_concurrent_ledger_appends(tmp_path: Path) -> None:
    SharedBrain(root=tmp_path)
    ctx = mp.get_context("spawn")
    procs = [ctx.Process(target=_worker, args=(str(tmp_path), w)) for w in range(4)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(120)
        assert p.exitcode == 0
    ledger = json.loads((tmp_path / "ledger.json").read_text())
    assert len(ledger) == 100
    assert sorted(e["version"] for e in ledger) == list(range(1, 101))
    assert SharedBrain(root=tmp_path).stats()["n_transactions"] == 100


def test_campaign_lock_serializes_version_allocation_and_evidence(tmp_path: Path) -> None:
    SharedBrain(root=tmp_path)
    ctx = mp.get_context("spawn")
    procs = [ctx.Process(target=_version_worker, args=(str(tmp_path), 20)) for _ in range(4)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(120)
        assert p.exitcode == 0
    brain = SharedBrain(root=tmp_path)
    ledger = brain.load_ledger()
    assert sorted(entry["version"] for entry in ledger) == list(range(1, 81))
    assert len(brain.validate_ledger_evidence(ledger)) == 80


def test_transaction_logging_and_stats(brain: SharedBrain) -> None:
    text = brain.read_playbook("agentA")
    brain.log_transaction("agentB", "write", "workspace/candidate_pipeline.py", 321, {"lines": 12})
    with pytest.raises(ValueError):
        brain.log_transaction("agentB", "delete", "x")
    rows = _rows(brain, "SELECT agent, op, target, tokens, detail_json FROM transactions ORDER BY id")
    assert rows[0] == ("agentA", "read", "playbook.md", estimate_tokens(text), None)
    assert rows[1][:4] == ("agentB", "write", "workspace/candidate_pipeline.py", 321)
    assert json.loads(rows[1][4]) == {"lines": 12}
    assert _rows(brain, "PRAGMA journal_mode")[0][0] == "wal"
    assert brain.stats() == {"n_transactions": 2, "n_versions": 0, "playbook_tokens": estimate_tokens(text),
                             "context_reads": 1, "context_read_tokens": estimate_tokens(text),
                             "context_write_tokens": 0, "conflicts_prevented": 0, "transactions_pruned": 0}


def test_record_version_and_reset(brain: SharedBrain) -> None:
    e = _entry(1, 0.4, True)
    brain.append_ledger(e)
    brain.record_version(e, "--- a\n+++ b\n", [{"seed": s, "composite_score": 0.4} for s in (11, 22, 33)],
                         [{"seed": 101, "composite_score": 0.3}])
    assert brain.stats()["n_versions"] == 1
    assert sorted(_rows(brain, "SELECT split, seed FROM seed_metrics")) == [
        ("holdout", 101), ("opt", 11), ("opt", 22), ("opt", 33)]
    brain.update_playbook("a", 2, ["something new"])
    brain.reset()
    assert brain.load_ledger() == [] and brain.next_version() == 1
    assert brain.playbook_path.read_text() == PLAYBOOK_SEED
    assert brain.stats() == {"n_transactions": 0, "n_versions": 0, "playbook_tokens": estimate_tokens(PLAYBOOK_SEED),
                             "context_reads": 0, "context_read_tokens": 0, "context_write_tokens": 0,
                             "conflicts_prevented": 0, "transactions_pruned": 0}


def test_mission_anchor_is_immutable_and_records_legacy_migration(brain: SharedBrain):
    import hashlib
    from agents.mission_contract import FrozenMission, normalize_goal_assertions

    document = {"goal": "legacy campaign", "source": "fallback",
                "assertions": normalize_goal_assertions([])}
    mission = FrozenMission(document)
    brain.append_ledger(_entry(4, 0.4, True))
    before = brain.ledger_path.read_bytes()
    anchor = brain.record_mission_anchor(
        mission.digest, mission.snapshot(), source="legacy_migration",
        legacy_ledger_sha256=hashlib.sha256(before).hexdigest(),
        legacy_entry_count=1, legacy_max_version=4)
    assert anchor["source"] == "legacy_migration"
    assert anchor["legacy_ledger_sha256"] == hashlib.sha256(before).hexdigest()
    assert anchor["legacy_entry_count"] == 1 and anchor["legacy_max_version"] == 4
    assert brain.ledger_path.read_bytes() == before
    assert brain.require_mission_anchor(mission.digest) == anchor
    changed = dict(document, goal="replacement")
    with pytest.raises(RuntimeError, match="persistent campaign anchor"):
        brain.record_mission_anchor(FrozenMission(changed).digest, changed, source="new_campaign")


def test_ledger_evidence_must_match_sqlite(brain: SharedBrain):
    entry = _entry(1, 0.4, True)
    brain.append_ledger(entry)
    brain.record_version(entry, "diff", [], [])
    assert brain.validate_ledger_evidence([entry]) == [{"version": 1, "matches": True}]
    altered = {**entry, "accepted": False}
    with pytest.raises(RuntimeError, match="conflicts with SQLite"):
        brain.validate_ledger_evidence([altered])
    with pytest.raises(RuntimeError, match="do not match SQLite"):
        brain.validate_ledger_evidence([])


def test_record_version_collision_fails_loudly(brain: SharedBrain):
    entry = _entry(1, 0.4, True)
    brain.record_version(entry, "diff", [], [])
    with pytest.raises(sqlite3.IntegrityError):
        brain.record_version(entry, "replacement", [], [])
