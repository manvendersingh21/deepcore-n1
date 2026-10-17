"""Bounded production context and retention preserve lifetime/evaluation evidence."""
import json
import sqlite3

from shared_brain.memory_store import SharedBrain, estimate_tokens


def test_read_context_caps_external_oversize_and_requested_budget(tmp_path):
    brain = SharedBrain(tmp_path)
    brain.playbook_path.write_text('evidence ' * 2000)
    for requested, expected in [(1500, 1499), (12, 12), (0, 0), (9000, 1499)]:
        returned = brain.read_context('VisionSpecialist', requested)
        assert estimate_tokens(returned) == expected
        with sqlite3.connect(brain.db_path) as db:
            tokens, detail = db.execute("SELECT tokens,detail_json FROM transactions ORDER BY id DESC LIMIT 1").fetchone()
        assert tokens == expected
        assert json.loads(detail)['truncated'] is True
    assert brain.stats()['context_reads'] == 4


def test_retention_preserves_versions_metrics_ledger_and_lifetime_counts(tmp_path, monkeypatch):
    monkeypatch.setattr('shared_brain.memory_store.MAX_TRANSACTIONS', 5)
    brain = SharedBrain(tmp_path)
    entry = {'version': 1, 'source': 'measured', 'sha256': 'a' * 64,
             'opt': {'composite_score': 80}, 'holdout': {'composite_score': 85}, 'accepted': True}
    brain.append_ledger(entry)
    brain.record_version(entry, 'actual diff', [{'seed': 11, 'composite_score': 80}],
                         [{'seed': 101, 'composite_score': 85}])
    brain.read_context('Architect', 10)
    brain._bump_conflicts(1)
    for index in range(12):
        brain.log_transaction('Verifier', 'write', 'ephemeral', index)
    with sqlite3.connect(brain.db_path) as db:
        assert db.execute('SELECT count(*) FROM transactions').fetchone()[0] == 5
        assert db.execute('SELECT count(*) FROM versions').fetchone()[0] == 1
        assert db.execute('SELECT count(*) FROM seed_metrics').fetchone()[0] == 2
    assert brain.load_ledger() == [entry]
    reopened = SharedBrain(tmp_path)
    assert reopened.stats()['transactions_pruned'] == 9
    assert reopened.stats()['context_reads'] == 1
    assert reopened.stats()['context_read_tokens'] == 10
    assert reopened.conflicts_prevented() == 1


def test_empty_delta_still_records_completed_agent_operation(tmp_path):
    brain = SharedBrain(tmp_path)
    assert brain.atomic_update('ControlSpecialist', {}) == {'merged': True, 'contended': 0}
    with sqlite3.connect(brain.db_path) as db:
        agent, target, detail = db.execute('SELECT agent,target,detail_json FROM transactions').fetchone()
    assert agent == 'ControlSpecialist' and target == 'agent_operation'
    assert json.loads(detail)['merged'] is True
