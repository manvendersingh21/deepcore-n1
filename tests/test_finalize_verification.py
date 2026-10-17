from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from scripts import finalize_verification as finalize
from shared_brain.memory_store import SharedBrain


def test_atomic_report_replaces_stale_pass(tmp_path):
    path = tmp_path / 'final.json'
    path.write_text('{"passed": true, "decision": "STOP_DONE"}')
    finalize.atomic_json_write(path, {'passed': False, 'decision': 'PING_HUMAN'})
    assert json.loads(path.read_text()) == {'passed': False, 'decision': 'PING_HUMAN'}
    assert not list(tmp_path.glob('*.tmp')) and not list(tmp_path.glob('.*.tmp'))


def test_private_evaluation_copy_is_read_only_and_temporary(tmp_path):
    source = tmp_path / 'accepted.py'
    source.write_text('pipeline source')
    original = source.read_bytes()
    with finalize.private_source_copy(source, tmp_path) as copy:
        assert copy.read_bytes() == original
        assert os.stat(copy).st_mode & 0o222 == 0
        assert copy != source
        copy_path = copy
    assert source.read_bytes() == original
    assert not copy_path.exists()


def test_failed_finalization_overwrites_previous_passing_report(tmp_path, monkeypatch):
    from core import integrity

    monkeypatch.setattr(finalize, 'ROOT', tmp_path)

    def fail_integrity():
        raise RuntimeError('injected integrity failure')

    monkeypatch.setattr(integrity, 'verify_integrity', fail_integrity)
    output = tmp_path / 'dev/final_verification.json'
    output.parent.mkdir(parents=True)
    output.write_text('{"passed": true, "decision": "STOP_DONE"}')
    assert finalize.run(output) == 1
    report = json.loads(output.read_text())
    assert report['passed'] is False
    assert report['decision'] == 'PING_HUMAN'
    assert report['status'] == 'error'
    assert 'injected integrity failure' in report['error']


def test_receipt_collection_failure_does_not_suppress_failure_report(tmp_path, monkeypatch):
    from core import integrity

    monkeypatch.setattr(finalize, 'ROOT', tmp_path)
    monkeypatch.setattr(integrity, 'verify_integrity', lambda: {})

    def fail_stats(self):
        raise OSError('injected stats read failure')

    monkeypatch.setattr(SharedBrain, 'stats', fail_stats)
    output = tmp_path / 'dev/final_verification.json'
    assert finalize.run(output) == 1
    report = json.loads(output.read_text())
    assert report['passed'] is False and report['status'] == 'error'
    assert any('context stats unavailable' in warning for warning in report['finalization_warnings'])
    assert 'No accepted benchmark version exists' in report['error']
