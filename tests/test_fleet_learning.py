"""Offline learner control-flow tests; stub outcomes are not fleet success evidence."""
import json

from agents.fleet_learning import FleetLearner, load_active_parameters, source_hashes


def outcome(parameters, seed):
    return {'seed': seed, 'sensor_based': True, 'debris_collected': 2 if parameters['detect_min_pixels'] < 40 else 1,
            'debris_total': 6, 'peak_force_n': 40., 'collector_peak_force_n': 4., 'pickup_misses': 1,
            'complete': False}


def test_fresh_baseline_and_candidate_promote_only_measured_improvement(tmp_path):
    calls = []
    def evaluate(parameters, seed):
        calls.append((parameters, seed))
        return outcome(parameters, seed)
    learner = FleetLearner(evaluate, tmp_path)
    entry = learner.run_iteration((11,))
    assert len(calls) == 2
    assert entry['accepted'] and entry['decision'] == 'PROMOTE'
    assert entry['source_hashes_before'] == entry['source_hashes_after'] == source_hashes()
    assert len(entry['fresh_run_receipts']) == 2
    assert load_active_parameters(tmp_path)['detect_min_pixels'] == 38
    assert learner.brain.stats()['context_reads'] == 3


def test_privileged_results_are_recorded_as_failure_without_proposal(tmp_path):
    learner = FleetLearner(lambda params, seed: {**outcome(params, seed), 'sensor_based': False}, tmp_path)
    entry = learner.run_iteration((11,))
    assert entry['decision'] == 'ERROR' and not entry['accepted']
    assert entry['failure_phase'] == 'baseline'
    assert 'sensor-based' in entry['error']
    assert json.loads(learner.path.read_text())[0]['decision'] == 'ERROR'


def test_malformed_actual_model_reply_is_persisted_with_baseline(tmp_path):
    class Client:
        available = True
        model = 'offline-test-stub'
        last_usage = {}
        last_error = None
        def chat(self, *args, **kwargs):
            return '[1,2,3]'
    entry = FleetLearner(outcome, tmp_path, client=Client()).run_iteration((11,))
    assert entry['decision'] == 'ERROR' and entry['failure_phase'] == 'proposal'
    assert entry['baseline_fresh']['runs'][0]['seed'] == 11
    assert len(entry['fresh_run_receipts']) == 1


def test_missing_required_llm_is_logged_without_rollout(tmp_path):
    def forbidden(*args):
        raise AssertionError('No physics should run before missing-client failure')
    entry = FleetLearner(forbidden, tmp_path, require_llm=True).run_iteration((11,))
    assert entry['decision'] == 'ERROR' and entry['failure_phase'] == 'client_preflight'
    assert entry['fresh_run_receipts'] == []


def test_source_mismatch_disables_stored_parameters(tmp_path):
    learner = FleetLearner(outcome, tmp_path)
    entry = learner.run_iteration((11,))
    entry['source_hashes_after'] = {'changed': 'invalid'}
    learner.path.write_text(json.dumps([entry]))
    assert load_active_parameters(tmp_path)['detect_min_pixels'] == 40


def test_run_start_rejects_tampered_accepted_parameter_hash(tmp_path):
    learner = FleetLearner(outcome, tmp_path)
    entry = learner.run_iteration((11,))
    entry['parameter_sha256'] = 'tampered'
    learner.path.write_text(json.dumps([entry]))
    second = learner.run_iteration((11,))
    assert second['previous_parameters']['detect_min_pixels'] == 40


def test_missing_required_source_fails_closed(tmp_path, monkeypatch):
    monkeypatch.setattr('agents.fleet_learning.REPO_ROOT', tmp_path)
    import pytest
    with pytest.raises(FileNotFoundError, match='Required fleet source'):
        source_hashes()
