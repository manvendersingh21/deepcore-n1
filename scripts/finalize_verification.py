"""Rerun the accepted RGB-D benchmark and report evidence without promoting code.

Optional real agent reviews produce unverified drafts only. Holdout diagnostics
are disabled; report output contains only whitelisted measured metrics.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import tempfile

ROOT = Path(__file__).resolve().parents[1]
METRICS = ('seed', 'vision_3d_rmse_mm', 'rubble_cleared_count', 'peak_core_contact_force_n',
           'composite_score', 'core_shattered', 'core_extracted', 'extraction_rate',
           'max_wrist_force_n', 'vision_latency_p99_ms', 'force_ok_25', 'passed', 'score_breakdown')


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json_write(path, value):
    """Replace a JSON receipt atomically, leaving no stale passing report on failure."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, allow_nan=False) + "\n"
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


@contextmanager
def private_source_copy(source, directory=ROOT / "dev"):
    """Yield a private read-only source copy for evaluation."""
    source = Path(source)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="accepted-source-", dir=directory) as temp_dir:
        snapshot = Path(temp_dir) / source.name
        shutil.copyfile(source, snapshot)
        os.chmod(snapshot, 0o444)
        if sha256(snapshot) != sha256(source):
            raise RuntimeError("private accepted-source copy does not match its source")
        yield snapshot


def public_metrics(value):
    return {key: value[key] for key in METRICS if key in value}


def validate_ledger(ledger):
    checks = []
    for entry in ledger:
        path = Path(entry['path'])
        path = path if path.is_absolute() else ROOT / path
        actual = sha256(path)
        checks.append({'version': entry['version'], 'path': str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
                       'expected_sha256': entry['sha256'], 'actual_sha256': actual,
                       'matches': actual == entry['sha256']})
    if not all(row['matches'] for row in checks):
        raise RuntimeError('Recorded source hash mismatch in ledger')
    return checks


def review_agents(orch, best, source_path, directory):
    """Call the actual architect and specialists; save proposals without evaluation/promotion."""
    from agents.contract_orchestrator import CV, CONTROL, PROMPT_AGENTS, atomic_write
    if not orch.client.available:
        raise RuntimeError('--review-agents requires an available configured Crusoe client')
    directory.mkdir(parents=True, exist_ok=True)
    orch.generation = max((int(entry.get('generation', 0)) for entry in orch.brain.load_ledger()), default=0) + 1
    contract = orch.architect(orch.generation, best, '')
    source = source_path.read_text()
    proposals = []
    for role in (CV, CONTROL):
        with orch.agent_operation(role, 'final_unverified_review') as context:
            code = orch.synth.try_edits(*orch._specialist_prompt(role, contract, source, best, context),
                                       source=source, agent=role)
            canonical = next(key for key, display in PROMPT_AGENTS.items() if display == role)
            proposal = {'agent': canonical, 'validated_static_only': code is not None,
                        'problems': list(orch.synth.last_problems), 'promoted': False}
            if code is not None:
                draft = directory / f'{canonical}_unverified.py'
                atomic_write(draft, code)
                proposal.update(path=str(draft), sha256=sha256(draft))
                source = code
            proposals.append(proposal)
    with orch.agent_operation('Independent Verifier', 'unverified_draft_receipt'):
        pass  # Evidence receipt only; drafts never enter acceptance or physics evaluation.
    summary = {'status': 'unverified agent review; not a promotion', 'contract': contract, 'proposals': proposals}
    (directory / 'review.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    return summary


def run(out=ROOT / 'dev/final_verification.json', review=False):
    from agents.contract_orchestrator import ContractOrchestrator, _check_assertion
    from agents.mission_contract import FrozenMission
    from core.integrity import verify_integrity
    from core.verifier_harness import evaluate, check_contract, OPTIMIZATION_SEEDS, HOLDOUT_SEEDS
    from shared_brain.memory_store import SharedBrain

    out = Path(out)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    report = {'created_at': datetime.now(timezone.utc).isoformat(), 'operation': 'accepted_benchmark_rerun',
              'decision': 'PING_HUMAN', 'passed': False, 'promotion_performed': False,
              'status': 'running',
              'scope': 'Locked sensor-only RGB-D excavation benchmark; separate fleet is privileged scripted simulation.'}
    # Persist a non-passing marker before any work. A crash or later receipt error
    # can no longer leave an older STOP_DONE report on disk.
    atomic_json_write(out, report)
    brain = None
    ledger_bytes = None
    try:
        report['core_hashes'] = verify_integrity()
        brain = SharedBrain(ROOT / 'shared_brain')
        ledger_bytes = brain.ledger_path.read_bytes()
        ledger = brain.load_ledger()
        report['ledger_source_hashes'] = validate_ledger(ledger)
        report['ledger_db_evidence'] = brain.validate_ledger_evidence(ledger)
        accepted = [entry for entry in ledger if entry.get('accepted')]
        if not accepted:
            raise RuntimeError('No accepted benchmark version exists')
        best = max(accepted, key=lambda entry: int(entry['version']))
        source_path = ROOT / best['path']
        source_hash = sha256(source_path)
        prior = [entry for entry in accepted if int(entry['version']) < int(best['version'])
                 and entry['sha256'] != source_hash]
        previous = max(prior, key=lambda entry: int(entry['version']), default=None)
        report['accepted_version'] = best['version']
        report['accepted_source_sha256'] = source_hash
        report['accepted_source_path'] = best['path']
        report['historical_accepted_metrics'] = {split: public_metrics(best.get(split, {})) for split in ('opt', 'holdout')}
        report['previous_accepted_reference'] = ({'version': previous['version'], 'sha256': previous['sha256'],
                                                'opt_composite': previous['opt']['composite_score'],
                                                'holdout_composite': previous['holdout']['composite_score']} if previous else None)
        mission_path = ROOT / 'shared_brain/generated_dod.json'
        if not mission_path.is_file():
            raise RuntimeError('generated_dod.json is missing for the accepted campaign')
        mission = FrozenMission(json.loads(mission_path.read_text()), mission_path)
        with brain.campaign_lock():
            anchor = brain.mission_anchor()
            if anchor is None:
                ledger_versions = [int(entry['version']) for entry in ledger]
                migration_source = 'legacy_migration' if ledger else 'existing_no_ledger'
                brain.record_mission_anchor(
                    mission.digest, mission.snapshot(), source=migration_source,
                    legacy_ledger_sha256=hashlib.sha256(ledger_bytes).hexdigest() if ledger else None,
                    legacy_entry_count=len(ledger) if ledger else None,
                    legacy_max_version=max(ledger_versions) if ledger_versions else None,
                )
            anchor = brain.require_mission_anchor(mission.digest)
        report['mission_anchor'] = {key: anchor[key] for key in (
            'sha256', 'source', 'recorded_at', 'legacy_ledger_sha256', 'legacy_entry_count', 'legacy_max_version')}
        report['mission'] = {'sha256': mission.digest, 'document': mission.snapshot()}
        orch = ContractOrchestrator(brain)
        orch.dod = mission.snapshot()
        if review:
            stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
            report['agent_review'] = review_agents(orch, best, source_path, ROOT / 'dev/agent_review_drafts' / stamp)
        if sha256(source_path) != source_hash or brain.ledger_path.read_bytes() != ledger_bytes:
            raise RuntimeError('Accepted source or ledger changed during agent review')
        mission.assert_unchanged()
        brain.require_mission_anchor(mission.digest)
        with private_source_copy(source_path, out.parent) as evaluation_source:
            report['evaluation_source_sha256'] = sha256(evaluation_source)
            opt = evaluate(evaluation_source, OPTIMIZATION_SEEDS, diagnostics=True)
            mission.assert_unchanged()
            brain.require_mission_anchor(mission.digest)
            hold = evaluate(evaluation_source, HOLDOUT_SEEDS, diagnostics=False)
        if 'autopsy' in hold:
            raise RuntimeError('Holdout evaluator unexpectedly exposed diagnostics')
        report['fresh_evaluation'] = {
            'opt': {'seeds': list(OPTIMIZATION_SEEDS), 'aggregate': public_metrics(opt['aggregate']),
                    'per_seed': [public_metrics(row) for row in opt['per_seed']]},
            'holdout': {'seeds': list(HOLDOUT_SEEDS), 'diagnostics': False,
                        'aggregate': public_metrics(hold['aggregate']),
                        'per_seed': [public_metrics(row) for row in hold['per_seed']]},
        }
        with orch.agent_operation('Independent Verifier', 'final_fixed_contract_check'):
            ok, reasons = check_contract(opt['aggregate'], hold['aggregate'],
                                         previous['opt']['composite_score'] if previous else None,
                                         previous['holdout']['composite_score'] if previous else None)
            mission.assert_unchanged()
            clauses = [{**clause, 'passed': _check_assertion(clause, opt['aggregate'], hold['aggregate'])}
                       for clause in mission.snapshot()['assertions']]
        report['law5'] = {'passed': ok, 'reasons': reasons}
        report['dod_assertions'] = clauses
        verify_integrity()
        validate_ledger(brain.load_ledger())
        brain.validate_ledger_evidence(brain.load_ledger())
        brain.require_mission_anchor(mission.digest)
        if sha256(source_path) != source_hash or brain.ledger_path.read_bytes() != ledger_bytes:
            raise RuntimeError('Accepted source or ledger changed during verification')
        report['evidence_unchanged'] = True
        passed = bool(ok and clauses and all(clause['passed'] for clause in clauses)
                      and opt['aggregate'].get('passed') and hold['aggregate'].get('passed'))
        report.update(passed=passed, decision='STOP_DONE' if passed else 'REPLAN', status='complete')
    except Exception as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'
        report['status'] = 'error'
    finally:
        if brain is not None:
            try:
                report['context_stats'] = brain.stats()
            except Exception as exc:
                report.setdefault('finalization_warnings', []).append(f'context stats unavailable: {type(exc).__name__}: {exc}')
            try:
                prompt_path = brain.root / 'evolved_agent_prompts.json'
                prompts = json.loads(prompt_path.read_text()) if prompt_path.exists() else {}
                report['prompt_versions'] = {agent: [entry['version'] for entry in versions]
                                             for agent, versions in prompts.items()}
            except Exception as exc:
                report.setdefault('finalization_warnings', []).append(f'prompt receipt unavailable: {type(exc).__name__}: {exc}')
            try:
                with sqlite3.connect(f'file:{brain.db_path}?mode=ro', uri=True) as db:
                    rows = db.execute("SELECT agent,target,detail_json FROM transactions WHERE target IN ('evolved_agent_prompts.json','operation_receipt') ORDER BY id DESC LIMIT 40").fetchall()
                report['recent_agent_receipts'] = [
                    {'agent': agent, 'target': target, 'detail': json.loads(detail) if detail else None}
                    for agent, target, detail in reversed(rows)]
            except Exception as exc:
                report.setdefault('finalization_warnings', []).append(f'agent receipts unavailable: {type(exc).__name__}: {exc}')
        if report.get('status') == 'running':
            report.update(status='error', passed=False, decision='PING_HUMAN',
                          error='finalization exited without a terminal outcome')
        atomic_json_write(out, report)
    print(json.dumps({'report': str(out), 'decision': report['decision'], 'passed': report['passed'], 'error': report.get('error')}))
    return 0 if report['passed'] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default='dev/final_verification.json')
    parser.add_argument('--review-agents', action='store_true', help='request real agent drafts before verifying accepted source; never promote drafts')
    args = parser.parse_args()
    return run(args.out, args.review_agents)


if __name__ == '__main__':
    raise SystemExit(main())
