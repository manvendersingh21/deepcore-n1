"""Measured, separately gated fleet parameter learning; never touches core ledger.

The evaluator must run a fresh rollout for each (parameters, seed) and return
sensor-provenance plus measured delivery/contact outcomes. No scores are loaded
from the excavation benchmark or inferred from prerecorded animations.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
from pathlib import Path
import tempfile
import os

REPO_ROOT = Path(__file__).resolve().parents[1]


def source_hashes():
    files = ['ui/mining_fleet.py', 'ui/fleet_perception.py', 'ui/industrial_arm.py', 'ui/rail_carts.py', 'ui/drilling_scene.py', 'agents/fleet_learning.py', 'scripts/learn_fleet.py']
    missing = [name for name in files if not (REPO_ROOT / name).is_file()]
    if missing:
        raise FileNotFoundError(f'Required fleet source files missing: {missing}')
    return {name: hashlib.sha256((REPO_ROOT / name).read_bytes()).hexdigest() for name in files}


DEFAULTS = {'detect_min_pixels': 40, 'detection_height_min_m': .012, 'track_memory_s': .8,
            'grasp_dwell_s': .5}
BOUNDS = {'detect_min_pixels': (20, 100), 'detection_height_min_m': (.006, .025),
          'track_memory_s': (.3, 2.0), 'grasp_dwell_s': (.3, 1.5)}


def validate_parameters(parameters):
    if not isinstance(parameters, dict) or set(parameters) - set(BOUNDS):
        raise ValueError('Unknown fleet parameter; only bounded declared knobs are allowed')
    out = dict(DEFAULTS)
    for name, value in parameters.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError(f'Nonfinite/non-numeric parameter {name}')
        low, high = BOUNDS[name]
        if not low <= value <= high or (name == 'detect_min_pixels' and int(value) != value):
            raise ValueError(f'Parameter {name} outside bounds {low}..{high}')
        out[name] = int(value) if name == 'detect_min_pixels' else float(value)
    return out


def _hash(parameters):
    return hashlib.sha256(json.dumps(parameters, sort_keys=True).encode()).hexdigest()


def measured_summary(runs):
    """Compute an explicit fleet-only objective from real evaluator outcomes."""
    cleaned = []
    for result in runs:
        if result.get('sensor_based') is not True:
            raise ValueError('Evaluator did not attest actual sensor-based policy wiring')
        fields = ('debris_collected', 'debris_total', 'peak_force_n', 'collector_peak_force_n', 'pickup_misses')
        if any(isinstance(result.get(key), bool) or not isinstance(result.get(key), (int, float)) or not math.isfinite(result[key]) for key in fields):
            raise ValueError('Missing/nonfinite measured fleet outcome')
        total, delivered = result['debris_total'], result['debris_collected']
        misses = result['pickup_misses']
        if total <= 0 or not 0 <= delivered <= total or not isinstance(misses, (int, float)) or misses < 0:
            raise ValueError('Invalid measured fleet counts')
        if result.get('errors'):
            raise ValueError('Evaluator reported runtime errors')
        cleaned.append({'seed': result['seed'], 'debris_collected': delivered, 'debris_total': total,
                        'delivery_fraction': delivered / total, 'peak_force_n': result['peak_force_n'],
                        'collector_peak_force_n': result['collector_peak_force_n'], 'pickup_misses': misses,
                        'sensor_based': True, 'complete': bool(result.get('complete', False))})
    if not cleaned:
        raise ValueError('No measured fleet runs')
    mean_fraction = sum(row['delivery_fraction'] for row in cleaned) / len(cleaned)
    peak = max(row['peak_force_n'] for row in cleaned)
    misses = sum(row['pickup_misses'] for row in cleaned) / len(cleaned)
    # Fleet objective is declared here and has no relationship to core composite.
    score = 100 * mean_fraction - .05 * peak - 2 * misses
    return {'score': score, 'mean_delivery_fraction': mean_fraction, 'peak_force_n': peak,
            'collector_peak_force_n': max(row['collector_peak_force_n'] for row in cleaned),
            'mean_pickup_misses': misses, 'runs': cleaned}


class FleetLearner:
    def __init__(self, evaluator, root='shared_brain/fleet_learning', client=None, event_fn=None, brain=None, require_llm=False):
        self.evaluator = evaluator
        self.root = Path(root)
        self.client = client
        self.require_llm = require_llm
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / 'ledger.json'
        self.event_fn = event_fn
        from shared_brain.memory_store import SharedBrain
        self.brain = brain or SharedBrain(self.root / 'memory')
        self._context = ''
        self._run_receipts = []

    def _emit(self, status, message, **extra):
        if self.event_fn:
            self.event_fn({'agent': 'Fleet Learner', 'status': status, 'message': message, **extra})

    @contextmanager
    def _operation(self, operation):
        self._context = self.brain.read_context('FleetLearner', 1100)
        outcome = 'error'
        try:
            yield
            outcome = 'completed'
        finally:
            self.brain.atomic_update('FleetLearner', {})
            self.brain.log_transaction('FleetLearner', 'write', 'fleet_operation_receipt', 0,
                                       {'operation': operation, 'outcome': outcome})
            self._context = ''


    def _load(self):
        return json.loads(self.path.read_text()) if self.path.exists() else []

    def _append(self, entry):
        with open(self.root / 'ledger.lock', 'a+') as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            rows = self._load()
            entry['attempt'] = len(rows)
            rows.append(entry)
            fd, temp = tempfile.mkstemp(dir=self.root, prefix='fleet_ledger_')
            try:
                with os.fdopen(fd, 'w') as stream:
                    json.dump(rows, stream, indent=2, allow_nan=False)
                    stream.write('\n')
                os.replace(temp, self.path)
            finally:
                if os.path.exists(temp):
                    os.unlink(temp)

    def _evaluate(self, parameters, seeds):
        runs = []
        for seed in seeds:
            # Source of outcomes is exclusively the supplied actual rollout evaluator.
            self._emit('running', 'Fresh sensor-based fleet rollout', seed=seed, parameters=parameters)
            with self._operation(f'evaluate_seed_{seed}'):
                result = self.evaluator(dict(parameters), int(seed))
            if int(result.get('seed', -1)) != int(seed):
                raise ValueError('Fresh rollout evaluator returned a different seed')
            self._run_receipts.append({'parameters_sha256': _hash(parameters),
                                       'measured': measured_summary([result])['runs'][0]})
            runs.append(result)
        return measured_summary(runs)

    def propose(self, current, measured):
        if self.client is not None and self.client.available:
            system = ('Improve this sensor-based mining fleet using ONLY bounded parameter changes. '
                      'Reply JSON {"parameters":{name:number},"rationale":"..."}. No source code. '
                      'Use only the supplied optimization outcomes; do not claim unevaluated success.')
            user = json.dumps({'shared_context': self._context, 'current': current, 'bounds': BOUNDS, 'measured_optimization': measured})
            text = self.client.chat([{'role': 'system', 'content': system}, {'role': 'user', 'content': user}],
                                    max_tokens=500, temperature=.2, timeout=60)
            if text:
                text = text.strip()
                if text.startswith('```'):
                    text = text.split('\n', 1)[1].rsplit('```', 1)[0]
                proposal = json.loads(text)
                if not isinstance(proposal, dict) or 'parameters' not in proposal or not isinstance(proposal['parameters'], dict) or set(proposal) - {'parameters', 'rationale'}:
                    raise ValueError('Unexpected model proposal fields')
                updated = validate_parameters({**current, **proposal['parameters']})
                return updated, {'source': f'llm:{self.client.model}', 'rationale': str(proposal.get('rationale', ''))[:600],
                                 'usage': dict(self.client.last_usage)}
            raise RuntimeError(self.client.last_error or 'No actual fleet proposal returned')
        updated = dict(current)
        # A measured miss triggers a bounded increase in visual track persistence
        # and reduction in minimum component area; no fabricated baseline success.
        if measured['mean_delivery_fraction'] < 1 or measured['mean_pickup_misses'] > 0:
            updated['detect_min_pixels'] = max(20, current['detect_min_pixels'] - 2)
            updated['track_memory_s'] = min(2, current['track_memory_s'] + .25)
            updated['grasp_dwell_s'] = min(1.5, current['grasp_dwell_s'] + .15)
        return validate_parameters(updated), {'source': 'measured_failure_parameter_update',
                                             'rationale': 'Bounded tracking and dwell changes after measured delivery/pickup failures'}

    def run_iteration(self, seeds=(11, 22, 33)):
        # Serialize the entire read/evaluate/promote cycle, not just the append.
        # Another learner reloads its starting controller only after this run ends.
        with open(self.root / 'run.lock', 'a+') as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            return self._run_iteration_locked(seeds)

    def _run_iteration_locked(self, seeds):
        seeds = tuple(int(seed) for seed in seeds)
        if not seeds or len(set(seeds)) != len(seeds) or any(seed not in (11, 22, 33) for seed in seeds):
            raise ValueError('Fleet learning uses unique optimization seeds 11/22/33 only')
        identity = source_hashes()
        current = load_active_parameters(self.root)
        entry = {'timestamp': datetime.now(timezone.utc).isoformat(), 'scope': 'sensor-based fleet optimization only',
                 'seeds': list(seeds), 'parameters': current, 'parameter_sha256': _hash(current),
                 'previous_parameters': current, 'previous_parameter_sha256': _hash(current),
                 'source_hashes_before': identity, 'accepted': False, 'decision': 'ERROR'}
        phase = 'baseline'
        self._run_receipts = []
        try:
            if self.require_llm and (self.client is None or not self.client.available):
                phase = 'client_preflight'
                raise RuntimeError('Actual Crusoe client unavailable; no model operation or physics run attempted')
            baseline = self._evaluate(current, seeds)
            entry['baseline_fresh'] = baseline
            self.brain.atomic_update('FleetLearner', {'section': 3, 'bullets': [
                f"Fleet optimization baseline seeds {list(seeds)}: delivery {baseline['mean_delivery_fraction']:.3f}, "
                f"pickup misses {baseline['mean_pickup_misses']:.3f}, peak tool {baseline['peak_force_n']:.3f} N."]})
            phase = 'proposal'
            self._emit('running', 'Proposing bounded fleet parameter changes from fresh optimization results', baseline=baseline)
            with self._operation('parameter_proposal'):
                candidate, proposal = self.propose(current, baseline)
            entry.update(parameters=candidate, parameter_sha256=_hash(candidate), proposal=proposal)
            if candidate == current:
                entry.update(decision='NO_CHANGE', reasons=['No parameter change proposed; no promotion'])
            else:
                phase = 'candidate'
                fresh = self._evaluate(candidate, seeds)
                safe = fresh['peak_force_n'] < 120 and fresh['collector_peak_force_n'] < 25
                improved = (fresh['score'] > baseline['score'] + .01
                            and fresh['mean_delivery_fraction'] >= baseline['mean_delivery_fraction'])
                entry.update(candidate_fresh=fresh, accepted=bool(safe and improved),
                             decision='PROMOTE' if safe and improved else 'REJECT',
                             reasons=[] if safe and improved else ['Candidate did not improve fresh fleet score/delivery within force limits'])
            if source_hashes() != identity:
                raise RuntimeError('Fleet source changed during evaluation; results cannot promote these parameters')
        except Exception as exc:
            entry.update(accepted=False, decision='ERROR', failure_phase=phase,
                         error=f'{type(exc).__name__}: {exc}', reasons=['Actual operation failed; no promotion'])
        entry['source_hashes_after'] = source_hashes()
        entry['fresh_run_receipts'] = list(self._run_receipts)
        self.brain.atomic_update('FleetLearner', {'section': 3 if entry['accepted'] else 4, 'bullets': [
            f"Fleet measured attempt: {entry['decision']}; parameters {entry['parameter_sha256'][:12]}; "
            f"{entry.get('error') or '; '.join(entry.get('reasons', [])) or 'fresh score improved within force limits'}"]})
        entry['context_stats'] = self.brain.stats()
        self._append(entry)
        self._emit('accepted' if entry['accepted'] else 'failed' if entry['decision'] == 'ERROR' else 'rejected',
                   f"Fleet learning {entry['decision']}; no core benchmark ledger changes", decision=entry['decision'])
        return entry


def load_active_parameters(root=REPO_ROOT / 'shared_brain/fleet_learning'):
    """Use promoted parameters only when the exact evaluated fleet source still matches."""
    path = Path(root) / 'ledger.json'
    rows = json.loads(path.read_text()) if path.exists() else []
    identity = source_hashes()
    active = dict(DEFAULTS)
    for row in rows:
        if not isinstance(row, dict) or not row.get('accepted') or row.get('decision') != 'PROMOTE':
            continue
        if row.get('source_hashes_before') != identity or row.get('source_hashes_after') != identity:
            continue
        try:
            parameters = validate_parameters(row.get('parameters', {}))
        except (TypeError, ValueError):
            continue
        if row.get('parameter_sha256') == _hash(parameters):
            active = parameters
    return active
