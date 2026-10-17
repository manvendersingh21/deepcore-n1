"""Run fresh sensor-based fleet optimization rollouts and a bounded parameter trial."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_active_parameters(root=ROOT / 'shared_brain/fleet_learning'):
    """Return only promoted parameters whose evaluated source identity still matches."""
    from agents.fleet_learning import load_active_parameters as load
    return load(root)


def run_learning(*, event_fn=None, use_llm=True, seeds=(11, 22, 33),
                 root=ROOT / 'shared_brain/fleet_learning'):
    from agents.crusoe_client import CrusoeClient
    from agents.fleet_learning import FleetLearner
    from ui.mining_fleet import run_mining_fleet
    def evaluator(parameters, seed):
        result = run_mining_fleet(controller_params=parameters, seed=seed)
        # Seed identifies the actual requested rollout; no measured metric is filled in.
        return {**result, 'seed': seed}
    client = CrusoeClient() if use_llm else None
    return FleetLearner(evaluator, root=root, client=client, event_fn=event_fn,
                        require_llm=use_llm).run_iteration(seeds=seeds)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds', nargs='+', type=int, default=[11, 22, 33])
    parser.add_argument('--root', type=Path, default=ROOT / 'shared_brain/fleet_learning')
    parser.add_argument('--no-llm', action='store_true', help='use measured-failure parameter adaptation without model calls')
    args = parser.parse_args()
    def progress(event):
        print(json.dumps({'type': 'event', **event}, allow_nan=False), flush=True)
    result = run_learning(event_fn=progress, use_llm=not args.no_llm, seeds=args.seeds, root=args.root)
    print(json.dumps({'type': 'result', 'entry': result}, allow_nan=False), flush=True)
    return 1 if result['decision'] == 'ERROR' else 0


if __name__ == '__main__':
    raise SystemExit(main())
