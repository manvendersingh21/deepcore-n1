"""Request one real bounded control edit using measured optimization data only."""
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from agents.contract_orchestrator import ContractOrchestrator, CONTROL
from agents.crusoe_client import CrusoeClient, FallbackSynthesizer
from core.verifier_harness import _aggregate

source_path, scores_path, output_path = map(Path, sys.argv[1:4])
source = source_path.read_text()
opt = [json.loads(line) for line in scores_path.read_text().splitlines()
       if line.strip() and json.loads(line).get('seed') in (11, 22, 33)]
assert {row['seed'] for row in opt} == {11, 22, 33}, 'all optimization seeds required'
best = {'sha256': hashlib.sha256(source.encode()).hexdigest(), 'opt': _aggregate(opt),
        'autopsy': {'optimization_results': opt}}
client = CrusoeClient()
synth = FallbackSynthesizer(client)
orch = ContractOrchestrator(None, client, synthesizer=synth, log_fn=lambda s: None)
contract = orch.derive_contract(1, best)
contract['control_focus'] = ('Preserve the successful extraction and rubble clearing on every optimization seed. '
    'Reduce contact force during guarded grasp/descent, especially the measured 15.38 N on seed 22. '
    'Use a general sensor-based policy, never branch on seeds or hardcode observed object poses. '
    'Keep perception code and the 11-second episode deadline intact.')
started = time.monotonic()
candidate = synth.try_edits(*orch._specialist_prompt(CONTROL, contract, source, best,
    'All optimization seeds currently extract safely. Changes must preserve this; no holdout diagnostics are supplied.'),
    source=source, agent=CONTROL)
if candidate:
    output_path.write_text(candidate)
receipt = {'source': str(source_path), 'source_sha256': best['sha256'],
    'origin': f'llm:{client.model}', 'proposal_mode': 'edits', 'valid': candidate is not None,
    'elapsed_s': round(time.monotonic() - started, 2), 'usage': client.last_usage,
    'problems': synth.last_problems, 'output': str(output_path) if candidate else None}
output_path.with_suffix('.receipt.json').write_text(json.dumps(receipt, indent=2))
print(json.dumps(receipt), flush=True)
