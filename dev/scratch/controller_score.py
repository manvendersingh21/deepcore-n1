import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import core.verifier_harness as vh
r=vh.evaluate(sys.argv[1], (int(sys.argv[2]),))["per_seed"][0]
print(json.dumps(r,sort_keys=True),flush=True)
