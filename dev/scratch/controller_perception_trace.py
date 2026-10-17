"""Optimization-only perception diagnosis; never imports into candidate pipeline."""
import sys, json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import core.verifier_harness as vh
import core.env_runner as er
seed=int(sys.argv[2])
if seed not in vh.OPTIMIZATION_SEEDS:
    raise SystemExit('Perception autopsies are restricted to optimization seeds')
base=vh.load_pipeline_class(sys.argv[1])
records=[]
class Trace(base):
    def step(self,t,rgb,depth,proprio):
        out=super().step(t,rgb,depth,proprio)
        if t<2.0 or self.phase=='APPROACH':
            records.append({'t':t,'phase':out['debug_overlay']['phase'],
                            'est':{n:p['pos'].tolist() for n,p in out['estimated_objects'].items()},
                            'bboxes':out['debug_overlay']['bboxes']})
        return out
rec=er.run_episode(Trace,seed)
s=vh.score_record(rec)
result={'score':vh._public(s),'autopsy':vh.autopsy(rec,s),'observations':records,
        'truth': [{'t': t, 'pos': {n: gt[n][0].tolist() for n in er.OBJECTS}}
                  for t,gt in zip(rec.t,rec.gt) if t<2.0 or round(t*20)%10==0]}
Path(sys.argv[3]).write_text(json.dumps(result,indent=2))
print(json.dumps(result['score']),flush=True)
