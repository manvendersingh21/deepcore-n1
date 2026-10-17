import sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import core.verifier_harness as vh
import core.env_runner as er
base=vh.load_pipeline_class(sys.argv[1])
class Trace(base):
    def step(self,t,rgb,depth,proprio):
        old=self.phase
        out=super().step(t,rgb,depth,proprio)
        if self.phase != old:
            print(round(t,2),old,'->',out['debug_overlay']['phase'],'ee',np.round(proprio['ee_pos'],3),'lock',self.lock,'gripper',proprio['gripper'],flush=True)
        return out
rec=er.run_episode(Trace,int(sys.argv[2]))
s=vh.score_record(rec)
print({k:s[k] for k in ['composite_score','core_shattered','core_extracted','peak_core_contact_force_n','rubble_cleared_count']})
print('final',rec.final)
