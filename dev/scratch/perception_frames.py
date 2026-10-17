import sys,json,cv2
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import core.verifier_harness as vh
import core.env_runner as er
seed=int(sys.argv[2])
assert seed in vh.OPTIMIZATION_SEEDS
base=vh.load_pipeline_class(sys.argv[1])
rows=[]
class Trace(base):
 def step(self,t,rgb,depth,proprio):
  out=super().step(t,rgb,depth,proprio)
  if any(abs(t-v)<.01 for v in [0.,1.,4.,6.5,8.5,10.5]):
   cv2.imwrite('dev/scratch/perception_%03d.png'%round(t*10),cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))
   masks=self._masks(rgb,depth)
   rows.append({'t':t,'track':{n:p['pos'].tolist() for n,p in out['estimated_objects'].items()},'measure':{n:None if self._measure(n,m,depth) is None else self._measure(n,m,depth)[0].tolist() for n,m in masks.items()}})
  return out
r=er.run_episode(Trace,seed)
Path('dev/scratch/perception_frames33.json').write_text(json.dumps({'obs':rows,'truth':[{'t':t,'core':gt['core'][0].tolist()} for t,gt in zip(r.t,r.gt) if any(abs(t-v)<.01 for v in [0.,1.,4.,6.5,8.5,10.5])]},indent=2))
