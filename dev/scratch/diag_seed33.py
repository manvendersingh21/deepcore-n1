"""Seed-33 diagnostic: per-physics-step context around the peak gated core force + phase timeline.
Usage: python dev/scratch/diag_seed33.py <pipeline.py> [seed]
Monkeypatches env_runner._log_physics at runtime only (no core/ edits)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import mujoco
import core.env_runner as er
import core.verifier_harness as vh

path = sys.argv[1]
seed = int(sys.argv[2]) if len(sys.argv) > 2 else 33
LOG = []
_orig = er._log_physics
_cur_phase = [""]


def _gname(m, g):
    return mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or f"g{g}"


def _log(ep, rec):
    _orig(ep, rec)
    m, d = ep.model, ep.data
    core_g = ep._gid["core"]
    hand, objs = [], []
    for i in range(d.ncon):
        g1, g2 = d.contact[i].geom1, d.contact[i].geom2
        if (g1 in ep._hand_gids and g2 in ep._obj_gids) or (g2 in ep._hand_gids and g1 in ep._obj_gids):
            hand.append(f"{_gname(m, g1)}-{_gname(m, g2)}")
        if core_g in (g1, g2):
            objs.append(_gname(m, g2 if g1 == core_g else g1))
    LOG.append(dict(t=d.time, ph=rec.phase[-1] if rec.phase else "", ee=ep.ee_pos().copy(),
                    core=d.xpos[ep._bid["core"]].copy(), gf=ep.core_force_now,
                    wf=ep.wrist_force_world().copy(), hand=hand, coreC=sorted(set(objs)),
                    grip=rec.gripper_cmd[-1] if rec.gripper_cmd else 1.0))


er._log_physics = _log
rec = er.run_episode(vh.load_pipeline_class(path), seed)
s = vh.score_record(rec)
print({k: s[k] for k in ("composite_score", "rubble_cleared_count", "peak_core_contact_force_n",
                         "max_wrist_force_n", "core_shattered", "core_extracted", "score_breakdown")})
gf = np.array([r["gf"] for r in LOG])
k = int(gf.argmax())
print(f"\nPEAK gated force {gf[k]:.2f} N at step {k} (sim t={LOG[k]['t']:.3f}; episode t={LOG[k]['t'] - 1.0:.3f}?)")
for j in range(max(0, k - 6), min(len(LOG), k + 4)):
    r = LOG[j]
    print(f"{'>>' if j == k else '  '} k={j} t={r['t']:.3f} {r['ph']:<20} gf={r['gf']:6.2f} wf={np.round(r['wf'], 2)} "
          f"ee={np.round(r['ee'], 4)} core={np.round(r['core'], 4)} grip={r['grip']:.2f} hand={r['hand']} coreC={r['coreC']}")
# first crossing of 18 N
over = np.nonzero(gf > 18.0)[0]
if over.size:
    print(f"first >18 N at step {over[0]}, all >18 N steps: {over.tolist()[:20]}")
# phase timeline
print("\nPHASE TIMELINE (episode t, from rec.t):")
prev = None
for t, ph, ee in zip(rec.t, rec.phase, rec.ee_pos):
    if ph != prev:
        print(f"  t={t:5.2f} {ph:<22} ee={np.round(ee, 3)}")
        prev = ph
print("\nFINAL:")
cxy = rec.final["core"]["pos"][:2]
for o in er.OBJECTS:
    p = rec.final[o]["pos"]
    print(f"  {o:9s} pos={np.round(p, 4)} dist_from_core_xy={np.linalg.norm(p[:2] - cxy):.3f} speed={rec.final[o]['speed']:.3f}")
print("GT at t=0:", {o: np.round(rec.gt[0][o][0], 4).tolist() for o in er.OBJECTS})
print("errors:", rec.errors[:5])
