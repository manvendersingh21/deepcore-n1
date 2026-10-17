"""LOCKED mathematical contract judge. Read-only after Phase 1 (SHA-256 guarded by core/integrity.py).

evaluate(pipeline_path, seeds) loads a candidate module, statically rejects privileged access, rolls out real
MuJoCo episodes (1400 x mj_step at 200 Hz, RGB-D rendered at 20 Hz) and scores them. Diagnostic autopsies are only
produced for OPTIMIZATION_SEEDS; holdout results expose aggregate numbers only.
"""
from __future__ import annotations

import ast
import importlib.util
import json
import sys
import uuid
from pathlib import Path

import numpy as np

from core import env_runner as er

OPTIMIZATION_SEEDS = (11, 22, 33)
HOLDOUT_SEEDS = (101, 102, 103)
FRACTURE_N = 18.0               # spec §1.4: core_shattered if peak core contact force > 18 N
FORCE_CONTRACT_N = 28.0         # Law 5 safety contract (25 N figure reported as `force_ok_25`)
LATENCY_CONTRACT_MS = 25.0
PLACE_TOL_XY_MM, PLACE_TOL_Z_MM, SETTLE_SPEED = 12.0, 6.0, 0.01
EXTRACT_XY_M, EXTRACT_SPEED = 0.045, 0.02
CLEAR_DIST_M = 0.09
VISION_RMSE_WINDOW_FRAMES = 40  # spec §1.4: vision RMSE over the first 40 vision frames
ERR_CLIP_MM = 150.0             # a missing / non-finite / teleporting estimate costs at most this per frame
FORBIDDEN_MODULES = {"mujoco", "core", "os", "sys", "subprocess", "shutil", "socket", "ctypes", "importlib",
                     "inspect", "gc", "builtins", "pathlib", "pickle", "multiprocessing", "threading", "signal",
                     "agents", "shared_brain", "ui", "tests", "http", "urllib", "requests", "httpx", "io"}
FORBIDDEN_NAMES = {"open", "exec", "eval", "compile", "__import__", "globals", "locals", "vars", "getattr",
                   "setattr", "delattr", "breakpoint", "input", "memoryview"}
FORBIDDEN_ATTRS = {"__globals__", "__subclasses__", "__builtins__", "__code__", "__closure__", "f_back", "f_globals",
                   "f_locals", "gi_frame", "tb_frame", "__dict__", "__mro__", "__bases__", "__loader__", "__spec__",
                   "load", "loadtxt", "genfromtxt", "fromfile", "memmap", "tofile", "save", "savez", "savetxt",
                   "imread", "imwrite", "VideoCapture"}


# ---------------------------------------------------------------------------------------------- static guard
def static_check(source: str) -> list[str]:
    """Returns a list of violations (empty == clean). Shared by agents/sandbox_exec.py."""
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return [f"SyntaxError: {e}"]
    bad = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name.split(".")[0] in FORBIDDEN_MODULES:
                    bad.append(f"forbidden import {a.name} (line {node.lineno})")
        elif isinstance(node, ast.ImportFrom):
            if node.level or (node.module or "").split(".")[0] in FORBIDDEN_MODULES:
                bad.append(f"forbidden import from {node.module} (line {node.lineno})")
        elif isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            bad.append(f"forbidden name {node.id} (line {node.lineno})")
        elif isinstance(node, ast.Attribute) and node.attr in FORBIDDEN_ATTRS:
            bad.append(f"forbidden attribute {node.attr} (line {node.lineno})")
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and (
                "core/" in node.value or "scene.xml" in node.value or "workspace/" in node.value):
            bad.append(f"forbidden path literal (line {node.lineno})")
    if not any(isinstance(n, ast.ClassDef) and n.name == "ExcavationPipeline" for n in tree.body):
        bad.append("missing class ExcavationPipeline")
    return bad


def load_pipeline_class(path: str | Path):
    path = Path(path)
    bad = static_check(path.read_text())
    if bad:
        raise PermissionError("candidate rejected by static guard: " + "; ".join(bad[:5]))
    name = f"candidate_{uuid.uuid4().hex[:8]}"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.ExcavationPipeline


# ---------------------------------------------------------------------------------------------- metrics
def _objects_gt_at(rec: er.EpisodeRecord, k: int) -> dict:
    return rec.gt[k]


def score_record(rec: er.EpisodeRecord) -> dict:
    """Computes the spec §1.4 metrics for one rollout. Pure function of the EpisodeRecord."""
    O = er.OBJECTS
    # ---- vision RMSE: 3D error over the first 40 vision frames (before contact moves the objects)
    errs, per_obj = [], {o: {"err": [], "bias": [], "occ_err": []} for o in O}
    n = min(VISION_RMSE_WINDOW_FRAMES, len(rec.gt))
    for k in range(n):
        gt, est = rec.gt[k], rec.est[k]
        for o in O:
            e = ERR_CLIP_MM
            if est is not None and o in est and np.all(np.isfinite(est[o][0])):
                diff = (est[o][0] - gt[o][0]) * 1000.0
                e = min(ERR_CLIP_MM, float(np.linalg.norm(diff)))
                if e < ERR_CLIP_MM:
                    per_obj[o]["bias"].append(diff)
            errs.append(e)
            per_obj[o]["err"].append(e)
            if rec.occluded[k]:
                per_obj[o]["occ_err"].append(e)
    vision_mm = float(np.mean(errs)) if errs else ERR_CLIP_MM

    # ---- peak core contact force, gated (spec §1.4): gripper within 0.07 m of core OR pad contact with rubble/core
    gate_f = np.asarray(rec.core_gate_force, dtype=np.float64)
    peak_core_f = float(gate_f.max()) if gate_f.size else 0.0
    max_wrist_f = float(np.linalg.norm(np.asarray(rec.force_vec), axis=1).max()) if rec.force_vec else 0.0
    core_shattered = bool(peak_core_f > FRACTURE_N or max_wrist_f > FORCE_CONTRACT_N)

    # ---- rubble clearance: boulders moved >= 9 cm from the core by episode end
    core_xy = rec.final["core"]["pos"][:2]
    cleared = sum(int(np.linalg.norm(rec.final[r]["pos"][:2] - core_xy) >= CLEAR_DIST_M) for r in ("rubble_a", "rubble_b"))

    # ---- extraction: core in hopper, at rest, not shattered
    hop = er.HOPPER_ZONE
    cf = rec.final["core"]
    in_hopper = (np.linalg.norm(cf["pos"][:2] - hop[:2]) <= EXTRACT_XY_M and 0.40 <= cf["pos"][2] <= 0.45
                 and cf["speed"] < EXTRACT_SPEED)
    core_extracted = bool(in_hopper and not core_shattered)

    p99 = float(np.percentile(rec.latency_ms, 99)) if rec.latency_ms else 1e3

    vision_score = 30.0 * max(0.0, 1.0 - (vision_mm / 40.0))
    clearance_score = 25.0 * (cleared / 2.0)
    extraction_score = 30.0 if core_extracted else 0.0
    safety_score = 0.0 if core_shattered else 10.0 * max(0.0, 1.0 - max(0.0, peak_core_f - 8.0) / 10.0)
    latency_score = 5.0 * max(0.0, 1.0 - (p99 / LATENCY_CONTRACT_MS))
    composite = round(vision_score + clearance_score + extraction_score + safety_score + latency_score, 2)

    return {
        "seed": rec.seed,
        "vision_3d_rmse_mm": round(vision_mm, 2),
        "rubble_cleared_count": int(cleared),
        "peak_core_contact_force_n": round(peak_core_f, 2),
        "max_wrist_force_n": round(max_wrist_f, 2),
        "core_shattered": core_shattered,
        "core_extracted": core_extracted,
        "vision_latency_p99_ms": round(p99, 2),
        "composite_score": composite,
        "score_breakdown": {"vision": round(vision_score, 2), "clearance": round(clearance_score, 2),
                            "extraction": round(extraction_score, 2), "safety": round(safety_score, 2),
                            "latency": round(latency_score, 2)},
        "force_ok_25": bool(max_wrist_f < 25.0),
        "pipeline_errors": list(rec.errors),
        "_per_obj": per_obj, "_final": rec.final, "_gate_force": gate_f, "_force": np.linalg.norm(
            np.asarray(rec.force_vec), axis=1) if rec.force_vec else np.zeros(1),
    }


def autopsy(rec: er.EpisodeRecord, s: dict) -> dict:
    """Structured failure report. Only ever generated for optimization seeds."""
    lines, objs = [], {}
    for o in er.OBJECTS:
        po = s["_per_obj"][o]
        bias = np.mean(po["bias"], axis=0) if po["bias"] else np.full(3, np.nan)
        info = {
            "mean_err_mm": round(float(np.mean(po["err"])), 1) if po["err"] else None,
            "bias_xyz_mm": [round(float(v), 1) for v in bias],
            "teleport_frames": int(sum(e >= 50.0 for e in po["err"])),
            "frames_scored": len(po["err"]),
            "occluded_mean_err_mm": round(float(np.mean(po["occ_err"])), 1) if po["occ_err"] else None,
            "final_pos_m": [round(float(v), 4) for v in s["_final"][o]["pos"]],
        }
        objs[o] = info
        lines.append(f"{o}: mean 3D err {info['mean_err_mm']}mm, bias xyz {info['bias_xyz_mm']}mm, "
                     f"{info['teleport_frames']}/{info['frames_scored']} frames >50mm, "
                     f"err while occluded {info['occluded_mean_err_mm']}mm")
    # force spike events above the fracture threshold (gated core contact)
    G = s["_gate_force"]
    spikes = []
    above = np.where(G > FRACTURE_N)[0]
    if above.size:
        groups = np.split(above, np.where(np.diff(above) > 5)[0] + 1)
        for g in groups[:6]:
            i = g[np.argmax(G[g])]
            spikes.append({"step": int(i), "t": round(float(rec.force_t[i]), 3), "force_n": round(float(G[i]), 1)})
            lines.append(f"core contact force {G[i]:.1f}N at t={rec.force_t[i]:.2f}s (gated, > {FRACTURE_N}N)")
    phases = []
    for t, ph in zip(rec.t, rec.phase):
        if not phases or phases[-1][1] != ph:
            phases.append((round(t, 2), ph))
    if s["rubble_cleared_count"] < 2:
        lines.append(f"only {s['rubble_cleared_count']}/2 rubble boulders cleared; last phase '{rec.phase[-1] if rec.phase else ''}'")
    if not s["core_extracted"]:
        f = s["_final"]["core"]
        lines.append(f"core NOT extracted: final xy {np.round(f['pos'][:2], 3).tolist()} "
                     f"speed {f['speed']:.3f} m/s (hopper {er.HOPPER_ZONE[:2].tolist()})")
    if rec.errors:
        lines.append(f"pipeline exceptions: {rec.errors[:3]}")
    return {"objects": objs, "core_force_spikes": spikes, "phase_timeline": phases[:40], "summary_lines": lines}


def _public(s: dict) -> dict:
    return {k: v for k, v in s.items() if not k.startswith("_")}


def _aggregate(per_seed: list[dict]) -> dict:
    mean_keys = ["vision_3d_rmse_mm", "rubble_cleared_count", "peak_core_contact_force_n",
                 "composite_score"]
    agg = {k: round(float(np.mean([s[k] for s in per_seed])), 2) for k in mean_keys}
    agg["core_shattered"] = bool(any(s["core_shattered"] for s in per_seed))
    agg["core_extracted"] = bool(all(s["core_extracted"] for s in per_seed))
    agg["extraction_rate"] = round(float(np.mean([s["core_extracted"] for s in per_seed])), 3)
    agg["max_wrist_force_n"] = round(float(max(s["max_wrist_force_n"] for s in per_seed)), 2)
    agg["vision_latency_p99_ms"] = round(float(max(s["vision_latency_p99_ms"] for s in per_seed)), 2)
    agg["force_ok_25"] = agg["max_wrist_force_n"] < 25.0
    agg["passed"] = bool(agg["max_wrist_force_n"] < FORCE_CONTRACT_N
                         and agg["vision_latency_p99_ms"] < LATENCY_CONTRACT_MS
                         and all(s["rubble_cleared_count"] == 2 for s in per_seed) and agg["core_extracted"]
                         and agg["composite_score"] >= 80.0)
    return agg


def check_contract(opt: dict, holdout: dict, best_opt: float | None, best_holdout: float | None) -> tuple[bool, list[str]]:
    """Spec Law 5 gate for Version N+1. `opt`/`holdout` are aggregates; best_* are the previous accepted scores."""
    reasons = []
    if best_opt is not None and not opt["composite_score"] > best_opt:
        reasons.append(f"opt composite {opt['composite_score']} not > {best_opt}")
    if best_holdout is not None and not holdout["composite_score"] >= best_holdout:
        reasons.append(f"holdout composite {holdout['composite_score']} < {best_holdout}")
    if opt["max_wrist_force_n"] >= FORCE_CONTRACT_N or holdout["max_wrist_force_n"] >= FORCE_CONTRACT_N:
        reasons.append("max wrist force >= 28 N")
    if opt["vision_latency_p99_ms"] >= LATENCY_CONTRACT_MS or holdout["vision_latency_p99_ms"] >= LATENCY_CONTRACT_MS:
        reasons.append("vision latency p99 >= 25 ms")
    if opt["core_shattered"] or holdout["core_shattered"]:
        reasons.append("core shattered")
    return (not reasons), reasons


def evaluate(pipeline_path: str | Path, seeds=OPTIMIZATION_SEEDS, diagnostics: bool | None = None) -> dict:
    from core.integrity import verify_integrity
    verify_integrity()
    seeds = tuple(int(s) for s in seeds)
    if diagnostics is None:
        diagnostics = set(seeds) <= set(OPTIMIZATION_SEEDS)
    if diagnostics and not set(seeds) <= set(OPTIMIZATION_SEEDS):
        raise PermissionError("diagnostic autopsies are never produced for holdout seeds")
    cls = load_pipeline_class(pipeline_path)
    per_seed, autopsies = [], {}
    for seed in seeds:
        rec = er.run_episode(cls, seed)
        s = score_record(rec)
        if diagnostics:
            autopsies[seed] = autopsy(rec, s)
        per_seed.append(_public(s))
    out = {"pipeline": str(pipeline_path), "seeds": list(seeds), "aggregate": _aggregate(per_seed),
           "per_seed": per_seed}
    if diagnostics:
        out["autopsy"] = autopsies
    else:
        for s in out["per_seed"]:
            s.pop("pipeline_errors", None)
    return out


if __name__ == "__main__":  # python -m core.verifier_harness PATH [--seeds ...] [--no-diagnostics] [--json-out F]
    import argparse
    ap = argparse.ArgumentParser(description="DeepCore-N1 locked verifier")
    ap.add_argument("pipeline")
    ap.add_argument("--seeds", nargs="+", type=int, default=list(OPTIMIZATION_SEEDS))
    ap.add_argument("--holdout", action="store_true", help="evaluate HOLDOUT_SEEDS (never diagnostics)")
    ap.add_argument("--no-diagnostics", action="store_true")
    ap.add_argument("--json-out", default=None)
    a = ap.parse_args()
    seeds = HOLDOUT_SEEDS if a.holdout else tuple(a.seeds)
    diag = False if (a.no_diagnostics or a.holdout) else None
    out = evaluate(a.pipeline, seeds, diagnostics=diag)
    text = json.dumps(out, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    if a.json_out:
        Path(a.json_out).write_text(text)
    else:
        print(text)
