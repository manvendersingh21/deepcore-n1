"""Deterministic fallback patch library (used when Crusoe inference is unavailable or returns bad code).

Each patch is a complete, standalone ExcavationPipeline file materialised from agents/patches/evolved_pipeline.py with a
cumulative set of verified mutations switched on. Ledger entries built from these are labelled `fallback_patch:<name>`.
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TEMPLATE = ROOT / "patches" / "evolved_pipeline.py"
OUT_DIR = ROOT / "patches" / "generated"

_STAGES = [
    {"name": "v1_hsv_iqr_z_ordering", "specialist": "both",
     "features": {"hsv_iqr": True, "z_ordering": True, "yaw": False, "tracker": False, "force_guard": False,
                  "gentle_place": False, "elevation_gate": True, "grasp_top": False, "core_grasp": False},
     "summary": "HSV segmentation + morphological closing + 25-75th IQR median depth back-projection with half-height "
                "correction; rubble sorted by descending Z and excavated to the dump zone before touching the core.",
     "playbook": {2: ["HSV bands + morph open/close + largest blob beat raw RGB thresholds under glare; use IQR-median "
                      "depth over the mask, never a single pixel (NaN/0.25/1.80 m spikes)."],
                  3: ["Excavate overburden first: sort rubble by descending Z, move each to the dump zone, then the core."]}},
    {"name": "v2_yaw_tracker", "specialist": "cv",
     "features": {"hsv_iqr": True, "z_ordering": True, "yaw": True, "tracker": True, "force_guard": False,
                  "gentle_place": False, "elevation_gate": True, "grasp_top": True, "core_grasp": False},
     "summary": "cv2.minAreaRect principal-axis yaw for grasp alignment + exponential tracker that rejects >6 cm jumps "
                "and holds tracks through dust plumes / hand occlusion.",
     "playbook": {2: ["minAreaRect long-axis yaw (image +u=-y, +v=-x) aligns the jaws; exponential tracker holding "
                      "through occlusion, rejecting >6 cm teleports."]}},
    {"name": "v3_force_guard_gentle_place", "specialist": "control",
     "features": {"hsv_iqr": True, "z_ordering": True, "yaw": True, "tracker": True, "force_guard": True,
                  "gentle_place": True, "elevation_gate": True, "grasp_top": True, "core_grasp": True},
     "summary": "Force-guarded compliant touchdown (fast to 2 cm above, then 12 mm creeps, stop at |Fz|>6.5 N) and a "
                "gentle low release of the core inside the extraction hopper.",
     "playbook": {3: ["Force-guarded touchdown: fast to 2 cm above the grasp height, then creep and stop on |Fz|>6.5 N; "
                      "release the core low (z~0.4245) in the hopper."]}},
]


def _materialise(stage: dict) -> Path:
    src = TEMPLATE.read_text()
    feats = "FEATURES = " + repr(stage["features"])
    src, n = re.subn(r"^FEATURES = \{[^}]*\}", feats, src, count=1, flags=re.M)
    if n != 1:
        raise RuntimeError("FEATURES line not found in evolved_pipeline.py")
    src = src.replace('"""Evolved DeepCore excavation pipeline (fallback patch library source).',
                      f'"""DeepCore fallback patch {stage["name"]}: {stage["summary"]}', 1)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"{stage['name']}.py"
    out.write_text(src)
    return out


def build_patches() -> list[dict]:
    return [{"name": s["name"], "path": str(_materialise(s)), "specialist": s["specialist"], "summary": s["summary"],
             "playbook": s["playbook"]} for s in _STAGES]


FALLBACK_PATCHES = build_patches()
