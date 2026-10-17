"""Mandatory end-to-end verification suite (spec Phase 6). Every number here comes from real MuJoCo rollouts."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def evolution(tmp_path_factory):
    """Runs baseline + 3 recursive generations once, in an isolated copy of the shared brain."""
    from shared_brain.memory_store import SharedBrain
    from agents.crusoe_client import CrusoeClient
    from agents.contract_orchestrator import ContractOrchestrator

    brain_dir = tmp_path_factory.mktemp("brain")
    for f in ("playbook.md", "ledger.json"):
        shutil.copy(ROOT / "shared_brain" / f, brain_dir / f)
    brain = SharedBrain(root=str(brain_dir))
    brain.reset()
    orch = ContractOrchestrator(brain, CrusoeClient(), log_fn=lambda *a, **k: None)
    orch.ensure_baseline()
    entries = orch.run(3, verify_improvement=True)
    return brain, brain.load_ledger(), entries


def test_sha256_integrity_and_no_privileged_leak():
    from core.integrity import verify_integrity, LOCKED_FILES
    from core.verifier_harness import static_check, load_pipeline_class
    hashes = verify_integrity()
    assert set(hashes) == set(LOCKED_FILES) and all(len(h) == 64 for h in hashes.values())
    cand = ROOT / "workspace" / "candidate_pipeline.py"
    src = cand.read_text()
    assert static_check(src) == []
    assert "mujoco" not in src and "MjData" not in src and "qpos" not in src and "xpos" not in src
    load_pipeline_class(cand)
    for evil in ("import mujoco\nclass ExcavationPipeline: pass",
                 "import os\nclass ExcavationPipeline: pass",
                 "class ExcavationPipeline:\n    x = open('core/scene.xml')",
                 "class ExcavationPipeline:\n    x = ().__class__.__mro__"):
        assert static_check(evil), evil


def test_baseline_v0_shatters_core():
    from core.verifier_harness import evaluate, OPTIMIZATION_SEEDS
    res = evaluate(ROOT / "workspace" / "baseline_v0.py", OPTIMIZATION_SEEDS)
    agg = res["aggregate"]
    assert agg["core_shattered"] is True
    assert agg["composite_score"] < 30.0
    assert all(s["core_shattered"] for s in res["per_seed"])


def test_recursive_evolution_v0_to_v3(evolution):
    _, ledger, _ = evolution
    accepted = [e for e in ledger if e["accepted"]]
    assert accepted[0]["version"] == 0 and len(accepted) >= 4
    opt = [e["opt"]["composite_score"] for e in accepted]
    hold = [e["holdout"]["composite_score"] for e in accepted]
    assert all(b > a for a, b in zip(opt, opt[1:])), opt
    assert all(b >= a for a, b in zip(hold, hold[1:])), hold
    best = accepted[-1]
    assert best["opt"]["composite_score"] > 85.0
    assert best["holdout"]["composite_score"] > 85.0
    assert best["holdout"]["core_shattered"] is False and best["holdout"]["core_extracted"] is True
    for e in accepted[1:]:
        assert e["source"].startswith(("llm:", "fallback_patch:"))


def test_shared_brain_persistence_and_token_budget(evolution):
    brain, ledger, _ = evolution
    root = Path(brain.root)
    on_disk = json.loads((root / "ledger.json").read_text())
    assert on_disk == ledger and len(on_disk) >= 4
    assert sorted({e["version"] for e in on_disk}) == list(range(len({e["version"] for e in on_disk})))
    text = (root / "playbook.md").read_text()
    for h in ("## 1. Camera Geometry & Back-Projection Equations",
              "## 2. Verified 3D Vision Filters (Dust/Glare/Occlusion Invariant)",
              "## 3. Verified Excavation Ordering & Force-Guarded Control Laws",
              "## 4. Graveyard of Rejected Hypotheses (DO NOT REPEAT)"):
        assert h in text
    assert -(-len(text) // 4) < 1500
    assert brain.stats()["n_versions"] >= 4


def test_hud_and_demo_video_generation(tmp_path):
    from ui.live_hud import HudState, render_frame
    import numpy as np
    frame = render_frame(HudState(), np.zeros((480, 640, 3), np.uint8), np.zeros((480, 640, 3), np.uint8))
    assert frame.shape == (900, 1600, 3) and frame.dtype == np.uint8 and frame.std() > 5
    from scripts.record_demo import main as record
    out = tmp_path / "submission_demo.mp4"
    record(out=str(out), fast=True)
    assert out.exists() and out.stat().st_size > 10_000
