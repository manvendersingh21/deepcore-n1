"""Unit tests for the agent layer. No MuJoCo: evaluation is injected, the LLM is absent or stubbed."""
from __future__ import annotations

import json
import socket
from pathlib import Path

import pytest

from agents import crusoe_client as cc
from agents.contract_orchestrator import ContractOrchestrator, HOLDOUT_SEEDS, OPT_SEEDS
from agents.crusoe_client import CrusoeClient, FallbackSynthesizer, extract_code
from shared_brain.memory_store import SharedBrain

PIPE_TMPL = '''import numpy as np


class ExcavationPipeline:
    """{tag}"""

    def __init__(self, cam_intrinsics, cam_pos, cam_rot):
        self.K = cam_intrinsics

    def step(self, t_sec, rgb, depth, proprio):
        return {{"estimated_objects": {{}}, "action": {{"target_pos": np.asarray(proprio["ee_pos"]),
                "target_yaw": 0.0, "gripper": 1.0}}, "debug_overlay": {{"bboxes": [], "phase": "{tag}"}}}}
'''


def _agg(score: float, rmse: float, force: float = 5.0, latency: float = 4.0) -> dict:
    return {"composite_score": score, "vision_3d_rmse_mm": rmse, "rubble_cleared_count": 2.0,
            "peak_core_contact_force_n": force, "core_shattered": False, "core_extracted": True,
            "extraction_rate": 1.0, "vision_latency_p99_ms": latency, "max_wrist_force_n": force, "fps": 50.0}


def _law5(opt, hold, best_opt, best_hold):
    """Same rule shape as core.verifier_harness.check_contract, kept local so the test needs no MuJoCo."""
    reasons = []
    if best_opt is not None and not opt["composite_score"] > best_opt:
        reasons.append("opt not improved")
    if best_hold is not None and not hold["composite_score"] >= best_hold:
        reasons.append("holdout regressed")
    if opt["core_shattered"] or hold["core_shattered"]:
        reasons.append("core shattered")
    return not reasons, reasons


class FakeEvaluator:
    """Scores a candidate by the `SCORE=<opt>/<holdout>/<rmse>` tag in its docstring; records every call."""

    def __init__(self):
        self.calls: list[tuple[str, tuple, bool]] = []

    def __call__(self, path, seeds, diagnostics=True):
        self.calls.append((Path(path).name, tuple(seeds), diagnostics))
        tag = Path(path).read_text().split('"""')[1]
        opt, hold, rmse = (float(x) for x in tag.split("=")[1].split("/"))
        holdout = tuple(seeds) == HOLDOUT_SEEDS
        score = hold if holdout else opt
        res = {"aggregate": _agg(score, rmse), "per_seed": [{"seed": s, **_agg(score, rmse)} for s in seeds]}
        if diagnostics:
            res["autopsy"] = {s: {"worst_frame": 12, "rmse_mm": rmse} for s in seeds}
        return {"ok": True, "result": res, "sandbox": "fake", "wall_s": 0.0}


@pytest.fixture()
def world(tmp_path, monkeypatch):
    monkeypatch.delenv("CRUSOE_API_KEY", raising=False)
    root = tmp_path / "repo"
    (root / "workspace").mkdir(parents=True)
    (root / "workspace" / "baseline_v0.py").write_text(PIPE_TMPL.format(tag="SCORE=20/18/40"))
    patches = []
    for i, (tag, owner) in enumerate((("SCORE=45/40/12", "cv"), ("SCORE=70/66/6", "control"),
                                      ("SCORE=88/85/4", "both")), start=1):
        p = root / "patches" / f"v{i}.py"
        p.parent.mkdir(exist_ok=True)
        p.write_text(PIPE_TMPL.format(tag=tag))
        patches.append({"name": f"v{i}_patch", "path": str(p), "specialist": owner, "summary": f"patch {i}",
                        "playbook": {2 if owner == "cv" else 3: [f"patch {i} verified rule"]}})
    brain = SharedBrain(tmp_path / "brain")
    client = CrusoeClient()
    synth = FallbackSynthesizer(client, brain, patches=patches, validator=lambda s: [])
    events: list[dict] = []
    ev = FakeEvaluator()
    orch = ContractOrchestrator(brain, client, log_fn=lambda m: None, event_fn=events.append, evaluator=ev,
                                checker=_law5, synthesizer=synth, root=root)
    return orch, brain, ev, events, root, patches


# ---------------------------------------------------------------- crusoe client
def test_client_without_key_never_touches_network(monkeypatch):
    monkeypatch.delenv("CRUSOE_API_KEY", raising=False)

    def boom(*a, **k):
        raise AssertionError("network access attempted")

    monkeypatch.setattr(socket, "create_connection", boom)
    monkeypatch.setattr(socket.socket, "connect", boom)
    client = CrusoeClient()
    assert not client.available
    assert client.chat([{"role": "user", "content": "hi"}]) is None
    assert client._client is None


def test_extract_code_picks_pipeline_block():
    text = ("Here:\n```json\n{\"a\": 1}\n```\n```python\nimport numpy as np\nx = 1\n```\n"
            "```python\nimport numpy as np\n\nclass ExcavationPipeline:\n    pass\n```\nDone.")
    code = extract_code(text)
    assert code is not None and code.startswith("import numpy") and "class ExcavationPipeline" in code
    assert "x = 1" not in code
    assert extract_code("```python\nprint(1)\n```") is None
    assert extract_code(None) is None
    assert "class ExcavationPipeline" in extract_code("class ExcavationPipeline:\n    pass\n")


def test_fallback_generation_mapping_and_clamp(world):
    _, _, _, _, _, patches = world
    synth = FallbackSynthesizer(CrusoeClient(), patches=patches)
    assert [synth.next_patch(g)["name"] for g in (0, 1, 2, 3, 7)] == ["v1_patch", "v1_patch", "v2_patch",
                                                                      "v3_patch", "v3_patch"]
    src, origin = synth.synthesize(2, "sys", "user")
    assert origin == "fallback_patch:v2_patch" and "SCORE=70/66/6" in src


def test_synthesize_accepts_valid_llm_and_rejects_invalid(monkeypatch, world):
    _, _, _, _, _, patches = world

    class StubClient(CrusoeClient):
        reply = ""

        @property
        def available(self):
            return True

        def chat(self, messages, temperature=0.2, max_tokens=4096, timeout=60):
            self.last_usage = {"prompt_tokens": 10, "completion_tokens": 20}
            return self.reply

    client = StubClient(api_key="k", model="stub-model")
    synth = FallbackSynthesizer(client, patches=patches, validator=lambda s: ["import os"] if "import os" in s else [])
    client.reply = "```python\nimport numpy as np\nclass ExcavationPipeline:\n    pass\n```"
    assert synth.synthesize(1, "s", "u")[1] == "llm:stub-model"
    client.reply = "```python\nimport os\nclass ExcavationPipeline:\n    pass\n```"
    src, origin = synth.synthesize(1, "s", "u")
    assert origin == "fallback_patch:v1_patch" and synth.last_problems == ["import os"]


# ---------------------------------------------------------------- sandbox
def test_sandbox_rejects_malicious_candidate(tmp_path):
    pytest.importorskip("core.verifier_harness", exc_type=ImportError)
    from agents.sandbox_exec import run_candidate

    bad = tmp_path / "evil.py"
    bad.write_text("import os\n\nclass ExcavationPipeline:\n    def step(self, *a):\n        os.system('rm -rf /')\n")
    res = run_candidate(bad, OPT_SEEDS)
    assert res["ok"] is False and "static check rejected" in res["error"] and "os" in res["error"]


def test_sandbox_builds_verifier_command():
    from agents.sandbox_exec import parse_result, verifier_cmd

    cmd = verifier_cmd("w/c.py", (101, 102), diagnostics=False, json_out="/tmp/x.json")
    assert cmd[1:4] == ["-m", "core.verifier_harness", "w/c.py"]
    assert cmd[cmd.index("--seeds") + 1:cmd.index("--json-out")] == ["101", "102"] and cmd[-1] == "--no-diagnostics"
    assert parse_result("/nonexistent.json", 'log\n{"aggregate": {"composite_score": 1}}') == {
        "aggregate": {"composite_score": 1}}


# ---------------------------------------------------------------- orchestrator
def test_baseline_then_three_improving_generations(world):
    orch, brain, ev, events, root, _ = world
    entries = orch.run(3, verify_improvement=True)
    ledger = brain.load_ledger()
    assert [e["version"] for e in ledger] == [0, 1, 2, 3] and all(e["accepted"] for e in ledger)
    assert [e["source"] for e in ledger] == ["baseline", "fallback_patch:v1_patch", "fallback_patch:v2_patch",
                                             "fallback_patch:v3_patch"]
    assert [e["opt"]["composite_score"] for e in entries] == [45.0, 70.0, 88.0]
    assert all(e["autopsy"] for e in ledger)  # opt autopsy stored on every accepted entry
    assert (root / "workspace" / "candidate_pipeline.py").read_text() == (root / "workspace/versions/v3.py").read_text()
    # holdout always evaluated without diagnostics
    assert all(d is False for _, seeds, d in ev.calls if seeds == HOLDOUT_SEEDS)
    assert all(d is True for _, seeds, d in ev.calls if seeds == OPT_SEEDS)
    playbook = brain.read_playbook("test")
    assert "v1 (fallback_patch:v1_patch): opt RMSE 40.0->12.0 mm" in playbook and "patch 3 verified rule" in playbook
    agents = {e["agent"] for e in events}
    assert {"Root Architect", "CV Specialist", "Excavation Control Specialist", "Independent Verifier"} <= agents
    assert brain.stats()["n_versions"] == 4


def test_regression_is_rejected_and_graveyarded(world):
    orch, brain, _, _, root, patches = world
    orch.ensure_baseline()
    Path(patches[0]["path"]).write_text(PIPE_TMPL.format(tag="SCORE=30/10/20"))  # opt up, holdout down
    entry = orch.run_generation(1)
    assert entry["accepted"] is False and "holdout regressed" in entry["reasons"]
    assert brain.best_accepted()["version"] == 0
    assert "baseline" in (root / "workspace" / "candidate_pipeline.py").read_text() or \
        "SCORE=20/18/40" in (root / "workspace" / "candidate_pipeline.py").read_text()
    assert "g1 fallback_patch:v1_patch" in brain.read_playbook("test").split("## 4.")[1]
    Path(patches[1]["path"]).write_text(PIPE_TMPL.format(tag="SCORE=19/18/40"))  # next generation: opt regresses
    with pytest.raises(RuntimeError, match="generation 2 rejected"):
        orch.run(1, verify_improvement=True)


def test_deterministic_contract_from_autopsy(world):
    orch, brain, _, _, _, _ = world
    base = orch.ensure_baseline()
    c = orch.architect(1, base, brain.read_playbook("t"))
    assert c["target_rmse_mm"] == 20.0 and c["max_force_cutoff_n"] == 18.0 and c["latency_ms"] == 25.0
    assert c["contract_id"].startswith("g1-") and c == orch.architect(1, base, "")
    assert json.loads(json.dumps(c)) == c


def test_llm_failure_retries_with_fallback(world, monkeypatch):
    orch, brain, ev, _, _, _ = world
    orch.ensure_baseline()
    bad_llm = PIPE_TMPL.format(tag="SCORE=10/10/50")
    monkeypatch.setattr(orch, "specialists", lambda *a: (bad_llm, "llm:stub"))
    entry = orch.run_generation(1)
    sources = [e["source"] for e in brain.load_ledger()]
    assert sources == ["baseline", "llm:stub", "fallback_patch:v1_patch"]
    assert entry["accepted"] and entry["version"] == 2
    import hashlib
    for e in brain.load_ledger():
        assert hashlib.sha256((orch.root / e["path"]).read_bytes()).hexdigest() == e["sha256"], e["path"]
