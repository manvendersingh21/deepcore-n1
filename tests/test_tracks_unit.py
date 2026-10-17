"""Track-wiring unit tests: --goal/DoD compilation, DECISION ledger fields, evolved prompts, drift probe.

Fully offline: evaluation is a fake scorer driven by docstring tags, CRUSOE_API_KEY is removed, no MuJoCo.
"""
from __future__ import annotations

import fcntl
import hashlib
import io
import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest

from agents.contract_orchestrator import ContractOrchestrator, HOLDOUT_SEEDS, OPT_SEEDS
from agents.crusoe_client import CrusoeClient, FallbackSynthesizer
from shared_brain.memory_store import SharedBrain, estimate_tokens
import main as cli

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
            "peak_core_contact_force_n": force, "max_wrist_force_n": force, "core_shattered": False,
            "core_extracted": True, "extraction_rate": 1.0, "vision_latency_p99_ms": latency, "fps": 50.0}


def _law5(opt, hold, best_opt, best_hold):
    """Same rule shape as core.verifier_harness.check_contract, kept local so the test needs no MuJoCo."""
    reasons = []
    if best_opt is not None and not opt["composite_score"] > best_opt:
        reasons.append(f"opt composite {opt['composite_score']} not > {best_opt}")
    if best_hold is not None and not hold["composite_score"] >= best_hold:
        reasons.append(f"holdout composite {hold['composite_score']} < {best_hold}")
    if opt["max_wrist_force_n"] >= 28.0 or hold["max_wrist_force_n"] >= 28.0:
        reasons.append("max wrist force >= 28 N")
    if opt["vision_latency_p99_ms"] >= 25.0 or hold["vision_latency_p99_ms"] >= 25.0:
        reasons.append("vision latency p99 >= 25 ms")
    if opt["core_shattered"] or hold["core_shattered"]:
        reasons.append("core shattered")
    return (not reasons), reasons


class FakeEvaluator:
    """Scores a candidate by the `SCORE=<opt>/<holdout>/<rmse>` tag in its docstring; records every call."""

    def __init__(self):
        self.calls: list[tuple[str, tuple, bool]] = []

    def __call__(self, path, seeds, diagnostics=True):
        self.calls.append((Path(path).name, tuple(seeds), diagnostics))
        tag = Path(path).read_text().split('"""')[1]
        opt, hold, rmse = (float(x) for x in tag.split("=")[1].split("/"))
        score = hold if tuple(seeds) == HOLDOUT_SEEDS else opt
        res = {"aggregate": _agg(score, rmse), "per_seed": [{"seed": s, **_agg(score, rmse)} for s in seeds]}
        if diagnostics:
            res["autopsy"] = {s: {"worst_frame": 12, "rmse_mm": rmse} for s in seeds}
        return {"ok": True, "result": res, "sandbox": "fake", "wall_s": 0.0}


class StubClient(CrusoeClient):
    reply: str | None = None

    @property
    def available(self) -> bool:
        return True

    def chat(self, messages, temperature=0.2, max_tokens=4096, timeout=60):
        return self.reply


@pytest.fixture()
def world(tmp_path, monkeypatch):
    monkeypatch.delenv("CRUSOE_API_KEY", raising=False)
    root = tmp_path / "repo"
    (root / "workspace").mkdir(parents=True)
    (root / "workspace" / "baseline_v0.py").write_text(PIPE_TMPL.format(tag="SCORE=20/18/40"))
    patches = []
    for i, (tag, owner) in enumerate((("SCORE=45/40/12", "cv"), ("SCORE=70/66/6", "control"),
                                      ("SCORE=88/86/4", "both")), start=1):
        p = root / "patches" / f"v{i}.py"
        p.parent.mkdir(exist_ok=True)
        p.write_text(PIPE_TMPL.format(tag=tag))
        patches.append({"name": f"v{i}_patch", "path": str(p), "specialist": owner, "summary": f"patch {i}",
                        "playbook": {2 if owner == "cv" else 3: [f"patch {i} verified rule"]}})
    brain = SharedBrain(tmp_path / "brain")
    client = CrusoeClient()
    synth = FallbackSynthesizer(client, brain, patches=patches, validator=lambda s: [])
    logs: list[str] = []
    orch = ContractOrchestrator(brain, client, log_fn=logs.append, evaluator=FakeEvaluator(),
                                checker=_law5, synthesizer=synth, root=root)
    return orch, brain, logs, root, patches


# ---------------------------------------------------------------- goal compilation (track 1)
def test_compile_goal_fallback_writes_dod(world):
    orch, brain, _, root, _ = world
    dod = orch.compile_goal("extract the core intact")
    path = root / "shared_brain" / "generated_dod.json"
    assert path.exists() and json.loads(path.read_text()) == dod
    assert dod["goal"] == "extract the core intact" and dod["source"] == "fallback"
    assert {(a["seeds"], a["metric"], a["op"]) for a in dod["assertions"]} >= {
        ("holdout", "core_shattered", "=="), ("holdout", "core_extracted", "=="),
        ("holdout", "composite_score", ">"), ("opt", "vision_latency_p99_ms", "<")}


def test_compile_goal_keeps_only_whitelisted_llm_assertions(world):
    orch = world[0]
    orch.client = StubClient(api_key="k", model="stub-model")
    orch.client.reply = json.dumps({"assertions": [
        {"seeds": "opt", "metric": "vision_3d_rmse_mm", "op": "<", "value": 8.0},
        {"seeds": "opt", "metric": "rmse_mm", "op": "<", "value": 8.0},
        {"seeds": "train", "metric": "composite_score", "op": ">", "value": 90},
        {"seeds": "holdout", "metric": "composite_score", "op": "~", "value": 90}]})
    dod = orch.compile_goal("maximize extraction")
    assert dod["source"] == "llm:stub-model"
    kept = [a for a in dod["assertions"] if a["metric"] == "vision_3d_rmse_mm"]
    assert kept == [{"seeds": "opt", "metric": "vision_3d_rmse_mm", "op": "<", "value": 8.0}]
    assert len(dod["assertions"]) == 5  # 1 valid LLM assertion + the 4 mandatory spec assertions


# ---------------------------------------------------------------- decisions + ledger fields (tracks 1+2)
def test_goal_driven_run_decisions_and_early_stop(world):
    orch, brain, logs, _, _ = world
    orch.compile_goal("clean dig: holdout composite above 85")
    entries = orch.run(5)
    assert [e["decision"] for e in entries] == ["CONTINUE", "CONTINUE", "STOP_DONE"]
    assert len(entries) == 3  # STOP_DONE halts before generations 4 and 5
    ledger = brain.load_ledger()
    assert [e["version"] for e in ledger] == [0, 1, 2, 3] and all(e["accepted"] for e in ledger)
    for e in ledger:
        assert e["decision"] in ContractOrchestrator.DECISIONS
        assert isinstance(e["conflicts_prevented"], int)
        assert isinstance(e["context_efficiency_gain"], int)
    tokens = [e["shared_brain_tokens"] for e in ledger]
    assert tokens == sorted(tokens)  # snapshot per entry, non-decreasing as the playbook grows
    assert tokens[-1] == estimate_tokens(brain.playbook_path.read_text())
    assert any("STOP_DONE" in m for m in logs)  # decide() verdicts printed per generation


def test_rejected_attempt_gets_replan_or_rollback_decision(world):
    orch, brain, _, _, patches = world
    orch.compile_goal("extract the core")
    orch.ensure_baseline()
    Path(patches[0]["path"]).write_text(PIPE_TMPL.format(tag="SCORE=45/10/12"))  # opt up, holdout down
    entry = orch.run_generation(1)
    assert entry["accepted"] is False and entry["decision"] == "REPLAN"
    rejected = [e for e in brain.load_ledger() if not e["accepted"]]
    assert len(rejected) == 1 and rejected[0]["decision"] in ("REPLAN", "ROLLBACK_DRIFT")


def test_opt_regression_gets_rollback_drift_decision(world):
    orch, brain, _, _, patches = world
    orch.compile_goal("extract the core")
    orch.ensure_baseline()
    Path(patches[0]["path"]).write_text(PIPE_TMPL.format(tag="SCORE=15/8/12"))  # composite regresses
    entry = orch.run_generation(1)
    assert entry["decision"] == "ROLLBACK_DRIFT" and entry["accepted"] is False
    assert brain.best_accepted()["version"] == 0  # drift never promoted


def test_drift_probe_records_rejected_negative_control(world):
    orch, brain, _, root, _ = world
    orch.compile_goal("extract the core")
    entries = orch.run(3)
    assert all(e["accepted"] for e in entries)
    probe = orch.drift_probe()
    assert probe is not None and probe["accepted"] is False
    assert probe["source"] == "drift_probe:baseline_v0"
    assert probe["decision"] in ("ROLLBACK_DRIFT", "REPLAN")
    assert brain.best_accepted()["version"] == 3
    best_src = (root / "workspace" / "versions" / "v3.py").read_text()
    assert (root / "workspace" / "candidate_pipeline.py").read_text() == best_src  # rejection never promoted


# ---------------------------------------------------------------- evolved prompts (track 3)
def test_evolved_agent_prompts_versioned_per_agent(world):
    orch, brain, _, _, _ = world
    orch.compile_goal("extract the core")
    orch.run(2)
    store = json.loads((brain.root / "evolved_agent_prompts.json").read_text())
    assert set(store) == {"RootArchitect", "VisionSpecialist", "ControlSpecialist", "IndependentVerifier"}
    for agent, versions in store.items():
        assert [v["version"] for v in versions] == [1, 2] and all("guidance" in v for v in versions)
    assert "12.0" in store["VisionSpecialist"][0]["guidance"]  # measured RMSE of v1, not a hardcoded number
    assert all("updated" in v for versions in store.values() for v in versions)


def test_prompt_only_atomic_update_leaves_playbook_and_counts_conflicts(world):
    brain = world[1]
    before = brain.playbook_path.read_text()
    out = brain.atomic_update("RootArchitect", {"evolved_prompt": {"guidance": "g1"}})
    assert out == {"merged": True, "contended": 0}
    assert brain.playbook_path.read_text() == before  # prompt-only delta never rewrites the playbook
    store = json.loads((brain.root / "evolved_agent_prompts.json").read_text())
    assert store["RootArchitect"][0]["version"] == 1
    lock = open(str(brain.playbook_path) + ".lock", "a+")  # simulate a second agent holding the lock
    fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
    try:
        out = brain.atomic_update("VisionSpecialist", {"evolved_prompt": {"guidance": "g2"}})
    finally:
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        lock.close()
    assert out["contended"] == 1 and brain.conflicts_prevented() == 1


# ---------------------------------------------------------------- context efficiency (track 2)
def test_context_efficiency_gain_from_transaction_log(tmp_path):
    brain = SharedBrain(tmp_path / "b")
    assert brain.context_efficiency_gain() == 0
    brain.read_playbook("RootArchitect")
    assert brain.context_efficiency_gain() == 0  # reads alone deposit nothing
    brain.update_playbook("RootArchitect", 2, ["depth IQR-median beats single pixel"])
    gain1 = brain.context_efficiency_gain()
    assert gain1 > 0
    brain.update_playbook("ControlSpecialist", 3, ["top-down rubble ordering"])
    gain2 = brain.context_efficiency_gain()
    assert gain2 > gain1  # computed from the log, so it moves as the log grows
    with sqlite3.connect(brain.db_path) as con:
        rows = con.execute("SELECT op, tokens FROM transactions WHERE target='playbook.md'").fetchall()
    expected = max(sum(t for op, t in rows if op == "write") - sum(t for op, t in rows if op == "read"), 0)
    assert gain2 == expected


# ---------------------------------------------------------------- CLI wiring (main.py)
def _render(table) -> str:
    from rich.console import Console
    con = Console(file=io.StringIO(), width=250)
    con.print(table)
    return con.file.getvalue()


def test_cli_goal_flag_and_defaults():
    args = cli.build_parser().parse_args(["run-evolution", "--goal", "dig safely"])
    assert args.goal == "dig safely" and args.iterations == 3 and not args.verify_improvement
    assert "holdout composite > 85" in cli.DEFAULT_GOAL


def test_evolution_session_offline(world, capsys):
    orch, brain, _, root, _ = world
    entries = cli._evolution_session(orch, SimpleNamespace(goal="clean dig", iterations=2,
                                                           verify_improvement=False))
    assert [e["version"] for e in entries] == [1, 2]
    dod = json.loads((root / "shared_brain" / "generated_dod.json").read_text())
    assert dod["goal"] == "clean dig"
    out = capsys.readouterr().out
    assert "DECISION" in out and "generated_dod.json" in out
    # drift probe fired because no generation was rejected
    assert any(e["source"] == "drift_probe:baseline_v0" and not e["accepted"]
               for e in brain.load_ledger())


def test_holdout_before_after_table(world):
    orch, brain, _, _, _ = world
    orch.compile_goal("extract the core")
    orch.run(3)
    text = _render(cli._holdout_table(brain.load_ledger()))
    assert "101-103" in text and "v0 baseline" in text and "v3" in text
    assert "composite_score" in text and "+68.00" in text  # 18.0 -> 86.0 on holdout
    assert "core_extracted" in text
    assert cli._holdout_table([]) is None


def test_reevaluates_holdout_without_autopsy(world):
    orch, _, _, _, _ = world
    orch.compile_goal("extract the core")
    orch.run(1)
    seeds = {(name, tuple(s)) for name, s, _ in orch.evaluator.calls}
    assert ("candidate_g1.py", OPT_SEEDS) in seeds
    assert ("candidate_g1.py", HOLDOUT_SEEDS) in seeds


def test_each_agent_consumes_latest_guidance_with_bounded_context(world, monkeypatch):
    orch, brain, _, _, _ = world
    orch.compile_goal('safe extraction')
    base = orch.ensure_baseline()
    for agent in ('RootArchitect', 'VisionSpecialist', 'ControlSpecialist', 'IndependentVerifier'):
        brain.atomic_update(agent, {'evolved_prompt': {'guidance': f'LATEST_{agent}'}})
    client = StubClient(api_key='k', model='stub')
    calls = []
    def chat(messages, **kwargs):
        calls.append(messages)
        return '{}'
    monkeypatch.setattr(client, 'chat', chat)
    orch.client = client
    proposal_prompts = []
    def edits(system, user, **kwargs):
        proposal_prompts.append(user)
        return None
    monkeypatch.setattr(orch.synth, 'try_edits', edits)
    orch.run_generation(1)
    assert any('LATEST_RootArchitect' in message['content'] for call in calls for message in call)
    assert 'LATEST_VisionSpecialist' in proposal_prompts[0]
    assert 'LATEST_ControlSpecialist' in proposal_prompts[1]
    with sqlite3.connect(brain.db_path) as db:
        rows = db.execute("SELECT agent,detail_json FROM transactions WHERE target='evolved_agent_prompts.json' AND op='read'").fetchall()
        receipts = db.execute("SELECT agent FROM transactions WHERE target='operation_receipt'").fetchall()
    assert {agent for agent, _ in rows} >= {'RootArchitect', 'VisionSpecialist', 'ControlSpecialist', 'IndependentVerifier'}
    assert all(json.loads(detail)['context_tokens'] < 1500 for _, detail in rows)
    assert all(json.loads(detail)['prompt_version'] == 1 for _, detail in rows if json.loads(detail)['prompt_version'] is not None)
    assert {agent for agent, in receipts} >= {'RootArchitect', 'VisionSpecialist', 'ControlSpecialist', 'IndependentVerifier'}


def test_agent_exception_still_logs_memory_update_and_error_receipt(world, monkeypatch):
    orch, brain, _, _, _ = world
    base = orch.ensure_baseline()
    orch.client = StubClient(api_key='k')
    def broken(*args, **kwargs):
        raise RuntimeError('proposal failed')
    monkeypatch.setattr(orch.client, 'chat', broken)
    with pytest.raises(RuntimeError, match='proposal failed'):
        orch.architect(1, base, '')
    with sqlite3.connect(brain.db_path) as db:
        agent, detail = db.execute("SELECT agent,detail_json FROM transactions WHERE target='operation_receipt' ORDER BY id DESC LIMIT 1").fetchone()
    assert agent == 'RootArchitect'
    assert json.loads(detail)['outcome'] == 'error'
    assert json.loads(detail)['operation'] == 'architect'


def test_orchestrator_dod_is_frozen_defensive_and_detects_disk_changes(world):
    orch, brain, _, root, _ = world
    original = orch.compile_goal('original goal')
    copy = orch.dod
    copy['assertions'].clear()
    assert orch.dod == original
    changed = dict(original, goal='changed goal')
    with pytest.raises(RuntimeError, match='already frozen'):
        orch.dod = changed
    assert orch.compile_goal('replacement goal') == original
    path = root / 'shared_brain/generated_dod.json'
    path.write_text(json.dumps(changed))
    with pytest.raises(RuntimeError, match='changed after'):
        orch._goal_met({'opt': _agg(90, 4), 'holdout': _agg(90, 4)})


def test_existing_ledger_reuses_mission_instead_of_recompiling(world):
    orch, brain, _, root, _ = world
    original = orch.compile_goal('original goal')
    orch.ensure_baseline()
    reloaded = ContractOrchestrator(brain, CrusoeClient(), root=root, evaluator=FakeEvaluator(), checker=_law5)
    assert reloaded.compile_goal('new request cannot replace active mission') == original
    assert json.loads((root / 'shared_brain/generated_dod.json').read_text()) == original


def test_persistent_mission_anchor_rejects_replacement_after_restart(world):
    orch, brain, _, root, _ = world
    original = orch.compile_goal('original goal')
    orch.ensure_baseline()
    path = root / 'shared_brain/generated_dod.json'
    path.write_text(json.dumps(dict(original, goal='weaker replacement goal')))
    reloaded = ContractOrchestrator(brain, CrusoeClient(), root=root, evaluator=FakeEvaluator(), checker=_law5)
    with pytest.raises(RuntimeError, match='persistent campaign anchor'):
        reloaded.compile_goal('new request')


def test_existing_ledger_without_mission_fails_closed(world):
    orch, brain, _, root, _ = world
    orch.compile_goal('original goal')
    orch.ensure_baseline()
    (root / 'shared_brain/generated_dod.json').unlink()
    reloaded = ContractOrchestrator(brain, CrusoeClient(), root=root, evaluator=FakeEvaluator(), checker=_law5)
    with pytest.raises(RuntimeError, match='anchor exists but generated_dod'):
        reloaded.compile_goal('replacement must fail')


def test_legacy_mission_migration_records_anchor_without_rewriting_ledger(world):
    from agents.mission_contract import normalize_goal_assertions

    orch, brain, _, root, _ = world
    path = root / 'shared_brain/generated_dod.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    document = {'goal': 'historical mission', 'source': 'legacy',
                'assertions': normalize_goal_assertions([])}
    path.write_text(json.dumps(document))
    brain.append_ledger({'version': 4, 'accepted': True, 'source': 'legacy', 'sha256': 'a' * 64,
                         'path': 'workspace/versions/v4.py'})
    ledger_bytes = brain.ledger_path.read_bytes()
    reloaded = ContractOrchestrator(brain, CrusoeClient(), root=root, evaluator=FakeEvaluator(), checker=_law5)
    assert reloaded.compile_goal('must retain historical mission') == document
    assert brain.ledger_path.read_bytes() == ledger_bytes
    anchor = brain.mission_anchor()
    assert anchor['source'] == 'legacy_migration'
    assert anchor['legacy_ledger_sha256'] == hashlib.sha256(ledger_bytes).hexdigest()
    assert anchor['legacy_entry_count'] == 1 and anchor['legacy_max_version'] == 4


def test_legacy_ledger_without_mission_is_not_recompiled(world):
    orch, brain, _, root, _ = world
    brain.append_ledger({'version': 2, 'accepted': True, 'source': 'legacy', 'sha256': 'a' * 64,
                         'path': 'workspace/versions/v2.py'})
    reloaded = ContractOrchestrator(brain, CrusoeClient(), root=root, evaluator=FakeEvaluator(), checker=_law5)
    with pytest.raises(RuntimeError, match='existing ledger has no generated_dod'):
        reloaded.compile_goal('do not replace missing mission')


def test_compilation_keeps_exact_mandatory_threshold_despite_weaker_llm_clause(world):
    orch = world[0]
    orch.client = StubClient(api_key='k', model='stub')
    orch.client.reply = json.dumps({'assertions': [{'seeds': 'holdout', 'metric': 'composite_score', 'op': '>', 'value': 0}]})
    dod = orch.compile_goal('weak suggestion')
    assert {'seeds': 'holdout', 'metric': 'composite_score', 'op': '>', 'value': 85.0} in dod['assertions']
    assert not orch._goal_met({'opt': _agg(90, 4), 'holdout': _agg(70, 4)})
