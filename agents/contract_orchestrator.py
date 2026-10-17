"""Recursive contract tree per generation: Root Architect -> [CV Specialist, Excavation Control Specialist] -> Verifier.

The architect turns the latest optimization-seed autopsy into a sub-contract, both specialists propose bounded
source edits (LLM via Crusoe, or a complete curated fallback patch), the parent merges them sequentially, and the
independent verifier evaluates the merge in the sandbox on optimization + held-out seeds and applies Law 5.
"""
from __future__ import annotations

from contextlib import contextmanager
from functools import wraps
import difflib
import hashlib
import json
import os
import re
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

from agents.crusoe_client import CrusoeClient, FallbackSynthesizer, estimate_tokens
from agents.mission_contract import FrozenMission, normalize_goal_assertions

REPO_ROOT = Path(__file__).resolve().parents[1]
OPT_SEEDS: tuple[int, ...] = (11, 22, 33)
HOLDOUT_SEEDS: tuple[int, ...] = (101, 102, 103)
FRACTURE_N = 18.0
LATENCY_CONTRACT_MS = 25.0

ARCHITECT, CV, CONTROL, VERIFIER = "Root Architect", "CV Specialist", "Excavation Control Specialist", "Independent Verifier"
PROMPT_AGENTS = {"RootArchitect": ARCHITECT, "VisionSpecialist": CV,
                 "ControlSpecialist": CONTROL, "IndependentVerifier": VERIFIER}
Evaluator = Callable[..., dict]  # (path, seeds, diagnostics) -> {"ok", "result"|"error", ...}
Checker = Callable[[dict, dict, Optional[float], Optional[float]], tuple]  # check_contract(opt, hold, best_opt, best_hold)

CANDIDATE_RULES = (
    "The resulting Python file must define `class ExcavationPipeline` with "
    "__init__(self, cam_intrinsics, cam_pos, cam_rot) and step(self, t_sec, rgb, depth, proprio) -> dict "
    "{'estimated_objects': {name: {'pos', 'yaw'}}, 'action': {'target_pos', 'target_yaw', 'gripper'}, "
    "'debug_overlay': {'bboxes', 'phase'}}. Only numpy/scipy/cv2/math imports. No os/sys/io/subprocess/mujoco/core "
    "imports, no open/eval/exec/getattr, no dunder access, no file IO. Back-projection: p_world = cam_pos + cam_rot @ "
    "[(u-cx)/fx*d, (v-cy)/fy*d, d]. Depth has NaN dropouts, 0.25/1.80 m spikes and a dust band in frames 40..70. "
    f"Peak core contact force must stay below {FRACTURE_N} N; step() p99 latency below {LATENCY_CONTRACT_MS} ms."
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as fh:
            fh.write(text)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def _default_evaluator(path, seeds, diagnostics=True) -> dict:
    from agents.sandbox_exec import run_candidate

    return run_candidate(path, seeds, diagnostics=diagnostics)


def _default_checker(opt, holdout, best_opt, best_holdout):
    from core.verifier_harness import check_contract

    return check_contract(opt, holdout, best_opt, best_holdout)


FALLBACK_DOD = [
    {"seeds": "holdout", "metric": "core_shattered", "op": "==", "value": False},
    {"seeds": "holdout", "metric": "core_extracted", "op": "==", "value": True},
    {"seeds": "holdout", "metric": "composite_score", "op": ">", "value": 85.0},
    {"seeds": "opt", "metric": "vision_latency_p99_ms", "op": "<", "value": 25.0},
]
_METRICS = {"composite_score", "vision_3d_rmse_mm", "rubble_cleared_count", "peak_core_contact_force_n",
            "max_wrist_force_n", "core_shattered", "core_extracted", "extraction_rate", "vision_latency_p99_ms"}


def _valid_assertion(a) -> bool:
    return (isinstance(a, dict) and a.get("seeds") in ("opt", "holdout") and a.get("metric") in _METRICS
            and a.get("op") in _OPS and isinstance(a.get("value"), (int, float, bool)))


def _parse_json_object(text: str):
    import re
    m = re.search(r"\{.*\}", text, re.S)
    try:
        return json.loads(m.group(0)) if m else None
    except json.JSONDecodeError:
        return None


_OPS = {"<": lambda a, b: a < b, "<=": lambda a, b: a <= b, ">": lambda a, b: a > b,
        ">=": lambda a, b: a >= b, "==": lambda a, b: a == b}


def _check_assertion(a: dict, opt: dict, hold: dict) -> bool:
    """One DoD assertion from generated_dod.json: {"seeds": "opt"|"holdout", "metric", "op", "value"}.
    Only whitelisted metric names and comparison operators are read; nothing is executed."""
    agg = opt if a.get("seeds") == "opt" else hold
    if a.get("metric") not in agg or a.get("op") not in _OPS:
        return False
    return bool(_OPS[a["op"]](agg[a["metric"]], a["value"]))


def _num(agg: Optional[dict], key: str, default: float = float("nan")) -> float:
    try:
        return float((agg or {}).get(key, default))
    except (TypeError, ValueError):
        return default


def _fmt(x: float, nd: int = 1) -> str:
    return "n/a" if x != x else f"{x:.{nd}f}"


def _agent_action(agent):
    def decorate(fn):
        @wraps(fn)
        def wrapped(self, *args, **kwargs):
            with self.agent_operation(agent, fn.__name__):
                return fn(self, *args, **kwargs)
        return wrapped
    return decorate


class ContractOrchestrator:
    """Runs and records generations of the recursive agent tree against the shared brain."""

    def __init__(self, brain, client: Optional[CrusoeClient] = None, log_fn: Callable[[str], None] = print,
                 event_fn: Optional[Callable[[dict], None]] = None, evaluator: Optional[Evaluator] = None,
                 checker: Optional[Checker] = None, synthesizer: Optional[FallbackSynthesizer] = None,
                 root: Path | str = REPO_ROOT):
        self.brain = brain
        self.client = client if client is not None else CrusoeClient()
        self.log_fn = log_fn
        self.event_fn = event_fn
        self.evaluator = evaluator or _default_evaluator
        self.checker = checker or _default_checker
        self.synth = synthesizer or FallbackSynthesizer(self.client, brain)
        self.root = Path(root)
        self.workspace = self.root / "workspace"
        self.versions = self.workspace / "versions"
        self.generation = 0
        self.failure_modes: list[str] = []
        self._mission = None
        self._operation_context = {}

    @property
    def dod(self):
        return self._mission.snapshot() if self._mission else None

    @dod.setter
    def dod(self, document):
        proposed = FrozenMission(document)
        if self._mission is not None:
            if proposed.digest != self._mission.digest:
                raise RuntimeError("mission contract is already frozen")
            self._assert_mission()
            return
        path = self.root / "shared_brain" / "generated_dod.json"
        self._mission = FrozenMission(document, path if path.exists() else None)

    def _assert_mission(self):
        if self._mission is not None:
            path = self.root / "shared_brain" / "generated_dod.json"
            self._mission.assert_unchanged(path)
            self.brain.require_mission_anchor(self._mission.digest)

    @contextmanager
    def agent_operation(self, agent, operation):
        canonical = next((key for key, display in PROMPT_AGENTS.items() if display == agent), agent)
        context = self.brain.read_context(canonical, max_tokens=1100)
        # Mutation prompts consume optimization evidence; holdout stays in the fixed gate.
        context = re.sub(r",? holdout composite [^\n]*", "", context)
        path = self.brain.root / "evolved_agent_prompts.json"
        guidance, version = "", None
        try:
            store = json.loads(path.read_text()) if path.exists() else {}
            versions = store.get(canonical, [])
            if versions:
                latest = versions[-1]
                guidance = str(latest.get("guidance", ""))[:1200]
                version = latest.get("version")
        except (OSError, ValueError, TypeError, AttributeError) as exc:
            self.brain.log_transaction(canonical, "read", "prompt_error", 0, {"error": str(exc)})
        suffix = f"\nEvolved guidance (v{version}):\n{guidance}" if guidance else ""
        combined = (context + suffix)[:1499 * 4]
        self.brain.log_transaction(canonical, "read", "evolved_agent_prompts.json", estimate_tokens(guidance),
                                   {"operation": operation, "prompt_version": version,
                                    "context_tokens": estimate_tokens(combined),
                                    "guidance_sha256": _sha(guidance) if guidance else None})
        prior = self._operation_context.get(agent)
        self._operation_context[agent] = combined
        outcome = "error"
        try:
            yield combined
            outcome = "completed"
        finally:
            self._operation_context[agent] = prior
            self.brain.atomic_update(canonical, {})
            self.brain.log_transaction(canonical, "write", "operation_receipt", 0,
                                       {"operation": operation, "outcome": outcome, "prompt_version": version})


    # ------------------------------------------------------------------ plumbing
    def _emit(self, agent: str, status: str, message: str) -> None:
        self.log_fn(f"[g{self.generation}] {agent:<30} {status:<9} {message}")
        if self.event_fn is not None:
            try:
                self.event_fn({"agent": agent, "status": status, "message": message, "generation": self.generation})
            except Exception as exc:  # the HUD must never break the evolution loop
                self.log_fn(f"event_fn error: {exc}")

    def _rel(self, path: Path) -> str:
        try:
            return str(path.relative_to(self.root))
        except ValueError:
            return str(path)

    def _abs(self, rel: str) -> Path:
        p = Path(rel)
        return p if p.is_absolute() else self.root / p

    @_agent_action(VERIFIER)
    def _evaluate(self, path: Path) -> tuple[Optional[dict], Optional[dict], str]:
        """Opt seeds with diagnostics, holdout without. Returns (opt_result, holdout_result, error)."""
        self._assert_mission()
        opt = self.evaluator(path, OPT_SEEDS, diagnostics=True)
        if not opt.get("ok"):
            return None, None, f"opt evaluation failed: {opt.get('error')}"
        hold = self.evaluator(path, HOLDOUT_SEEDS, diagnostics=False)
        if not hold.get("ok"):
            return opt["result"], None, f"holdout evaluation failed: {hold.get('error')}"
        return opt["result"], hold["result"], ""

    def _entry(self, version: int, source: str, contract_id: str, opt: dict, hold: dict, accepted: bool,
               reasons: list, code: str, path: Path) -> dict:
        return {"version": version, "source": source, "contract_id": contract_id,
                "opt": opt.get("aggregate", {}), "holdout": hold.get("aggregate", {}), "accepted": bool(accepted),
                "reasons": [str(r) for r in reasons], "timestamp": _now(), "sha256": _sha(code),
                "path": self._rel(path), "autopsy": opt.get("autopsy", {}), "generation": self.generation,
                "shared_brain_tokens": self.brain.stats()["playbook_tokens"],
                "conflicts_prevented": self.brain.conflicts_prevented(),
                "context_efficiency_gain": self.brain.context_efficiency_gain()}

    # ------------------------------------------------------------------ baseline
    def ensure_baseline(self) -> dict:
        """Evaluate and register workspace/baseline_v0.py as version 0 when the ledger is empty."""
        with self.brain.campaign_lock():
            ledger = self.brain.load_ledger()
            if ledger:
                return self.brain.best_accepted() or ledger[0]
            base = self.workspace / "baseline_v0.py"
            code = base.read_text()
            self._emit(VERIFIER, "running", "evaluating baseline_v0 on opt + holdout seeds")
            opt, hold, err = self._evaluate(base)
            if err:
                self._emit(VERIFIER, "failed", err)
                raise RuntimeError(f"baseline evaluation failed: {err}")
            v0 = self.versions / "v0.py"
            atomic_write(v0, code)
            atomic_write(self.workspace / "candidate_pipeline.py", code)
            entry = self._entry(0, "baseline", "baseline", opt, hold, True, ["baseline reference"], code, v0)
            entry["decision"] = self.decide(0, entry, entry, self.failure_modes)["decision"]
            self.brain.append_ledger(entry)
            self.brain.record_version(entry, "", opt.get("per_seed", []), hold.get("per_seed", []))
            self._emit(VERIFIER, "accepted", f"v0 baseline composite {_fmt(_num(entry['opt'], 'composite_score'), 3)}")
            return entry

    # ------------------------------------------------------------------ architect
    def derive_contract(self, generation: int, best: dict) -> dict:
        """Deterministic sub-contract from the best version's measured opt metrics."""
        agg = best.get("opt", {})
        rmse = _num(agg, "vision_3d_rmse_mm")
        cleared = _num(agg, "rubble_cleared_count", 0.0)
        force = _num(agg, "peak_core_contact_force_n", 0.0)
        target = round(max(3.0, 0.5 * rmse), 1) if rmse == rmse else 10.0
        seed = f"{generation}:{best.get('sha256', '')}:{target}"
        return {"contract_id": f"g{generation}-{_sha(seed)[:8]}", "target_rmse_mm": target,
                "require_top_down_rubble_ordering": bool(cleared < 2.0 or force >= FRACTURE_N),
                "max_force_cutoff_n": FRACTURE_N, "latency_ms": LATENCY_CONTRACT_MS,
                "baseline_composite": _num(agg, "composite_score"), "origin": "deterministic"}

    @_agent_action(ARCHITECT)
    def architect(self, generation: int, best: dict, playbook: str) -> dict:
        playbook = self._operation_context.get(ARCHITECT) or playbook
        contract = self.derive_contract(generation, best)
        if self.client.available:
            sys_p = ("You are the Root Architect of a recursive robot-excavation agent tree. Reply with ONE JSON object "
                     "with keys contract_id, target_rmse_mm, require_top_down_rubble_ordering, max_force_cutoff_n, "
                     "latency_ms, cv_focus, control_focus.")
            user = (f"Playbook:\n{playbook}\n\nBest opt metrics:\n{json.dumps(best.get('opt', {}), default=str)}\n\n"
                    f"Opt autopsy:\n{json.dumps(best.get('autopsy', {}), default=str)[:6000]}")
            text = self.client.chat([{"role": "system", "content": sys_p}, {"role": "user", "content": user}],
                                    temperature=0.1, max_tokens=600)
            self.brain.log_transaction(ARCHITECT, "write", "contract", tokens=estimate_tokens(sys_p + user + (text or "")),
                                       detail={"llm": text is not None})
            m = re.search(r"\{.*\}", text or "", re.DOTALL)
            try:
                llm = json.loads(m.group(0)) if m else {}
            except json.JSONDecodeError:
                llm = {}
            if llm:
                try:
                    contract["target_rmse_mm"] = max(1.0, float(llm.get("target_rmse_mm", contract["target_rmse_mm"])))
                except (TypeError, ValueError):
                    pass
                contract["require_top_down_rubble_ordering"] = bool(
                    llm.get("require_top_down_rubble_ordering", contract["require_top_down_rubble_ordering"]))
                contract.update({k: str(llm[k])[:400] for k in ("cv_focus", "control_focus") if k in llm})
                contract["origin"] = f"llm:{self.client.model}"
        # Safety limits are never negotiable, whoever wrote the contract.
        contract["max_force_cutoff_n"] = FRACTURE_N
        contract["latency_ms"] = LATENCY_CONTRACT_MS
        return contract

    # ------------------------------------------------------------------ specialists
    def _specialist_prompt(self, role: str, contract: dict, source: str, best: dict, playbook: str) -> tuple[str, str]:
        focus = {CV: "3D object pose estimation from the corrupted overhead RGB-D (RMSE, dust/glare/occlusion "
                     "robustness, latency). Keep the control logic unchanged unless it depends on your estimates.",
                 CONTROL: "excavation ordering (rubble_a off the top first, then rubble_b, then the core to the "
                          "hopper) and force-guarded control from wrist_force. Keep the vision logic intact."}[role]
        system = (f"You are the {role} in a recursive agent tree improving a robot excavation pipeline. {focus} "
                  f"{CANDIDATE_RULES} Return only a JSON object with schema "
                  '{"edits":[{"old":"exact existing source text","new":"replacement source text"}],'
                  '"rationale":"brief measured reason"}. '
                  "Propose one to three small edits. Each old string must occur exactly once in the supplied source; "
                  "preserve whitespace and use JSON escapes for newlines. Edits are applied in order. "
                  "Keep the response under 1400 tokens. Do not return a complete file or Markdown.")
        assigned_focus = contract.get("cv_focus" if role == CV else "control_focus", "")
        user = (f"Sub-contract:\n{json.dumps(contract)}\n\nPlaybook (verified knowledge, obey the graveyard):\n{playbook}\n\n"
                f"Architect's focus for this specialist:\n{assigned_focus or focus}\n\n"
                f"Measured opt metrics of the current best:\n{json.dumps(best.get('opt', {}), default=str)}\n\n"
                f"Opt-seed autopsy:\n{json.dumps(best.get('autopsy', {}), default=str)[:8000]}\n\n"
                f"Current pipeline source:\n```python\n{source}\n```")
        return system, user

    def specialists(self, generation: int, contract: dict, best_src: str, best: dict, playbook: str) -> tuple[Optional[str], str]:
        """Merge bounded CV edits, then Control edits; reject invalid proposals before real evaluation."""
        if not self.client.available:
            return None, ""
        merged, origin = best_src, ""
        for role in (CV, CONTROL):
            self._emit(role, "running", f"bounded LLM edit proposal on {self.client.model}")
            with self.agent_operation(role, "specialist_proposal") as context:
                code = self.synth.try_edits(*self._specialist_prompt(role, contract, merged, best, context),
                                          source=merged, agent=role)
            if code is None:
                self._emit(role, "failed", "; ".join(map(str, self.synth.last_problems))[:200] or "no valid code")
                continue
            merged, origin = code, f"llm:{self.client.model}"
            self._emit(role, "done", f"valid proposal ({len(code.splitlines())} lines)")
        return (merged, origin) if origin else (None, "")

    def fallback(self, generation: int, best: dict) -> tuple[str, str, dict]:
        patch = self.synth.next_patch(generation, best.get("autopsy"))
        src = self.synth.patch_source(patch)
        owner = patch.get("specialist", "both")
        for role, key in ((CV, "cv"), (CONTROL, "control")):
            with self.agent_operation(role, "fallback_patch"):
                mine = owner in (key, "both")
                self._emit(role, "done", f"fallback patch {patch['name']}: {patch.get('summary', '')}" if mine
                           else f"no change (patch {patch['name']} owned by {owner})")
        self.brain.log_transaction("Synthesizer", "read", f"fallback_patch:{patch['name']}", tokens=estimate_tokens(src),
                                   detail={"generation": generation, "specialist": owner})
        return src, f"fallback_patch:{patch['name']}", patch

    # ------------------------------------------------------------------ verifier
    @_agent_action(VERIFIER)
    def _verify(self, generation: int, code: str, origin: str, contract: dict, best: dict) -> tuple[bool, list, dict, dict, Path]:
        self._assert_mission()
        label = "_llm" if origin.startswith("llm:") else ""  # each attempt keeps its own file so ledger sha256 matches
        path = self.versions / f"candidate_g{generation}{label}.py"
        atomic_write(path, code)
        self._emit(VERIFIER, "running", f"{origin}: sandbox eval on opt {OPT_SEEDS} + holdout {HOLDOUT_SEEDS}")
        opt, hold, err = self._evaluate(path)
        if err:
            return False, [err], opt or {}, hold or {}, path
        best_opt = (best.get("opt") or {}).get("composite_score")
        best_hold = (best.get("holdout") or {}).get("composite_score")
        with self.agent_operation(VERIFIER, "fixed_contract_gate"):
            self._assert_mission()
            ok, reasons = self.checker(opt["aggregate"], hold["aggregate"], best_opt, best_hold)
        reasons = list(reasons)
        rmse = _num(opt["aggregate"], "vision_3d_rmse_mm")
        if ok and rmse == rmse and rmse > contract["target_rmse_mm"]:
            reasons.append(f"note: RMSE {rmse:.1f} mm above sub-contract target {contract['target_rmse_mm']} mm")
        return bool(ok), reasons, opt, hold, path

    def _findings(self, version: int, origin: str, best: dict, entry: dict) -> list[str]:
        a, b = best.get("opt", {}), entry["opt"]
        return [f"v{version} ({origin}): opt RMSE {_fmt(_num(a, 'vision_3d_rmse_mm'))}->"
                f"{_fmt(_num(b, 'vision_3d_rmse_mm'))} mm, composite {_fmt(_num(a, 'composite_score'), 3)}->"
                f"{_fmt(_num(b, 'composite_score'), 3)}, holdout composite {_fmt(_num(entry['holdout'], 'composite_score'), 3)}",
                f"v{version}: cleared {_fmt(_num(b, 'rubble_cleared_count'))} rubble, peak core force "
                f"{_fmt(_num(b, 'peak_core_contact_force_n'))} N (<{FRACTURE_N}), latency p99 "
                f"{_fmt(_num(b, 'vision_latency_p99_ms'))} ms, extraction {_fmt(_num(b, 'extraction_rate'), 2)}"]

    def _evolved_prompts(self, generation: int, version: int, contract: dict, entry: dict) -> dict[str, dict]:
        """Prompt deltas per agent for evolved_agent_prompts.json, derived from the contract and measured metrics."""
        rmse, target = _num(entry["opt"], "vision_3d_rmse_mm"), contract.get("target_rmse_mm")
        force, cleared = _num(entry["opt"], "peak_core_contact_force_n"), _num(entry["opt"], "rubble_cleared_count")
        dod = "met" if self._goal_met(entry) else "open"
        basis = {"generation": generation, "basis": f"v{version}", "contract_id": contract["contract_id"]}
        return {
            "RootArchitect": {**basis, "guidance": f"DoD {dod}; next sub-contract target RMSE <= {target} mm, "
                                                    f"top-down rubble ordering={contract.get('require_top_down_rubble_ordering')}"},
            "VisionSpecialist": {**basis, "guidance": f"measured 3D RMSE {_fmt(rmse)} mm vs target {target} mm on "
                                                      f"opt seeds; hold the pose budget next generation"},
            "ControlSpecialist": {**basis, "guidance": f"measured {_fmt(cleared)}/2 rubble cleared, peak core force "
                                                       f"{_fmt(force)} N (<{FRACTURE_N}), extraction rate "
                                                       f"{_fmt(_num(entry['opt'], 'extraction_rate'), 2)}"},
            "IndependentVerifier": {**basis, "guidance": f"v{version} passed Law 5: "
                                                         f"{'; '.join(map(str, entry.get('reasons', [])))[:160] or 'all gates green'}; DoD {dod}"},
        }

    @_agent_action(VERIFIER)
    def _accept(self, generation: int, code: str, origin: str, contract: dict, best: dict, best_src: str,
                opt: dict, hold: dict, reasons: list, patch: Optional[dict]) -> dict:
        with self.brain.campaign_lock():
            version = self.brain.next_version()
            vpath = self.versions / f"v{version}.py"
            atomic_write(vpath, code)
            atomic_write(self.workspace / "candidate_pipeline.py", code)
            diff = "".join(difflib.unified_diff(best_src.splitlines(True), code.splitlines(True),
                                                fromfile=best.get("path", "best"), tofile=self._rel(vpath)))
            findings = self._findings(version, origin, best,
                                      {"opt": opt.get("aggregate", {}), "holdout": hold.get("aggregate", {})})
            owner = (patch or {}).get("specialist", "both")
            sections = {"cv": [2], "control": [3]}.get(owner, [2, 3])
            for sec in sections:
                self.brain.update_playbook(VERIFIER, sec, findings[:1] if sec == 2 else findings[1:])
            for sec, bullets in ((patch or {}).get("playbook") or {}).items():
                self.brain.update_playbook(VERIFIER, int(sec), list(bullets))
            entry = self._entry(version, origin, contract["contract_id"], opt, hold, True, reasons, code, vpath)
            self._emit(VERIFIER, "accepted", findings[0])
            entry["decision"] = self.decide(generation, entry, best, self.failure_modes)["decision"]
            self.brain.append_ledger(entry)
            self.brain.record_version(entry, diff, opt.get("per_seed", []), hold.get("per_seed", []))
        for agent_id, prompt in self._evolved_prompts(generation, version, contract, entry).items():
            self.brain.atomic_update(agent_id, {"evolved_prompt": prompt})
        return entry

    @_agent_action(VERIFIER)
    def _reject(self, generation: int, code: str, origin: str, contract: dict, opt: dict, hold: dict,
                reasons: list, path: Path, best: dict) -> dict:
        with self.brain.campaign_lock():
            version = self.brain.next_version()
            entry = self._entry(version, origin, contract["contract_id"], opt, hold, False, reasons, code, path)
            why = "; ".join(map(str, reasons))[:300] or "contract not met"
            self.brain.update_playbook(VERIFIER, 4, [f"g{generation} {origin} ({contract['contract_id']}): rejected - {why}"])
            self._emit(VERIFIER, "rejected", why)
            entry["decision"] = self.decide(generation, entry, best, self.failure_modes)["decision"]
            self.brain.append_ledger(entry)
            self.brain.record_version(entry, "", opt.get("per_seed", []), hold.get("per_seed", []))
            return entry

    # ------------------------------------------------------------------ generation
    @_agent_action(ARCHITECT)
    def compile_goal(self, goal: str) -> dict:
        """Step 1: compile a one-line goal into shared_brain/generated_dod.json (assertions, machine-checkable).
        Crusoe proposes the assertions; anything outside the whitelist is dropped, and the fixed DoD is the fallback."""
        with self.brain.campaign_lock():
            path = self.root / "shared_brain" / "generated_dod.json"
            ledger = self.brain.load_ledger()
            anchor = self.brain.mission_anchor()
            if self._mission is not None:
                self._assert_mission()
                return self.dod
            if anchor is not None:
                if not path.exists():
                    raise RuntimeError("persistent mission anchor exists but generated_dod.json is missing")
                mission = FrozenMission(json.loads(path.read_text(encoding="utf-8")), path)
                self.brain.require_mission_anchor(mission.digest)
                self._mission = mission
                return self.dod
            if ledger and not path.exists():
                raise RuntimeError("existing ledger has no generated_dod.json; refusing to replace the campaign mission")

            if path.exists():
                dod = json.loads(path.read_text(encoding="utf-8"))
                source = dod.get("source", "existing")
                anchor_source = "legacy_migration" if ledger else "existing_no_ledger"
            else:
                source, assertions = "fallback", []
                anchor_source = "new_campaign"
                client = self.client
                if client is not None and client.available:
                    prompt = (
                        "Compile this goal into a JSON object {\"assertions\": [..]}. Each assertion is "
                        "{\"seeds\": \"opt\"|\"holdout\", \"metric\": one of composite_score, vision_3d_rmse_mm, "
                        "rubble_cleared_count, peak_core_contact_force_n, max_wrist_force_n, core_shattered, core_extracted, "
                        "extraction_rate, vision_latency_p99_ms, \"op\": one of < <= > >= ==, \"value\": number or bool}. "
                        f"Context:\n{self._operation_context.get(ARCHITECT, '')}\n"
                        f"Goal: {goal}\nReturn only the JSON object."
                    )
                    text = client.chat([{"role": "user", "content": prompt}], max_tokens=1024, timeout=90)
                    parsed = _parse_json_object(text) if text else None
                    if parsed and isinstance(parsed.get("assertions"), list):
                        assertions = [a for a in parsed["assertions"] if _valid_assertion(a)]
                        source = f"llm:{client.model}"
                assertions = normalize_goal_assertions(assertions)
                dod = {"goal": goal, "source": source, "assertions": assertions}

            mission = FrozenMission(dod)
            if not path.exists():
                atomic_write(path, json.dumps(dod, indent=2) + "\n")
            mission = FrozenMission(dod, path)
            ledger_hash = hashlib.sha256(self.brain.ledger_path.read_bytes()).hexdigest() if ledger else None
            legacy_max = max((int(entry["version"]) for entry in ledger), default=None)
            self.brain.record_mission_anchor(
                mission.digest, mission.snapshot(), source=anchor_source,
                legacy_ledger_sha256=ledger_hash,
                legacy_entry_count=len(ledger) if ledger else None,
                legacy_max_version=legacy_max,
            )
            self._mission = mission
            self._emit(ARCHITECT, "dod", f"{len(mission.snapshot()['assertions'])} assertions from {source}")
            return self.dod

    DECISIONS = ("CONTINUE", "REPLAN", "ROLLBACK_DRIFT", "STOP_DONE", "PING_HUMAN")

    @_agent_action(VERIFIER)
    def decide(self, generation: int, entry: dict, best: dict, failure_modes: list[str]) -> dict:
        """Structured state after each verifier run. Drift = regressed composite or a repeated failure mode."""
        opt = _num(entry.get("opt"), "composite_score", float("nan"))
        best_opt = _num(best.get("opt"), "composite_score", float("nan"))
        mode = ";".join(sorted(set(entry.get("reasons") or ["accepted"])))[:120]
        repeated = failure_modes.count(mode) >= 1 and not entry.get("accepted")
        if entry.get("accepted") and self._goal_met(entry):
            verdict, why = "STOP_DONE", "DoD satisfied on optimization and holdout seeds"
        elif entry.get("accepted"):
            verdict, why = "CONTINUE", "accepted, DoD not yet met"
        elif repeated:
            verdict, why = "ROLLBACK_DRIFT", f"DRIFT_DETECTED -> ROLLBACK_AND_REPLAN (repeated failure: {mode})"
        elif not (opt == opt) or (best_opt == best_opt and opt < best_opt):
            verdict, why = "ROLLBACK_DRIFT", f"DRIFT_DETECTED -> ROLLBACK_AND_REPLAN (opt {opt} < best {best_opt})"
        else:
            verdict, why = "REPLAN", "rejected by the contract gate, new branch next"
        failure_modes.append(mode)
        self._emit(VERIFIER, verdict, why)
        return {"generation": generation, "decision": verdict, "reason": why}

    def _goal_met(self, entry: dict) -> bool:
        self._assert_mission()
        dod = self.dod
        if not dod or not dod.get("assertions"):
            return False
        agg_o, agg_h = entry.get("opt") or {}, entry.get("holdout") or {}
        return all(_check_assertion(a, agg_o, agg_h) for a in dod.get("assertions", []))

    def run_generation(self, generation: int) -> dict:
        """One full tree pass. Returns the final ledger entry (accepted or rejected) of this generation."""
        self.generation = generation
        best = self.brain.best_accepted() or self.ensure_baseline()
        best_src = self._abs(best["path"]).read_text()
        playbook = ""  # Each agent reads its own bounded context immediately before operating.
        self._emit(ARCHITECT, "running", f"reading playbook + autopsy of v{best['version']}")
        contract = self.architect(generation, best, playbook)
        self._emit(ARCHITECT, "done", f"{contract['contract_id']}: RMSE<={contract['target_rmse_mm']} mm, "
                                      f"top-down={contract['require_top_down_rubble_ordering']}, F<{FRACTURE_N} N")
        code, origin = self.specialists(generation, contract, best_src, best, playbook)
        patch: Optional[dict] = None
        if code is None:
            code, origin, patch = self.fallback(generation, best)
        while True:
            if _sha(code) == best.get("sha256"):
                dup = self.versions / f"candidate_g{generation}_dup.py"
                atomic_write(dup, code)
                return self._reject(generation, code, origin, contract, {}, {},
                                    ["candidate identical to current best"], dup, best)
            ok, reasons, opt, hold, path = self._verify(generation, code, origin, contract, best)
            if ok:
                return self._accept(generation, code, origin, contract, best, best_src, opt, hold, reasons, patch)
            entry = self._reject(generation, code, origin, contract, opt, hold, reasons, path, best)
            if not origin.startswith("llm:"):
                return entry
            self._emit(ARCHITECT, "retry", "LLM candidate failed the contract; retrying with fallback patch")
            code, origin, patch = self.fallback(generation, best)

    def drift_probe(self) -> Optional[dict]:
        """Negative control: resubmit the accepted v0 baseline as a real candidate against the current best.
        The gate is expected to reject it (decision ROLLBACK_DRIFT/REPLAN), which records a genuine rejected
        attempt in ledger.json and proves the Law 5 gate still blocks known-bad candidates."""
        ledger = self.brain.load_ledger()
        best = self.brain.best_accepted()
        v0 = next((e for e in ledger if e.get("version") == 0 and e.get("accepted")), None)
        if best is None or v0 is None or int(best.get("version", 0)) == 0:
            return None
        code = self._abs(v0["path"]).read_text()
        if _sha(code) == best.get("sha256"):
            return None
        self.generation = max((int(e.get("generation", 0)) for e in ledger), default=0) + 1
        contract = self.derive_contract(self.generation, best)
        origin = "drift_probe:baseline_v0"
        self._emit(VERIFIER, "running", f"{origin}: negative control, resubmitting {v0['path']} against v{best['version']}")
        ok, reasons, opt, hold, path = self._verify(self.generation, code, origin, contract, best)
        if ok:
            return self._accept(self.generation, code, origin, contract, best,
                                self._abs(best["path"]).read_text(), opt, hold, reasons, None)
        return self._reject(self.generation, code, origin, contract, opt, hold, reasons, path, best)

    def run(self, iterations: int, verify_improvement: bool = False) -> list[dict]:
        """Run `iterations` generations; optionally enforce strict opt improvement and non-decreasing holdout.
        Stops early when decide() returns STOP_DONE (DoD satisfied on opt + holdout)."""
        prev = self.brain.best_accepted() or self.ensure_baseline()
        start = max((int(e.get("generation", 0)) for e in self.brain.load_ledger()), default=0)
        entries = []
        for g in range(start + 1, start + iterations + 1):
            entry = self.run_generation(g)
            entries.append(entry)
            if verify_improvement:
                if not entry.get("accepted"):
                    raise RuntimeError(f"generation {g} rejected: {entry.get('reasons')}")
                d_opt = _num(entry["opt"], "composite_score") - _num(prev["opt"], "composite_score")
                d_hold = _num(entry["holdout"], "composite_score") - _num(prev["holdout"], "composite_score")
                if not d_opt > 0 or d_hold < 0:
                    raise RuntimeError(f"generation {g} did not improve: opt delta {d_opt:+.4f}, holdout delta {d_hold:+.4f}")
            if entry.get("accepted"):
                prev = entry
            if entry.get("decision") == "STOP_DONE":
                self._emit(ARCHITECT, "halt", "DoD satisfied on optimization and holdout seeds - stopping early")
                break
        return entries
