#!/usr/bin/env python
"""DeepCore-N1 CLI: recursive-contract evolution loop, live HUD, full verification, demo recording.

Heavy imports (mujoco-backed verifier, OpenCV HUD, OpenAI client) happen lazily inside each
subcommand so `--help` and argument parsing stay instant and dependency-free.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path
from typing import Optional

REPO = Path(__file__).resolve().parent
DEFAULT_GOAL = ("Excavate rubble_a and rubble_b to the dump zone, then extract the intact core to the hopper: "
                "holdout composite > 85, core never shattered, extracted on all holdout seeds, "
                "vision latency p99 < 25 ms")


def _fmt(value, nd: int = 1) -> str:
    """Format a metric; NaN/missing becomes 'n/a'."""
    try:
        x = float(value)
    except (TypeError, ValueError):
        return "n/a"
    return "n/a" if x != x else f"{x:.{nd}f}"


# ---------------------------------------------------------------------- run-evolution
def cmd_run_evolution(args: argparse.Namespace) -> int:
    from agents.contract_orchestrator import ContractOrchestrator
    from agents.crusoe_client import CrusoeClient
    from shared_brain.memory_store import SharedBrain

    brain = SharedBrain(REPO / "shared_brain")
    if args.reset:
        brain.reset()
        print("shared brain reset: ledger + versions cleared, playbook reseeded")
    client = CrusoeClient()
    if not client.available:
        print("CRUSOE_API_KEY not set - generations will use the deterministic fallback patch library")
    orchestrator = ContractOrchestrator(brain, client)
    try:
        _evolution_session(orchestrator, args)
    except RuntimeError as exc:
        print(f"\nverify-improvement FAILED: {exc}", file=sys.stderr)
        return 1
    return 0


def _evolution_session(orchestrator, args: argparse.Namespace) -> list[dict]:
    """run-evolution flow, separated from CLI wiring so it is testable with an injected orchestrator."""
    goal = (args.goal or DEFAULT_GOAL).strip()
    dod = orchestrator.compile_goal(goal)  # step 1: frozen DoD before any generation runs
    print(f"goal compiled by the Root Architect: {len(dod['assertions'])} assertions ({dod['source']}) "
          f"-> shared_brain/generated_dod.json")
    orchestrator.ensure_baseline()
    entries = orchestrator.run(args.iterations, verify_improvement=args.verify_improvement)
    _print_generations(entries)
    _print_decisions(entries)
    if not any(not e.get("accepted") for e in orchestrator.brain.load_ledger()):
        probe = orchestrator.drift_probe()
        if probe is not None:
            _print_decisions([probe])
    _print_holdout_comparison(orchestrator.brain.load_ledger())
    return entries


def _print_decisions(entries: list[dict]) -> None:
    print("\nDECISION log (ContractOrchestrator.decide):")
    for e in entries:
        status = "accepted" if e.get("accepted") else "REJECTED"
        print(f"  g{e.get('generation', '?')}: DECISION {e.get('decision', 'n/a')} "
              f"[{status} v{e.get('version', '?')} {e.get('source', '?')}]")


def _print_generations(entries: list[dict]) -> None:
    from rich.console import Console
    from rich.table import Table

    table = Table(title="DeepCore-N1 generations", header_style="bold cyan")
    for col in ("version", "source", "opt composite", "holdout composite", "rmse mm",
                "peak core force N", "extracted", "accepted", "decision"):
        table.add_column(col, justify="left" if col == "source" else "right")
    for e in entries:
        opt, hold = e.get("opt") or {}, e.get("holdout") or {}
        table.add_row(str(e.get("version", "?")), str(e.get("source", "?")),
                      _fmt(opt.get("composite_score"), 3), _fmt(hold.get("composite_score"), 3),
                      _fmt(opt.get("vision_3d_rmse_mm")), _fmt(opt.get("peak_core_contact_force_n")),
                      "yes" if opt.get("core_extracted") else "no",
                      "yes" if e.get("accepted") else "NO", str(e.get("decision", "n/a")))
    Console().print(table)


_HOLDOUT_METRICS = ("composite_score", "vision_3d_rmse_mm", "rubble_cleared_count", "peak_core_contact_force_n",
                    "extraction_rate", "vision_latency_p99_ms", "core_shattered", "core_extracted")


def _holdout_table(ledger: list[dict]) -> object:
    """Rich table: accepted v0 baseline vs latest accepted version on holdout seeds 101-103."""
    from rich.table import Table

    v0 = next((e for e in ledger if e.get("version") == 0 and e.get("accepted") and e.get("holdout")), None)
    latest = max((e for e in ledger if e.get("accepted") and int(e.get("version", 0)) != 0 and e.get("holdout")),
                 key=lambda e: int(e.get("version", 0)), default=None)
    if v0 is None:
        return None
    after = latest if latest is not None else v0
    table = Table(title=f"holdout seeds 101-103: v0 baseline vs v{after['version']} "
                        f"({after.get('source', '?')})", header_style="bold cyan")
    for col in ("metric", "v0 baseline", f"after (v{after['version']})", "delta"):
        table.add_column(col, justify="left" if col == "metric" else "right")
    for key in _HOLDOUT_METRICS:
        a, b = (v0["holdout"] or {}).get(key), (after["holdout"] or {}).get(key)
        if isinstance(a, bool) or isinstance(b, bool):
            row = (key, str(a).lower(), str(b).lower(), "changed" if a != b else "same")
        else:
            fa, fb = _fmt(a, 2), _fmt(b, 2)
            delta = f"{float(b) - float(a):+.2f}" if "n/a" not in (fa, fb) else "n/a"
            row = (key, fa, fb, delta)
        table.add_row(*row)
    return table


def _print_holdout_comparison(ledger: list[dict]) -> None:
    from rich.console import Console

    table = _holdout_table(ledger)
    if table is None:
        print("\nholdout comparison: no v0 baseline entry in the ledger yet")
        return
    Console().print(table)


# ---------------------------------------------------------------------- run-hud
def cmd_run_hud(args: argparse.Namespace) -> int:
    import inspect

    from ui.live_hud import LiveHud

    if args.seed is not None and "seed" in inspect.signature(LiveHud.__init__).parameters:
        LiveHud(seed=args.seed).run()
    else:
        LiveHud().run()
    return 0


def cmd_run_web(args: argparse.Namespace) -> int:
    from ui.web_server import serve
    serve(host=args.host, port=args.port)
    return 0


def cmd_run_drill(args: argparse.Namespace) -> int:
    from ui.drilling_hud import run
    run(seed=args.seed)
    return 0


def cmd_run_mine(args: argparse.Namespace) -> int:
    from ui.mining_hud import run, record
    if args.out:
        record(args.out, seed=args.seed)
    else:
        run(seed=args.seed)
    return 0


# ---------------------------------------------------------------------- verify-all
def _previous_best(candidate: Path) -> tuple[Optional[float], Optional[float]]:
    """Composite scores of the newest accepted ledger version whose code differs from `candidate`."""
    from shared_brain.memory_store import SharedBrain

    sha = hashlib.sha256(candidate.read_text().encode()).hexdigest()
    older = [e for e in SharedBrain(REPO / "shared_brain").load_ledger()
             if e.get("accepted") and e.get("sha256") != sha]
    if not older:
        return None, None
    best = max(older, key=lambda e: int(e.get("version", 0)))
    return (best.get("opt") or {}).get("composite_score"), (best.get("holdout") or {}).get("composite_score")


def cmd_verify_all(_: argparse.Namespace) -> int:
    from core.integrity import verify_integrity
    from core.verifier_harness import HOLDOUT_SEEDS, OPTIMIZATION_SEEDS, check_contract, evaluate

    locked = verify_integrity()
    print(f"core integrity OK: {len(locked)} locked files match core/LOCK.json")
    candidate = REPO / "workspace" / "candidate_pipeline.py"
    if not candidate.exists():
        print(f"no candidate at {candidate} - run `main.py run-evolution` first", file=sys.stderr)
        return 1
    opt = evaluate(candidate, OPTIMIZATION_SEEDS, diagnostics=True)
    hold = evaluate(candidate, HOLDOUT_SEEDS, diagnostics=False)
    for split, result in (("opt", opt), ("holdout", hold)):
        print(f"\n{split} aggregate (seeds {tuple(result['seeds'])}):")
        for key in sorted(result["aggregate"]):
            print(f"  {key:<28} {result['aggregate'][key]}")
    best_opt, best_hold = _previous_best(candidate)
    ok, reasons = check_contract(opt["aggregate"], hold["aggregate"], best_opt, best_hold)
    ref = f" (vs previous accepted opt={best_opt}, holdout={best_hold})" if best_opt is not None else ""
    print(f"\nLaw 5 contract: {'PASS' if ok else 'FAIL'}{ref}")
    for reason in reasons:
        print(f"  - {reason}")
    return 0 if ok else 1


# ---------------------------------------------------------------------- record-demo
def cmd_record_demo(args: argparse.Namespace) -> int:
    from scripts.record_demo import main as record_demo_main

    record_demo_main(out=args.out, seconds=args.seconds)
    return 0


# ---------------------------------------------------------------------- CLI
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="DeepCore-N1: recursive agent tree improving a MuJoCo excavation pipeline under a locked verifier.")
    sub = parser.add_subparsers(dest="command", required=True)

    ev = sub.add_parser("run-evolution", help="run recursive contract-tree generations and print the ledger table")
    ev.add_argument("--goal", default=None, metavar="ONE_LINE",
                    help=f"one-line mission goal compiled into shared_brain/generated_dod.json (default: {DEFAULT_GOAL[:60]}...)")
    ev.add_argument("--iterations", type=int, default=3, help="number of generations to run (default: 3)")
    ev.add_argument("--verify-improvement", action="store_true",
                    help="exit 1 unless every generation is accepted and strictly improves the opt composite")
    ev.add_argument("--reset", action="store_true",
                    help="wipe the shared brain (ledger, versions, playbook) before running")
    ev.set_defaults(fn=cmd_run_evolution)

    hud = sub.add_parser("run-hud", help="open the 1600x900 live HUD (0-3 versions, E evolve, W occlude, R reseed, Q quit)")
    hud.add_argument("--seed", type=int, default=None, help="initial episode seed (default: HUD chooses)")
    hud.set_defaults(fn=cmd_run_hud)

    web = sub.add_parser("run-web", help="open the interactive 3D mining control room in a browser")
    web.add_argument("--host", default="127.0.0.1")
    web.add_argument("--port", type=int, default=8080)
    web.set_defaults(fn=cmd_run_web)

    drill = sub.add_parser("run-drill", help="open native drilling with measured thrust, torque and depth")
    drill.add_argument("--seed", type=int, default=101)
    drill.set_defaults(fn=cmd_run_drill)

    mine = sub.add_parser('run-mine', help='open the native two-robot CMU mine operation')
    mine.add_argument('--seed', type=int, default=101)
    mine.add_argument('--out', default=None, help='record a native episode to this MP4 instead of opening a window')
    mine.set_defaults(fn=cmd_run_mine)

    va = sub.add_parser("verify-all",
                        help="check core integrity, re-evaluate the candidate on opt+holdout seeds, apply Law 5")
    va.set_defaults(fn=cmd_verify_all)

    demo = sub.add_parser("record-demo", help="record the submission demo video from real episodes + ledger data")
    demo.add_argument("--out", default="submission_demo.mp4", help="output video path (default: submission_demo.mp4)")
    demo.add_argument("--seconds", type=int, default=60, help="target video length in seconds (default: 60)")
    demo.set_defaults(fn=cmd_record_demo)
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.fn(args)
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
