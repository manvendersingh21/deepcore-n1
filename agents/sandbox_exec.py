"""Isolated evaluation of a candidate pipeline: AST guard first, then the locked verifier in a sandboxed process."""
from __future__ import annotations

import json
import os
import resource
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
MEM_LIMIT_BYTES = 6 * 1024 ** 3
_ENV_PASSTHROUGH = ("PATH", "HOME", "LANG", "LC_ALL", "TMPDIR", "MUJOCO_GL", "PYOPENGL_PLATFORM", "DISPLAY",
                    "VIRTUAL_ENV", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MUJOCO_EGL_DEVICE_ID",
                    "CUDA_VISIBLE_DEVICES", "__EGL_VENDOR_LIBRARY_FILENAMES")


def _limits(cpu_s: int):
    """preexec_fn: cap CPU seconds everywhere and address space where it is safe (not macOS)."""

    def apply() -> None:
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_s, cpu_s + 5))
        if sys.platform != "darwin" and hasattr(resource, "RLIMIT_AS"):
            try:
                resource.setrlimit(resource.RLIMIT_AS, (MEM_LIMIT_BYTES, MEM_LIMIT_BYTES))
            except (ValueError, OSError):
                pass

    return apply


def _sandbox_env() -> dict:
    env = {k: os.environ[k] for k in _ENV_PASSTHROUGH if k in os.environ}
    # BLAS/OpenMP pools reserve stack space per host CPU. On large GPU hosts
    # those reservations can exhaust RLIMIT_AS before MuJoCo loads its scene.
    # Keep the memory cap and bound library parallelism for each evaluation.
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        env[name] = "1"
    env["PYTHONPATH"] = str(REPO_ROOT)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def _static_problems(source: str) -> list[str]:
    from core.verifier_harness import static_check

    return list(static_check(source))


def verifier_cmd(path: str, seeds: Sequence[int], diagnostics: bool, json_out: str) -> list[str]:
    cmd = [sys.executable, "-m", "core.verifier_harness", str(path), "--seeds", *map(str, seeds), "--json-out", json_out]
    return cmd if diagnostics else cmd + ["--no-diagnostics"]


def parse_result(json_out: str, stdout: str) -> dict:
    """Prefer the --json-out file; fall back to the last JSON object printed on stdout."""
    try:
        if os.path.exists(json_out):
            with open(json_out) as fh:
                return json.load(fh)
        start = stdout.find("{")
        if start >= 0:
            return json.loads(stdout[start:])
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"unreadable verifier output: {exc}") from exc
    raise ValueError("verifier produced no JSON result")


def run_candidate(path: str | os.PathLike, seeds: Sequence[int], diagnostics: bool = True, timeout_s: float = 600,
                  use_nemoclaw: Optional[bool] = None) -> dict:
    """Evaluate `path` on `seeds`. Returns {"ok", "result"|"error", "sandbox", "wall_s"}; never raises."""
    t0 = time.monotonic()
    path = Path(path)
    if not path.is_absolute():
        path = REPO_ROOT / path

    def fail(msg: str, sandbox: str = "static") -> dict:
        return {"ok": False, "error": msg, "sandbox": sandbox, "wall_s": round(time.monotonic() - t0, 3)}

    try:
        source = path.read_text()
    except OSError as exc:
        return fail(f"cannot read candidate: {exc}")
    try:
        problems = _static_problems(source)
    except Exception as exc:
        return fail(f"static check crashed: {type(exc).__name__}: {exc}")
    if problems:
        return fail("static check rejected candidate: " + "; ".join(map(str, problems)))

    nemoclaw = shutil.which("nemoclaw") if use_nemoclaw in (None, True) else None
    with tempfile.TemporaryDirectory(prefix="deepcore_eval_") as tmp:
        out = os.path.join(tmp, "result.json")
        cmd = verifier_cmd(str(path), seeds, diagnostics, out)
        sandbox = "subprocess"
        if nemoclaw:
            # NEMOCLAW_ARGS lets operators choose the sandbox profile, e.g. "run --policy strict --".
            cmd = [nemoclaw, *shlex.split(os.environ.get("NEMOCLAW_ARGS", "run --")), *cmd]
            sandbox = "nemoclaw"
        try:
            proc = subprocess.run(cmd, cwd=str(REPO_ROOT), env=_sandbox_env(), capture_output=True, text=True,
                                  timeout=timeout_s, preexec_fn=_limits(int(timeout_s) + 30))
        except subprocess.TimeoutExpired:
            return fail(f"evaluation timed out after {timeout_s:.0f}s", sandbox)
        except OSError as exc:
            return fail(f"cannot launch sandbox: {exc}", sandbox)
        if proc.returncode != 0:
            tail = (proc.stderr or proc.stdout or "").strip()[-2000:]
            return fail(f"verifier exited {proc.returncode}: {tail}", sandbox)
        try:
            result = parse_result(out, proc.stdout)
        except ValueError as exc:
            return fail(str(exc), sandbox)
    return {"ok": True, "result": result, "sandbox": sandbox, "wall_s": round(time.monotonic() - t0, 3)}


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Sandboxed candidate evaluation")
    ap.add_argument("path")
    ap.add_argument("--seeds", type=int, nargs="+", default=[11, 22, 33])
    ap.add_argument("--no-diagnostics", action="store_true")
    a = ap.parse_args()
    r = run_candidate(a.path, a.seeds, diagnostics=not a.no_diagnostics)
    print(json.dumps({k: v for k, v in r.items() if k != "result"} | {"aggregate": r.get("result", {}).get("aggregate")},
                     indent=2, default=str))
