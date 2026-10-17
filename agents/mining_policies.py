"""Adapters that reuse accepted excavation control logic in the mining demo.

Only the accepted pipeline's gripper closure law is adapted here. This module
does not run its camera perception, excavation planner, or core recovery task.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
LEDGER_PATH = ROOT / "shared_brain" / "ledger.json"
POLICY_CADENCE_S = 0.05
EVALUATION_SCOPE = "accepted on core recovery; fleet transfer not benchmark-verified"


def _load_latest_accepted() -> tuple[ModuleType, dict[str, Any], Path]:
    """Load the newest ledger-accepted pipeline, rejecting missing or stale source."""
    if not LEDGER_PATH.is_file():
        raise RuntimeError(f"accepted mining policy ledger is missing: {LEDGER_PATH}")
    try:
        entries = json.loads(LEDGER_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read accepted mining policy ledger: {exc}") from exc
    if not isinstance(entries, list):
        raise RuntimeError("accepted mining policy ledger must contain a list of entries")
    accepted = [entry for entry in entries if isinstance(entry, dict) and entry.get("accepted")]
    if not accepted:
        raise RuntimeError("no accepted excavation policy is recorded in shared_brain/ledger.json")
    record = max(accepted, key=lambda entry: int(entry.get("version", -1)))
    relpath = record.get("path")
    expected_hash = record.get("sha256")
    if not relpath or not expected_hash:
        raise RuntimeError("latest accepted excavation policy has no recorded source path or SHA-256")
    path = (ROOT / str(relpath)).resolve()
    if not path.is_file():
        raise RuntimeError(f"latest accepted excavation policy source is missing: {path}")
    actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual_hash != expected_hash:
        raise RuntimeError(
            f"latest accepted excavation policy hash mismatch for {path}: "
            f"ledger={expected_hash}, actual={actual_hash}"
        )

    module_name = f"_deepcore_accepted_policy_v{int(record['version'])}_{actual_hash[:12]}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot create an import spec for accepted excavation policy: {path}")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        raise RuntimeError(f"cannot import accepted excavation policy {path}: {exc}") from exc
    if not callable(getattr(module, "ExcavationPipeline", None)):
        raise RuntimeError(f"accepted source has no ExcavationPipeline class: {path}")
    if not callable(getattr(module.ExcavationPipeline, "_track_grip", None)):
        raise RuntimeError(f"accepted ExcavationPipeline has no _track_grip method: {path}")
    if not hasattr(module, "FINGER_OPEN") or not hasattr(module, "TOUCH_N"):
        raise RuntimeError(f"accepted source is missing FINGER_OPEN or TOUCH_N: {path}")
    return module, record, path


class AcceptedGraspPolicy:
    """20 Hz adapter for the accepted version's measured-gap gripper controller.

    ``close_gap_m`` accepts the current physical pad gap and returns a desired
    physical gap in meters. Calls inside one 50 ms controller interval return
    the cached target. Call ``reset`` at the start of each grasp and after each
    release so the accepted controller's rate limiter and contact latch restart.
    """

    def __init__(self) -> None:
        self._module, record, self._source_path = _load_latest_accepted()
        self.version = int(record["version"])
        self.source = str(record.get("source", ""))
        self.sha256 = str(record["sha256"])
        self.safe_touch_n = float(self._module.TOUCH_N)
        self.evaluation_scope = EVALUATION_SCOPE
        self._finger_open_m = float(self._module.FINGER_OPEN)
        if not np.isfinite(self._finger_open_m) or self._finger_open_m <= 0:
            raise RuntimeError(f"accepted policy has invalid FINGER_OPEN: {self._finger_open_m}")
        if not np.isfinite(self.safe_touch_n) or self.safe_touch_n <= 0:
            raise RuntimeError(f"accepted policy has invalid TOUCH_N: {self.safe_touch_n}")
        self.reset()

    @property
    def metadata(self) -> dict[str, Any]:
        """Source evidence and transfer scope suitable for demo/API metadata."""
        return {
            "policy": "accepted_gripper_closure",
            "version": self.version,
            "source": self.source,
            "source_path": str(self._source_path),
            "sha256": self.sha256,
            "safe_touch_n": self.safe_touch_n,
            "evaluation_scope": self.evaluation_scope,
            "update_cadence_hz": 1.0 / POLICY_CADENCE_S,
        }

    def reset(self) -> None:
        """Start a fresh closure attempt without invoking perception or a fake step."""
        self._pipeline = self._module.ExcavationPipeline(
            np.eye(3, dtype=np.float64),
            np.zeros(3, dtype=np.float64),
            np.eye(3, dtype=np.float64),
        )
        self._last_update_s: float | None = None
        self._target_gap_m = self._finger_open_m

    def close_gap_m(self, measured_gap_m: float, now_s: float) -> float:
        """Return the accepted controller's next target gap in meters.

        The accepted pipeline expects normalized gripper width. Its real
        ``_track_grip`` method is called at most once per 50 ms, and its output
        is converted back into the accepted source's meter-scale opening.
        """
        gap = float(measured_gap_m)
        now = float(now_s)
        if not np.isfinite(gap) or not np.isfinite(now):
            raise ValueError("measured_gap_m and now_s must be finite numbers")
        # A simulation clock reset denotes a new attempt; avoid stale cadence
        # state even if the caller omitted reset().
        if self._last_update_s is not None and now < self._last_update_s:
            self.reset()
        if self._last_update_s is None or now - self._last_update_s >= POLICY_CADENCE_S:
            measured = float(np.clip(gap, 0.0, self._finger_open_m))
            normalized = measured / self._finger_open_m
            command = float(self._pipeline._track_grip("rubble_a", {"gripper": normalized}))
            if not np.isfinite(command):
                raise RuntimeError("accepted _track_grip returned a non-finite command")
            self._target_gap_m = float(np.clip(command, 0.0, 1.0) * self._finger_open_m)
            self._last_update_s = now
        return self._target_gap_m
