"""Validation and tamper checks for machine-readable mission contracts."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any


METRIC_TYPES = {
    "composite_score": "number",
    "vision_3d_rmse_mm": "number",
    "rubble_cleared_count": "number",
    "peak_core_contact_force_n": "number",
    "max_wrist_force_n": "number",
    "core_shattered": "bool",
    "core_extracted": "bool",
    "extraction_rate": "number",
    "vision_latency_p99_ms": "number",
}
NUMERIC_OPERATORS = {"<", "<=", ">", ">=", "=="}
BOOLEAN_OPERATORS = {"=="}

MANDATORY_ASSERTIONS = (
    {"seeds": "holdout", "metric": "core_shattered", "op": "==", "value": False},
    {"seeds": "holdout", "metric": "core_extracted", "op": "==", "value": True},
    {"seeds": "holdout", "metric": "composite_score", "op": ">", "value": 85.0},
    {"seeds": "opt", "metric": "vision_latency_p99_ms", "op": "<", "value": 25.0},
)


def _clean_assertion(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    seeds, metric, op = value.get("seeds"), value.get("metric"), value.get("op")
    if not isinstance(seeds, str) or seeds not in {"opt", "holdout"}:
        return None
    if not isinstance(metric, str) or metric not in METRIC_TYPES:
        return None
    expected_type = METRIC_TYPES[metric]
    if expected_type == "bool":
        if not isinstance(op, str) or op not in BOOLEAN_OPERATORS or type(value.get("value")) is not bool:
            return None
        clean_value = value["value"]
    else:
        raw = value.get("value")
        if not isinstance(op, str) or op not in NUMERIC_OPERATORS or isinstance(raw, bool) or not isinstance(raw, (int, float)):
            return None
        clean_value = float(raw)
        if not math.isfinite(clean_value):
            return None
    return {"seeds": seeds, "metric": metric, "op": op, "value": clean_value}


def _numeric_interval(assertion: dict[str, Any]) -> tuple[float | None, bool, float | None, bool]:
    """Return lower, lower-inclusive, upper, upper-inclusive for one comparison."""
    op, value = assertion["op"], float(assertion["value"])
    if op == "==":
        return value, True, value, True
    if op in (">", ">="):
        return value, op == ">=", None, False
    return None, False, value, op == "<="


def _compatible_with_mandatory(assertion: dict[str, Any]) -> bool:
    for required in MANDATORY_ASSERTIONS:
        if (assertion["seeds"], assertion["metric"]) != (required["seeds"], required["metric"]):
            continue
        if METRIC_TYPES[assertion["metric"]] == "bool":
            return assertion["value"] is required["value"]
        low_a, low_inc_a, high_a, high_inc_a = _numeric_interval(assertion)
        low_b, low_inc_b, high_b, high_inc_b = _numeric_interval(required)
        lows = [(x, inc) for x, inc in ((low_a, low_inc_a), (low_b, low_inc_b)) if x is not None]
        highs = [(x, inc) for x, inc in ((high_a, high_inc_a), (high_b, high_inc_b)) if x is not None]
        if lows and highs:
            low, low_inc = max(lows, key=lambda pair: pair[0])
            high, high_inc = min(highs, key=lambda pair: pair[0])
            if low > high or (low == high and not (low_inc and high_inc)):
                return False
    return True


def _canonical_json(value: Any) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError(f"mission contract is not finite JSON data: {exc}") from exc


def normalize_goal_assertions(assertions: Any) -> list[dict[str, Any]]:
    """Keep safe whitelisted assertions and add every mandatory contract clause.

    Invalid entries, incompatible metric/operator/value combinations, and
    assertions that contradict a mandatory clause are discarded. A weaker
    clause is retained as an additional condition, but the exact mandatory
    clause is still appended, so it cannot relax the mission.
    """
    if not isinstance(assertions, (list, tuple)):
        assertions = []
    normalized: list[dict[str, Any]] = []
    seen: set[bytes] = set()
    for raw in assertions:
        assertion = _clean_assertion(raw)
        if assertion is None or not _compatible_with_mandatory(assertion):
            continue
        key = _canonical_json(assertion)
        if key not in seen:
            normalized.append(assertion)
            seen.add(key)
    for required in MANDATORY_ASSERTIONS:
        clause = dict(required)
        key = _canonical_json(clause)
        if key not in seen:
            normalized.append(clause)
            seen.add(key)
    return normalized


class FrozenMission:
    """Immutable canonical snapshot of a normalized mission document.

    The optional path is only pinned/read. This class never writes the mission
    document. Call ``assert_unchanged(path)`` before evaluation to detect file
    changes; a path supplied to the constructor is checked by default.
    """

    __slots__ = ("_canonical_bytes", "_digest", "_path", "_sealed")

    def __init__(self, document: dict[str, Any], path: str | Path | None = None):
        if not isinstance(document, dict):
            raise ValueError("mission contract document must be a JSON object")
        assertions = document.get("assertions")
        if not isinstance(assertions, list) or not assertions:
            raise ValueError("mission contract must contain a nonempty assertions list")
        clean = [_clean_assertion(a) for a in assertions]
        if any(a is None for a in clean):
            raise ValueError("mission contract contains an invalid or incompatible assertion")
        normalized = normalize_goal_assertions(assertions)
        mandatory_keys = {_canonical_json(dict(a)) for a in MANDATORY_ASSERTIONS}
        supplied_keys = {_canonical_json(a) for a in clean if a is not None}
        if not mandatory_keys.issubset(supplied_keys):
            raise ValueError("mission contract is missing one or more exact mandatory assertions")
        if len(normalized) != len(assertions):
            raise ValueError("mission contract assertions must be normalized and deduplicated")

        frozen = json.loads(_canonical_json(document).decode("utf-8"))
        canonical = _canonical_json(frozen)
        object.__setattr__(self, "_canonical_bytes", canonical)
        object.__setattr__(self, "_digest", hashlib.sha256(canonical).hexdigest())
        object.__setattr__(self, "_path", Path(path).expanduser().resolve() if path is not None else None)
        object.__setattr__(self, "_sealed", True)
        if self._path is not None:
            self.assert_unchanged()

    def __setattr__(self, name: str, value: Any) -> None:
        if getattr(self, "_sealed", False):
            raise AttributeError("FrozenMission is immutable")
        object.__setattr__(self, name, value)

    @property
    def digest(self) -> str:
        return self._digest

    @property
    def sha256(self) -> str:
        return self._digest

    def snapshot(self) -> dict[str, Any]:
        """Return an independent deep copy of the frozen mission document."""
        return json.loads(self._canonical_bytes.decode("utf-8"))

    def assert_unchanged(self, path: str | Path | None = None) -> None:
        """Raise if the frozen bytes or optional pinned file no longer match."""
        if hashlib.sha256(self._canonical_bytes).hexdigest() != self._digest:
            raise RuntimeError("frozen mission in-memory integrity check failed")
        target = Path(path).expanduser().resolve() if path is not None else self._path
        if target is None:
            return
        try:
            current = json.loads(target.read_text(encoding="utf-8"))
            current_bytes = _canonical_json(current)
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            raise RuntimeError(f"cannot validate frozen mission file {target}: {exc}") from exc
        if current_bytes != self._canonical_bytes:
            raise RuntimeError(f"mission contract file changed after it was frozen: {target}")
