"""Safety tests for normalized and frozen mission assertions."""
import json

import pytest

from agents.mission_contract import (
    MANDATORY_ASSERTIONS,
    FrozenMission,
    normalize_goal_assertions,
)


def _mandatory(assertions):
    return all(any(a == required for a in assertions) for required in MANDATORY_ASSERTIONS)


def test_normalizer_filters_invalid_and_nonfinite_assertions():
    result = normalize_goal_assertions([
        {"seeds": "holdout", "metric": "composite_score", "op": ">", "value": float("nan")},
        {"seeds": "holdout", "metric": "core_shattered", "op": ">", "value": 0},
        {"seeds": "train", "metric": "composite_score", "op": ">", "value": 90},
        {"seeds": "holdout", "metric": "unknown", "op": ">", "value": 1},
        {"seeds": "opt", "metric": "vision_latency_p99_ms", "op": "<", "value": float("inf")},
        {"seeds": "holdout", "metric": "extraction_rate", "op": ">", "value": True},
        {"seeds": "holdout", "metric": "rubble_cleared_count", "op": ">=", "value": 2},
    ])
    assert len(result) == 5  # one valid model assertion plus four mandatory clauses
    assert result[0] == {"seeds": "holdout", "metric": "rubble_cleared_count", "op": ">=", "value": 2.0}
    assert _mandatory(result)


def test_weaker_llm_thresholds_cannot_relax_mandatory_contract():
    result = normalize_goal_assertions([
        {"seeds": "holdout", "metric": "core_shattered", "op": "==", "value": True},
        {"seeds": "holdout", "metric": "core_extracted", "op": "==", "value": False},
        {"seeds": "holdout", "metric": "composite_score", "op": ">", "value": 1},
        {"seeds": "opt", "metric": "vision_latency_p99_ms", "op": "<", "value": 100},
    ])
    assert _mandatory(result)
    assert {"seeds": "holdout", "metric": "composite_score", "op": ">", "value": 1.0} in result
    assert {"seeds": "opt", "metric": "vision_latency_p99_ms", "op": "<", "value": 100.0} in result
    assert not any(a.get("value") is True and a.get("metric") == "core_shattered" for a in result)
    assert not any(a.get("value") is False and a.get("metric") == "core_extracted" for a in result)


def test_contradictory_numeric_contract_assertions_are_dropped():
    result = normalize_goal_assertions([
        {"seeds": "holdout", "metric": "composite_score", "op": "<=", "value": 85},
        {"seeds": "opt", "metric": "vision_latency_p99_ms", "op": ">=", "value": 25},
    ])
    assert _mandatory(result)
    assert not any(a.get("op") == "<=" and a.get("metric") == "composite_score" for a in result)
    assert not any(a.get("op") == ">=" and a.get("metric") == "vision_latency_p99_ms" for a in result)


def test_frozen_mission_returns_deep_copies_and_detects_file_tampering(tmp_path):
    document = {"goal": "extract safely", "source": "test", "assertions": normalize_goal_assertions([])}
    path = tmp_path / "generated_dod.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    frozen = FrozenMission(document, path=path)
    digest = frozen.digest

    document["assertions"].clear()
    snapshot = frozen.snapshot()
    snapshot["goal"] = "mutated copy"
    snapshot["assertions"].clear()
    assert frozen.digest == digest
    assert _mandatory(frozen.snapshot()["assertions"])
    frozen.assert_unchanged()

    changed = frozen.snapshot()
    changed["goal"] = "tampered on disk"
    path.write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(RuntimeError, match="changed after it was frozen"):
        frozen.assert_unchanged()


@pytest.mark.parametrize("document", [
    {"assertions": []},
    {"assertions": [{"seeds": "holdout", "metric": "composite_score", "op": ">", "value": 1}]},
])
def test_frozen_mission_requires_nonempty_exact_mandatory_assertions(document):
    with pytest.raises(ValueError):
        FrozenMission(document)


def test_frozen_mission_rejects_invalid_assertions_even_if_mandatory_present():
    assertions = normalize_goal_assertions([])
    assertions.append({"seeds": "holdout", "metric": "composite_score", "op": ">", "value": float("nan")})
    with pytest.raises(ValueError):
        FrozenMission({"assertions": assertions})
