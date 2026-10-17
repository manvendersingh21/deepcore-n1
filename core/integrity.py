"""SHA-256 guard for the locked judge. `python -m core.integrity --lock` writes core/LOCK.json once (end of Phase 1)."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

CORE = Path(__file__).parent
LOCKED_FILES = ("verifier_harness.py", "scene.xml", "env_runner.py", "sensor_corruption.py")
LOCK_PATH = CORE / "LOCK.json"


class IntegrityError(RuntimeError):
    pass


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def current_hashes() -> dict:
    return {f: sha256_of(CORE / f) for f in LOCKED_FILES}


def write_lock() -> dict:
    h = current_hashes()
    LOCK_PATH.write_text(json.dumps(h, indent=2) + "\n")
    return h


def verify_integrity() -> dict:
    if not LOCK_PATH.exists():
        raise IntegrityError("core/LOCK.json missing - run `python -m core.integrity --lock` after Phase 1")
    expected = json.loads(LOCK_PATH.read_text())
    actual = current_hashes()
    bad = [f for f in LOCKED_FILES if expected.get(f) != actual[f]]
    if bad:
        raise IntegrityError(f"locked judge files modified: {bad}")
    return actual


if __name__ == "__main__":
    if "--lock" in sys.argv:
        print(json.dumps(write_lock(), indent=2))
    else:
        print(json.dumps(verify_integrity(), indent=2))
