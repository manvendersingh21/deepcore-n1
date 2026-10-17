"""Headless unit tests for ui/live_hud.render_frame (synthetic inputs, no MuJoCo)."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from ui import live_hud as hud

ROOT = Path(__file__).resolve().parents[1]


def _images() -> tuple[np.ndarray, np.ndarray]:
    yy, xx = np.mgrid[0:480, 0:640]
    cam = np.stack([40 + xx * 0.08, 38 + yy * 0.06, 36 + 0 * xx], -1).astype(np.uint8)
    noise = np.random.default_rng(0).integers(0, 18, cam.shape, dtype=np.uint8)
    cam = cv2.add(cam, noise)
    for (cx, cy, w, h, ang), col in (((330, 250, 54, 54, 18), (34, 229, 245)), ((300, 215, 88, 60, 10), (232, 168, 58)),
                                     ((382, 262, 78, 58, -25), (255, 79, 216))):
        cv2.fillPoly(cam, [np.round(cv2.boxPoints(((cx, cy), (w, h), ang))).astype(np.int32)], col)
    persp = np.stack([30 + yy * 0.12, 32 + yy * 0.1, 40 + yy * 0.08], -1).astype(np.uint8)
    cv2.rectangle(persp, (0, 300), (640, 480), (70, 64, 58), -1)
    cv2.rectangle(persp, (280, 260), (360, 320), (34, 229, 245), -1)
    return cam, persp


def _state(ledger: list[dict], shattered: bool = False) -> hud.HudState:
    return hud.HudState(
        seed=101, phase="render-test",
        bboxes=[("core", 330, 250, 54, 54, 18)],
        depth_conf={"core": 0.5},
        rmse_mm=float("nan"),
        core_force_now=0.0, peak_core_force=0.0, shattered=shattered,
        wrist_force=np.zeros(3),
        tree_status={"architect": "idle", "cv": "idle", "control": "idle", "verifier": "idle"},
        playbook_bullets=[],
        ledger=ledger)


def test_render_frame_shape_and_content():
    cam, persp = _images()
    frame = hud.render_frame(_state(hud.load_ledger()), cam, persp)
    assert frame.shape == (900, 1600, 3) and frame.dtype == np.uint8
    assert frame.std() > 10 and len(np.unique(frame.reshape(-1, 3), axis=0)) > 1000
    out = ROOT / "dev" / "hud_preview.png"
    out.parent.mkdir(exist_ok=True)
    assert cv2.imwrite(str(out), frame) and out.stat().st_size > 0


def test_render_frame_shattered_flash_and_empty_ledger():
    cam, persp = _images()
    s = _state([], shattered=True)
    a = hud.render_frame(s, cam, persp)
    s.frame_idx = 1
    b = hud.render_frame(s, cam, persp)
    assert a.shape == b.shape == (900, 1600, 3)
    assert np.any(a != b), "shatter badge must flash by frame parity"
    cv2.imwrite(str(ROOT / "dev" / "hud_preview_empty.png"), a)


def test_render_frame_without_images():
    frame = hud.render_frame(hud.HudState(), None, None)
    assert frame.shape == (900, 1600, 3) and frame.std() > 3


def test_ledger_helpers(tmp_path):
    led = hud.load_ledger()
    assert isinstance(led, list)
    assert hud.load_ledger(tmp_path / "missing.json") == []
    best = hud.best_holdout_entry(led)
    if best is not None:
        assert best["accepted"] and best["version"] != 0 and best["holdout"]
    latest = hud.latest_holdout_entry(led)
    if latest is not None:
        assert latest["accepted"] and latest["holdout"]
        assert latest["version"] == max(e["version"] for e in led
                                         if e.get("accepted") and e.get("version", 0) != 0 and e.get("holdout"))
    accepted = [e["version"] for e in led if e.get("accepted") and "version" in e]
    assert hud.newest_accepted_version(led) == (max(accepted) if accepted else None)


def test_depth_confidence_and_tree():
    depth = np.full((480, 640), 0.6, np.float32)
    depth[:, :320] = np.nan
    conf = hud.depth_confidence(depth, [("core", 480, 240, 40, 40, 0), ("rubble_a", 320, 240, 100, 40, 0)])
    assert conf["core"] == 1.0 and 0.4 < conf["rubble_a"] < 0.6
    st, msg = hud.tree_update({k: "idle" for k in hud.TREE_NODES},
                              {"agent": "Excavation Control Specialist", "status": "running", "message": "x"})
    assert st["control"] == "running" and "Control" in msg
    st, _ = hud.tree_update(st, {"agent": "Independent Verifier", "status": "rejected", "message": "y"})
    assert st["verifier"] == "fail"


def _badge_state(**kw) -> hud.HudState:
    return hud.HudState(seed=101, phase=kw.pop("phase", "-"), core_force_now=3.0, peak_core_force=25.0,
                        wrist_force=np.array([1.0, 2.0, 2.0]), ledger=[], **kw)


def test_badge_text_per_state():
    cases = {
        "not-started": (_badge_state(), "NOT STARTED"),
        "in-progress": (_badge_state(started=True, phase="control:grasp"), "IN PROGRESS  control:grasp"),
        "shattered": (_badge_state(started=True, shattered=True, episode_done=True), "CORE SHATTERED"),
        "intact-not-extracted": (_badge_state(started=True, episode_done=True), "NOT EXTRACTED"),
        "extracted": (_badge_state(started=True, episode_done=True, extracted=True), "CORE EXTRACTED"),
    }
    for name, (state, want) in cases.items():
        assert hud.badge_text(state)[0] == want, name
    # extraction only counts at episode end: a mid-run hopper pass is still in progress
    mid = _badge_state(started=True, extracted=True, phase="control:place")
    assert hud.badge_text(mid)[0] == "IN PROGRESS  control:place"


def test_badge_renders_for_shattered_and_intact_not_extracted():
    cam, persp = _images()
    shattered = _badge_state(started=True, shattered=True, episode_done=True)
    intact = _badge_state(started=True, episode_done=True)
    out = ROOT / "dev"
    out.mkdir(exist_ok=True)
    for name, st in (("shattered", shattered), ("intact_not_extracted", intact)):
        frame = hud.render_frame(st, cam, persp)
        assert frame.shape == (900, 1600, 3)
        assert cv2.imwrite(str(out / f"hud_badge_check_{name}.png"), frame)
