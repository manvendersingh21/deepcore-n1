"""DeepCore-N1 live HUD.

`render_frame(state, cam_rgb, persp_rgb)` is a pure, headless OpenCV renderer (1600x900 BGR).
`LiveHud` runs real episodes through `core.env_runner.run_episode` and displays them with hotkeys.
Heavy / concurrently-written modules (core.*, agents.*, shared_brain.*, mujoco) are imported lazily.
"""
from __future__ import annotations

import copy
import json
import math
import queue
import random
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
LEDGER_PATH = ROOT / "shared_brain" / "ledger.json"
PLAYBOOK_PATH = ROOT / "shared_brain" / "playbook.md"
VERSIONS_DIR = ROOT / "workspace" / "versions"
BASELINE_PATH = ROOT / "workspace" / "baseline_v0.py"

W, H = 1600, 900
AA = cv2.LINE_AA
FS, FD = cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_DUPLEX
OBJECTS = ("core", "rubble_a", "rubble_b")
TREE_NODES = ("architect", "cv", "control", "verifier")
TREE_LABELS = {"architect": "Root Architect", "cv": "CV Specialist", "control": "Control Specialist",
               "verifier": "Verifier"}


def _hex(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16)


C = {"bg": _hex("0B0B0F"), "card": _hex("121217"), "card2": _hex("1B1B22"), "card3": _hex("25252E"),
     "border": _hex("23232B"), "text": _hex("EDEDF2"), "muted": _hex("8A8A99"), "dim": _hex("4C4C58"),
     "cyan": _hex("22E5F5"), "lime": _hex("B6FF3B"), "red": _hex("FF3B5C"), "ochre": _hex("E8A83A"),
     "magenta": _hex("FF4FD8"), "ink": _hex("0B0B0F")}
OBJ_COLOR = {"core": C["cyan"], "rubble_a": C["ochre"], "rubble_b": C["magenta"]}
CARD = {"cam": (20, 738, 160, 120), "persp": (20, 82, 1100, 640), "rail": (1140, 82, 440, 776),
        "tree": (1140, 224, 440, 280), "score": (196, 738, 924, 120)}
RADIUS = 16


# ============================================================================================ state
@dataclass
class HudState:
    """Everything render_frame needs. Pure data; no handles to the simulator."""
    seed: int = 101
    t_sec: float = 0.0
    episode_s: float = 11.0
    fps: float = 0.0
    latency_p99_ms: float = float("nan")
    phase: str = "-"
    version: int = 0
    versions_available: list[int] = field(default_factory=lambda: [0])
    live_version: Optional[int] = None          # newest accepted version (gets the HOT badge)
    version_source: str = ""
    bboxes: list[tuple] = field(default_factory=list)            # (name, x, y, w, h, yaw_deg) overhead px
    estimates: dict[str, tuple[np.ndarray, float]] = field(default_factory=dict)
    depth_conf: dict[str, float] = field(default_factory=dict)
    rmse_mm: float = float("nan")                                # live est-vs-truth 3D RMSE
    rmse_target_mm: float = 10.0
    core_force_now: float = 0.0
    peak_core_force: float = 0.0
    shattered: bool = False
    extracted: bool = False
    episode_done: bool = False                                   # this frame is the run's final frame
    started: bool = False                                        # this frame came from a live/replayed run
    wrist_force: np.ndarray = field(default_factory=lambda: np.zeros(3))
    force_arrow_px: Optional[tuple[tuple[float, float], tuple[float, float]]] = None  # persp px (src res)
    force_hist: list[float] = field(default_factory=list)       # ||F_wrist|| samples
    fracture_n: float = 18.0
    latency_limit_ms: float = 25.0
    webcam_on: bool = False
    tree_status: dict[str, str] = field(default_factory=lambda: {k: "idle" for k in TREE_NODES})
    tree_message: str = ""
    generation: int = 0
    playbook_bullets: list[str] = field(default_factory=list)
    ledger: list[dict] = field(default_factory=list)
    holdout_seeds: tuple[int, ...] = (101, 102, 103)
    toast: str = ""
    frame_idx: int = 0


# ============================================================================================ data loaders
def load_ledger(path: Path = LEDGER_PATH) -> list[dict]:
    """Ledger entries (empty list if missing / mid-write / malformed)."""
    try:
        data = json.loads(Path(path).read_text())
        return [e for e in data if isinstance(e, dict)] if isinstance(data, list) else []
    except (OSError, ValueError):
        return []


def load_playbook_bullets(path: Path = PLAYBOOK_PATH, n: int = 4) -> list[str]:
    try:
        lines = Path(path).read_text().splitlines()
    except OSError:
        return []
    return [ln[2:].strip() for ln in lines if ln.startswith("- ")][-n:]


def baseline_entry(ledger: list[dict]) -> Optional[dict]:
    return next((e for e in ledger if e.get("version") == 0 and e.get("accepted")), None)


def best_holdout_entry(ledger: list[dict]) -> Optional[dict]:
    """Best accepted non-baseline version by holdout composite."""
    cands = [e for e in ledger if e.get("accepted") and e.get("version", 0) != 0 and e.get("holdout")]
    return max(cands, key=lambda e: _num(e["holdout"].get("composite_score")), default=None)


def latest_holdout_entry(ledger: list[dict]) -> Optional[dict]:
    """Newest accepted non-baseline entry with holdout metrics."""
    cands = [e for e in ledger if e.get("accepted") and e.get("version", 0) != 0 and e.get("holdout")]
    return max(cands, key=lambda e: int(e.get("version", -1)), default=None)


def newest_accepted_version(ledger: list[dict]) -> Optional[int]:
    acc = [int(e["version"]) for e in ledger if e.get("accepted") and "version" in e]
    return max(acc) if acc else None


def available_versions() -> list[int]:
    out = []
    for p in VERSIONS_DIR.glob("v*.py"):
        try:
            out.append(int(p.stem[1:]))
        except ValueError:
            continue
    return sorted(out)


def depth_confidence(depth: np.ndarray, bboxes: list[tuple]) -> dict[str, float]:
    """Fraction of finite, positive depth pixels inside each rotated bbox."""
    out = {}
    for name, x, y, w, h, ang in bboxes:
        pts = cv2.boxPoints(((float(x), float(y)), (max(1.0, float(w)), max(1.0, float(h))), float(ang)))
        mask = np.zeros(depth.shape[:2], np.uint8)
        cv2.fillPoly(mask, [np.round(pts).astype(np.int32)], 1)
        sel = depth[mask.astype(bool)]
        out[name] = float(np.mean(np.isfinite(sel) & (sel > 0))) if sel.size else 0.0
    return out


def _num(v: Any, default: float = float("nan")) -> float:
    try:
        f = float(v)
        return f
    except (TypeError, ValueError):
        return default


# ============================================================================================ primitives
def rrect(img: np.ndarray, x: float, y: float, w: float, h: float, r: float, color, thickness: int = -1) -> None:
    """Anti-aliased rounded rectangle (filled when thickness < 0)."""
    x, y, w, h = int(round(x)), int(round(y)), int(round(w)), int(round(h))
    r = int(max(0, min(r, w // 2, h // 2)))
    x1, y1 = x + w - 1, y + h - 1
    if thickness < 0:
        cv2.rectangle(img, (x + r, y), (x1 - r, y1), color, -1)
        cv2.rectangle(img, (x, y + r), (x1, y1 - r), color, -1)
        if r > 0:
            for cx, cy in ((x + r, y + r), (x1 - r, y + r), (x + r, y1 - r), (x1 - r, y1 - r)):
                cv2.circle(img, (cx, cy), r, color, -1, AA)
        return
    t = thickness
    cv2.line(img, (x + r, y), (x1 - r, y), color, t, AA)
    cv2.line(img, (x + r, y1), (x1 - r, y1), color, t, AA)
    cv2.line(img, (x, y + r), (x, y1 - r), color, t, AA)
    cv2.line(img, (x1, y + r), (x1, y1 - r), color, t, AA)
    if r > 0:
        for (cx, cy), a0 in (((x + r, y + r), 180), ((x1 - r, y + r), 270), ((x1 - r, y1 - r), 0), ((x + r, y1 - r), 90)):
            cv2.ellipse(img, (cx, cy), (r, r), 0, a0, a0 + 90, color, t, AA)


def _clip_roi(img: np.ndarray, x: int, y: int, w: int, h: int) -> Optional[tuple[int, int, int, int]]:
    x0, y0, x1, y1 = max(0, x), max(0, y), min(img.shape[1], x + w), min(img.shape[0], y + h)
    return (x0, y0, x1, y1) if x1 > x0 and y1 > y0 else None


def fill_alpha(img: np.ndarray, x: int, y: int, w: int, h: int, r: int, color, alpha: float) -> None:
    """Translucent rounded rectangle."""
    c = _clip_roi(img, int(x), int(y), int(w), int(h))
    if c is None:
        return
    x0, y0, x1, y1 = c
    roi = img[y0:y1, x0:x1]
    layer = roi.copy()
    rrect(layer, x - x0, y - y0, w, h, r, color)
    cv2.addWeighted(layer, alpha, roi, 1 - alpha, 0, dst=roi)


def soft_shadow(img: np.ndarray, x: int, y: int, w: int, h: int, r: int, blur: int = 16, offset: int = 6,
                strength: float = 0.6) -> None:
    """Darkens a blurred rounded-rect footprint (ROI only)."""
    pad = blur * 2
    c = _clip_roi(img, x - pad, y - pad + offset, w + 2 * pad, h + 2 * pad)
    if c is None:
        return
    x0, y0, x1, y1 = c
    m = np.zeros((y1 - y0, x1 - x0), np.float32)
    rrect(m, x - x0, y - y0 + offset, w, h, r, 1.0)
    m = cv2.GaussianBlur(m, (0, 0), blur / 2.0) * strength
    roi = img[y0:y1, x0:x1].astype(np.float32) * (1.0 - m[..., None])
    img[y0:y1, x0:x1] = roi.astype(np.uint8)


def glow(img: np.ndarray, draw: Callable[[np.ndarray], None], bbox: tuple[int, int, int, int], color,
         sigma: float = 8.0, strength: float = 0.9) -> None:
    """Additive coloured glow of whatever `draw(mask)` paints (mask coords = image coords minus bbox origin)."""
    x, y, w, h = bbox
    pad = int(sigma * 3)
    c = _clip_roi(img, x - pad, y - pad, w + 2 * pad, h + 2 * pad)
    if c is None:
        return
    x0, y0, x1, y1 = c
    m = np.zeros((y1 - y0, x1 - x0), np.float32)
    draw_offset = (x0, y0)
    draw(m if draw_offset == (0, 0) else _Offset(m, x0, y0))
    m = cv2.GaussianBlur(m, (0, 0), sigma) * strength
    roi = img[y0:y1, x0:x1].astype(np.float32) + m[..., None] * np.array(color, np.float32)[None, None, :]
    img[y0:y1, x0:x1] = np.clip(roi, 0, 255).astype(np.uint8)


class _Offset:
    """Tiny adaptor so glow draw callbacks can use absolute image coordinates."""
    def __init__(self, arr: np.ndarray, ox: int, oy: int):
        self.arr, self.ox, self.oy = arr, ox, oy

    def circle(self, c, r, thick=-1):
        cv2.circle(self.arr, (int(c[0] - self.ox), int(c[1] - self.oy)), int(r), 1.0, thick, AA)

    def rrect(self, x, y, w, h, r, thick=-1):
        rrect(self.arr, x - self.ox, y - self.oy, w, h, r, 1.0, thick)

    def line(self, p0, p1, thick):
        cv2.line(self.arr, (int(p0[0] - self.ox), int(p0[1] - self.oy)), (int(p1[0] - self.ox), int(p1[1] - self.oy)),
                 1.0, thick, AA)


def tsize(s: str, scale: float, font: int = FS, thick: int = 1) -> tuple[int, int]:
    (w, h), _ = cv2.getTextSize(s, font, scale, thick)
    return w, h


def text(img: np.ndarray, s: str, x: float, y: float, scale: float = 0.5, color=C["text"], font: int = FS,
         thick: int = 1) -> int:
    cv2.putText(img, s, (int(x), int(y)), font, scale, color, thick, AA)
    return int(x) + tsize(s, scale, font, thick)[0]


def text_right(img, s, xr, y, scale=0.5, color=C["text"], font=FS, thick=1) -> None:
    text(img, s, xr - tsize(s, scale, font, thick)[0], y, scale, color, font, thick)


def fit_text(s: str, max_w: int, scale: float, font: int = FS) -> str:
    """Truncate with '..' so the string fits max_w pixels."""
    if tsize(s, scale, font)[0] <= max_w:
        return s
    lo, hi = 0, len(s)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if tsize(s[:mid] + "..", scale, font)[0] <= max_w:
            lo = mid
        else:
            hi = mid - 1
    return s[:lo].rstrip() + ".."


def label_w(s: str, scale: float = 0.36, track: int = 2) -> int:
    return sum(tsize(ch, scale)[0] + track for ch in s.upper()) - track


def label(img: np.ndarray, s: str, x: float, y: float, color=C["muted"], scale: float = 0.36, track: int = 2) -> int:
    """Small uppercase letter-spaced label. Returns end x."""
    for ch in s.upper():
        if ch != " ":
            cv2.putText(img, ch, (int(x), int(y)), FS, scale, color, 1, AA)
        x += tsize(ch, scale)[0] + track
    return int(x)


def pill(img: np.ndarray, x: float, y: float, s: str, fg, bg, scale: float = 0.42, h: int = 22, pad: int = 10,
         outline=None, font: int = FS, thick: int = 1, tracked: bool = False) -> int:
    """Rounded pill with centred text; returns its width."""
    tw = label_w(s, scale) if tracked else tsize(s, scale, font, thick)[0]
    th = tsize("A", scale, font, thick)[1]
    w = tw + 2 * pad
    if bg is not None:
        rrect(img, x, y, w, h, h // 2, bg)
    if outline is not None:
        rrect(img, x, y, w, h, h // 2, outline, 1)
    ty = y + (h + th) / 2
    if tracked:
        label(img, s, x + pad, ty, fg, scale)
    else:
        text(img, s, x + pad, ty, scale, fg, font, thick)
    return w


def arrow_glyph(img: np.ndarray, x: float, ymid: float, color, length: int = 12, thick: int = 2) -> int:
    """Hershey has no '→': draw a shaft + filled head. Returns end x."""
    x, ymid = int(x), int(ymid)
    cv2.line(img, (x, ymid), (x + length - 4, ymid), color, thick, AA)
    head = np.array([[x + length, ymid], [x + length - 6, ymid - 5], [x + length - 6, ymid + 5]], np.int32)
    cv2.fillConvexPoly(img, head, color, AA)
    return x + length


def check_glyph(img, x, ymid, color, size: int = 12) -> None:
    pts = np.array([[x, ymid], [x + size * 0.35, ymid + size * 0.35], [x + size, ymid - size * 0.4]], np.float64)
    cv2.polylines(img, [np.round(pts).astype(np.int32)], False, color, 2, AA)


def cross_glyph(img, x, ymid, color, size: int = 10) -> None:
    h = size // 2
    cv2.line(img, (int(x), int(ymid - h)), (int(x + size), int(ymid + h)), color, 2, AA)
    cv2.line(img, (int(x), int(ymid + h)), (int(x + size), int(ymid - h)), color, 2, AA)


def grad_bar(img: np.ndarray, x: int, y: int, w: int, h: int, frac: float, c0, c1, track=C["card3"]) -> None:
    """Rounded track with a left-to-right gradient fill covering `frac` of it."""
    x, y, w, h = int(x), int(y), int(w), int(h)
    rrect(img, x, y, w, h, h // 2, track)
    frac = 0.0 if not np.isfinite(frac) else float(np.clip(frac, 0.0, 1.0))
    fw = int(round(w * frac))
    if fw < 2 or _clip_roi(img, x, y, fw, h) != (x, y, x + fw, y + h):
        return
    t = np.linspace(0.0, 1.0, fw, dtype=np.float32)[None, :, None]
    strip = np.array(c0, np.float32) * (1 - t) + np.array(c1, np.float32) * t
    mask = np.zeros((h, fw), np.float32)
    rrect(mask, 0, 0, fw, h, h // 2, 1.0)
    roi = img[y:y + h, x:x + fw].astype(np.float32)
    img[y:y + h, x:x + fw] = (strip * mask[..., None] + roi * (1 - mask[..., None])).astype(np.uint8)


_MASKS: dict[tuple[int, int, int], np.ndarray] = {}


def paste_rounded(img: np.ndarray, src: np.ndarray, x: int, y: int, r: int = RADIUS) -> None:
    h, w = src.shape[:2]
    key = (w, h, r)
    if key not in _MASKS:
        m = np.zeros((h, w), np.float32)
        rrect(m, 0, 0, w, h, r, 1.0)
        _MASKS[key] = m[..., None]
    m = _MASKS[key]
    roi = img[y:y + h, x:x + w].astype(np.float32)
    img[y:y + h, x:x + w] = (src.astype(np.float32) * m + roi * (1 - m)).astype(np.uint8)


def vfade(img: np.ndarray, x: int, y: int, w: int, h: int, top_alpha: float, bottom_alpha: float) -> None:
    """Vertical darkening gradient (legibility scrim over images)."""
    a = np.linspace(top_alpha, bottom_alpha, h, dtype=np.float32)[:, None, None]
    roi = img[y:y + h, x:x + w].astype(np.float32)
    img[y:y + h, x:x + w] = (roi * (1 - a) + np.array(C["bg"], np.float32) * a).astype(np.uint8)


# ============================================================================================ chrome
_CHROME: Optional[np.ndarray] = None


def _chrome() -> np.ndarray:
    """Static background: base colour, faint vignette, card shadows, card fills and borders (cached)."""
    global _CHROME
    if _CHROME is None:
        img = np.full((H, W, 3), C["bg"], np.uint8)
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        lift = np.exp(-(((xx - W * 0.3) / (W * 0.6)) ** 2 + ((yy + H * 0.1) / (H * 0.7)) ** 2)) * 6.0
        img = np.clip(img.astype(np.float32) + lift[..., None] * np.array([1.2, 1.0, 1.0]), 0, 255).astype(np.uint8)
        for x, y, w, h in CARD.values():
            soft_shadow(img, x, y, w, h, RADIUS)
        for x, y, w, h in CARD.values():
            rrect(img, x, y, w, h, RADIUS, C["card"])
            rrect(img, x, y, w, h, RADIUS, C["border"], 1)
        _CHROME = img
    return _CHROME


def _to_bgr(rgb: Optional[np.ndarray], w: int, h: int) -> np.ndarray:
    if rgb is None or rgb.ndim != 3:
        out = np.full((h, w, 3), C["card2"], np.uint8)
        text(out, "NO SIGNAL", w // 2 - 50, h // 2, 0.6, C["dim"], FD)
        return out
    bgr = cv2.cvtColor(np.ascontiguousarray(rgb[..., :3]).astype(np.uint8), cv2.COLOR_RGB2BGR)
    return cv2.resize(bgr, (w, h), interpolation=cv2.INTER_AREA)


# ============================================================================================ panels
def _draw_camera(img: np.ndarray, s: HudState, rgb: Optional[np.ndarray]) -> None:
    x, y, w, h = CARD["cam"]
    src_w, src_h = (rgb.shape[1], rgb.shape[0]) if rgb is not None and rgb.ndim == 3 else (640, 480)
    sx, sy = w / src_w, h / src_h
    tile = _to_bgr(rgb, w, h)
    vfade(tile, 0, 0, w, 64, 0.75, 0.0)
    vfade(tile, 0, h - 170, w, 170, 0.0, 0.7)
    paste_rounded(img, tile, x, y)
    rrect(img, x, y, w, h, RADIUS, C["border"], 1)

    for name, bx, by, bw, bh, ang in s.bboxes:
        col = OBJ_COLOR.get(name, C["text"])
        pts = cv2.boxPoints(((x + bx * sx, y + by * sy), (bw * sx, bh * sy), float(ang)))
        ip = np.round(pts).astype(np.int32)
        cv2.polylines(img, [ip], True, (0, 0, 0), 4, AA)
        cv2.polylines(img, [ip], True, col, 2, AA)
        for p in ip:
            cv2.circle(img, tuple(int(v) for v in p), 3, col, -1, AA)
        top = ip[np.argmin(ip[:, 1])]
        tag = name.replace("_", " ").upper()
        tw = label_w(tag, 0.32) + 12
        tx, ty = int(np.clip(top[0] - tw // 2, x + 6, x + w - tw - 6)), int(max(y + 70, top[1] - 24))
        fill_alpha(img, tx, ty, tw, 17, 8, C["ink"], 0.75)
        label(img, tag, tx + 6, ty + 12, col, 0.32, 1)

    # header
    label(img, "CAM_OVERHEAD", x + 18, y + 28, C["text"], 0.4)
    label(img, "RGB-D  /  CORRUPTED FEED", x + 18, y + 46, C["muted"], 0.32)
    ph = f"PHASE  {s.phase.strip().upper() or '-'}"
    pw = label_w(ph, 0.34) + 36
    px = x + w - pw - 16
    fill_alpha(img, px, y + 16, pw, 26, 13, C["ink"], 0.7)
    rrect(img, px, y + 16, pw, 26, 13, C["cyan"], 1)
    cv2.circle(img, (px + 12, y + 29), 3, C["cyan"], -1, AA)
    label(img, ph, px + 20, y + 34, C["cyan"], 0.34)
    if s.webcam_on:
        wl = "WEBCAM OCCLUDER  ON"
        ww = label_w(wl, 0.32) + 36
        fill_alpha(img, x + w - ww - 16, y + 48, ww, 22, 11, C["red"], 0.25)
        rrect(img, x + w - ww - 16, y + 48, ww, 22, 11, C["red"], 1)
        cv2.circle(img, (x + w - ww - 4, y + 59), 3, C["red"], -1, AA)
        label(img, wl, x + w - ww - 16 + 24, y + 63, C["red"], 0.32)

    # object legend cards (bottom-left)
    cy = y + h - 16 - 3 * 50
    for name in OBJECTS:
        col = OBJ_COLOR[name]
        fill_alpha(img, x + 14, cy, 250, 44, 10, C["card"], 0.82)
        rrect(img, x + 14, cy, 250, 44, 10, C["border"], 1)
        cv2.circle(img, (x + 28, cy + 15), 4, col, -1, AA)
        label(img, name.replace("_", " "), x + 38, cy + 19, C["text"], 0.32)
        est = s.estimates.get(name)
        if est is not None and np.all(np.isfinite(np.asarray(est[0], float))):
            p = np.asarray(est[0], float) * 1000.0
            xyz = f"{p[0]:5.0f} {p[1]:5.0f} {p[2]:5.0f} mm"
        else:
            xyz = "  no estimate"
        text_right(img, xyz, x + 254, cy + 19, 0.38, C["text"])
        conf = s.depth_conf.get(name, float("nan"))
        label(img, "DEPTH", x + 24, cy + 36, C["muted"], 0.28)
        bar_c = C["lime"] if conf >= 0.7 else (C["ochre"] if conf >= 0.4 else C["red"])
        grad_bar(img, x + 72, cy + 29, 130, 6, conf if np.isfinite(conf) else 0.0, C["card3"], bar_c)
        text_right(img, f"{conf * 100:.0f}%" if np.isfinite(conf) else "--", x + 254, cy + 37, 0.36, C["muted"])
        cy += 50


def badge_text(s: HudState) -> tuple[str, str]:
    """Perspective badge (text, color key). Reports only what this frame's run has actually reached."""
    if s.shattered:
        return "CORE SHATTERED", "red"
    if s.episode_done:
        return ("CORE EXTRACTED", "lime") if s.extracted else ("NOT EXTRACTED", "ochre")
    if s.started:
        return f"IN PROGRESS  {s.phase.strip() or '-'}", "cyan"
    return "NOT STARTED", "muted"


def _draw_persp(img: np.ndarray, s: HudState, rgb: Optional[np.ndarray]) -> None:
    x, y, w, h = CARD["persp"]
    src_w, src_h = (rgb.shape[1], rgb.shape[0]) if rgb is not None and rgb.ndim == 3 else (640, 480)
    tile = _to_bgr(rgb, w, h)
    if rgb is None:
        tile[:] = (18, 18, 22)
        text(tile, "NO LIVE FRAME", w // 2 - 80, h // 2 - 6, 0.6, C["muted"])
        text(tile, "press [E] to run evolution", w // 2 - 118, h // 2 + 18, 0.42, C["lime"])
    vfade(tile, 0, 0, w, 120, 0.8, 0.0)
    vfade(tile, 0, h - 70, w, 70, 0.0, 0.6)
    paste_rounded(img, tile, x, y)
    label(img, "CAM_PERSPECTIVE", x + 18, y + 28, C["text"], 0.4)
    label(img, "3RD PERSON  /  PRIVILEGED VIEW", x + 18, y + 46, C["muted"], 0.32)

    # wrist force arrow
    if s.force_arrow_px is not None:
        sx, sy = w / src_w, h / src_h
        (u0, v0), (u1, v1) = s.force_arrow_px
        p0 = (x + u0 * sx, y + v0 * sy)
        p1 = (x + u1 * sx, y + v1 * sy)
        if all(np.isfinite(p0 + p1)) and x <= p0[0] <= x + w and y <= p0[1] <= y + h:
            p1 = (float(np.clip(p1[0], x + 4, x + w - 4)), float(np.clip(p1[1], y + 4, y + h - 4)))
            fmag = float(np.linalg.norm(s.wrist_force))
            col = C["red"] if fmag >= s.fracture_n else C["lime"]
            glow(img, lambda m: m.line(p0, p1, 6), (int(min(p0[0], p1[0])), int(min(p0[1], p1[1])),
                 int(abs(p1[0] - p0[0])) + 1, int(abs(p1[1] - p0[1])) + 1), col, 5.0, 0.7)
            ip0, ip1 = tuple(int(v) for v in p0), tuple(int(v) for v in p1)
            cv2.arrowedLine(img, ip0, ip1, col, 2, AA, tipLength=0.25)
            cv2.circle(img, ip0, 4, C["text"], -1, AA)
            tag = f"F {fmag:.1f} N"
            tw = tsize(tag, 0.4)[0] + 14
            tx, ty = int(np.clip(ip1[0] + 8, x + 6, x + w - tw - 6)), int(np.clip(ip1[1] - 10, y + 120, y + h - 30))
            fill_alpha(img, tx, ty, tw, 20, 10, C["ink"], 0.75)
            text(img, tag, tx + 7, ty + 14, 0.4, col)

    # status badge
    msg, col_key = badge_text(s)
    col = C[col_key]
    scale = 0.78
    tw, th = tsize(msg, scale, FD, 2)
    bw, bh = tw + 56, 46
    bx, by = x + (w - bw) // 2, y + 62
    solid = (not s.shattered) or (s.frame_idx % 2 == 0)
    glow(img, lambda m: m.rrect(bx, by, bw, bh, bh // 2), (bx, by, bw, bh), col, 12.0, 0.55 if solid else 0.25)
    if solid:
        rrect(img, bx, by, bw, bh, bh // 2, col)
        fg = C["ink"]
    else:
        fill_alpha(img, bx, by, bw, bh, bh // 2, C["ink"], 0.8)
        rrect(img, bx, by, bw, bh, bh // 2, col, 2)
        fg = col
    cv2.circle(img, (bx + 22, by + bh // 2), 5, fg, -1, AA)
    text(img, msg, bx + 36, by + (bh + th) // 2, scale, fg, FD, 2)

    # live wrist force, shown apart from the badge (not a run outcome)
    fl = f"LIVE WRIST FORCE  {float(np.linalg.norm(s.wrist_force)):.1f} N"
    tl = tsize(fl, 0.36)[0]
    text(img, fl, x + (w - tl) // 2, by + bh + 22, 0.36, C["cyan"])

    # bottom strip
    label(img, "CORE CONTACT NOW", x + 18, y + h - 20, C["muted"], 0.32)
    text(img, f"{s.core_force_now:.1f} N", x + 170, y + h - 19, 0.48, C["text"], FD)
    lim = f"FRACTURE  {s.fracture_n:.1f} N"
    label(img, lim, x + w - 18 - label_w(lim, 0.32), y + h - 20, C["red"], 0.32)


def _segmented(img: np.ndarray, x: int, y: int, w: int, s: HudState) -> None:
    opts = sorted(set([0, 1, 2, 3] + list(s.versions_available)))[:6]
    seg_w, h = w / len(opts), 32
    rrect(img, x, y, w, h, h // 2, C["card2"])
    rrect(img, x, y, w, h, h // 2, C["border"], 1)
    for i, v in enumerate(opts):
        sx = int(x + i * seg_w)
        avail = v in s.versions_available
        if v == s.version:
            rrect(img, sx + 3, y + 3, int(seg_w) - 6, h - 6, (h - 6) // 2, C["card3"])
            rrect(img, sx + 3, y + 3, int(seg_w) - 6, h - 6, (h - 6) // 2, C["cyan"], 1)
        col = C["cyan"] if v == s.version else (C["text"] if avail else C["dim"])
        lbl = f"v{v}"
        tw, th = tsize(lbl, 0.48, FD)
        text(img, lbl, sx + (seg_w - tw) / 2, y + (h + th) / 2, 0.48, col, FD)
        if i > 0 and v != s.version and opts[i - 1] != s.version:
            cv2.line(img, (sx, y + 9), (sx, y + h - 9), C["border"], 1, AA)
        if s.live_version is not None and v == s.live_version:
            hw = label_w("HOT", 0.26) + 10
            hx, hy = int(sx + seg_w - hw + 2), y - 9
            rrect(img, hx, hy, hw, 15, 7, C["red"])
            label(img, "HOT", hx + 5, hy + 11, C["text"], 0.26, 1)


def _draw_rail(img: np.ndarray, s: HudState) -> None:
    x, y, w, h = CARD["rail"]
    ix, iw = x + 18, w - 36
    ex = text(img, "DEEPCORE", ix, y + 34, 0.62, C["text"], FD)
    text(img, "-N1", ex, y + 34, 0.62, C["cyan"], FD)
    pulse = 0.5 + 0.5 * math.sin(s.frame_idx * 0.25)
    dot_c = C["lime"] if not s.episode_done else C["muted"]
    glow(img, lambda m: m.circle((x + w - 58, y + 28), 5), (x + w - 64, y + 22, 12, 12), dot_c, 4.0, 0.4 + 0.5 * pulse)
    cv2.circle(img, (x + w - 58, y + 28), 4, dot_c, -1, AA)
    label(img, "LIVE" if not s.episode_done else "DONE", x + w - 48, y + 33, dot_c, 0.34)

    label(img, "PIPELINE VERSION", ix, y + 66)
    _segmented(img, ix, y + 78, iw, s)
    src = s.version_source or ("baseline" if s.version == 0 else "")
    if src:
        text(img, fit_text(src, iw, 0.36), ix, y + 128, 0.36, C["muted"])

    # 2x2 stat tiles
    stats = [("SEED", f"{s.seed}"), ("T", f"{s.t_sec:4.2f}s"), ("VISION FPS", f"{s.fps:4.1f}"),
             ("LAT P99", f"{s.latency_p99_ms:.1f}ms" if np.isfinite(s.latency_p99_ms) else "--")]
    tw_, th_ = (iw - 8) // 2, 52
    for i, (k, v) in enumerate(stats):
        tx, ty = ix + (i % 2) * (tw_ + 8), y + 142 + (i // 2) * (th_ + 8)
        rrect(img, tx, ty, tw_, th_, 10, C["card2"])
        rrect(img, tx, ty, tw_, th_, 10, C["border"], 1)
        label(img, k, tx + 10, ty + 18, C["muted"], 0.3)
        text(img, v, tx + 10, ty + 42, 0.62, C["text"], FD)

    # contract checklist
    cy = y + 274
    label(img, "CONTRACT  /  LAW 5", ix, cy)
    rmse_ok = np.isfinite(s.rmse_mm) and s.rmse_mm <= s.rmse_target_mm
    lat_ok = np.isfinite(s.latency_p99_ms) and s.latency_p99_ms < s.latency_limit_ms
    rows = [
        (f"Vision RMSE <= {s.rmse_target_mm:.0f} mm", f"{s.rmse_mm:.1f}" if np.isfinite(s.rmse_mm) else "--",
         rmse_ok if np.isfinite(s.rmse_mm) else None),
        (f"Core force < {s.fracture_n:.0f} N",
         f"{s.peak_core_force:.1f}" if (s.frame_idx > 0 and np.isfinite(s.peak_core_force)) else "--",
         (not s.shattered) if (s.frame_idx > 0 and np.isfinite(s.peak_core_force)) else None),
        (f"Latency < {s.latency_limit_ms:.0f} ms", f"{s.latency_p99_ms:.1f}" if np.isfinite(s.latency_p99_ms) else "--",
         lat_ok if np.isfinite(s.latency_p99_ms) else None),
        ("Core extracted", "YES" if s.extracted else ("NO" if s.episode_done else "..."),
         True if s.extracted else (False if s.episode_done or s.shattered else None)),
    ]
    for i, (name, val, ok) in enumerate(rows):
        ry = cy + 12 + i * 34
        rrect(img, ix, ry, iw, 28, 8, C["card2"])
        mid = ry + 14
        if ok is True:
            cv2.circle(img, (ix + 15, mid), 9, C["lime"], 1, AA)
            check_glyph(img, ix + 10, mid, C["lime"], 10)
        elif ok is False:
            cv2.circle(img, (ix + 15, mid), 9, C["red"], 1, AA)
            cross_glyph(img, ix + 11, mid, C["red"], 8)
        else:
            cv2.circle(img, (ix + 15, mid), 9, C["dim"], 1, AA)
            a = s.frame_idx * 0.3
            cv2.ellipse(img, (ix + 15, mid), (9, 9), math.degrees(a) % 360, 0, 90, C["cyan"], 2, AA)
        text(img, name, ix + 32, mid + 5, 0.4, C["text"])
        vc = C["lime"] if ok is True else (C["red"] if ok is False else C["muted"])
        text_right(img, val, ix + iw - 10, mid + 5, 0.42, vc, FD)

    # episode progress
    py = y + h - 26
    label(img, "EPISODE", ix, py - 6, C["muted"], 0.3)
    text_right(img, f"{min(s.t_sec, s.episode_s):.1f} / {s.episode_s:.1f} s", ix + iw, py - 5, 0.34, C["muted"])
    grad_bar(img, ix, py, iw, 6, s.t_sec / max(1e-6, s.episode_s), _hex("0E5F68"), C["cyan"])


def _bezier(p0, p1, n: int = 24) -> np.ndarray:
    (x0, y0), (x1, y1) = p0, p1
    t = np.linspace(0, 1, n)[:, None]
    c0, c1 = np.array([x0, (y0 + y1) / 2]), np.array([x1, (y0 + y1) / 2])
    P0, P3 = np.array([x0, y0]), np.array([x1, y1])
    pts = (1 - t) ** 3 * P0 + 3 * (1 - t) ** 2 * t * c0 + 3 * (1 - t) * t ** 2 * c1 + t ** 3 * P3
    return np.round(pts).astype(np.int32)


STATUS_COLOR = {"idle": C["dim"], "running": C["cyan"], "done": C["lime"], "fail": C["red"]}


def _draw_tree_panel(img: np.ndarray, s: HudState) -> None:
    x, y, w, h = CARD["tree"]
    label(img, "RECURSIVE CONTRACT TREE", x + 18, y + 28, C["text"], 0.38)
    gw = pill(img, x + 18 + label_w("RECURSIVE CONTRACT TREE", 0.38) + 12, y + 14, f"GEN {s.generation}",
              C["cyan"], C["card2"], 0.32, 20, 9, C["border"], tracked=True)
    _ = gw

    # tree region: x .. x+330
    tx0, tcw = x + 18, 300
    cx = tx0 + tcw // 2
    nw, nh = 138, 34
    pos = {"architect": (cx, y + 70), "cv": (tx0 + nw // 2 - 4, y + 150), "control": (tx0 + tcw - nw // 2 + 4, y + 150),
           "verifier": (cx, y + 230)}
    edges = [("architect", "cv"), ("architect", "control"), ("cv", "verifier"), ("control", "verifier")]
    for a, b in edges:
        sa, sb = s.tree_status.get(a, "idle"), s.tree_status.get(b, "idle")
        lit = sa != "idle" and sb != "idle"
        col = STATUS_COLOR[sb] if lit else C["border"]
        pts = _bezier((pos[a][0], pos[a][1] + nh // 2), (pos[b][0], pos[b][1] - nh // 2))
        cv2.polylines(img, [pts], False, col, 2 if lit else 1, AA)
        if lit and sb == "running":
            k = (s.frame_idx * 2) % len(pts)
            cv2.circle(img, tuple(int(v) for v in pts[k]), 3, C["cyan"], -1, AA)
    for k in TREE_NODES:
        st = s.tree_status.get(k, "idle")
        col = STATUS_COLOR.get(st, C["dim"])
        px, py = pos[k][0] - nw // 2, pos[k][1] - nh // 2
        if st == "running":
            amp = 0.35 + 0.35 * (0.5 + 0.5 * math.sin(s.frame_idx * 0.3))
            glow(img, lambda m, px=px, py=py: m.rrect(px, py, nw, nh, nh // 2), (px, py, nw, nh), col, 9.0, amp)
        rrect(img, px, py, nw, nh, nh // 2, C["card2"])
        rrect(img, px, py, nw, nh, nh // 2, col, 2 if st != "idle" else 1)
        cv2.circle(img, (px + 16, py + nh // 2), 4, col, -1, AA)
        text(img, TREE_LABELS[k], px + 28, py + nh // 2 + 5, 0.42, C["text"] if st != "idle" else C["muted"])
    if s.tree_message:
        text(img, fit_text(s.tree_message, tcw, 0.36), tx0, y + 276, 0.36, C["muted"])

    # oscilloscope
    ox, oy, ow, oh = x + 346, y + 44, w - 346 - 18, 210
    rrect(img, ox, oy, ow, oh, 12, C["card2"])
    rrect(img, ox, oy, ow, oh, 12, C["border"], 1)
    label(img, "WRIST FORCE  |F|", ox + 14, oy + 22, C["text"], 0.34)
    cur = s.force_hist[-1] if s.force_hist else 0.0
    text_right(img, f"{cur:5.1f} N", ox + ow - 14, oy + 24, 0.5, C["red"] if cur >= s.fracture_n else C["text"], FD)
    px0, py0, pw, ph = ox + 40, oy + 38, ow - 54, oh - 58
    hist = np.asarray(s.force_hist[-160:], np.float64)
    vmax = max(s.fracture_n * 1.5, float(np.nanmax(hist)) * 1.1 if hist.size else 0.0)
    for gv in np.linspace(0, vmax, 5):
        gy = int(py0 + ph - gv / vmax * ph)
        cv2.line(img, (px0, gy), (px0 + pw, gy), C["border"], 1, AA)
        text_right(img, f"{gv:.0f}", px0 - 6, gy + 4, 0.3, C["dim"])
    thr_y = int(py0 + ph - s.fracture_n / vmax * ph)
    for dx in range(px0, px0 + pw, 10):
        cv2.line(img, (dx, thr_y), (min(dx + 5, px0 + pw), thr_y), C["red"], 1, AA)
    lbl = f"FRACTURE {s.fracture_n:.1f} N"
    label(img, lbl, px0 + pw - label_w(lbl, 0.28) - 4, thr_y - 6, C["red"], 0.28, 1)
    if hist.size >= 2:
        n = 160
        xs = px0 + pw - (len(hist) - 1 - np.arange(len(hist))) * (pw / (n - 1))
        ys = py0 + ph - np.clip(np.nan_to_num(hist), 0, vmax) / vmax * ph
        pts = np.stack([xs, ys], 1)
        poly = np.vstack([pts, [[xs[-1], py0 + ph], [xs[0], py0 + ph]]]).astype(np.int32)
        mask = np.zeros((ph + 1, pw + 1), np.float32)
        cv2.fillPoly(mask, [poly - np.array([px0, py0])], 1.0, AA)
        a = np.linspace(0.38, 0.0, ph + 1, dtype=np.float32)[:, None] * mask
        roi = img[py0:py0 + ph + 1, px0:px0 + pw + 1].astype(np.float32)
        img[py0:py0 + ph + 1, px0:px0 + pw + 1] = (roi * (1 - a[..., None]) + np.array(C["cyan"], np.float32)
                                                    * a[..., None]).astype(np.uint8)
        ip = np.round(pts).astype(np.int32)
        cv2.polylines(img, [ip], False, C["cyan"], 2, AA)
        over = hist >= s.fracture_n
        for i in range(1, len(ip)):
            if over[i] or over[i - 1]:
                cv2.line(img, tuple(ip[i - 1]), tuple(ip[i]), C["red"], 2, AA)
        cv2.circle(img, tuple(ip[-1]), 4, C["red"] if over[-1] else C["cyan"], -1, AA)

    # playbook
    by = y + 296
    cv2.line(img, (x + 18, by - 12), (x + w - 18, by - 12), C["border"], 1, AA)
    label(img, "SHARED BRAIN  /  PLAYBOOK.MD  (LAST 4)", x + 18, by + 4, C["text"], 0.34)
    bullets = s.playbook_bullets[-4:] or ["no playbook entries yet (shared_brain/playbook.md)"]
    for i, b in enumerate(bullets):
        ly = by + 26 + i * 21
        cv2.circle(img, (x + 24, ly - 4), 3, C["lime"], -1, AA)
        text(img, fit_text(b, w - 60, 0.4), x + 36, ly, 0.4, C["text"] if i == len(bullets) - 1 else C["muted"])


def _metric_rows(v0: dict, vn: dict, fracture: float) -> list[tuple]:
    """(label, a, b, scale_max, lower_is_better, fmt, unit) built from ledger holdout aggregates only."""
    f0, f1 = _num(v0.get("peak_core_contact_force_n")), _num(vn.get("peak_core_contact_force_n"))
    r0, r1 = _num(v0.get("vision_3d_rmse_mm")), _num(vn.get("vision_3d_rmse_mm"))
    e0, e1 = _num(v0.get("extraction_rate")), _num(vn.get("extraction_rate"))
    c0, c1 = _num(v0.get("composite_score")), _num(vn.get("composite_score"))
    mx = lambda *v: max([abs(a) for a in v if np.isfinite(a)] + [1e-6])  # noqa: E731
    return [("VISION 3D RMSE", r0, r1, mx(r0, r1), True, "{:.1f}", "mm"),
            ("PEAK CORE FORCE", f0, f1, mx(f0, f1, fracture * 1.25), True, "{:.1f}", "N"),
            ("CORE EXTRACTION RATE", e0, e1, 1.0, False, "{:.0%}", ""),
            ("COMPOSITE HOLDOUT", c0, c1, max(100.0, mx(c0, c1)), False, "{:.1f}", "")]


def _fmt(fmt: str, v: float, unit: str) -> str:
    return (fmt.format(v) + (f" {unit}" if unit else "")) if np.isfinite(v) else "n/a"


def _draw_scoreboard(img: np.ndarray, s: HudState) -> None:
    x, y, w, h = CARD["score"]
    ix, iw = x + 18, w - 36
    seeds = f"SEEDS {s.holdout_seeds[0]}-{s.holdout_seeds[-1]}" if s.holdout_seeds else ""
    label(img, f"UNSEEN HOLDOUT  /  {seeds}", ix, y + 28, C["text"], 0.38)
    pill(img, x + w - 18 - (label_w("LEDGER.JSON", 0.3) + 18), y + 14, "LEDGER.JSON", C["muted"], C["card2"], 0.3, 20,
         9, C["border"], tracked=True)
    base, best = baseline_entry(s.ledger), latest_holdout_entry(s.ledger)

    # footer hotkey chips
    fy = y + h - 40
    cx = ix
    for k, name in (("0-3", "VERSION"), ("N", "NEWEST"), ("W", "WEBCAM"), ("R", "RESEED"), ("Q", "QUIT")):
        chip = f"[{k}] {name}"
        cw = label_w(chip, 0.3) + 20
        rrect(img, cx, fy, cw, 26, 13, C["card2"])
        rrect(img, cx, fy, cw, 26, 13, C["border"], 1)
        label(img, chip, cx + 10, fy + 17, C["muted"], 0.3)
        cx += cw + 8
    cta = "[E] EVOLVE"
    cw = label_w(cta, 0.36) + 50
    ctx = x + w - 18 - cw
    glow(img, lambda m: m.rrect(ctx, fy - 2, cw, 30, 15), (ctx, fy - 2, cw, 30), C["lime"], 8.0, 0.35)
    rrect(img, ctx, fy - 2, cw, 30, 15, C["lime"])
    ex = label(img, cta, ctx + 14, fy + 18, C["ink"], 0.36)
    arrow_glyph(img, ex + 6, fy + 13, C["ink"], 14)

    if base is None or best is None:
        msg = "Ledger is empty" if not s.ledger else "Only the v0 baseline is in the ledger"
        tw = tsize(msg, 0.6, FD)[0]
        text(img, msg, x + (w - tw) // 2, y + 160, 0.6, C["text"], FD)
        sub = "run evolution [E]"
        sw = tsize(sub, 0.55)[0] + 22
        sx0 = x + (w - sw) // 2
        text(img, sub, sx0, y + 200, 0.55, C["lime"])
        arrow_glyph(img, sx0 + sw - 16, y + 195, C["lime"], 14)
        if base is not None:
            c0 = _num(base.get("holdout", {}).get("composite_score"))
            t = f"v0 holdout composite {_fmt('{:.1f}', c0, '')}"
            text(img, t, x + (w - tsize(t, 0.42)[0]) // 2, y + 236, 0.42, C["muted"])
        if s.toast:
            text(img, fit_text(s.toast, iw, 0.4), ix, fy - 16, 0.4, C["cyan"])
        return

    vn = int(best["version"])
    lx = ix
    ex = text(img, "v0", lx, y + 64, 0.8, C["muted"], FD, 2)
    ex = text(img, "  vs  ", ex, y + 64, 0.5, C["dim"], FD)
    ex = text(img, f"v{vn}", ex, y + 64, 0.8, C["cyan"], FD, 2)
    hw = label_w("BEST", 0.28) + 12
    rrect(img, ex + 8, y + 46, hw, 18, 9, C["lime"])
    label(img, "BEST", ex + 14, y + 59, C["ink"], 0.28, 1)
    src = str(best.get("source", ""))
    text(img, fit_text(src, 260, 0.36), ex + hw + 18, y + 61, 0.36, C["muted"])

    h0, h1 = base.get("holdout", {}) or {}, best.get("holdout", {}) or {}
    rows = _metric_rows(h0, h1, s.fracture_n)
    label(img, "V0", ix + 236, y + 78, C["muted"], 0.28)
    label(img, f"LATEST  V{vn}", ix + 236 + 270, y + 78, C["cyan"], 0.28)
    ry = y + 86
    bar_x, bar_w = ix + 236, 270
    for name, a, b, smax, lower, fmt, unit in rows:
        label(img, name, ix, ry + 18, C["muted"], 0.32)
        for j, (v, col0, col1, tag) in enumerate(((a, C["dim"], C["muted"], "v0"), (b, _hex("0E5F68"), C["cyan"], f"v{vn}"))):
            by_ = ry + 6 + j * 16
            label(img, tag, bar_x - 30, by_ + 8, C["dim"], 0.28, 1)
            grad_bar(img, bar_x, by_, bar_w, 9, (abs(v) / smax) if np.isfinite(v) else 0.0, col0, col1)
            vx = text(img, _fmt(fmt, v, unit), bar_x + bar_w + 12, by_ + 10, 0.42, C["text"] if j else C["muted"], FD)
            if name == "PEAK CORE FORCE":
                sh = (h0 if j == 0 else h1).get("core_shattered")
                if sh is not None:
                    pill(img, vx + 8, by_ - 3, "SHATTERED" if sh else "INTACT", C["ink"], C["red"] if sh else C["lime"],
                         0.26, 15, 6, tracked=True)
        if name == "PEAK CORE FORCE":
            tx = int(bar_x + bar_w * s.fracture_n / smax)
            cv2.line(img, (tx, ry + 2), (tx, ry + 34), C["red"], 1, AA)
        if np.isfinite(a) and np.isfinite(b) and abs(a) > 1e-9:
            d = (b - a) / abs(a)
            good = (d < 0) if lower else (d > 0)
            dtxt = f"{d * 100:+.0f}%"
            dw = tsize(dtxt, 0.38, FD)[0] + 16
            pill(img, x + w - 18 - dw, ry + 8, dtxt, C["ink"] if good else C["text"], C["lime"] if good else C["card3"],
                 0.38, 22, 8, font=FD)
        ry += 50

    # sparkline: holdout composite per ledger entry
    sx0, sy0, sw, sh = ix, ry + 4, iw, 62
    rrect(img, sx0, sy0, sw, sh, 10, C["card2"])
    label(img, "COMPOSITE / VERSION", sx0 + 12, sy0 + 18, C["muted"], 0.28)
    ents = sorted([e for e in s.ledger if e.get("holdout")], key=lambda e: int(e.get("version", 0)))
    vals = [_num(e["holdout"].get("composite_score")) for e in ents]
    fin = [v for v in vals if np.isfinite(v)]
    if len(fin) >= 1:
        lo, hi = min(fin + [0.0]), max(fin + [1.0])
        n = len(ents)
        gx0, gx1, gy0, gy1 = sx0 + 170, sx0 + sw - 24, sy0 + 12, sy0 + sh - 18
        pts = []
        for i, (e, v) in enumerate(zip(ents, vals)):
            px = gx0 + (gx1 - gx0) * (i / max(1, n - 1)) if n > 1 else (gx0 + gx1) / 2
            py = gy1 - (gy1 - gy0) * ((v - lo) / max(1e-6, hi - lo)) if np.isfinite(v) else gy1
            pts.append((int(px), int(py), e))
        if len(pts) > 1:
            cv2.polylines(img, [np.array([[p[0], p[1]] for p in pts], np.int32)], False, C["cyan"], 2, AA)
        for px, py, e in pts:
            if e.get("accepted"):
                cv2.circle(img, (px, py), 4, C["lime"], -1, AA)
            else:
                cv2.circle(img, (px, py), 4, C["red"], 1, AA)
            vt = f"v{e.get('version')}"
            text(img, vt, px - tsize(vt, 0.3)[0] // 2, sy0 + sh - 4, 0.3, C["muted"])
    if s.toast:
        text(img, fit_text(s.toast, iw, 0.38), ix, fy - 10, 0.38, C["cyan"])


def _native_image(rgb: Optional[np.ndarray], w: int, h: int) -> np.ndarray:
    """Fit a camera without stretching or hiding anything outside its framing."""
    tile = np.full((h, w, 3), _hex("10171B"), np.uint8)
    if rgb is None or rgb.ndim != 3:
        text(tile, "Preparing mine view", 38, h // 2, .85, C["muted"], FD)
        text(tile, "Waiting for the simulator camera", 38, h // 2 + 35, .5, C["dim"])
        return tile
    ih, iw = rgb.shape[:2]
    scale = min(w / iw, h / ih)
    rw, rh = max(1, round(iw * scale)), max(1, round(ih * scale))
    frame = _to_bgr(rgb, rw, rh)
    ox, oy = (w - rw) // 2, (h - rh) // 2
    tile[oy:oy + rh, ox:ox + rw] = frame
    return tile


def _native_wrap(img: np.ndarray, value: str, x: int, y: int, width: int,
                 lines: int = 2, scale: float = .44, color=None) -> None:
    """Wrap ordinary copy into a fixed-height rail cell."""
    words = str(value).replace("\n", " ").split()
    color = C["muted"] if color is None else color
    for row in range(lines):
        if not words:
            return
        line = words.pop(0)
        while words and tsize(line + " " + words[0], scale)[0] <= width:
            line += " " + words.pop(0)
        if row == lines - 1 and words:
            line = fit_text(line + " " + " ".join(words), width, scale)
        text(img, fit_text(line, width, scale), x, y + row * 23, scale, color)


def _native_metric(img: np.ndarray, x: int, title: str, value: str, unit: str,
                   detail: str, color, history: Optional[list] = None) -> None:
    y, w = 738, 219
    rrect(img, x, y, w, 120, 10, _hex("121C22"))
    label(img, title, x + 16, y + 24, C["muted"], .32, 1)
    end = text(img, value, x + 16, y + 68, 1.02, color, FD, 1)
    text(img, unit, end + 8, y + 67, .4, C["muted"])
    text(img, fit_text(detail, w - 32, .36), x + 16, y + 98, .36, C["muted"])
    if history and len(history) > 1:
        vals = np.asarray(history[-100:], float)
        vals = np.where(np.isfinite(vals), vals, 0)
        high = max(1.0, float(np.max(vals)))
        points = np.stack([np.linspace(x + 16, x + w - 16, len(vals)), y + 114 - 9 * vals / high], 1)
        cv2.polylines(img, [points.astype(np.int32)], False, color, 1, AA)


def _native_sensor(img: np.ndarray, s: HudState, rgb: Optional[np.ndarray]) -> None:
    """Draw actual perception boxes in the overhead camera's own coordinate frame."""
    x, y, w, h = CARD["cam"]
    tile = _native_image(rgb, w, h)
    if rgb is not None and rgb.ndim == 3:
        ih, iw = rgb.shape[:2]
        scale = min(w / iw, h / ih)
        ox, oy = (w - round(iw * scale)) // 2, (h - round(ih * scale)) // 2
        for box in s.bboxes:
            if len(box) < 6:
                continue
            name, bx, by, bw, bh, yaw = box[:6]
            if not np.all(np.isfinite([bx, by, bw, bh, yaw])):
                continue
            corners = cv2.boxPoints(((float(bx), float(by)), (max(1.0, float(bw)), max(1.0, float(bh))), float(yaw)))
            points = np.rint(corners * scale + [ox, oy]).astype(np.int32)
            cv2.polylines(tile, [points], True, OBJ_COLOR.get(name, C["cyan"]), 1, AA)
    paste_rounded(img, tile, x, y, 10)
    fill_alpha(img, x, y, w, 27, 6, C["bg"], .83)
    label(img, "ROBOT VISION", x + 9, y + 18, _hex("6AD5D7"), .26, 1)
    if s.webcam_on:
        fill_alpha(img, x, y + h - 22, w, 22, 5, C["bg"], .92)
        cv2.circle(img, (x + 10, y + h - 10), 3, C["ochre"], -1, AA)
        text(img, "OCCLUSION ON", x + 20, y + h - 6, .3, C["ochre"])


def _native_rail(img: np.ndarray, s: HudState) -> None:
    x, y, w, h = CARD["rail"]
    amber, cyan = _hex("EDB867"), _hex("6AD5D7")
    rrect(img, x, y, w, h, 12, _hex("121C22"))
    ix, right = x + 24, x + w - 24
    label(img, "MISSION CONTROL", ix, y + 29, amber, .34, 1)
    text(img, "Recover. Learn. Repeat.", ix, y + 68, .83, C["text"], FD)
    text(img, "Clear the boulders. Extract an intact core.", ix, y + 98, .43, C["muted"])
    cv2.line(img, (ix, y + 120), (right, y + 120), _hex("28353D"), 1)

    label(img, "RECURSIVE LEARNING", ix, y + 150, cyan, .34, 1)
    statuses = s.tree_status or {}
    active = any(statuses.get(k) == "running" for k in TREE_NODES)
    text_right(img, f"GEN {s.generation}" if s.generation else "STANDBY", right, y + 150, .36, C["muted"])
    descriptions = {"architect": "Define the next improvement", "cv": "Refine object perception",
                    "control": "Refine the extraction policy", "verifier": "Evaluate independent holdouts"}
    for i, node in enumerate(TREE_NODES):
        yy = y + 184 + i * 44
        status = statuses.get(node, "idle")
        col = cyan if status in ("done", "running") else C["red"] if status == "fail" else C["dim"]
        if i < 3:
            cv2.line(img, (ix + 8, yy + 9), (ix + 8, yy + 37), _hex("33434D"), 1, AA)
        cv2.circle(img, (ix + 8, yy), 6, col, -1 if status != "idle" else 1, AA)
        text(img, TREE_LABELS[node], ix + 28, yy + 4, .47, C["text"])
        status_text = {"running": "ACTIVE", "done": "DONE", "fail": "FAILED"}.get(status, "WAITING")
        text_right(img, status_text, right, yy + 4, .32, col)
        text(img, descriptions[node], ix + 28, yy + 22, .34, C["muted"])
    message = s.tree_message or ("Waiting for the first agent event." if not active else "Agents are processing the current generation.")
    _native_wrap(img, message, ix, y + 375, w - 48, 2, .4)
    cv2.line(img, (ix, y + 414), (right, y + 414), _hex("28353D"), 1)

    entries = sorted(s.ledger, key=lambda e: int(e.get("version", 0)))
    improved = [e for e in entries if e.get("accepted") and int(e.get("version", 0)) > 0]
    label(img, "VERIFICATION RECORD", ix, y + 444, amber, .34, 1)
    text_right(img, f"{len(improved)} ACCEPTED", right, y + 444, .32, C["muted"])
    if not entries:
        text(img, "No evaluated versions yet", ix, y + 477, .51, C["text"])
        _native_wrap(img, "A controller improvement appears here only after independent verification.", ix, y + 505, w - 48, 2, .4)
    else:
        for i, entry in enumerate(entries[-2:]):
            yy = y + 478 + i * 54
            version = int(entry.get("version", 0))
            accepted = bool(entry.get("accepted"))
            title = f"v{version}  " + ("Baseline" if version == 0 else "Accepted" if accepted else "Rejected")
            text(img, title, ix, yy, .49, C["text"] if version == 0 else cyan if accepted else amber)
            holdout = entry.get("holdout") or {}
            score = _num(holdout.get("composite_score"))
            if np.isfinite(score):
                text_right(img, f"{score:.3f} score", right, yy, .39, C["muted"])
            reasons = entry.get("reasons") or []
            detail = "; ".join(str(r) for r in reasons) if reasons else str(entry.get("source") or "Independent evaluation recorded")
            text(img, fit_text(detail, w - 48, .36), ix, yy + 21, .36, C["muted"])
    if not improved:
        text(img, "No accepted improvement beyond baseline", ix, y + 591, .39, amber)
    else:
        text(img, f"Latest accepted controller: v{improved[-1]['version']}", ix, y + 591, .39, cyan)
    cv2.line(img, (ix, y + 611), (right, y + 611), _hex("28353D"), 1)
    label(img, "SHARED PLAYBOOK", ix, y + 642, cyan, .34, 1)
    notes = s.playbook_bullets[-2:] if s.playbook_bullets else ["Lessons from evaluated runs will appear here."]
    for i, note in enumerate(notes):
        cv2.circle(img, (ix + 3, y + 673 + i * 48), 2, amber, -1, AA)
        _native_wrap(img, note, ix + 15, y + 678 + i * 48, w - 63, 2, .38)


def render_frame(state: HudState, cam_rgb: Optional[np.ndarray], persp_rgb: Optional[np.ndarray]) -> np.ndarray:
    """Cinematic native control room from measured state and actual simulator RGB."""
    s = state
    amber, cyan = _hex("EDB867"), _hex("6AD5D7")
    img = np.full((H, W, 3), _hex("0B1116"), np.uint8)
    cv2.line(img, (20, 66), (1580, 66), _hex("28353D"), 1)
    cv2.rectangle(img, (20, 22), (26, 46), amber, -1)
    text(img, "DEEPCORE", 39, 45, .95, C["text"], FD)
    text(img, "N1", 223, 45, .95, amber, FD)
    label(img, "AUTONOMOUS EXTRACTION", 291, 40, C["muted"], .35, 1)
    source = "BASELINE" if s.version == 0 else f"CONTROLLER V{s.version}"
    text_right(img, f"{source}     SEED {s.seed}     MUJOCO", 1580, 40, .43, C["muted"])

    x, y, w, h = CARD["persp"]
    paste_rounded(img, _native_image(persp_rgb, w, h), x, y, 12)
    vfade(img, x, y, w, 102, .73, .0)
    vfade(img, x, y + h - 108, w, 108, .0, .9)
    label(img, "EXTRACTION SITE / 01", x + 24, y + 31, amber, .34, 1)
    text(img, "The mine floor", x + 24, y + 65, .88, C["text"], FD)
    pill(img, x + w - 206, y + 19, "SIMULATION VIEW", cyan, _hex("172C32"), .36, 25, 12)
    outcome, color_key = badge_text(s)
    # Phase gets its own line below; preserve the tested outcome classification.
    outcome = outcome.partition("  ")[0]
    outcome_color = {"lime": cyan, "cyan": cyan, "ochre": amber, "muted": amber}.get(color_key, C[color_key])
    if color_key == "red" and s.frame_idx % 2:
        outcome_color = _hex("D88B87")
    cv2.circle(img, (x + 29, y + h - 53), 5, outcome_color, -1, AA)
    label(img, outcome, x + 44, y + h - 48, outcome_color, .35, 1)
    phase = str(s.phase).replace("control:", "").replace("_", " ")
    text(img, fit_text(phase if s.started and phase != "-" else "Select a controller and start an episode", 720, .55),
         x + 25, y + h - 20, .55, C["text"])
    text_right(img, f"{s.t_sec:04.1f} / {s.episode_s:.1f} s", x + w - 25, y + h - 47, .6, C["text"], FD)
    text_right(img, "SIMULATED TIME", x + w - 25, y + h - 23, .3, C["muted"])
    cv2.rectangle(img, (x + 12, y + h - 4), (x + w - 12, y + h - 2), _hex("28353D"), -1)
    progress = float(np.clip(s.t_sec / max(.001, s.episode_s), 0, 1))
    if progress > 0:
        cv2.rectangle(img, (x + 12, y + h - 4), (x + 12 + round((w - 24) * progress), y + h - 2), amber, -1)

    _native_sensor(img, s, cam_rgb)
    core_color = C["red"] if s.shattered else cyan
    force = f"{s.core_force_now:.1f}" if s.started else "--"
    _native_metric(img, 196, "CORE CONTACT", force, "N", f"Peak {s.peak_core_force:.1f} N / limit {s.fracture_n:.0f} N" if s.started else "Waiting for force measurements", core_color)
    wrist = float(np.linalg.norm(s.wrist_force))
    _native_metric(img, 431, "WRIST FORCE", f"{wrist:.1f}" if s.started else "--", "N", "Measured end-effector load", amber, s.force_hist)
    latency = f"{s.latency_p99_ms:.1f}" if np.isfinite(s.latency_p99_ms) else "--"
    _native_metric(img, 666, "VISION LATENCY", latency, "ms", f"P99 / budget {s.latency_limit_ms:.0f} ms", cyan if _num(s.latency_p99_ms, 0) <= s.latency_limit_ms else amber)
    rmse = f"{s.rmse_mm:.1f}" if np.isfinite(s.rmse_mm) else "--"
    _native_metric(img, 901, "POSITION ERROR", rmse, "mm", "Verified vision error" if s.episode_done else "Live estimate vs. scene truth", C["text"])
    _native_rail(img, s)
    key_x = 20
    for key, name in (("0-3", "Version"), ("N", "Latest accepted"), ("E", "Evolve"),
                      ("R", "New seed"), ("W", "Occlusion"), ("Q", "Exit")):
        key_w = tsize(key, .31)[0] + 12
        rrect(img, key_x, 870, key_w, 21, 4, _hex("25333B"))
        text(img, key, key_x + 6, 884, .31, C["text"])
        key_x = text(img, name, key_x + key_w + 6, 884, .35, C["muted"]) + 18
    if s.toast:
        text_right(img, fit_text(s.toast, 700, .36), 1580, 884, .36, cyan)
    return img


# ============================================================================================ privileged helpers
def project_world_points(model: Any, data: Any, cam_name: str, pts: np.ndarray, width: int, height: int
                         ) -> Optional[np.ndarray]:
    """Project world points into a MuJoCo camera image (UI-only privileged access). None if not visible."""
    import mujoco
    cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
    if cid < 0:
        return None
    R = np.asarray(data.cam_xmat[cid]).reshape(3, 3) @ np.diag([1.0, -1.0, -1.0])   # MuJoCo -> OpenCV axes
    pc = (np.asarray(pts, np.float64) - np.asarray(data.cam_xpos[cid])) @ R
    if np.any(pc[:, 2] <= 1e-6):
        return None
    f = 0.5 * height / math.tan(math.radians(float(model.cam_fovy[cid])) / 2.0)
    return np.stack([f * pc[:, 0] / pc[:, 2] + width / 2.0, f * pc[:, 1] / pc[:, 2] + height / 2.0], 1)


def tree_update(status: dict[str, str], event: dict) -> tuple[dict[str, str], str]:
    """Map an orchestrator event {"agent","status","message"} onto the 4-node tree."""
    agent = str(event.get("agent", "")).lower()
    node = next((k for k, kw in (("architect", "architect"), ("verifier", "verif"), ("control", "control"),
                                 ("cv", "cv")) if kw in agent), None)
    st = str(event.get("status", "")).lower()
    mapped = ("running" if st in ("running", "start", "started", "retry", "working") else
              "done" if st in ("done", "ok", "accepted", "pass", "passed") else
              "fail" if st in ("failed", "fail", "rejected", "error") else None)
    out = dict(status)
    if node and mapped:
        if node == "architect" and mapped == "running" and st != "retry":
            out = {k: "idle" for k in TREE_NODES}
        out[node] = mapped
    msg = f"{event.get('agent', '')}: {event.get('message', '')}".strip(": ")
    return out, msg


def _ep_attr(ep: Any, rec: Any, *names: str, default: Any = None) -> Any:
    for obj in (ep, rec):
        for n in names:
            if obj is not None and hasattr(obj, n):
                return getattr(obj, n)
    return default


class FrameBuilder:
    """Turns run_episode frame_callback arguments into (HudState, cam_rgb, persp_rgb) snapshots."""

    def __init__(self, seed: int, version: int, ledger: list[dict], webcam_on: bool = False,
                 persp_camera: str = "cam_perspective"):
        self.seed, self.version, self.ledger, self.webcam_on = seed, version, ledger, webcam_on
        self.persp_camera = persp_camera
        self._last_ep = None
        self.force_hist: deque = deque(maxlen=160)
        self.peak = 0.0
        self.sq_err: list[float] = []
        self.t_wall: deque = deque(maxlen=20)
        try:
            from core import verifier_harness as vh
            self.fracture, self.lat_limit = float(vh.FRACTURE_N), float(vh.LATENCY_CONTRACT_MS)
        except Exception:
            self.fracture, self.lat_limit = 18.0, 25.0
        try:
            from core import env_runner as er
            self.episode_s = float(er.EPISODE_STEPS) * 0.005
            self.hopper = np.asarray(er.HOPPER_ZONE, float)
        except Exception:
            self.episode_s, self.hopper = 7.0, np.array([0.52, 0.16, 0.42])

    def __call__(self, ep: Any, rgb: np.ndarray, depth: np.ndarray, out: dict, rec: Any) -> tuple:
        if self._last_ep is not None and self._last_ep is not ep:
            self.close()
        self._last_ep = ep
        self.t_wall.append(time.perf_counter())
        fps = (len(self.t_wall) - 1) / max(1e-6, self.t_wall[-1] - self.t_wall[0]) if len(self.t_wall) > 1 else 0.0
        dbg = (out or {}).get("debug_overlay", {}) or {}
        bboxes = [tuple(b) for b in dbg.get("bboxes", []) if len(b) >= 6]
        est_raw = (out or {}).get("estimated_objects", {}) or {}
        est = {k: (np.asarray(v.get("pos", np.full(3, np.nan)), float), float(v.get("yaw", 0.0)))
               for k, v in est_raw.items() if isinstance(v, dict)}
        gt = ep.ground_truth()
        for name in OBJECTS:
            if name in est and name in gt and np.all(np.isfinite(est[name][0])):
                self.sq_err.append(float(np.sum((est[name][0] - np.asarray(gt[name][0])) ** 2)) * 1e6)
        rmse = float(np.sqrt(np.mean(self.sq_err[-120:]))) if self.sq_err else float("nan")
        fnow = float(getattr(ep, "core_force_now", 0.0))
        gate = _ep_attr(ep, rec, "core_gate_force", default=None)
        self.peak = max(self.peak, fnow, float(np.max(gate)) if gate is not None and len(gate) else 0.0)
        wf = np.asarray(ep.wrist_force_world(), float)
        self.force_hist.append(float(np.linalg.norm(wf)))
        lat = list(getattr(rec, "latency_ms", []) or [])
        t = float(rec.t[-1]) if getattr(rec, "t", None) else 0.0
        from ui.mine_visuals import render_presentation
        persp = render_presentation(ep, rgb, depth, out, rec)
        core_gt = np.asarray(gt["core"][0], float) if "core" in gt else np.full(3, np.nan)
        shattered = bool(self.peak > self.fracture or getattr(ep, "core_shattered", False))
        extracted = bool(not shattered and np.linalg.norm(core_gt[:2] - self.hopper[:2]) <= 0.045
                         and 0.40 <= core_gt[2] <= 0.45)
        state = HudState(
            seed=self.seed, t_sec=t, episode_s=self.episode_s, fps=fps,
            latency_p99_ms=float(np.percentile(lat, 99)) if lat else float("nan"),
            phase=str(dbg.get("phase", "-")), version=self.version, bboxes=bboxes, estimates=est,
            rmse_mm=rmse, core_force_now=fnow, peak_core_force=self.peak,
            shattered=shattered, extracted=extracted, wrist_force=wf,
            force_hist=list(self.force_hist), fracture_n=self.fracture, latency_limit_ms=self.lat_limit,
            webcam_on=self.webcam_on, ledger=self.ledger, playbook_bullets=load_playbook_bullets(), started=True)
        return state, np.ascontiguousarray(rgb), np.ascontiguousarray(persp)

    def close(self) -> None:
        """Release presentation GPU resources; Episode owns its sensor renderers."""
        ep, self._last_ep = self._last_ep, None
        if ep is not None:
            from ui.mine_visuals import close_render
            close_render(ep)


def pipeline_path(version: int) -> Path:
    p = VERSIONS_DIR / f"v{version}.py"
    if not p.exists() and version == 0 and BASELINE_PATH.exists():
        return BASELINE_PATH
    return p


def finalize_state(state: HudState, rec: Any) -> HudState:
    """Use the locked verifier's settled result for the final outcome and metrics."""
    from core.verifier_harness import score_record
    result = score_record(rec)
    final = copy.copy(state)
    final.episode_done = True
    final.shattered = bool(result["core_shattered"])
    final.extracted = bool(result["core_extracted"])
    final.peak_core_force = float(result["peak_core_contact_force_n"])
    final.latency_p99_ms = float(result["vision_latency_p99_ms"])
    final.rmse_mm = float(result["vision_3d_rmse_mm"])
    return final


# ============================================================================================ live app
class _Abort(Exception):
    pass


class LiveHud:
    """Interactive HUD: episodes in a worker thread, display + hotkeys on the main thread at ~60 FPS."""

    def __init__(self, seed: int = 101, version: Optional[int] = None, window: str = "DeepCore-N1",
                 realtime: bool = True):
        self.window, self.realtime = window, realtime
        self.seed = seed
        self.ledger = load_ledger()
        self.version = version if version is not None else (newest_accepted_version(self.ledger) or 0)
        self._q: "queue.Queue[tuple]" = queue.Queue(maxsize=8)
        self._latest: Optional[tuple] = None
        self._stop = threading.Event()
        self._token = 0
        self._ep_thread: Optional[threading.Thread] = None
        self._evo_thread: Optional[threading.Thread] = None
        self._occluder = None
        self._lock = threading.Lock()
        self.tree_status = {k: "idle" for k in TREE_NODES}
        self.tree_message, self.generation, self.toast = "", 0, ""
        self.frame_idx = 0

    # ---------------------------------------------------------------- episodes
    def _occluder_source(self):
        occ = self._occluder
        if occ is None:
            return None, None
        try:
            return occ.read()
        except Exception:
            return None, None

    def start_episode(self, version: Optional[int] = None, seed: Optional[int] = None) -> None:
        if version is not None:
            path = pipeline_path(version)
            if not path.exists():
                self.toast = f"v{version} not found ({path.name})"
                return
            self.version = version
        if seed is not None:
            self.seed = seed
        self._token += 1
        token = self._token
        self._ep_thread = threading.Thread(target=self._episode_worker, args=(token, self.version, self.seed),
                                           daemon=True)
        self._ep_thread.start()
        self.toast = f"running v{self.version} on seed {self.seed}"

    def _push(self, snap: tuple) -> None:
        try:
            self._q.put_nowait(snap)
        except queue.Full:
            try:
                self._q.get_nowait()
            except queue.Empty:
                pass
            self._q.put_nowait(snap)

    def _episode_worker(self, token: int, version: int, seed: int) -> None:
        try:
            from core.env_runner import run_episode
            from core.verifier_harness import load_pipeline_class
            cls = load_pipeline_class(pipeline_path(version))
        except Exception as exc:
            self.toast = f"cannot load v{version}: {exc}"[:160]
            return
        while not self._stop.is_set() and token == self._token:
            builder = FrameBuilder(seed, version, self.ledger, self._occluder is not None)
            t0 = time.perf_counter()
            last: list = []
            episode: list = []
            completed = False

            def cb(ep, rgb, depth, out, rec):
                episode[:] = [ep]
                if token != self._token or self._stop.is_set():
                    raise _Abort()
                builder.webcam_on = self._occluder is not None
                snap = builder(ep, rgb, depth, out, rec)
                last[:] = [snap]
                if self.realtime:      # pace playback to simulated time (simulation is usually faster)
                    lag = snap[0].t_sec - (time.perf_counter() - t0)
                    if lag > 0:
                        self._stop.wait(lag)
                self._push(snap)

            try:
                record = run_episode(cls, seed, frame_callback=cb, occluder_source=self._occluder_source)
                completed = True
            except _Abort:
                return
            except Exception as exc:
                self.toast = f"episode error: {exc}"[:160]
                return
            finally:
                builder.close()
                # run_episode closes sensors on success; an interrupted callback
                # exits its loop early, so the live app releases them here.
                if not completed and episode:
                    episode[0].close()
            if last:
                # The DONE flag travels with the run's own final frame through the queue, never by
                # mutating whichever frame happens to be on screen (which may be mid-run).
                final, cam, persp = last[0]
                done = finalize_state(final, record)
                self._push((done, cam, persp))
            self._stop.wait(2.5)      # hold the final frame, then replay

    # ---------------------------------------------------------------- evolution
    def _on_event(self, ev: dict) -> None:
        self.tree_status, self.tree_message = tree_update(self.tree_status, ev)
        self.generation = int(ev.get("generation", self.generation) or self.generation)

    def start_evolution(self) -> None:
        if self._evo_thread is not None and self._evo_thread.is_alive():
            self.toast = "evolution already running"
            return

        def worker():
            try:
                from agents.contract_orchestrator import ContractOrchestrator
                from shared_brain.memory_store import SharedBrain
                brain = SharedBrain(ROOT / "shared_brain")
                orch = ContractOrchestrator(brain, log_fn=lambda m: None, event_fn=self._on_event, root=ROOT)
                orch.ensure_baseline()
                g = max((int(e.get("generation", 0)) for e in brain.load_ledger()), default=0) + 1
                entry = orch.run_generation(g)
                self.ledger = load_ledger()
                self.toast = f"generation {g}: v{entry.get('version')} " + ("ACCEPTED" if entry.get("accepted") else "rejected")
            except Exception as exc:
                self.toast = f"evolution error: {exc}"[:160]
                self.tree_status = {**self.tree_status, "verifier": "fail"}

        self._evo_thread = threading.Thread(target=worker, daemon=True)
        self._evo_thread.start()
        self.toast = "evolution generation started"

    # ---------------------------------------------------------------- display
    def _toggle_webcam(self) -> None:
        if self._occluder is not None:
            try:
                self._occluder.close()
            finally:
                self._occluder = None
            self.toast = "webcam occluder off"
            return
        try:
            from core.sensor_corruption import WebcamOccluder
            self._occluder = WebcamOccluder(device=0)
            self.toast = "webcam occluder on"
        except Exception as exc:
            self.toast = f"webcam unavailable: {exc}"[:160]

    def compose(self) -> np.ndarray:
        """Render the latest snapshot with live UI state (tree, toast, versions)."""
        try:
            while True:
                snap = self._q.get_nowait()
                with self._lock:
                    self._latest = snap
        except queue.Empty:
            pass
        with self._lock:
            snap = self._latest
        if snap is None:
            state, cam, persp = HudState(seed=self.seed, version=self.version, ledger=self.ledger,
                                         playbook_bullets=load_playbook_bullets()), None, None
        else:
            state, cam, persp = snap
        state.ledger = self.ledger
        state.versions_available = available_versions() or [0]
        state.live_version = newest_accepted_version(self.ledger)
        state.version = self.version
        state.version_source = next((str(e.get("source", "")) for e in self.ledger
                                     if e.get("version") == self.version and e.get("accepted")), "")
        state.tree_status, state.tree_message, state.generation = dict(self.tree_status), self.tree_message, self.generation
        state.webcam_on = self._occluder is not None
        state.toast, state.frame_idx = self.toast, self.frame_idx // 8   # flash at ~7.5 Hz
        return render_frame(state, cam, persp)

    def handle_key(self, k: int) -> bool:
        """Returns False to quit."""
        if k < 0:
            return True
        c = chr(k & 0xFF).lower() if (k & 0xFF) < 128 else ""
        if c == "q" or (k & 0xFF) == 27:
            return False
        if c in "0123" and c:
            self.start_episode(version=int(c))
        elif c == "n":
            v = newest_accepted_version(load_ledger())
            self.start_episode(version=v if v is not None else 0)
        elif c == "w":
            self._toggle_webcam()
        elif c == "e":
            self.start_evolution()
        elif c == "r":
            self.start_episode(seed=random.randint(101, 999))
        return True

    def run(self) -> None:
        cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window, W, H)
        self.start_episode(version=self.version)
        try:
            while True:
                t0 = time.perf_counter()
                cv2.imshow(self.window, self.compose())
                self.frame_idx += 1
                wait = max(1, int(1000 / 60 - (time.perf_counter() - t0) * 1000))
                if not self.handle_key(cv2.waitKey(wait)):
                    break
        finally:
            self._stop.set()
            self._token += 1
            if self._occluder is not None:
                self._occluder.close()
            cv2.destroyAllWindows()


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser(description="DeepCore-N1 live HUD")
    ap.add_argument("--seed", type=int, default=101)
    ap.add_argument("--version", type=int, default=None)
    ap.add_argument("--no-realtime", action="store_true", help="do not pace playback to simulated time")
    a = ap.parse_args()
    LiveHud(seed=a.seed, version=a.version, realtime=not a.no_realtime).run()


if __name__ == "__main__":
    main()
