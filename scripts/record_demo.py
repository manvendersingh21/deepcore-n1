"""Records the DeepCore-N1 submission video from real episodes and the real ledger.

Segments (default 60 s): 12 s independent drilling material model, 10 s benchmark v0,
12 s recorded ledger recap, 26 s best accepted recovery on the same holdout seed.
These are separate genuine simulations, not one continuous physical episode.
Episode frames are time-stretched to fill each segment.
Optional --mining-clip prepends an already-recorded fleet MP4 without running a drilling simulation
(default 60 s split: 24 s fleet, 8 s baseline, 10 s ledger recap, 18 s accepted recovery).
Usage: python -m scripts.record_demo [--out submission_demo.mp4] [--seconds 60] [--fps 30] [--fast] [--mining-clip FILE]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterator, Optional

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ui import live_hud as hud  # noqa: E402

SPLIT = (20.0, 15.0, 25.0)
DRILLING_SPLIT = (12.0, 10.0, 12.0, 26.0)
MINING_CLIP_SPLIT = (24.0, 8.0, 10.0, 18.0)


def capture_episode(version: int, seed: int, ledger: list[dict], jpeg_q: int = 94,
                    source_tiles: Optional[dict[str, np.ndarray]] = None) -> list[bytes]:
    """Run one real episode and return its HUD frames plus the settled verifier outcome."""
    from core.env_runner import run_episode
    from core.verifier_harness import load_pipeline_class

    path = hud.pipeline_path(version)
    cls = load_pipeline_class(path)
    builder = hud.FrameBuilder(seed, version, ledger)
    frames: list[bytes] = []
    acc = [e for e in ledger if e.get("accepted")]
    live = hud.newest_accepted_version(ledger)
    src = next((str(e.get("source", "")) for e in acc if e.get("version") == version), "")
    avail = hud.available_versions() or [0]

    last: list = []
    episode: list = []
    completed = False

    def cb(ep, rgb, depth, out, rec):
        episode[:] = [ep]
        state, cam, persp = builder(ep, rgb, depth, out, rec)
        state.versions_available, state.live_version, state.version_source = avail, live, src
        state.frame_idx = len(frames) // 3
        state.toast = f"replaying v{version} on unseen holdout seed {seed}"
        last[:] = [state, cam, persp]
        frames.append(_encode(hud.render_frame(state, cam, persp), jpeg_q))

    try:
        record = run_episode(cls, seed, frame_callback=cb)
        completed = True
    finally:
        try:
            builder.close()
        finally:
            # Callback failures and interrupts bypass run_episode's normal close.
            if not completed and episode:
                episode[0].close()
    if not frames:
        raise RuntimeError(f"episode v{version} seed {seed} produced no frames")
    state, cam, persp = last
    state = hud.finalize_state(state, record)
    if source_tiles is not None:
        # Keep the actual final camera feeds for the recap backdrop; cropping a HUD
        # frame also captures its labels, borders, and outcome overlays.
        source_tiles["cam"] = np.ascontiguousarray(cam.copy())
        source_tiles["persp"] = np.ascontiguousarray(persp.copy())
    frames.append(_encode(hud.render_frame(state, cam, persp), jpeg_q))
    for k in range(max(1, len(frames) // 10)):          # hold the outcome on screen (badge keeps flashing)
        state.frame_idx += 1
        frames.append(_encode(hud.render_frame(state, cam, persp), jpeg_q))
    return frames


def _encode(img: np.ndarray, q: int) -> bytes:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
    if not ok:
        raise RuntimeError("jpeg encode failed")
    return buf.tobytes()


def _decode(b: bytes) -> np.ndarray:
    return cv2.imdecode(np.frombuffer(b, np.uint8), cv2.IMREAD_COLOR)


def write_stretched(writer: cv2.VideoWriter, frames: list[bytes], n_out: int) -> int:
    """Time-stretches captured frames to exactly n_out output frames (repeat / skip by index)."""
    last_i, last = -1, None
    for k in range(n_out):
        i = min(len(frames) - 1, int(k * len(frames) / max(1, n_out)))
        if i != last_i:
            last, last_i = _decode(frames[i]), i
        writer.write(last)
    return n_out


def inspect_mining_clip(path: str | Path) -> dict:
    """Validate a recorded fleet clip and read its timing; retain no decoded frames."""
    clip = Path(path).expanduser().resolve()
    if not clip.is_file():
        raise ValueError(f"mining clip does not exist or is not a file: {clip}")
    try:
        with clip.open("rb") as f:
            f.read(1)
    except OSError as exc:
        raise ValueError(f"mining clip is not readable: {clip}: {exc}") from exc

    cap = cv2.VideoCapture(str(clip))
    try:
        if not cap.isOpened():
            raise ValueError(f"cannot decode mining clip: {clip}")
        source_fps = float(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        ok, first = cap.read()
        if not ok or first is None:
            raise ValueError(f"mining clip contains no decodable frames: {clip}")
        first_h, first_w = first.shape[:2]
        if frame_count <= 0:
            # Some containers omit a frame count; count in a streaming pass.
            frame_count = 1
            while True:
                ok, _ = cap.read()
                if not ok:
                    break
                frame_count += 1
        if source_fps <= 0 or not np.isfinite(source_fps):
            raise ValueError(f"mining clip has invalid source frame rate: {source_fps}")
        if frame_count <= 0:
            raise ValueError("mining clip has no frames")
    finally:
        cap.release()

    sidecar = clip.with_suffix(".json")
    summary, sidecar_error = None, None
    if sidecar.exists():
        try:
            with sidecar.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                summary = data.get("fleet_summary", data.get("summary", data.get("result", data)))
            else:
                summary = data
        except (OSError, json.JSONDecodeError) as exc:
            sidecar_error = f"{type(exc).__name__}: {exc}"
    return {"path": clip, "fps": source_fps, "frame_count": frame_count,
            "duration_s": frame_count / source_fps, "width": first_w, "height": first_h,
            "sidecar": sidecar if sidecar.exists() else None,
            "summary": summary, "sidecar_error": sidecar_error}


def _fit_mining_frame(frame: np.ndarray) -> np.ndarray:
    """Fit a clip frame into the native HUD canvas with aspect-preserving black bars."""
    h, w = frame.shape[:2]
    if (w, h) == (hud.W, hud.H):
        return frame
    scale = min(hud.W / w, hud.H / h)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    resized = cv2.resize(frame, (nw, nh), interpolation=interpolation)
    canvas = np.zeros((hud.H, hud.W, 3), dtype=np.uint8)
    x, y = (hud.W - nw) // 2, (hud.H - nh) // 2
    canvas[y:y + nh, x:x + nw] = resized
    return canvas


def write_mining_clip(writer: cv2.VideoWriter, info: dict, n_out: int) -> int:
    """Stream and index-resample a clip directly to the writer using one frame of memory."""
    cap = cv2.VideoCapture(str(info["path"]))
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"cannot reopen mining clip: {info['path']}")
    frame_idx, frame = -1, None
    try:
        for k in range(n_out):
            target_idx = min(info["frame_count"] - 1,
                             int(k * info["frame_count"] / max(1, n_out)))
            while frame_idx < target_idx:
                ok, frame = cap.read()
                if not ok or frame is None:
                    raise RuntimeError(f"mining clip ended before source frame {target_idx}")
                frame_idx += 1
            writer.write(_fit_mining_frame(frame))
    finally:
        cap.release()
    return n_out


def evolution_frames(ledger: list[dict], playbook: list[str], cam: Optional[np.ndarray],
                     persp: Optional[np.ndarray], n_out: int, seed: int) -> Iterator[np.ndarray]:
    """Show recorded ledger entries as static recap slides, without inventing agent activity."""
    entries = sorted(ledger, key=lambda e: int(e.get("version", 0)))
    newest = hud.newest_accepted_version(entries)
    for k in range(n_out):
        if not entries:
            message = "No ledger history is recorded."
            version, source, generation = 0, "", 0
            toast = "No recorded learning history in shared_brain/ledger.json"
        else:
            index = min(len(entries) - 1, int(k * len(entries) / max(1, n_out)))
            entry = entries[index]
            version = int(entry.get("version", 0))
            source = str(entry.get("source", "")).strip()
            generation = int(entry.get("generation", 0) or 0)
            verdict = "accepted" if entry.get("accepted") else "rejected"
            reasons = "; ".join(map(str, entry.get("reasons", [])))
            suffix = f" ({reasons})" if reasons and not entry.get("accepted") else ""
            origin = f" from {source}" if source else ""
            message = f"Recorded v{version}{origin}: {verdict}{suffix}"
            toast = f"Recorded ledger entry {index + 1}/{len(entries)}; no live agent activity"
        s = hud.HudState(seed=seed, phase="-", version=version,
                         versions_available=hud.available_versions() or [0],
                         live_version=newest, version_source=source,
                         tree_message=message, generation=generation,
                         playbook_bullets=playbook[-4:], ledger=entries, frame_idx=0,
                         toast=toast)
        yield hud.render_frame(s, cam, persp)


def main(out: str = "submission_demo.mp4", seconds: float = 60, fps: int = 30, opt_seed: int = 11,
         holdout_seed: int = 101, fast: bool = False, include_drilling: bool = True,
         mining_clip: Optional[str] = None) -> Path:
    if fast:
        seconds, fps = min(seconds, 6.0), 10
    clip_info = inspect_mining_clip(mining_clip) if mining_clip is not None else None
    out_path = Path(out) if Path(out).is_absolute() else ROOT / out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ledger = hud.load_ledger()
    best = hud.best_holdout_entry(ledger)
    best_v = int(best["version"]) if best else hud.newest_accepted_version(ledger)
    if best_v is None or best_v == 0:
        print("record_demo: ledger has no accepted version beyond v0 - final segment replays v0 (run evolution first)")
        best_v = 0
    if clip_info is not None:
        split = MINING_CLIP_SPLIT
        names = ["mining_fleet", "baseline", "ledger_recap", "accepted_recovery"]
    else:
        split = DRILLING_SPLIT if include_drilling else SPLIT
        names = (["drilling"] if include_drilling else []) + ["baseline", "ledger_recap", "accepted_recovery"]
    seg = [round(seconds * f / sum(split) * fps) for f in split]
    seg[-1] += round(seconds * fps) - sum(seg)
    segment_counts = dict(zip(names, seg))
    mine_note = f"; recorded fleet clip {clip_info['path']}" if clip_info is not None else ""
    print(f"record_demo: opt seed {opt_seed} (reference only), holdout seed {holdout_seed}; v0 vs v{best_v}; "
          f"{seconds:.0f}s @ {fps} fps -> {out_path}{mine_note}")

    source_tiles: dict[str, np.ndarray] = {}
    drilling_frames: list[bytes] = []
    drilling_summary = None
    if include_drilling and clip_info is None:
        from ui.drilling_hud import capture
        drilling_frames, drilling_summary = capture(seed=holdout_seed)
        if not drilling_frames:
            raise RuntimeError("drilling simulation produced no frames")
    v0_frames = capture_episode(0, holdout_seed, ledger, source_tiles=source_tiles)
    best_frames = v0_frames if best_v == 0 else capture_episode(best_v, holdout_seed, ledger)
    cam, persp = source_tiles["cam"], source_tiles["persp"]

    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (hud.W, hud.H))
    if not writer.isOpened():
        raise RuntimeError(f"cannot open video writer for {out_path}")
    try:
        n = 0
        if clip_info is not None:
            n += write_mining_clip(writer, clip_info, segment_counts["mining_fleet"])
        elif include_drilling:
            n += write_stretched(writer, drilling_frames, segment_counts["drilling"])
        n += write_stretched(writer, v0_frames, segment_counts["baseline"])
        evo = evolution_frames(ledger, hud.load_playbook_bullets(n=12), cam, persp,
                               segment_counts["ledger_recap"], holdout_seed)
        for f in evo:
            writer.write(f)
            n += 1
        n += write_stretched(writer, best_frames, segment_counts["accepted_recovery"])
    finally:
        writer.release()
    if not out_path.exists() or out_path.stat().st_size == 0:
        raise RuntimeError(f"video {out_path} was not written")
    from core.env_runner import SIM_DT, VISION_EVERY
    cadence = {
        "benchmark_callback_hz": 1.0 / (SIM_DT * VISION_EVERY),
        "benchmark_frames_include_settled_outcome_and_hold": True,
        "baseline_captured_frames": len(v0_frames),
        "recovery_captured_frames": len(best_frames),
        "ledger_recap": "static recorded entries; no live agent activity",
        "playback": "captured episode frames repeated or skipped to fit segment duration",
    }
    mining_fleet_clip = None
    if clip_info is not None:
        segment_s = segment_counts["mining_fleet"] / fps
        mining_fleet_clip = {
            "path": str(clip_info["path"]),
            "summary_sidecar": str(clip_info["sidecar"]) if clip_info["sidecar"] else None,
            "summary": clip_info["summary"],
            "summary_sidecar_error": clip_info["sidecar_error"],
        }
        cadence["mining_fleet_clip"] = {
            "source_fps": clip_info["fps"],
            "source_frames": clip_info["frame_count"],
            "source_duration_s": clip_info["duration_s"],
            "source_resolution": [clip_info["width"], clip_info["height"]],
            "output_frames": segment_counts["mining_fleet"],
            "output_fps": fps,
            "segment_duration_s": segment_s,
            "source_to_segment_time_ratio": clip_info["duration_s"] / segment_s if segment_s else None,
            "frame_resampling": "sequential decode; output frame k selects floor(k * source_frames / output_frames), repeating or skipping frames",
            "resolution_handling": "aspect-preserving resize with black letterbox bars to 1600x900 when source resolution differs",
            "resolution_resized": (clip_info["width"], clip_info["height"]) != (hud.W, hud.H),
            "output_resolution": [hud.W, hud.H],
            "continuity": "recorded fleet footage; separate from the later benchmark and recovery episodes",
        }
    elif include_drilling:
        from ui.drilling_scene import FRAME_INTERVAL_S
        cadence.update(drilling_callback_hz=1.0 / FRAME_INTERVAL_S,
                       drilling_captured_frames=len(drilling_frames))
    metadata = {
        "video": out_path.name, "output_fps": fps, "output_frames": n,
        "duration_s": n / fps, "seed": holdout_seed,
        "baseline_version": 0, "chosen_accepted_version": best_v if any(
            entry.get("accepted") and int(entry.get("version", -1)) == best_v for entry in ledger) else None,
        "recovery_version": best_v,
        "drilling_summary": drilling_summary,
        "mining_fleet_clip": mining_fleet_clip,
        "sequence": ("Recorded mining fleet clip, then a separate v0 episode, static ledger recap, and separate accepted-version recovery episode"
                     if clip_info is not None else
                     "Independent drilling material model, then separate benchmark recovery episodes and recorded ledger recap"
                     if include_drilling else "Separate benchmark recovery episodes and recorded ledger recap"),
        "segments": [{"name": name, "frames": segment_counts[name],
                      "duration_s": segment_counts[name] / fps} for name in names],
        "source_cadence": cadence,
    }
    out_path.with_suffix(".json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    print(f"record_demo: wrote {n} frames ({n / fps:.1f}s), {out_path.stat().st_size / 1e6:.1f} MB")
    return out_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="submission_demo.mp4")
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--opt-seed", type=int, default=11)
    ap.add_argument("--holdout-seed", type=int, default=101)
    ap.add_argument("--fast", action="store_true", help="short preview: <=6 s at 10 fps (full 1600x900 frames)")
    ap.add_argument("--no-drilling", action="store_true", help="skip the live drilling simulation; a supplied mining clip is still included")
    ap.add_argument("--mining-clip", help="prepend an existing native fleet MP4 instead of running drilling simulation")
    a = ap.parse_args()
    main(a.out, a.seconds, a.fps, a.opt_seed, a.holdout_seed, a.fast, not a.no_drilling, a.mining_clip)
