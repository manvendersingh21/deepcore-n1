"""Native presentation of measured MuJoCo drilling, at 1600 x 900.

The presentation owns its renderer and model copy. Neither the simulation's
model nor its sensor renderer is changed. All motion, loads, and progress come
from ``ui.drilling_scene``; the surrounding quarry rocks are static scenery.
"""
from __future__ import annotations

from collections import deque
from copy import copy
from pathlib import Path
import time
from typing import Any

import cv2
import numpy as np

from ui.live_hud import AA, FD, FS, _hex, fill_alpha, text, tsize


WIDTH, HEIGHT = 1600, 900
VIEW = (20, 58, 1560, 720)
BG = _hex("0C1217")
FG = _hex("E8EDEB")
MUTED = _hex("91A2A8")
CYAN = _hex("65DCCB")
AMBER = _hex("DEA75B")
RED = _hex("EF765D")
RULE = _hex("29373E")


def _number(telemetry: dict, key: str) -> float:
    try:
        value = float(telemetry[key])
        return value if np.isfinite(value) else float("nan")
    except (KeyError, TypeError, ValueError):
        return float("nan")


def _formatted(value: float, places: int = 1) -> str:
    return f"{value:.{places}f}" if np.isfinite(value) else "--"


def _texture_assets(model: Any, mujoco: Any) -> None:
    """Upload photographs into the private visual model before renderer creation."""
    assets = Path(__file__).resolve().parents[1] / "assets"
    files = {
        "tex_earth": "soil_ambientcg_Ground112/Ground112_1K-JPG_Color.jpg",
        "tex_rock": "rock_ambientcg_Rock064/Rock064_1K-JPG_Color.jpg",
        "rock_texture": "rock_ambientcg_Rock064/Rock064_1K-JPG_Color.jpg",
        "tex_wall": "rock_ambientcg_Rock064/Rock064_1K-JPG_Color.jpg",
    }
    for name, filename in files.items():
        tid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_TEXTURE, name)
        if tid < 0 or model.tex_nchannel[tid] != 3:
            continue
        image = cv2.imread(str(assets / filename))
        if image is None:
            continue
        width, height = int(model.tex_width[tid]), int(model.tex_height[tid])
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        start = int(model.tex_adr[tid])
        model.tex_data[start:start + width * height * 3] = image.reshape(-1)


def _quarry_rocks() -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Static outcrop outside the physical rig, in its world coordinate system."""
    rng = np.random.default_rng(8042)
    rocks = []
    for i in range(98):
        if i < 48:
            x, y = rng.uniform(-.65, 1.45), rng.uniform(.66, .97)
            radius = rng.uniform(.065, .17)
        elif i < 76:
            x, y = rng.uniform(-.67, -.40), rng.uniform(-.60, .85)
            radius = rng.uniform(.055, .15)
        else:
            x, y = rng.uniform(-.4, 1.25), rng.uniform(-.78, -.57)
            radius = rng.uniform(.008, .030)
        size = np.array([radius, radius * rng.uniform(.65, 1.3), radius * rng.uniform(.45, .9)])
        pos = np.array([x, y, .36 + size[2] * .55])
        angle = rng.uniform(-np.pi, np.pi)
        c, s = np.cos(angle), np.sin(angle)
        mat = np.array([c, -s, 0, s, c, 0, 0, 0, 1], dtype=float)
        shade = rng.uniform(.63, .88)
        rgba = np.array([shade, shade * .91, shade * .79, 1], np.float32)
        rocks.append((size, pos, mat, rgba))
    return rocks


class _Presentation:
    def __init__(self, ep: Any):
        import mujoco

        self.mujoco = mujoco
        from ui.ar_overlays import VisualModel
        self.visual = VisualModel(ep.model)
        self.model = self.visual.model
        self.model.vis.global_.offwidth = VIEW[2]
        self.model.vis.global_.offheight = VIEW[3]
        self.model.vis.global_.fovy = 39
        _texture_assets(self.model, mujoco)
        self.camera = mujoco.MjvCamera()
        self.camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.camera.lookat[:] = (.42, 0, .64)
        self.camera.distance = 1.65
        self.camera.azimuth = 126
        self.camera.elevation = -36
        self.fleet = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'drill_mobile_base') >= 0
        if self.fleet:
            self.camera.lookat[:] = (-.05, .25, .66)
            self.camera.distance = 4.1
            self.camera.azimuth = 112
            self.camera.elevation = -23
        self.option = mujoco.MjvOption()
        self.option.sitegroup[:] = 0
        self.rocks = []  # Faceted mesh environment replaces primitive quarry dressing.
        self._sensor_packet = None
        self._sensor_at = -1.
        self.renderer = mujoco.Renderer(self.model, VIEW[3], VIEW[2], max_geom=6500)
        self.names = {i: mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, i) or ""
                      for i in range(self.model.ngeom)}

    def _add_rocks(self, scene: Any, texid: int, matid: int) -> None:
        mj = self.mujoco
        for i, (size, pos, mat, rgba) in enumerate(self.rocks):
            if scene.ngeom >= scene.maxgeom:
                break
            geom = scene.geoms[scene.ngeom]
            kind = mj.mjtGeom.mjGEOM_BOX if i % 3 == 0 else mj.mjtGeom.mjGEOM_ELLIPSOID
            mj.mjv_initGeom(geom, kind, size, pos, mat, rgba)
            geom.category = mj.mjtCatBit.mjCAT_DECOR
            geom.objtype = mj.mjtObj.mjOBJ_UNKNOWN
            geom.objid = geom.segid = -1
            geom.matid = matid
            geom.texid = texid
            geom.texuniform = 1
            geom.texrepeat[:] = (2, 2)
            geom.specular = .04
            geom.shininess = .02
            scene.ngeom += 1

    def _project(self, point: np.ndarray) -> np.ndarray | None:
        """Project against the actual presentation camera built by MuJoCo."""
        cameras = self.renderer.scene.camera
        pos = .5 * (np.asarray(cameras[0].pos) + np.asarray(cameras[1].pos))
        forward = np.asarray(cameras[0].forward, dtype=float)
        up = np.asarray(cameras[0].up, dtype=float)
        right = np.cross(forward, up)
        delta = np.asarray(point, dtype=float) - pos
        depth = float(np.dot(delta, forward))
        if depth <= 1e-6:
            return None
        cam = cameras[0]
        near = float(cam.frustum_near)
        bottom, top = float(cam.frustum_bottom), float(cam.frustum_top)
        height = top - bottom
        if near <= 0 or height <= 0:
            return None
        # Perspective frustum, with a centered horizontal field of view.
        px = VIEW[2] / 2 + np.dot(delta, right) * near / depth * VIEW[3] / height
        py = (top - np.dot(delta, up) * near / depth) * VIEW[3] / height
        return np.array([px + VIEW[0], py + VIEW[1]], dtype=float)

    def render(self, ep: Any, telemetry: dict) -> tuple[np.ndarray, tuple | None]:
        mj = self.mujoco
        # Removed layers become transparent in the simulation. Propagate that
        # state into the visual copy on every frame, including completion.
        visual_data = self.visual.sync(ep)
        self.renderer.update_scene(visual_data, camera=self.camera, scene_option=self.option)
        scene = self.renderer.scene
        scene.flags[mj.mjtRndFlag.mjRND_SHADOW] = True
        scene.flags[mj.mjtRndFlag.mjRND_FOG] = False
        scene.flags[mj.mjtRndFlag.mjRND_HAZE] = False
        if scene.nlight:
            scene.lights[0].diffuse[:] = (.72, .78, .86)
            scene.lights[0].ambient[:] = (.11, .12, .14)
            scene.lights[0].cutoff = 80
            scene.lights[0].exponent = 1
        if scene.nlight > 1:
            scene.lights[1].diffuse[:] = (.22, .27, .34)
            scene.lights[1].ambient[:] = (.025, .03, .035)
        rock_texture = rock_material = -1
        for geom in scene.geoms[:scene.ngeom]:
            if geom.objtype != mj.mjtObj.mjOBJ_GEOM:
                continue
            name = self.names.get(geom.objid, "")
            if name.startswith('cmu_tunnel_'):
                # The measured survey remains in the source assets. Its floating
                # cutaway sheets obscure the robots inside this synthetic vault.
                geom.rgba[3] = 0
                continue
            if name == "rock_base":
                rock_texture = geom.texid
                rock_material = geom.matid
            if name.startswith(("rock_", "stratum_", "bore_wall_", "scree_")):
                geom.rgba[:3] = (.36, .39, .40)
                geom.specular = .06
                geom.texrepeat[:] = (1, 1)
            elif name in ("world_floor", "pit_floor"):
                geom.rgba[:3] = (.28, .30, .31)
                geom.texrepeat[:] = (6, 6)
        from ui.ar_overlays import add_headlamp, add_sensor_overlays
        add_headlamp(scene, ep)
        estimates, command = {}, None
        if self.fleet:
            rgb = getattr(ep, '_last_sensor_rgb', None)
            depth = getattr(ep, '_last_sensor_depth', None)
            if rgb is not None and depth is not None and (float(telemetry['t']) - self._sensor_at >= .199 or self._sensor_packet is None):
                from ui.sensor_overlay import camera_packet
                cid = self.mujoco.mj_name2id(ep.model, self.mujoco.mjtObj.mjOBJ_CAMERA, 'cam_overhead')
                h,w = depth.shape
                focal = h/2 / np.tan(np.deg2rad(ep.model.cam_fovy[cid])/2)
                rotation = ep.data.cam_xmat[cid].reshape(3,3) @ np.diag([1.,-1.,-1.])
                self._sensor_packet = camera_packet(rgb, depth, [[focal,0,w/2],[0,focal,h/2],[0,0,1]],
                                                     ep.data.cam_xpos[cid], rotation)
                self._sensor_at = float(telemetry['t'])
            estimates = {f"debris_{track['track_id']}": {'pos': track['pos'], 'orientation_observed': False}
                         for track in telemetry.get('sensor_tracks', [])}
            if hasattr(ep, '_arm'): command = ep._arm.goal
        add_sensor_overlays(scene, self._sensor_packet, estimates, command,
                            telemetry.get('collector_tcp_m', ep.ee_pos()))
        vector = None
        force = _number(telemetry, "axial_force_n")
        # The extended scene telemetry supplies the full contact reaction. The
        # stable API also supports an axial-only vector for this vertical drill.
        force_world = np.asarray(telemetry.get("contact_force_world_n", [0, 0, force]), float)
        point = np.asarray(telemetry.get("contact_position_m", ep.ee_pos()), float)
        if (force_world.shape == (3,) and point.shape == (3,) and np.all(np.isfinite(force_world))
                and np.all(np.isfinite(point))):
            magnitude = float(np.linalg.norm(force_world))
            if magnitude > .5:
                end = point + force_world / magnitude * min(.18, magnitude * .0013)
                start_px, end_px = self._project(point), self._project(end)
                if start_px is not None and end_px is not None:
                    vector = (start_px, end_px, magnitude)
        return self.renderer.render().copy(), vector

    def close(self) -> None:
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None


def _rounded_image(frame: np.ndarray, image_bgr: np.ndarray, box: tuple, radius: int = 12) -> None:
    x, y, w, h = box
    scaled = cv2.resize(image_bgr, (w, h), interpolation=cv2.INTER_AREA)
    frame[y:y + h, x:x + w] = scaled
    # Only clip the four small corners; avoid a full-frame compositing buffer.
    for cx, cy, start in ((x + radius, y + radius, 180),
                          (x + w - radius - 1, y + radius, 270),
                          (x + radius, y + h - radius - 1, 90),
                          (x + w - radius - 1, y + h - radius - 1, 0)):
        if start == 180:
            ox, oy = x, y
        elif start == 270:
            ox, oy = x + w - radius, y
        elif start == 90:
            ox, oy = x, y + h - radius
        else:
            ox, oy = x + w - radius, y + h - radius
        tile = frame[oy:oy + radius, ox:ox + radius]
        yy, xx = np.ogrid[oy:oy + radius, ox:ox + radius]
        tile[(xx - cx) ** 2 + (yy - cy) ** 2 > radius ** 2] = BG


def _force_vector(frame: np.ndarray, vector: tuple | None, limit: float, axial: float) -> None:
    if vector is None:
        return
    a, b, magnitude = vector
    x, y, w, h = VIEW
    if not (x + 8 < a[0] < x + w - 8 and y + 8 < a[1] < y + h - 8):
        return
    b = np.clip(b, [x + 10, y + 10], [x + w - 10, y + h - 10])
    p0, p1 = tuple(np.round(a).astype(int)), tuple(np.round(b).astype(int))
    color = RED if np.isfinite(limit) and axial > limit else CYAN
    cv2.arrowedLine(frame, p0, p1, _hex("14242A"), 6, AA, tipLength=.20)
    cv2.arrowedLine(frame, p0, p1, color, 2, AA, tipLength=.20)
    cv2.circle(frame, p0, 4, color, -1, AA)
    label = f"{magnitude:.1f} N"
    label_w = tsize(label, .48, FS, 1)[0] + 22
    lx = int(np.clip(b[0] + 18, x + 8, x + w - label_w - 8))
    ly = int(np.clip(b[1] - 30, y + 8, y + h - 60))
    fill_alpha(frame, lx, ly, label_w, 47, 7, BG, .87)
    text(frame, "CONTACT", lx + 11, ly + 15, .27, MUTED)
    text(frame, label, lx + 11, ly + 35, .48, color)


def _metric(frame: np.ndarray, x: int, label: str, value: float, unit: str,
            caption: str, color: tuple, places: int = 1) -> None:
    text(frame, label, x, 809, .35, MUTED)
    end = text(frame, _formatted(value, places), x, 849, 1.04, color, FD, 1)
    text(frame, unit, end + 10, 846, .43, MUTED)
    text(frame, caption, x, 876, .34, MUTED)


class DrillingFrameBuilder:
    """Callable ``(episode, telemetry) -> BGR frame`` with explicit cleanup."""

    def __init__(self, seed: int = 101):
        self.seed = int(seed)
        self._ep = None
        self._presentation = None
        self.force_history: deque[float] = deque(maxlen=140)
        self._last_t = None
        self.last_frame: np.ndarray | None = None
        self.last_telemetry: dict = {}

    def __call__(self, ep: Any, telemetry: dict) -> np.ndarray:
        if self._ep is not ep:
            self.close()
            self._ep = ep
            self._presentation = _Presentation(ep)
        t = _number(telemetry, "t")
        force = _number(telemetry, "axial_force_n")
        limit = _number(telemetry, "limit_force_n")
        if t != self._last_t:
            self.force_history.append(force)
            self._last_t = t
        scene_rgb, vector = self._presentation.render(ep, telemetry)
        sensor_rgb = ep.render_camera()
        frame = np.full((HEIGHT, WIDTH, 3), BG, dtype=np.uint8)
        text(frame, "DEEPCORE N1", 24, 36, .64, FG, FD, 1)
        cv2.line(frame, (218, 18), (218, 39), RULE, 1, AA)
        text(frame, "Simulated rock layers", 237, 34, .43, MUTED)
        phase = str(telemetry.get("phase", "")).replace("_", " ").strip().capitalize()
        if phase == "Spinup":
            phase = "Spinning up"
        complete = bool(telemetry.get("complete", False))
        phase_color = CYAN if complete or not (np.isfinite(limit) and force > limit) else RED
        clock = f"{t:04.1f} s" if np.isfinite(t) else "-- s"
        clock_width = tsize(clock, .45)[0]
        text(frame, clock, 1576 - clock_width, 35, .45, MUTED)
        phase_width = tsize(phase, .47)[0]
        phase_x = 1548 - clock_width - phase_width
        text(frame, phase, phase_x, 35, .47, phase_color)
        cv2.circle(frame, (phase_x - 14, 29), 3, phase_color, -1, AA)
        _rounded_image(frame, cv2.cvtColor(scene_rgb, cv2.COLOR_RGB2BGR), VIEW)
        _force_vector(frame, vector, limit, force)

        # A small, unaltered sensor image preserves the actual robot camera view.
        sx, sy, sw, sh = 1378, 608, 182, 137
        fill_alpha(frame, sx - 8, sy - 27, sw + 16, sh + 35, 8, BG, .85)
        text(frame, "ROBOT CAMERA", sx, sy - 9, .29, MUTED)
        _rounded_image(frame, cv2.cvtColor(sensor_rgb, cv2.COLOR_RGB2BGR), (sx, sy, sw, sh), 4)

        peak = _number(telemetry, "peak_force_n")
        force_caption = f"Peak {_formatted(peak)} / limit {_formatted(limit, 0)} N"
        force_color = RED if np.isfinite(limit) and force > limit else CYAN
        _metric(frame, 28, "AXIAL THRUST", force, "N", force_caption, force_color)
        _metric(frame, 428, "SPINDLE TORQUE", _number(telemetry, "torque_nm"), "Nm",
                "Measured motor torque", AMBER, 2)
        removed = _number(telemetry, "layers_removed")
        total = _number(telemetry, "layers_total")
        depth_caption = f"{_formatted(removed, 0)} / {_formatted(total, 0)} layers removed"
        max_depth = _number(telemetry, "max_depth_mm")
        if np.isfinite(max_depth):
            depth_caption += f" / max {max_depth:.1f} mm"
        _metric(frame, 828, "DRILL DEPTH", _number(telemetry, "depth_mm"), "mm", depth_caption, FG)
        _metric(frame, 1228, "ROTATION", _number(telemetry, "rpm"), "rpm", "Measured spindle speed", FG, 0)
        for x in (403, 803, 1203):
            cv2.line(frame, (x, 798), (x, 877), RULE, 1, AA)
        # Short load history beside the thrust readout; every point is a sample.
        history = np.asarray(self.force_history, float)
        if history.size > 1 and np.all(np.isfinite(history)):
            scale = max(float(np.max(history)), limit if np.isfinite(limit) else 1.0, 1.0)
            xx = np.linspace(243, 382, history.size)
            yy = 850 - np.clip(history / scale, 0, 1) * 30
            points = np.column_stack((xx, yy)).round().astype(np.int32)
            cv2.polylines(frame, [points], False, CYAN, 1, AA)
        self.last_frame = frame
        self.last_telemetry = dict(telemetry)
        return frame

    def close(self) -> None:
        if self._presentation is not None:
            try:
                self._presentation.close()
            finally:
                self._presentation = None
                self._ep = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


def capture(seed: int = 101, jpeg_quality: int = 94) -> tuple[list[bytes], dict]:
    """Capture the 20 Hz simulation callbacks as JPEG frames and a true summary."""
    from ui.drilling_scene import run_drilling

    frames: list[bytes] = []
    with DrillingFrameBuilder(seed) as builder:
        def on_frame(ep, telemetry):
            frame = builder(ep, telemetry)
            ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)])
            if not ok:
                raise RuntimeError("Could not encode the native drilling frame")
            frames.append(encoded.tobytes())

        summary = run_drilling(seed=seed, frame_callback=on_frame)
    return frames, summary


class _QuitDrilling(Exception):
    pass


def run(seed: int = 101) -> dict:
    """Run a paced native OpenCV window; Q, Escape, or closing it exits cleanly."""
    from ui.drilling_scene import run_drilling

    window = "DeepCore N1 | Drilling"
    builder = DrillingFrameBuilder(seed)
    started_at = time.perf_counter()
    summary: dict = {}
    opened = False
    try:
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        opened = True
        cv2.resizeWindow(window, WIDTH, HEIGHT)

        def on_frame(ep, telemetry):
            cv2.imshow(window, builder(ep, telemetry))
            if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q"), 27):
                raise _QuitDrilling
            if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                raise _QuitDrilling
            due = started_at + float(telemetry["t"])
            while time.perf_counter() < due:
                if cv2.waitKey(5) & 0xFF in (ord("q"), ord("Q"), 27):
                    raise _QuitDrilling

        summary = run_drilling(seed=seed, frame_callback=on_frame)
        # Leave the measured final state visible until the viewer closes it.
        while cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) >= 1:
            if cv2.waitKey(50) & 0xFF in (ord("q"), ord("Q"), 27):
                break
    except _QuitDrilling:
        summary = {**builder.last_telemetry, "aborted": True, "seed": int(seed),
                   "material_model": "breakable layers"}
    finally:
        builder.close()
        if opened:
            try:
                cv2.destroyWindow(window)
            except cv2.error:
                pass
    return summary


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=101)
    args = parser.parse_args()
    print(json.dumps(run(seed=args.seed), indent=2))
