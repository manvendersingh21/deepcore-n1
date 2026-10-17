"""Version 0 — deliberately naive excavation pipeline (spec Phase 2).

Flaws (on purpose, so the recursive agents have something real to fix):
  * raw RGB channel thresholds (break under headlamp glare / dust haze)
  * depth read at ONE pixel (the mask centroid) -> NaN dropouts and 0.25/1.80 m dust spikes go straight through
  * reports the visible top surface as the object centre (no half-height correction)
  * ignores the overburden rubble and lunges straight down onto the buried core at t=0.5 s, no force monitoring
"""
import numpy as np
import cv2

NAMES = ("core", "rubble_a", "rubble_b")


class ExcavationPipeline:
    def __init__(self, cam_intrinsics: np.ndarray, cam_pos: np.ndarray, cam_rot: np.ndarray):
        self.K = np.asarray(cam_intrinsics, dtype=np.float64)
        self.cam_pos = np.asarray(cam_pos, dtype=np.float64)
        self.cam_rot = np.asarray(cam_rot, dtype=np.float64)
        self.grasped_at = None

    def _masks(self, rgb: np.ndarray) -> dict:
        r, g, b = (rgb[..., i].astype(np.int16) for i in range(3))
        return {
            "core": (g > 150) & (b > 150) & (r < 100),
            "rubble_a": (r > 150) & (g > 80) & (g < 160) & (b < 80),
            "rubble_b": (r > 150) & (b > 120) & (g < 100),
        }

    def _backproject(self, u: float, v: float, d: float) -> np.ndarray:
        fx, fy, cx, cy = self.K[0, 0], self.K[1, 1], self.K[0, 2], self.K[1, 2]
        return self.cam_pos + self.cam_rot @ np.array([(u - cx) / fx * d, (v - cy) / fy * d, d])

    def step(self, t_sec: float, rgb: np.ndarray, depth: np.ndarray, proprio: dict) -> dict:
        est, bboxes = {}, []
        for name, m in self._masks(rgb).items():
            vs, us = np.nonzero(m)
            if us.size < 20:
                est[name] = {"pos": np.full(3, np.nan), "yaw": 0.0}
                continue
            u, v = float(us.mean()), float(vs.mean())
            d = float(depth[int(v), int(u)])  # single pixel: NaN / spikes pass straight through
            est[name] = {"pos": self._backproject(u, v, d), "yaw": 0.0}
            (cx, cy), (w, h), ang = cv2.minAreaRect(np.stack([us, vs], 1).astype(np.float32))
            bboxes.append((name, cx, cy, w, h, ang))

        core = est["core"]["pos"]
        target_xy = core[:2] if np.all(np.isfinite(core)) else np.asarray(proprio["ee_pos"])[:2]
        if t_sec < 0.5:
            phase, target, grip = "HOVER_OVER_CORE", np.array([target_xy[0], target_xy[1], 0.60]), 1.0
        elif t_sec < 1.6:
            phase, target, grip = "LUNGE_STRAIGHT_DOWN", np.array([target_xy[0], target_xy[1], 0.46]), 1.0
        elif t_sec < 2.0:
            phase, target, grip = "CLOSE_GRIPPER", np.array([target_xy[0], target_xy[1], 0.46]), 0.0
        else:
            phase, target, grip = "CARRY_TO_HOPPER", np.array([0.52, 0.16, 0.50]), 0.0
        return {
            "estimated_objects": est,
            "action": {"target_pos": target, "target_yaw": 0.0, "gripper": grip},
            "debug_overlay": {"bboxes": bboxes, "phase": phase},
        }
