"""Seeded, deterministic RGB-D corruption applied to every frame before the candidate pipeline sees it.

Everything here is a pure function of (seed, frame_idx, clean frame) so a rollout is bit-reproducible.
Live webcam hand occlusion is opt-in (HUD only) and never used by the benchmark.
"""
from __future__ import annotations

import numpy as np
import cv2

HAND_HEIGHT_Z = 0.65          # world height the occluder/hand is rendered at (m)
DEPTH_SIGMA_M = 0.003
EDGE_DROPOUT_P = 0.70         # dropout prob inside the depth-edge band
TOTAL_DROPOUT_FRAC = 0.12     # total NaN/zero fraction targeted per frame
SALT_PEPPER_FRAC = 0.03
OCCLUSION_FRAMES = 15
OCCLUSION_WIDTH_FRAC = 0.25


class SensorCorruption:
    def __init__(self, seed: int, height: int = 480, width: int = 640, cam_height_above_table: float = 0.65):
        self.seed = int(seed)
        self.h, self.w = height, width
        rng = np.random.default_rng([self.seed, 7919])
        self.gamma = float(rng.uniform(0.75, 1.35))
        self.channel_gain = rng.uniform(0.85, 1.15, size=3).astype(np.float32)
        self.pixel_sigma = float(rng.uniform(3.0, 7.0))
        # specular glare blobs (overhead room lights washing out RGB)
        yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
        glare = np.zeros((height, width), np.float32)
        for _ in range(int(rng.integers(2, 4))):
            cx, cy = rng.uniform(0.1 * width, 0.9 * width), rng.uniform(0.15 * height, 0.95 * height)
            sx, sy = rng.uniform(18, 45), rng.uniform(12, 35)
            glare += rng.uniform(130, 230) * np.exp(-(((xx - cx) / sx) ** 2 + ((yy - cy) / sy) ** 2) / 2)
        self.glare = np.clip(glare, 0, 255)[..., None]
        # occluder sweep schedule (frames are 20 Hz vision frames)
        self.occ_start = int(rng.integers(6, 13))
        self.occ_dir = 1 if rng.random() < 0.5 else -1
        self.occ_depth_value = (0.40 + cam_height_above_table) - HAND_HEIGHT_Z  # camera z - occluder z
        self._occ_tex = (rng.uniform(40, 80, size=(height, width, 1)) * np.ones((1, 1, 3))).astype(np.float32)
        self.webcam = None

    # ------------------------------------------------------------------ occlusion
    def occlusion_mask(self, frame_idx: int) -> np.ndarray | None:
        k = frame_idx - self.occ_start
        if k < 0 or k >= OCCLUSION_FRAMES:
            return None
        band = int(OCCLUSION_WIDTH_FRAC * self.w)
        # sweep across the central workspace (where all bricks + targets live)
        lo, hi = int(0.10 * self.w), int(0.90 * self.w) - band
        frac = k / (OCCLUSION_FRAMES - 1)
        x0 = int(lo + (hi - lo) * (frac if self.occ_dir > 0 else 1 - frac))
        m = np.zeros((self.h, self.w), bool)
        m[:, x0:x0 + band] = True
        return m

    # ------------------------------------------------------------------ main entry
    def corrupt(self, rgb: np.ndarray, depth: np.ndarray, frame_idx: int, external_mask: np.ndarray | None = None,
                external_rgb: np.ndarray | None = None):
        """Returns (rgb_uint8, depth_float32_with_nans, occluded_bool)."""
        rng = np.random.default_rng([self.seed, int(frame_idx), 104729])
        h, w = self.h, self.w

        # ---- RGB: gamma, white balance, glare, pixel noise
        img = rgb.astype(np.float32) / 255.0
        img = np.power(img, self.gamma) * 255.0 * self.channel_gain
        img = img + self.glare
        img += rng.normal(0.0, self.pixel_sigma, size=img.shape).astype(np.float32)

        # ---- Depth: noise, edge dropouts, blob dropouts, salt & pepper
        d = depth.astype(np.float32).copy()
        d += rng.normal(0.0, DEPTH_SIGMA_M, size=d.shape).astype(np.float32) * (d / 0.65)
        gx = cv2.Sobel(depth, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(depth, cv2.CV_32F, 0, 1, ksize=3)
        edge = cv2.dilate(((np.abs(gx) + np.abs(gy)) > 0.02).astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        drop = edge & (rng.random((h, w)) < EDGE_DROPOUT_P)
        remaining = max(0.0, TOTAL_DROPOUT_FRAC - drop.mean())
        if remaining > 0:
            field = cv2.resize(rng.random((h // 16, w // 16)).astype(np.float32), (w, h), interpolation=cv2.INTER_CUBIC)
            field += 0.25 * rng.random((h, w)).astype(np.float32)
            thr = np.quantile(field, 1.0 - remaining)
            drop |= field > thr
        nan_or_zero = rng.random((h, w)) < 0.5
        d[drop & nan_or_zero] = np.nan
        d[drop & ~nan_or_zero] = 0.0
        sp = rng.random((h, w)) < SALT_PEPPER_FRAC
        d[sp] = np.where(rng.random(int(sp.sum())) < 0.5, 0.2, 2.5)

        # ---- Occluders (synthetic band, or a live webcam hand composited by the HUD)
        occluded = False
        mask = self.occlusion_mask(frame_idx)
        if mask is not None:
            occluded = True
            alpha = 0.92
            img[mask] = (1 - alpha) * img[mask] + alpha * self._occ_tex[mask]
            d[mask] = self.occ_depth_value + rng.normal(0, DEPTH_SIGMA_M, size=int(mask.sum()))
        if external_mask is not None and external_mask.any():
            occluded = True
            if external_rgb is not None:
                img[external_mask] = external_rgb[external_mask].astype(np.float32)
            else:
                img[external_mask] = (200, 160, 140)
            d[external_mask] = self.occ_depth_value

        return np.clip(img, 0, 255).astype(np.uint8), d.astype(np.float32), occluded


class WebcamOccluder:
    """Live judge-hand stressor: grabs webcam frames and extracts a hand mask (MediaPipe if available, skin threshold otherwise)."""

    def __init__(self, device: int = 0, height: int = 480, width: int = 640):
        self.h, self.w = height, width
        self.cap = cv2.VideoCapture(device)
        self._mp = None
        try:  # optional, heavy
            import mediapipe as mp  # noqa: F401
            self._mp = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=0)
        except Exception:
            self._mp = None

    @property
    def ok(self) -> bool:
        return self.cap is not None and self.cap.isOpened()

    def read(self):
        """Returns (mask_bool HxW, rgb_uint8 HxWx3) or (None, None)."""
        if not self.ok:
            return None, None
        ok, frame = self.cap.read()
        if not ok:
            return None, None
        frame = cv2.resize(cv2.flip(frame, 1), (self.w, self.h))
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        skin = cv2.inRange(ycrcb, (0, 135, 85), (255, 180, 135)) > 0
        if self._mp is not None:
            seg = self._mp.process(rgb).segmentation_mask
            if seg is not None:
                skin &= seg > 0.5
        skin = cv2.morphologyEx(skin.astype(np.uint8), cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
        skin = cv2.morphologyEx(skin, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8)) > 0
        return skin, rgb

    def close(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
