"""Presentation packets derived solely from the actual corrupted camera samples.

No simulator poses enter backprojection or segmentation. Object brackets use
the policy's estimates; trajectory endpoints use its issued command.
"""
from __future__ import annotations

import base64
import inspect

import cv2
import numpy as np


def _jpeg(rgb):
    small = cv2.resize(np.asarray(rgb, np.uint8), (320, 240))
    ok, jpg = cv2.imencode('.jpg', cv2.cvtColor(small, cv2.COLOR_RGB2BGR),
                           [cv2.IMWRITE_JPEG_QUALITY, 75])
    return base64.b64encode(jpg).decode() if ok else None


def camera_packet(rgb, depth, K, camera_pos, camera_rotation, mask_provider=None):
    """Sample observed optical depth; invalid/dropout samples remain absent."""
    rgb, depth = np.asarray(rgb), np.asarray(depth, float)
    K = np.asarray(K, float).reshape(3, 3)
    camera_pos = np.asarray(camera_pos, float).reshape(3)
    rotation = np.asarray(camera_rotation, float).reshape(3, 3)
    h, w = depth.shape
    if rgb.shape != (h, w, 3) or not np.isfinite(K).all() or min(K[0, 0], K[1, 1]) <= 0:
        raise ValueError('Invalid aligned RGB-D/calibration')
    yy, xx = np.mgrid[0:h:12, 0:w:12]
    values = depth[yy, xx]
    valid = np.isfinite(values) & (values > .05) & (values < 8)
    u, v, d = xx[valid], yy[valid], values[valid]
    rays = np.column_stack(((u - K[0, 2]) / K[0, 0] * d,
                            (v - K[1, 2]) / K[1, 1] * d, d))
    points = rays @ rotation.T + camera_pos
    # Display crop only: keep corrupted points above/below surfaces, no repair.
    keep = ((points[:, 0] > -3) & (points[:, 0] < 4)
            & (points[:, 1] > -2) & (points[:, 1] < 2) & (points[:, 2] > .25) & (points[:, 2] < 2.5))
    points, colors = points[keep], rgb[v[keep], u[keep]]
    stride = max(1, int(np.ceil(len(points) / 1400)))
    range_m = float(np.clip(np.nanmedian(d) if d.size else .6, .1, 2.5))
    pixels = np.array([[0, 0], [w, 0], [w, h], [0, h]], float)
    rays = np.column_stack(((pixels[:, 0] - K[0, 2]) / K[0, 0] * range_m,
                            (pixels[:, 1] - K[1, 2]) / K[1, 1] * range_m,
                            np.full(4, range_m)))
    valid_depth = np.isfinite(depth) & (depth > .05) & (depth < 8)
    normalized = np.zeros(depth.shape, np.uint8)
    normalized[valid_depth] = np.clip(depth[valid_depth] / max(range_m * 1.5, .1) * 255, 0, 255).astype(np.uint8)
    depth_rgb = cv2.cvtColor(cv2.applyColorMap(normalized, cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)
    depth_rgb[~valid_depth] = 0
    mask_rgb = None
    mask_source = 'No segmentation output supplied by this controller'
    if mask_provider is not None:
        # Replay the exact selected pipeline's pure segmentation stage on the
        # same corrupted input, outside its timed control step.
        method = mask_provider._masks
        masks = method(rgb, depth) if len(inspect.signature(method).parameters) == 2 else method(rgb)
        mask_rgb = np.zeros_like(rgb)
        for name, color in [('core', (60, 235, 241)), ('rubble_a', (236, 175, 77)), ('rubble_b', (188, 136, 241))]:
            if name in masks:
                mask_rgb[np.asarray(masks[name], bool)] = color
        mask_source = 'Selected pipeline segmentation stage, replayed on its actual corrupted input'
    return {
        'provenance': 'actual_corrupted_rgbd',
        'points': np.round(points[::stride], 5).reshape(-1).tolist(),
        'colors': np.round(colors[::stride].astype(float) / 255, 3).reshape(-1).tolist(),
        'camera_pos': camera_pos.tolist(), 'camera_rot': rotation.reshape(-1).tolist(),
        'frustum_corners': (rays @ rotation.T + camera_pos).tolist(),
        'K': K.tolist(), 'image_size': [w, h],
        'depth_image': _jpeg(depth_rgb), 'mask_image': _jpeg(mask_rgb) if mask_rgb is not None else None,
        'mask_source': mask_source,
    }
