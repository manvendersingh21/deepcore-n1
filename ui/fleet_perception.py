"""Debris perception from corrupted RGB-D and static calibration only.

No simulator handles, body identifiers, contact state, or object ground truth are
accepted. The first frame is the pre-discharge background observation.
"""
from __future__ import annotations

import cv2
import numpy as np
from collections import deque
import warnings


class FleetObserver:
    def __init__(self, params=None):
        params = params or {}
        self.min_pixels = int(np.clip(params.get('detect_min_pixels', 40), 20, 100))
        self.height_min = float(np.clip(params.get('detection_height_min_m', .012), .006, .03))
        self.memory_s = float(np.clip(params.get('track_memory_s', .8), .25, 2.0))
        self.background = None
        self._depth_window = deque(maxlen=5)
        self._background_window = []
        self.tracks = {}
        self._next_id = 0
        self._consumed = []
        self.stats = {'frames': 0, 'detections': 0, 'empty_frames': 0}

    def mark_consumed(self, track_id):
        """Policy feedback after an observed physical pickup/release, not a GT pose."""
        track = self.tracks.pop(int(track_id), None)
        if track is not None:
            self._consumed.append(np.asarray(track['pos'], float))
            self._consumed = self._consumed[-6:]

    mark_collected = mark_consumed

    def observe(self, rgb, depth, calibration, robot_proprio, t):
        rgb, depth = np.asarray(rgb), np.asarray(depth, dtype=float)
        if rgb.ndim != 3 or rgb.shape[:2] != depth.shape or rgb.shape[2] != 3:
            raise ValueError('Expected aligned RGB HxWx3 and metric optical depth HxW')
        K = np.asarray(calibration['K'], float).reshape(3, 3)
        camera = np.asarray(calibration['cam_pos'], float).reshape(3)
        rotation = np.asarray(calibration['cam_rot'], float).reshape(3, 3)
        if not np.isfinite(K).all() or min(K[0, 0], K[1, 1]) <= 0:
            raise ValueError('Invalid camera calibration')
        t = float(t)
        self.stats['frames'] += 1
        # Median observations suppress independent depth noise/dropouts before
        # foreground segmentation. Every estimate uses sensor samples only.
        measured = depth.astype(np.float32).copy()
        measured[~np.isfinite(measured) | (measured <= .05) | (measured >= 12)] = np.nan
        self._depth_window.append(measured)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            temporal = np.nanmedian(np.stack(self._depth_window), axis=0)
        depth = cv2.medianBlur(np.nan_to_num(temporal, nan=0).astype(np.float32), 5)
        if self.background is None:
            self._background_window.append(depth.copy())
            if len(self._background_window) >= 5:
                self.background = np.median(self._background_window, axis=0)
                self._background_window.clear()
            return []
        v, u = np.indices(depth.shape)
        valid = np.isfinite(depth) & (depth > .05) & (depth < 12)
        # Work only with valid sensor pixels; 0/NaN/spike corruption is not repaired from ideal state.
        yy, xx = np.nonzero(valid)
        d = depth[yy, xx]
        rays = np.column_stack(((xx - K[0, 2]) / K[0, 0] * d,
                                (yy - K[1, 2]) / K[1, 1] * d, d))
        world = rays @ rotation.T + camera
        floor = float(calibration.get('floor_z', .4))
        roi = calibration.get('discharge_roi', [-.1, .65, .10, 1.10])
        xmin, xmax, ymin, ymax = map(float, roi)
        keep = ((world[:, 0] >= xmin) & (world[:, 0] <= xmax)
                & (world[:, 1] >= ymin) & (world[:, 1] <= ymax)
                & (world[:, 2] > floor + self.height_min) & (world[:, 2] < floor + .065))
        colors = rgb[yy, xx].astype(float)
        # Warm mineral colors reject cyan collector, blue joints and yellow mast.
        keep &= ((colors[:, 0] > colors[:, 2] * 1.10)
                 & (colors[:, 1] > colors[:, 2] * 1.03)
                 & (colors[:, 0] < colors[:, 1] * 1.65))
        background = self.background[yy, xx]
        keep &= ~np.isfinite(background) | ((background - d) > .008)
        # Robot poses are own-body proprioception, used only to reject chassis pixels.
        for position in (robot_proprio or {}).get('base_positions', []):
            position = np.asarray(position, float)
            if position.size >= 2:
                keep &= np.linalg.norm(world[:, :2] - position[:2], axis=1) > .20
        mask = np.zeros(depth.shape, np.uint8)
        mask[yy[keep], xx[keep]] = 1
        # Bridge dropout gaps for labeling only; never invent depth samples.
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        points = np.full((*depth.shape, 3), np.nan)
        points[yy[keep], xx[keep]] = world[keep]
        detections = []
        for label in range(1, count):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area < self.min_pixels:
                continue
            cloud = points[labels == label]
            cloud = cloud[np.isfinite(cloud).all(axis=1)]
            if len(cloud) < self.min_pixels:
                continue
            lo, hi = np.percentile(cloud, [10, 90], axis=0)
            extent = hi - lo
            if max(extent[:2]) > .105 or min(extent[:2]) < .020 or extent[2] > .085:
                continue
            cloud = cloud[np.all((cloud >= lo - .008) & (cloud <= hi + .008), axis=1)]
            if len(cloud) < self.min_pixels:
                continue
            pos = np.median(cloud, axis=0)
            # A fragment presents a coherent raised top surface. Thin residual
            # floor-noise islands are rejected even if a few pixels survived.
            if pos[2] < floor + .028:
                continue
            # Observed top surface -> known nominal fragment center, never a body pose.
            pos[2] = max(floor + .02, pos[2] - .021)
            if any(np.linalg.norm(pos[:2] - old[:2]) < .035 for old in self._consumed):
                continue
            confidence = min(.99, .45 + len(cloud) / 70)
            detections.append((confidence, pos))
        detections.sort(key=lambda value: value[0], reverse=True)
        self.tracks = {key: track for key, track in self.tracks.items() if t - track['last_seen'] <= self.memory_s}
        unmatched = set(self.tracks)
        for confidence, pos in detections[:6]:
            candidates = [(np.linalg.norm(np.asarray(self.tracks[key]['pos'])[:2] - pos[:2]), key) for key in unmatched]
            distance, key = min(candidates, default=(float('inf'), None))
            if distance < .075:
                unmatched.remove(key)
                previous = self.tracks[key]
                pos = .65 * pos + .35 * np.asarray(previous['pos'])
                born = previous['born_at']
            elif len(self.tracks) < 6:
                key, born = self._next_id, t
                self._next_id += 1
            else:
                continue
            self.tracks[key] = {'track_id': key, 'pos': pos.tolist(), 'confidence': confidence,
                                'last_seen': t, 'born_at': born, 'provenance': 'corrupted_rgbd'}
        self.stats['detections'] += len(detections[:6])
        self.stats['empty_frames'] += not bool(detections)
        return [{**track, 'age_s': t - track['last_seen'], 'visible': t == track['last_seen']}
                for track in sorted(self.tracks.values(), key=lambda value: value['track_id'])]
