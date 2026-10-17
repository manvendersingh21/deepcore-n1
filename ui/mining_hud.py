"""Native MuJoCo view of the measured two-robot mining episode."""
from __future__ import annotations

import json
from pathlib import Path
import time

import cv2
import numpy as np

from ui.drilling_hud import (DrillingFrameBuilder, BG, FG, MUTED, CYAN, AMBER,
                             WIDTH, HEIGHT, _metric, _force_vector)
from ui.live_hud import text, tsize, FD


class MiningFrameBuilder(DrillingFrameBuilder):
    def __call__(self, ep, telemetry):
        frame = super().__call__(ep, telemetry)
        # Replace drilling-only headings with the measured crew activity.
        frame[12:45, 230:850] = BG
        text(frame, 'SUBTERRANEAN ROBOTICS / MINING CREW', 237, 34, .43, MUTED)
        phases = telemetry.get('machine_phases', {})
        for x, name, color, phase in (
            (42, '01  DRILLER', AMBER, phases.get('driller', phases.get('drill', telemetry.get('driller_phase', '')))),
            (620, '02  COLLECTOR', CYAN, phases.get('collector', telemetry.get('collector_phase', ''))),
        ):
            cv2.rectangle(frame, (x - 10, 76), (x + 250, 130), BG, -1)
            text(frame, name, x, 97, .35, color, FD)
            text(frame, str(phase).replace('_', ' ').capitalize(), x, 119, .43, FG)
        tcp = telemetry.get('collector_tcp_m')
        if tcp is not None:
            pixel = self._presentation._project(np.asarray(tcp) + [0, 0, .07])
            if pixel is not None and 25 < pixel[0] < 1320 and 145 < pixel[1] < 750:
                x, y = np.round(pixel).astype(int)
                cv2.rectangle(frame, (x, y - 21), (x + 172, y + 9), BG, -1)
                text(frame, f"GRIP CONTACT {telemetry.get('collector_contact_force_n', 0):.1f} N",
                     x + 8, y, .36, CYAN)
        for contact in sorted(telemetry.get('collector_contacts', []),
                              key=lambda c: np.linalg.norm(c['force_world_n']), reverse=True)[:2]:
            force = np.asarray(contact['force_world_n'], dtype=float)
            magnitude = float(np.linalg.norm(force))
            if magnitude < .2:
                continue
            point = np.asarray(contact['position_m'], dtype=float)
            a = self._presentation._project(point)
            b = self._presentation._project(point + force / magnitude * min(.12, .035 + magnitude * .008))
            if a is not None and b is not None:
                _force_vector(frame, (a, b, magnitude), float('nan'), magnitude)
        for i, center in enumerate(telemetry.get('cart_centers', [])):
            travel = telemetry.get('cart_travel_m', [0, 0, 0])[i]
            count = telemetry.get('cart_counts', [0, 0, 0])[i]
            departed = telemetry.get('cart_departed', [False, False, False])[i]
            pixel = self._presentation._project(np.array([center[0] + travel, center[1], .72]))
            if pixel is not None and 25 < pixel[0] < 1320 and 145 < pixel[1] < 750:
                x, y = np.round(pixel).astype(int)
                label = f'CART {i + 1}  {count}/{telemetry.get("cart_capacity", 2)}'
                if departed:
                    label += '  DISPATCHED'
                elif travel < -.03:
                    label += '  DEPARTING'
                cv2.rectangle(frame, (x - 5, y - 19), (x + tsize(label, .33)[0] + 8, y + 8), BG, -1)
                text(frame, label, x, y, .33, CYAN)
        frame[790:899, 1210:1599] = BG
        count = telemetry.get('debris_collected', 0)
        produced = telemetry.get('debris_produced', 0)
        _metric(frame, 1228, 'DEBRIS RECOVERED', float(count), f'/ {produced}',
                'Measured fragments loaded into rail carts', CYAN, 0)
        return frame


def record(out: str, seed: int = 101, fps: int = 60) -> dict:
    """Write the native episode as it runs, without holding full frames in memory."""
    from ui.mining_fleet import run_mining_fleet
    from scripts.learn_fleet import load_active_parameters
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (WIDTH, HEIGHT))
    if not writer.isOpened():
        raise RuntimeError(f'Could not open video output {path}')
    count = 0
    frame_times = []
    with MiningFrameBuilder(seed) as builder:
        def frame(ep, telemetry):
            nonlocal count
            started = time.perf_counter()
            image = builder(ep, telemetry)
            frame_times.append((time.perf_counter()-started)*1000)
            writer.write(image)
            if count % 100 == 0:
                cv2.imwrite(str(path.with_name(f'{path.stem}_{count:04d}.jpg')), image)
                print(json.dumps({'frame': count, 't': telemetry['t'],
                                  'phase': telemetry['phase'],
                                  'collected': telemetry.get('debris_collected')}), flush=True)
            count += 1
        try:
            result = run_mining_fleet(seed=seed, frame_callback=frame, controller_params=load_active_parameters(),
                                      frame_interval_s=1/fps)
        finally:
            writer.release()
    path.with_suffix('.json').write_text(json.dumps({'seed': seed, 'frames': count, 'encoded_fps': fps,
        'frame_source': 'Distinct actual physics states sampled at the requested callback schedule; sensors remain 20 Hz',
        'native_frame_build_ms_p50': float(np.percentile(frame_times,50)) if frame_times else None,
        'native_frame_build_ms_p95': float(np.percentile(frame_times,95)) if frame_times else None,
        'native_60fps_p95_met': bool(frame_times and np.percentile(frame_times,95)<=1000/60),
        **result}, indent=2))
    return result


def run(seed: int = 101) -> dict:
    from ui.mining_fleet import run_mining_fleet
    from scripts.learn_fleet import load_active_parameters
    window = 'DeepCore | Mine + collect'
    start = time.perf_counter()
    class Quit(Exception):
        pass
    with MiningFrameBuilder(seed) as builder:
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window, WIDTH, HEIGHT)
        def frame(ep, telemetry):
            cv2.imshow(window, builder(ep, telemetry))
            while True:
                if cv2.waitKey(5) & 0xff in (27, ord('q')) or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                    raise Quit()
                if time.perf_counter() >= start + telemetry['t']:
                    break
        try:
            return run_mining_fleet(seed=seed, frame_callback=frame, controller_params=load_active_parameters())
        except Quit:
            return builder.last_telemetry
        finally:
            cv2.destroyWindow(window)
