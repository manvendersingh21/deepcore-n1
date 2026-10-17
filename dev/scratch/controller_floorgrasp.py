"""Evolved DeepCore excavation pipeline (fallback patch library source).

FEATURES selects which verified mutations are active; agents/fallback_library.py materialises v1/v2/v3 from this one
source so every generation is a complete, standalone ExcavationPipeline file.
"""
import numpy as np
import cv2

FEATURES = {"hsv_iqr": True, "z_ordering": True, "yaw": True, "tracker": True, "force_guard": True, "gentle_place": True,
            "elevation_gate": True, "grasp_top": True, "core_grasp": True}

NAMES = ("core", "rubble_a", "rubble_b")
HALF_H = {"core": 0.022, "rubble_a": 0.020, "rubble_b": 0.020}
# HSV bands (OpenCV H in [0,180)): cyan core, ochre cap rock, magenta slag
HSV_BANDS = {"core": [((80, 90, 70), (100, 255, 255))],
             "rubble_a": [((8, 170, 120), (22, 255, 220))],
             "rubble_b": [((145, 70, 60), (172, 255, 255))]}
DUMP = {"rubble_a": np.array([0.30, -0.16]), "rubble_b": np.array([0.22, -0.20])}
HOPPER = np.array([0.52, 0.16])
SAFE_Z = 0.56
TOUCH_N = 5.5                 # wrist |Fz| that halts the descent (spec: 5.5 N)
ELEV_MIN_Z = 0.412           # world z above pit floor (0.400) for a pixel to count as an object
FINGER_OPEN = 0.09            # gripper cmd 1.0 == 90 mm pad-to-pad gap (INTERFACE)
GRASP_W = {"core": 0.044, "rubble_a": 0.048, "rubble_b": 0.048}   # nominal grasped width (m)
SQUEEZE = {"core": 0.016, "rubble_a": 0.016, "rubble_b": 0.016}   # closure past contact for a friction hold
PLACE_Z = {"core": 0.433, "rubble_a": 0.495, "rubble_b": 0.495}   # air release; deepest dangle (pad edge + diagonal) clears
SOFT_OPEN = 0.80               # partial release: clears object corners and hopper walls, no snap


class ExcavationPipeline:
    def __init__(self, cam_intrinsics: np.ndarray, cam_pos: np.ndarray, cam_rot: np.ndarray):
        self.K = np.asarray(cam_intrinsics, dtype=np.float64)
        self.cam_pos = np.asarray(cam_pos, dtype=np.float64)
        self.cam_rot = np.asarray(cam_rot, dtype=np.float64)
        self.track = {n: None for n in NAMES}       # (pos, yaw)
        self.seen = {n: 0 for n in NAMES}
        self.cand = {n: None for n in NAMES}        # consistent re-detection candidate (pos, count)
        self.queue = None                            # excavation order, decided after the first good frames
        self.phase, self.phase_t, self.lock = "SURVEY", 0.0, None
        self.grip_hold, self.stall, self.grip_cmd = None, 0, 1.0  # latched hold, stall ticks, rate-limited cmd
        self.tries = {n: 0 for n in NAMES}     # grasp attempts per object (max 2-3)
        self.dumped = {n: False for n in NAMES}  # rubble that already reached the dump
        self.kernel = np.ones((5, 5), np.uint8)

    # ------------------------------------------------------------------ perception
    def _masks(self, rgb: np.ndarray, depth: np.ndarray) -> dict:
        if not FEATURES["hsv_iqr"]:
            r, g, b = (rgb[..., i].astype(np.int16) for i in range(3))
            return {"core": (g > 150) & (b > 150) & (r < 100), "rubble_a": (r > 150) & (g > 80) & (g < 160) & (b < 80),
                    "rubble_b": (r > 150) & (b > 120) & (g < 100)}
        hsv = cv2.cvtColor(cv2.GaussianBlur(rgb, (5, 5), 0), cv2.COLOR_RGB2HSV)
        out = {}
        elevated = np.isfinite(depth) & (depth > 0.3) & ((self.cam_pos[2] - depth) > ELEV_MIN_Z)
        for n, bands in HSV_BANDS.items():
            m = np.zeros(rgb.shape[:2], np.uint8)
            for lo, hi in bands:
                m |= cv2.inRange(hsv, lo, hi)
            if FEATURES.get("elevation_gate"):  # objects sit above the floor; the flat dump marker does not
                m &= elevated.astype(np.uint8) * 255
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN, self.kernel)
            m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, self.kernel)
            k, lab, stats, _ = cv2.connectedComponentsWithStats(m)
            if k > 1:  # keep the blob that looks most like the object: near the core footprint, not the largest
                out[n] = self._pick_blob(lab, stats, k, hsv, n)
            else:
                out[n] = m > 0
        return out

    def _pick_blob(self, lab, stats, k, hsv, name):
        # the dump-zone marker shares the ochre band and is often bigger than the boulder: prefer the blob whose
        # colour is closest to the band centre and whose area is boulder-sized, with the core as a reference point
        best, best_score = None, -1e9
        centre = np.mean(HSV_BANDS[name][0], axis=0)
        for i in range(1, k):
            area = stats[i, cv2.CC_STAT_AREA]
            if area < 60 or area > 9000:
                continue
            col = np.median(hsv[lab == i], axis=0)
            score = -float(np.abs(col[0] - centre[0])) - 0.02 * float(area)
            if score > best_score:
                best, best_score = i, score
        return (lab == best) if best is not None else np.zeros(lab.shape, bool)

    def _backproject(self, u, v, d):
        fx, fy, cx, cy = self.K[0, 0], self.K[1, 1], self.K[0, 2], self.K[1, 2]
        pts = np.stack([(u - cx) / fx * d, (v - cy) / fy * d, d], -1)
        return self.cam_pos + pts @ self.cam_rot.T

    def _measure(self, name, mask, depth):
        vs, us = np.nonzero(mask)
        if us.size < 60:
            return None
        d = depth[vs, us]
        ok = np.isfinite(d) & (d > 0.3) & (d < 1.0)
        if FEATURES["hsv_iqr"]:
            if ok.sum() < 30:
                return None
            q1, q3 = np.percentile(d[ok], [25, 75])
            ok &= (d >= q1) & (d <= q3)
            pts = self._backproject(us[ok].astype(np.float64), vs[ok].astype(np.float64), d[ok].astype(np.float64))
            p = np.median(pts, axis=0)
            p[2] -= HALF_H[name]
        else:
            u, v = float(us.mean()), float(vs.mean())
            p = self._backproject(np.array([u]), np.array([v]), np.array([float(depth[int(v), int(u)])]))[0]
        yaw = 0.0
        (cx, cy), (w, h), ang = cv2.minAreaRect(np.stack([us, vs], 1).astype(np.float32))
        if FEATURES["yaw"]:
            # image long axis -> world yaw (image +u = world -y, image +v = world -x)
            a = np.deg2rad(ang if w >= h else ang + 90.0)
            du, dv = np.cos(a), np.sin(a)
            yaw = float(np.arctan2(-du, -dv))
            yaw = (yaw + np.pi / 2) % np.pi - np.pi / 2
        return p, yaw, (name, float(cx), float(cy), float(w), float(h), float(ang)), int(us.size)

    def _update_tracks(self, rgb, depth, ee_pos):
        masks = self._masks(rgb, depth)
        bboxes = []
        for n in NAMES:
            m = self._measure(n, masks[n], depth)
            if m is None:
                continue
            p, yaw, box, npx = m
            bboxes.append(box)
            prev = self.track[n]
            if FEATURES["tracker"] and prev is not None:
                jump = np.linalg.norm(p - prev[0])
                held = self.lock is not None and self.lock[0] == n and self.phase in ("GRASP", "LIFT", "TRANSPORT", "PLACE")
                if jump > 0.06 and not held:   # dust plume / hand: reject teleports, hold the track
                    if not FEATURES.get("core_grasp"):
                        continue
                    c = self.cand[n]
                    if c is not None and np.linalg.norm(p - c[0]) < 0.03:
                        c = (0.5 * (c[0] + p), c[1] + 1)
                    else:
                        c = (p.copy(), 1)
                    if c[1] < 3:               # not yet consistent: keep the old lock
                        self.cand[n] = c
                        continue
                    self.cand[n] = None        # 3 consistent re-detections: the old lock was wrong
                    self.track[n] = (p, yaw)
                    self.seen[n] += 1
                    continue
                a = 0.35 if self.seen[n] > 3 else 0.8
                p = (1 - a) * prev[0] + a * p
                dy = (yaw - prev[1] + np.pi / 2) % np.pi - np.pi / 2
                yaw = prev[1] + a * dy
            self.track[n] = (p, yaw)
            self.seen[n] += 1
        return bboxes

    # ------------------------------------------------------------------ control
    def _order(self):
        rubble = [n for n in ("rubble_a", "rubble_b") if self.track[n] is not None]
        if FEATURES["z_ordering"]:
            if FEATURES.get("core_grasp"):
                # lowest first: the leaner at the core's edge sits ~2 cm lower than the boulder stacked on
                # top; its side grasp is reliable and removing it un-shadows and un-blocks the one above
                rubble.sort(key=lambda n: self.track[n][0][2])
            else:
                rubble.sort(key=lambda n: -self.track[n][0][2])  # highest overburden first
            return rubble + ["core"]
        return ["core"]

    def _grasp_z(self, name, p):
        # core: grip 4 mm above its centre (pads stay clear of the floor).  rubble: grip the top 10 mm.
        if name == "core" and FEATURES.get("core_grasp"):
            return p[2] + 0.004
        if name != "core" and FEATURES.get("grasp_top"):
            gz = p[2] + HALF_H[name] - 0.010
            if FEATURES.get("core_grasp") and self.track["core"] is not None \
                    and p[2] > 0.438 and np.linalg.norm(self.track["core"][0][:2] - p[:2]) < 0.06:
                gz = max(gz, 0.4625)   # pads bottom (grip-16mm) must clear the core's top face
            return gz
        return p[2] + 0.002

    def _goto(self, t, phase):
        self.phase, self.phase_t = phase, t

    def _track_grip(self, name, proprio):
        """Ramp the pads closed (<0.25 m/s), then servo SQUEEZE below the measured contact width.

        The measured pad gap equals the object's chord while the pads rest on its faces, so re-targeting the hold
        every tick keeps the squeeze (and friction) constant even as the carried object rotates or sags. The
        commanded gap only ever moves down at a bounded rate, so latching never snaps the fingers shut.
        """
        width = FINGER_OPEN * float(proprio["gripper"])
        gap_cmd = GRASP_W[name] - SQUEEZE[name]
        in_contact = gap_cmd < width < GRASP_W[name] + 0.020
        self.stall = self.stall + 1 if in_contact and self.grip_cmd * FINGER_OPEN < width - 0.002 else 0
        if self.stall >= 3:    # fingers resting on the faces for 3 ticks: latch and track the contact width
            self.grip_hold = float(np.clip((width - SQUEEZE[name]) / FINGER_OPEN, 0.05, 1.0))
        target = self.grip_hold if self.grip_hold is not None else gap_cmd / FINGER_OPEN
        self.grip_cmd = max(target, self.grip_cmd - 0.12)
        return self.grip_cmd

    def _held(self, name, proprio):
        """True while an object stalls the pads wider than the empty-gripper command."""
        return FINGER_OPEN * float(proprio["gripper"]) > GRASP_W[name] - SQUEEZE[name] + 0.003

    def step(self, t_sec: float, rgb: np.ndarray, depth: np.ndarray, proprio: dict) -> dict:
        ee = np.asarray(proprio["ee_pos"], dtype=np.float64)
        fz = abs(float(np.asarray(proprio["wrist_force"])[2]))
        bboxes = self._update_tracks(rgb, depth, ee)
        dt = t_sec - self.phase_t
        target, yaw, grip = ee.copy(), float(proprio["ee_yaw"]), 1.0

        if self.phase == "SURVEY":
            target = np.array([0.42, 0.0, SAFE_Z])
            found = all(self.track[n] is not None for n in NAMES)
            if (t_sec >= 1.45 and found) or (FEATURES["force_guard"] and t_sec >= 1.8):
                self.queue = self._order()
                self._goto(t_sec, "APPROACH")
        if self.phase != "SURVEY" and self.queue:
            name = self.queue[0]
            if FEATURES.get("core_grasp") and self.phase == "APPROACH" and name != "core" \
                    and t_sec > 6.5 and "core" in self.queue:
                self.queue, name = ["core"], "core"     # time budget: the core cycle needs ~4 s
            if FEATURES.get("core_grasp") and self.phase == "APPROACH" and name != "core" and self.tries[name] > 0 \
                    and self.track[name] is not None and self.track["core"] is not None \
                    and np.linalg.norm(self.track[name][0][:2] - self.track["core"][0][:2]) > 0.12:
                self.dumped[name] = True                # dropped in flight but already clear of the core: done
                self.queue.pop(0)
                self.lock = None
                name = self.queue[0] if self.queue else "core"
            if self.phase == "APPROACH":
                if name == "core" and FEATURES.get("core_grasp") and self.track["core"] is not None:
                    # never dig the core through overburden: re-queue rubble still parked on top of it
                    over = next((r for r in ("rubble_a", "rubble_b")
                                 if t_sec <= 6.5 and not self.dumped[r] and self.tries[r] < 2 and self.track[r] is not None
                                 and self.track[r][0][2] > self.track["core"][0][2]
                                 and np.linalg.norm(self.track[r][0][:2] - self.track["core"][0][:2]) < 0.05), None)
                    if over is not None:
                        self.queue.insert(0, over)
                        name = over
                if self.track[name] is None and FEATURES["force_guard"]:
                    p, oy = np.array([0.42, 0.0, 0.44]), 0.0   # lost track: hover over the nominal site
                else:
                    p, oy = self.track[name]
                if name != "core" and self.track["core"] is not None:
                    delta = p[:2] - self.track["core"][0][:2]
                    distance = np.linalg.norm(delta)
                    if 0.015 < distance < 0.08:
                        p = p.copy()
                        p[:2] += delta / distance * 0.010
                if self.lock is None or self.lock[0] != name:
                    self.grip_hold, self.stall, self.grip_cmd = None, 0, 1.0
                self.lock = (name, p.copy(), oy)
                target, yaw = np.array([p[0], p[1], SAFE_Z]), oy
                if (np.linalg.norm(ee[:2] - p[:2]) < 0.006 and dt > 0.10) or (FEATURES["force_guard"] and dt > 2.5):
                    self._goto(t_sec, "DESCEND")
            elif self.phase == "DESCEND":
                _, p, oy = self.lock
                yaw = oy
                gz = self._grasp_z(name, p)
                fast = gz + 0.02
                if FEATURES.get("core_grasp"):
                    for o in NAMES:    # end the dive clear of every neighbour still parked at the site
                        t = self.track[o]
                        if o != name and t is not None and np.linalg.norm(t[0][:2] - p[:2]) < 0.06:
                            fast = max(fast, t[0][2] + 0.064)
                if FEATURES["force_guard"] and ee[2] > fast + 0.01:
                    target = np.array([p[0], p[1], fast])            # fast to just above, then creep
                elif FEATURES["force_guard"]:
                    target = np.array([p[0], p[1], max(gz, ee[2] - 0.012)])
                else:
                    target = np.array([p[0], p[1], gz])
                if FEATURES.get("core_grasp") and name == "core":
                    grip = 0.80   # pre-shape: opening clears the core but not a neighbour ~45 mm off
                if (FEATURES["force_guard"] and fz > TOUCH_N) or abs(ee[2] - gz) < 0.003 or (FEATURES["force_guard"] and dt > 2.5):
                    self._goto(t_sec, "GRASP")
            elif self.phase == "GRASP":
                _, p, oy = self.lock
                if FEATURES.get("core_grasp"):
                    grip = self._track_grip(name, proprio)   # close on the faces, hold by friction (spec §2b)
                else:
                    grip = 0.0
                target, yaw = np.array([p[0], p[1], ee[2]]), oy
                bottomed = self.grip_cmd <= (GRASP_W[name] - SQUEEZE[name]) / FINGER_OPEN + 0.005
                early = self.stall >= 5 or (self.grip_hold is None and bottomed and dt > 0.35)
                if (FEATURES.get("core_grasp") and early) or dt > (0.55 if FEATURES.get("core_grasp") else 0.20):
                    self._goto(t_sec, "LIFT")
            elif self.phase == "LIFT":
                _, p, oy = self.lock
                grip = self._track_grip(name, proprio) if FEATURES.get("core_grasp") else 0.0
                target, yaw = np.array([p[0], p[1], SAFE_Z]), oy
                if ee[2] > SAFE_Z - 0.02:
                    budget = 3 if t_sec < 7.0 else 2   # a third attempt only while the clock allows
                    if FEATURES.get("core_grasp") and not self._held(name, proprio) and self.tries[name] < budget:
                        self.tries[name] += 1        # lifted an empty gripper: set it down and re-grasp
                        self.lock = None
                        self._goto(t_sec, "APPROACH")
                    else:
                        self._goto(t_sec, "TRANSPORT")
            elif self.phase == "TRANSPORT":
                dst = HOPPER if name == "core" else DUMP[name]
                target = np.array([dst[0], dst[1], SAFE_Z])
                grip = self._track_grip(name, proprio) if FEATURES.get("core_grasp") else 0.0
                yaw = self.lock[2]
                if np.linalg.norm(ee[:2] - dst) < 0.006 or (FEATURES["force_guard"] and dt > 3.0):
                    budget = 3 if t_sec < 7.0 else 2
                    if FEATURES.get("core_grasp") and not self._held(name, proprio) and self.tries[name] < budget:
                        self.tries[name] += 1        # dropped in flight: go back for it
                        self.lock = None
                        self._goto(t_sec, "APPROACH")
                    else:
                        self._goto(t_sec, "PLACE")
            elif self.phase == "PLACE":
                dst = HOPPER if name == "core" else DUMP[name]
                grip = self._track_grip(name, proprio) if FEATURES.get("core_grasp") else 0.0
                yaw = self.lock[2]
                if FEATURES["gentle_place"]:
                    # air release: creep the last 3 cm and open at height, so a deep-dangling object never
                    # gets driven into the ground (fz trigger + timeout back the creep off)
                    place_z = PLACE_Z[name]
                    hover = place_z + 0.02 if ee[2] > place_z + 0.03 else max(place_z, ee[2] - 0.012)
                    target = np.array([dst[0], dst[1], hover])
                    if ee[2] <= place_z + 0.001 or (FEATURES["force_guard"] and fz > 4.5) or dt > 1.5:
                        self._goto(t_sec, "RELEASE")
                else:
                    place_z = 0.43
                    target = np.array([dst[0], dst[1], place_z])
                    if abs(ee[2] - place_z) < 0.004:
                        self._goto(t_sec, "RELEASE")
            elif self.phase == "RELEASE":
                dst = HOPPER if name == "core" else DUMP[name]
                open_cmd = SOFT_OPEN if FEATURES["gentle_place"] else 1.0
                target, grip, yaw = np.array([dst[0], dst[1], ee[2]]), open_cmd, self.lock[2]
                if dt > 0.12:
                    self._goto(t_sec, "RETREAT")
            elif self.phase == "RETREAT":
                dst = HOPPER if name == "core" else DUMP[name]
                grip = SOFT_OPEN if FEATURES["gentle_place"] else 1.0
                target = np.array([dst[0], dst[1], SAFE_Z])
                if ee[2] > SAFE_Z - 0.03:
                    done = self.queue.pop(0)
                    if done != "core":
                        self.dumped[done] = True
                    self.lock = None
                    self._goto(t_sec, "APPROACH" if self.queue else "DONE")
        if self.phase == "DONE":
            target, grip = np.array([HOPPER[0], HOPPER[1], SAFE_Z]), 1.0

        est = {n: {"pos": (self.track[n][0].copy() if self.track[n] is not None else np.full(3, np.nan)),
                   "yaw": float(self.track[n][1]) if self.track[n] is not None else 0.0} for n in NAMES}
        cur = self.queue[0] if self.queue else "-"
        return {"estimated_objects": est,
                "action": {"target_pos": target, "target_yaw": float(yaw), "gripper": float(grip)},
                "debug_overlay": {"bboxes": bboxes, "phase": f"{self.phase}:{cur}"}}
