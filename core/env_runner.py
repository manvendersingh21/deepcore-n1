"""Excavation-basin rollout loop with a privileged/unprivileged boundary.

The candidate pipeline only ever receives (rgb, depth, camera intrinsics/extrinsics, proprio). Ground truth is
recorded here for core/verifier_harness.py and the HUD and is never handed to the pipeline.

Conventions (see INTERFACE.md):
  * physics 200 Hz (dt=0.005); before t=0 the world is pre-settled for 1.0 s (not counted); an episode is
    EPISODE_STEPS=2200 mj_step calls (11.0 s), followed by SETTLE_STEPS=100 steps with the gripper forced open.
  * vision + pipeline.step() at 20 Hz (every 10 physics steps) -> 140 calls; the returned action is held in between.
  * camera extrinsics are OpenCV-convention: X right, Y down, Z forward.
    p_world = cam_pos + cam_rot @ [(u-cx)/fx*d, (v-cy)/fy*d, d], with d = depth[v, u] (z-depth, metres).
  * action.target_pos is the grip centre (between finger pads) in world metres; it passes through a
    velocity/acceleration-limited setpoint filter (2.0 m/s, 20 m/s^2) and is mapped to joints by Jacobian IK.
  * core_force_now: |wrist_force_world - F_rest| (F_rest = static hand-weight reading at t=0) while the core
    gate is open (grip centre within 0.07 m of the core centre OR a pad/palm touching core/rubble), else 0.
"""
from __future__ import annotations

import time
import zlib
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

from core.sensor_corruption import SensorCorruption

SCENE_PATH = Path(__file__).with_name("scene.xml")
SIM_DT = 0.005
EPISODE_STEPS = 2200                     # 11.0 s horizon (project-owner authorised pre-lock change)
VISION_EVERY = 10
SETTLE_STEPS = 100
PRESETTLE_STEPS = 200                      # 1.0 s of physics before t=0
IMG_H, IMG_W = 480, 640
OBJECTS = ("core", "rubble_a", "rubble_b")
PIT_FLOOR_Z = 0.400
CORE_HALF = 0.022
NOMINAL_CORE_XY = np.array([0.42, 0.0])
CORE_XY_JITTER = 0.045
CORE_YAW_JITTER = np.deg2rad(30.0)
DUMP_ZONE = np.array([0.30, -0.16, 0.42])
HOPPER_ZONE = np.array([0.52, 0.16, 0.42])
GATE_RADIUS = 0.07
WORKSPACE_LO = np.array([0.12, -0.38, 0.40])
WORKSPACE_HI = np.array([0.78, 0.38, 0.85])
HOME_POS = np.array([0.26, -0.27, 0.68])   # parked off to the side so t=0 overhead frames see the dig site
MAX_EE_SPEED = 2.0
MAX_EE_ACCEL = 20.0
MAX_YAW_RATE = 6.0
FINGER_MAX = 0.045
REFLEX_N = 10.0                              # wrist-force compliance threshold (N)
GANTRY_Z0 = 0.80                           # grip-site height when slide_z == 0
RUBBLE_A_HALF = np.array([0.036, 0.024, 0.020])
RUBBLE_B_HALF = np.array([0.032, 0.024, 0.020])
# rubble_a: lies flat on the core top, shifted to the core's top-left (core frame +y, -x)
RUBBLE_A_OFFSET = np.array([-0.014, 0.019])
RUBBLE_A_YAW = np.deg2rad(10.0)
# rubble_b: long axis points at the core, lower end on the pit floor, bottom face resting on the core's right
# (core frame -y) top edge. LEAN_L = distance along the bottom face from the floor corner to the core edge.
RUBBLE_B_LEAN_L = 0.058
RUBBLE_B_YAW_SKEW = np.deg2rad(8.0)
SETTLE_TOL_XY = 0.03


def yaw_from_quat(q: np.ndarray) -> float:
    w, x, y, z = q
    return float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def _quat_from_mat(R: np.ndarray) -> np.ndarray:
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.ascontiguousarray(R, dtype=np.float64).reshape(9))
    return q


def _rotz(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


# ------------------------------------------------------------------ procedural textures (cached per process)
_TEX_CACHE: dict = {}


def _fractal_noise(n: int, seed: int, octaves: int = 6) -> np.ndarray:
    rng = np.random.default_rng(seed)
    acc, amp, tot = np.zeros((n, n)), 1.0, 0.0
    for o in range(octaves):
        g = 2 ** (o + 2)
        coarse = rng.random((g + 1, g + 1))
        coarse[-1, :], coarse[:, -1] = coarse[0, :], coarse[:, 0]       # tileable
        t = np.linspace(0, g, n, endpoint=False)
        i0 = t.astype(int)
        f = t - i0
        f = f * f * (3 - 2 * f)
        a = coarse[i0][:, i0] * (1 - f)[None, :] + coarse[i0][:, i0 + 1] * f[None, :]
        b = coarse[i0 + 1][:, i0] * (1 - f)[None, :] + coarse[i0 + 1][:, i0 + 1] * f[None, :]
        acc += amp * (a * (1 - f)[:, None] + b * f[:, None])
        tot += amp
        amp *= 0.55
    acc /= tot
    return (acc - acc.min()) / (np.ptp(acc) + 1e-9)


def _texture_rgb(name: str, n: int) -> np.ndarray:
    key = (name, n)
    if key in _TEX_CACHE:
        return _TEX_CACHE[key]
    h = zlib.crc32(name.encode()) % 1000   # stable across processes (hash() is salted per process)
    base = _fractal_noise(n, 100 + h)
    grit = np.random.default_rng(200 + h).random((n, n))
    pal = {  # (dark, light) earth tones
        "tex_earth": ((0.10, 0.075, 0.05), (0.42, 0.32, 0.22)),
        "tex_wall": ((0.08, 0.06, 0.045), (0.36, 0.27, 0.19)),
        "tex_rock": ((0.22, 0.21, 0.20), (0.62, 0.60, 0.57)),
        "tex_steel": ((0.25, 0.25, 0.26), (0.55, 0.53, 0.50)),
    }[name]
    v = np.clip(0.75 * base + 0.25 * grit, 0, 1)
    if name == "tex_earth":       # dark damp patches + light gravel speckles
        v = np.where(_fractal_noise(n, 300 + h, 4) > 0.62, v * 0.55, v)
        v = np.where(grit > 0.985, 1.0, v)
    if name == "tex_wall":        # horizontal strata
        rows = np.linspace(0, 18 * np.pi, n)[:, None]
        v = np.clip(v * (0.8 + 0.2 * np.sin(rows + 3 * base)), 0, 1)
    if name == "tex_steel":       # rust streaks
        v = np.clip(v - 0.3 * (_fractal_noise(n, 400 + h, 3) > 0.7), 0, 1)
    lo, hi = np.array(pal[0]), np.array(pal[1])
    rgb = (255 * (lo + v[..., None] * (hi - lo))).astype(np.uint8)
    _TEX_CACHE[key] = rgb
    return rgb


def _apply_textures(m: mujoco.MjModel) -> None:
    for name in ("tex_earth", "tex_wall", "tex_rock", "tex_steel"):
        tid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_TEXTURE, name)
        w, h, c = m.tex_width[tid], m.tex_height[tid], m.tex_nchannel[tid]
        if w != h or c != 3:
            continue
        adr = m.tex_adr[tid]
        m.tex_data[adr:adr + w * h * c] = _texture_rgb(name, int(w)).reshape(-1)


@dataclass
class EpisodeRecord:
    seed: int
    t: list = field(default_factory=list)                  # vision frame times
    gt: list = field(default_factory=list)                 # {obj: (pos(3), yaw)} per vision frame
    est: list = field(default_factory=list)                # {obj: (pos(3), yaw)} or None per vision frame
    occluded: list = field(default_factory=list)           # bool per vision frame
    ee_pos: list = field(default_factory=list)
    gripper_cmd: list = field(default_factory=list)
    phase: list = field(default_factory=list)
    latency_ms: list = field(default_factory=list)
    force_t: list = field(default_factory=list)            # physics-rate time stamps
    force_vec: list = field(default_factory=list)          # physics-rate wrist force (world frame, raw)
    core_gate_force: list = field(default_factory=list)    # physics-rate gated core-contact force (N)
    core_gate_open: list = field(default_factory=list)     # physics-rate gate state
    final: dict = field(default_factory=dict)              # {obj: {"pos", "yaw", "speed"}} after settle
    errors: list = field(default_factory=list)


class Episode:
    """One seeded rollout world. Owns MjModel/MjData; exposes only sensor data to pipelines."""

    def __init__(self, seed: int, render_perspective: bool = False):
        self.seed = int(seed)
        self.model = mujoco.MjModel.from_xml_path(str(SCENE_PATH))
        _apply_textures(self.model)
        self.data = mujoco.MjData(self.model)
        m = self.model
        self._jid = {n: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n) for n in
                     ("slide_x", "slide_y", "slide_z", "hinge_yaw", "finger_left", "finger_right")}
        self._qadr = {n: m.jnt_qposadr[j] for n, j in self._jid.items()}
        self._dadr = {n: m.jnt_dofadr[j] for n, j in self._jid.items()}
        self._bid = {o: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, o) for o in OBJECTS}
        self._gid = {o: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, o) for o in OBJECTS}
        self._hand_gids = {mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, g) for g in ("pad_left", "pad_right", "palm")}
        self._obj_gids = set(self._gid.values())
        self._grip_sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "grip_site")
        self._wrist_sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "wrist_site")
        self._cam_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "cam_overhead")
        self._force_adr = m.sensor_adr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "wrist_force")]
        self._torque_adr = m.sensor_adr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "wrist_torque")]
        self._arm_dofs = [self._dadr[n] for n in ("slide_x", "slide_y", "slide_z", "hinge_yaw")]

        self.settle_attempts = 0
        self._randomize()
        self.renderer = mujoco.Renderer(m, IMG_H, IMG_W)
        self._scene_opt = mujoco.MjvOption()
        self._scene_opt.sitegroup[:] = 0
        self._scene_opt.sitegroup[0] = 1           # zone markers (group 0) are painted on the floor; volumes hidden
        self.persp_renderer = mujoco.Renderer(m, IMG_H, IMG_W) if render_perspective else None
        self.corruption = SensorCorruption(self.seed, IMG_H, IMG_W)

        # intrinsics / extrinsics (OpenCV convention)
        fy = (IMG_H / 2) / np.tan(np.deg2rad(m.cam_fovy[self._cam_id]) / 2)
        self.camera_matrix = np.array([[fy, 0, IMG_W / 2], [0, fy, IMG_H / 2], [0, 0, 1]], dtype=np.float64)
        self.cam_pos = self.data.cam_xpos[self._cam_id].copy()
        self.cam_rot = self.data.cam_xmat[self._cam_id].reshape(3, 3) @ np.diag([1.0, -1.0, -1.0])

        self.sp_pos = self.ee_pos().copy()
        self.sp_vel = np.zeros(3)
        self.sp_yaw = 0.0
        self.step_count = 0
        self.frame_idx = 0
        self.start_pos = {o: self.data.xpos[self._bid[o]].copy() for o in OBJECTS}
        self._f_rest = self.wrist_force_world()
        self.core_gate_open = False
        self.core_force_now = 0.0

    # ------------------------------------------------------------------ setup
    def _place(self, name: str, pos: np.ndarray, R: np.ndarray) -> None:
        m, d = self.model, self.data
        qadr = m.jnt_qposadr[m.body_jntadr[self._bid[name]]]
        d.qpos[qadr:qadr + 3] = pos
        d.qpos[qadr + 3:qadr + 7] = _quat_from_mat(R)

    def _pose_objects(self, cxy, cyaw, a_off, a_yaw, b_lean_l, b_skew) -> None:
        Rc = _rotz(cyaw)
        core_top = PIT_FLOOR_Z + 2 * CORE_HALF
        self._place("core", np.array([cxy[0], cxy[1], PIT_FLOOR_Z + CORE_HALF + 0.0005]), Rc)
        # rubble_a flat on the core top
        pa = np.r_[cxy + (Rc[:2, :2] @ a_off), core_top + RUBBLE_A_HALF[2] + 0.001]
        self._place("rubble_a", pa, _rotz(cyaw + a_yaw))
        # rubble_b leaning: outward direction u = core-frame -y (right edge); x-axis of the box runs up toward the core
        T = 2 * RUBBLE_B_HALF[2]
        sin_t = (2 * CORE_HALF) / b_lean_l
        th = np.arcsin(np.clip(sin_t, 0, 0.98))
        u = Rc @ np.array([0.0, -1.0, 0.0])
        u = _rotz(b_skew) @ u
        ex = np.r_[-np.cos(th) * u[:2], np.sin(th)]
        ez = np.r_[np.sin(th) * u[:2], np.cos(th)]
        ey = np.cross(ez, ex)
        R = np.column_stack([ex, ey, ez])
        s_floor = CORE_HALF + b_lean_l * np.cos(th)
        corner = np.r_[cxy + s_floor * u[:2], PIT_FLOOR_Z]       # lower bottom edge on the floor
        centre = corner + RUBBLE_B_HALF[0] * ex + (T / 2 + 0.001) * ez
        self._place("rubble_b", centre, R)

    def _check_settled(self) -> dict:
        gt = self.ground_truth()
        c = gt["core"][0]
        res = {"core_z": float(c[2]), "min_z": float(min(self.data.xpos[self._bid[o]][2] for o in OBJECTS))}
        for r in ("rubble_a", "rubble_b"):
            res[f"{r}_dxy"] = float(np.linalg.norm(gt[r][0][:2] - c[:2]))
            res[f"{r}_touch"] = self._touching(self._gid[r], self._gid["core"])
        return res

    def _touching(self, g1: int, g2: int) -> bool:
        d = self.data
        for i in range(d.ncon):
            pair = {d.contact[i].geom1, d.contact[i].geom2}
            if pair == {g1, g2}:
                return True
        return False

    def _settle_ok(self, s: dict) -> bool:
        return (s["rubble_a_dxy"] <= SETTLE_TOL_XY and s["rubble_a_touch"] and s["rubble_b_touch"]
                and s["min_z"] >= 0.39 and 0.415 <= s["core_z"] <= 0.430)

    def _randomize(self):
        m, d = self.model, self.data
        rng = np.random.default_rng([self.seed, 31337])
        key = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_LIGHT, "key")
        m.light_pos[key] += rng.uniform(-0.12, 0.12, 3) * np.array([1, 1, 0.5])
        m.light_diffuse[key] *= rng.uniform(0.85, 1.1)
        cxy = NOMINAL_CORE_XY + rng.uniform(-CORE_XY_JITTER, CORE_XY_JITTER, 2)
        cyaw = rng.uniform(-CORE_YAW_JITTER, CORE_YAW_JITTER)
        a_off = RUBBLE_A_OFFSET + rng.uniform(-0.002, 0.002, 2)
        a_yaw = RUBBLE_A_YAW + rng.uniform(-0.12, 0.12)
        b_l, b_skew = RUBBLE_B_LEAN_L + rng.uniform(-0.002, 0.002), RUBBLE_B_YAW_SKEW * rng.uniform(-1, 1)
        for attempt in range(8):
            self.settle_attempts = attempt + 1
            mujoco.mj_resetData(m, d)
            self._pose_objects(cxy, cyaw, a_off, a_yaw, b_l, b_skew)
            d.qpos[self._qadr["slide_x"]] = HOME_POS[0]
            d.qpos[self._qadr["slide_y"]] = HOME_POS[1]
            d.qpos[self._qadr["slide_z"]] = HOME_POS[2] - GANTRY_Z0
            d.qpos[self._qadr["finger_left"]] = d.qpos[self._qadr["finger_right"]] = FINGER_MAX
            d.ctrl[:] = [HOME_POS[0], HOME_POS[1], HOME_POS[2] - GANTRY_Z0, 0.0, FINGER_MAX, FINGER_MAX]
            mujoco.mj_forward(m, d)
            for _ in range(PRESETTLE_STEPS):
                mujoco.mj_step(m, d)
            self.settle_report = self._check_settled()
            if self._settle_ok(self.settle_report):
                break
            # adjust only the failing placement, then retry from scratch: pull rubble_a toward the core centre;
            # lay rubble_b flatter (a steep lean slides off the core edge)
            s = self.settle_report
            if not (s["rubble_a_touch"] and s["rubble_a_dxy"] <= SETTLE_TOL_XY) or s["core_z"] > 0.430:
                a_off = a_off * 0.8
            if not s["rubble_b_touch"]:
                b_l = min(0.066, b_l + 0.003)
        d.time = 0.0

    # ------------------------------------------------------------------ unprivileged observations
    def ee_pos(self) -> np.ndarray:
        return self.data.site_xpos[self._grip_sid].copy()

    def wrist_force_world(self) -> np.ndarray:
        R = self.data.site_xmat[self._wrist_sid].reshape(3, 3)
        return R @ self.data.sensordata[self._force_adr:self._force_adr + 3]

    def proprio(self) -> dict:
        d = self.data
        R = d.site_xmat[self._wrist_sid].reshape(3, 3)
        width = d.qpos[self._qadr["finger_left"]] + d.qpos[self._qadr["finger_right"]]
        return {
            "ee_pos": self.ee_pos(),
            "ee_yaw": float(d.qpos[self._qadr["hinge_yaw"]]),
            "gripper": float(np.clip(width / (2 * FINGER_MAX), 0.0, 1.0)),
            "wrist_force": R @ d.sensordata[self._force_adr:self._force_adr + 3],
            "wrist_torque": R @ d.sensordata[self._torque_adr:self._torque_adr + 3],
        }

    def render_clean(self):
        self.renderer.update_scene(self.data, camera=self._cam_id, scene_option=self._scene_opt)
        rgb = self.renderer.render().copy()
        self.renderer.enable_depth_rendering()
        self.renderer.update_scene(self.data, camera=self._cam_id, scene_option=self._scene_opt)
        depth = self.renderer.render().copy()
        self.renderer.disable_depth_rendering()
        return rgb, depth

    def observe(self, external_mask=None, external_rgb=None):
        rgb, depth = self.render_clean()
        return self.corruption.corrupt(rgb, depth, self.frame_idx, external_mask, external_rgb)

    def render_perspective(self, camera: str = "cam_perspective") -> np.ndarray:
        if self.persp_renderer is None:
            self.persp_renderer = mujoco.Renderer(self.model, IMG_H, IMG_W)
        self.persp_renderer.update_scene(self.data, camera=camera, scene_option=self._scene_opt)
        return self.persp_renderer.render().copy()

    # ------------------------------------------------------------------ privileged (verifier / HUD only)
    def ground_truth(self) -> dict:
        return {o: (self.data.xpos[self._bid[o]].copy(), yaw_from_quat(self.data.xquat[self._bid[o]]))
                for o in OBJECTS}

    def object_speed(self, o: str) -> float:
        v = np.zeros(6)
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, self._bid[o], v, 0)
        return float(np.linalg.norm(v[3:]))

    def _update_core_gate(self) -> None:
        d = self.data
        near = np.linalg.norm(self.ee_pos() - d.xpos[self._bid["core"]]) < GATE_RADIUS
        touch = False
        if not near:
            for i in range(d.ncon):
                g1, g2 = d.contact[i].geom1, d.contact[i].geom2
                if (g1 in self._hand_gids and g2 in self._obj_gids) or (g2 in self._hand_gids and g1 in self._obj_gids):
                    touch = True
                    break
        self.core_gate_open = bool(near or touch)
        self.core_force_now = float(np.linalg.norm(self.wrist_force_world() - self._f_rest)) if self.core_gate_open else 0.0

    # ------------------------------------------------------------------ control
    def apply_action(self, action: dict, dt: float = SIM_DT):
        target = np.asarray(action["target_pos"], dtype=np.float64).reshape(3)
        if not np.all(np.isfinite(target)):
            target = self.sp_pos.copy()
        target = np.clip(target, WORKSPACE_LO, WORKSPACE_HI)
        err = target - self.sp_pos
        dist = np.linalg.norm(err)
        v_des = np.zeros(3) if dist < 1e-9 else err / dist * min(MAX_EE_SPEED, np.sqrt(2 * MAX_EE_ACCEL * dist))
        dv = v_des - self.sp_vel
        dvn = np.linalg.norm(dv)
        if dvn > MAX_EE_ACCEL * dt:
            dv *= MAX_EE_ACCEL * dt / dvn
        self.sp_vel += dv
        # compliance reflex (200 Hz): a downward Z command while the wrist already carries > 11 N is clamped
        # to the current height, so 50 ms candidate ticks cannot drive the arm through the core
        # only for controlled descents (|target_z - ee_z| < 0.04), not for the high-speed lunge
        if (target[2] < self.ee_pos()[2] and abs(target[2] - self.ee_pos()[2]) < 0.04
                and np.linalg.norm(self.wrist_force_world()) > REFLEX_N):
            self.sp_pos[2] = self.ee_pos()[2]
            self.sp_vel[2] = 0.0
        self.sp_pos = self.sp_pos + self.sp_vel * dt
        tyaw = float(action.get("target_yaw", 0.0))
        tyaw = float(np.clip(tyaw if np.isfinite(tyaw) else self.sp_yaw, -3.2, 3.2))
        self.sp_yaw += float(np.clip(tyaw - self.sp_yaw, -MAX_YAW_RATE * dt, MAX_YAW_RATE * dt))

        m, d = self.model, self.data
        jacp, jacr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        mujoco.mj_jacSite(m, d, jacp, jacr, self._grip_sid)
        J = np.vstack([jacp[:, self._arm_dofs], jacr[2:3, self._arm_dofs]])
        task_err = np.concatenate([self.sp_pos - self.ee_pos(), [self.sp_yaw - d.qpos[self._qadr["hinge_yaw"]]]])
        dq = np.linalg.pinv(J) @ task_err
        q = np.array([d.qpos[self._qadr[n]] for n in ("slide_x", "slide_y", "slide_z", "hinge_yaw")])
        d.ctrl[0:4] = q + dq
        g = action.get("gripper", 1.0)
        g = float(np.clip(g, 0.0, 1.0)) if np.isfinite(g) else 1.0
        d.ctrl[4] = d.ctrl[5] = FINGER_MAX * g

    def physics_step(self):
        mujoco.mj_step(self.model, self.data)
        self.step_count += 1
        self._update_core_gate()

    def close(self):
        for r in (self.renderer, self.persp_renderer):
            if r is not None:
                r.close()


def _parse_estimates(out) -> dict | None:
    try:
        est = {}
        for o in OBJECTS:
            e = out["estimated_objects"][o]
            est[o] = (np.asarray(e["pos"], dtype=np.float64).reshape(3), float(e.get("yaw", 0.0)))
        return est
    except Exception:
        return None


def _log_physics(ep: Episode, rec: EpisodeRecord) -> None:
    rec.force_t.append(ep.data.time)
    rec.force_vec.append(ep.wrist_force_world())
    rec.core_gate_force.append(ep.core_force_now)
    rec.core_gate_open.append(ep.core_gate_open)


def run_episode(pipeline_cls, seed: int, frame_callback=None, occluder_source=None) -> EpisodeRecord:
    """Roll out one seeded episode. `pipeline_cls(camera_matrix, cam_pos, cam_rot)` must build the candidate.

    frame_callback(episode, rgb, depth, out, record) is invoked after each vision step (HUD / video only).
    occluder_source() -> (mask, rgb) supplies live webcam hand occlusion (HUD only).
    """
    ep = Episode(seed, render_perspective=frame_callback is not None)
    rec = EpisodeRecord(seed=seed)
    try:
        pipe = pipeline_cls(ep.camera_matrix.copy(), ep.cam_pos.copy(), ep.cam_rot.copy())
    except Exception as e:  # a candidate that cannot even construct holds still for the whole episode
        rec.errors.append(f"init: {type(e).__name__}: {e}")
        pipe = None
    action = {"target_pos": ep.ee_pos(), "target_yaw": 0.0, "gripper": 1.0}

    for k in range(EPISODE_STEPS):
        if k % VISION_EVERY == 0:
            ext_mask, ext_rgb = occluder_source() if occluder_source is not None else (None, None)
            rgb, depth, occluded = ep.observe(ext_mask, ext_rgb)
            out = None
            t0 = time.perf_counter()
            if pipe is not None:
                try:
                    out = pipe.step(k * SIM_DT, rgb.copy(), depth.copy(), ep.proprio())
                except Exception as e:
                    if len(rec.errors) < 20:
                        rec.errors.append(f"t={k * SIM_DT:.2f}: {type(e).__name__}: {e}")
            lat = (time.perf_counter() - t0) * 1000.0
            if isinstance(out, dict) and isinstance(out.get("action"), dict) and "target_pos" in out["action"]:
                action = out["action"]
            rec.t.append(k * SIM_DT)
            rec.gt.append(ep.ground_truth())
            rec.est.append(_parse_estimates(out) if out is not None else None)
            rec.occluded.append(bool(occluded))
            rec.ee_pos.append(ep.ee_pos())
            try:
                rec.gripper_cmd.append(float(action.get("gripper", 1.0)))
            except Exception:
                rec.gripper_cmd.append(1.0)
            dbg = out.get("debug_overlay", {}) if isinstance(out, dict) else {}
            rec.phase.append(str(dbg.get("phase", "")) if isinstance(dbg, dict) else "")
            rec.latency_ms.append(lat)
            ep.frame_idx += 1
            if frame_callback is not None:
                frame_callback(ep, rgb, depth, out, rec)
        try:
            ep.apply_action(action)
        except Exception as e:
            rec.errors.append(f"bad action at t={k * SIM_DT:.2f}: {e}")
            action = {"target_pos": ep.sp_pos.copy(), "target_yaw": ep.sp_yaw, "gripper": 1.0}
        ep.physics_step()
        _log_physics(ep, rec)

    # settle: release the gripper, hold the arm, and let gravity decide where everything ends up
    hold = {"target_pos": ep.sp_pos.copy(), "target_yaw": ep.sp_yaw, "gripper": 1.0}
    for _ in range(SETTLE_STEPS):
        ep.apply_action(hold)
        ep.physics_step()
        _log_physics(ep, rec)
    gt = ep.ground_truth()
    rec.final = {o: {"pos": gt[o][0], "yaw": gt[o][1], "speed": ep.object_speed(o)} for o in OBJECTS}
    ep.close()
    return rec
