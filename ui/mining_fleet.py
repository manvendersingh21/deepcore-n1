"""Two actuated mining rovers with contact-driven drilling and debris pickup.

Mobile bases use ideal planar motor drives. Rock removal reuses the discrete
contact-work model from drilling_scene. Six free bodies are created at the
drill's discharge when two-layer removal milestones actually occur. The
collector closes physical fingers, activates a weld only after both fingers
contact the same chunk, and releases it into actuated rail carts. Pickup
targets come from corrupted RGB-D; contact and load measurements gate grasp
and cart departure. Independent physics bookkeeping verifies the result.
"""
from __future__ import annotations

from collections.abc import Callable
import math
import xml.etree.ElementTree as ET
from typing import Any

import mujoco
import numpy as np

from ui import drilling_scene as drilling
from ui.industrial_arm import ArmController, attach_ur5e
from agents.mining_policies import AcceptedGraspPolicy
from core.sensor_corruption import SensorCorruption
from ui.fleet_perception import FleetObserver
from ui.rail_carts import (CART_CAPACITY, CART_CENTERS, add_rail_carts,
                           cart_interior_bounds, cart_slot_positions)


DT = drilling.DT
FRAME_INTERVAL_S = .05
DURATION_S = 80.0
DEBRIS_COUNT = len(CART_CENTERS) * CART_CAPACITY
DRILL_START = np.array([-.45, -.25])
DRILL_WORK = np.array([.06, 0.0])
COLLECTOR_START = np.array([1.15, -.60])
GRIP_OFFSET_X = -.42
BIN_CENTER = np.array([1.05, .35])
FLOOR_Z = .4
SENSOR_CAMERA_POSITION = (.42, .52, 1.65)
SENSOR_FOVY_DEG = 50.0


def _append(parent, source: str):
    element = ET.fromstring(source)
    parent.append(element)
    return element


def _mobile_body(name: str, start: np.ndarray, material: str) -> ET.Element:
    body = ET.fromstring(f'''<body name="{name}" pos="{start[0]} {start[1]} .49">
      <joint name="{name}_x" type="slide" axis="1 0 0" range="-2 2" damping="10"/>
      <joint name="{name}_y" type="slide" axis="0 1 0" range="-2 2" damping="10"/>
      <inertial pos="0 0 0" mass="28" diaginertia=".28 .38 .42"/>
      <geom class="decor" name="{name}" type="box" size=".18 .12 .045" material="{material}"/>
      <geom class="decor" name="{name}_deck" type="box" pos="0 0 .051" size=".155 .115 .009" material="dark"/>
      <geom class="decor" name="{name}_battery" type="box" pos="-.065 0 .105" size=".07 .09 .045" material="{material}"/>
    </body>''')
    for side in (-1, 1):
        _append(body, f'''<geom class="decor" name="{name}_track_{side}" type="box"
                 pos="0 {side * .145} -.025" size=".19 .03 .055" material="dark"/>''')
        for i in range(5):
            _append(body, f'''<geom class="decor" name="{name}_wheel_{side}_{i}" type="cylinder"
                pos="{-.14 + i * .07} {side * .177} -.025" euler="{math.pi / 2} 0 0"
                size=".038 .004" material="steel"/>''')
        for i in range(9):
            _append(body, f'''<geom class="decor" type="box" pos="{-.172 + i * .043} {side * .145} -.080"
                size=".009 .032 .008" material="dark"/>''')
    return body


def _scene_xml(seed: int) -> str:
    root = ET.fromstring(drilling._scene_xml(seed))
    root.set("model", "deepcore_cooperative_mining_fleet")
    world, assets, actuators = root.find("worldbody"), root.find("asset"), root.find("actuator")
    _append(assets, '<material name="collector_teal" rgba=".05 .58 .56 1" specular=".3"/>')
    _append(assets, '<material name="debris_material" rgba=".62 .48 .32 1" texture="tex_rock" texrepeat="1 1"/>')
    _append(assets, '<material name="bin_material" rgba=".29 .34 .38 1" specular=".2"/>')
    keep = ("rock_layer_", "stratum_", "bore_wall_")
    drill_feed = world.find("body[@name='drill_feed']")
    world.remove(drill_feed)
    for geom in list(world.findall("geom")):
        name = geom.get("name", "")
        if name not in ("world_floor", "pit_floor", "rock_base", "bore_bottom") and not name.startswith(keep):
            world.remove(geom)
    floor = world.find("geom[@name='pit_floor']")
    floor.set("pos", ".75 0 .38")
    floor.set("size", "1.85 1.10 .02")
    floor.set("contype", "2")
    floor.set("conaffinity", "5")
    world.find("geom[@name='world_floor']").set("pos", "0 0 .399")
    overview = world.find("camera[@name='overview']")
    overview.set("pos", "3.1 -3.6 3.0")
    overview.set("xyaxes", drilling._camera_axes((3.1, -3.6, 3.0), (.70, .1, .68)))
    overview.set("fovy", "44")
    overhead = world.find("camera[@name='cam_overhead']")
    overhead.set("pos", drilling._vector(SENSOR_CAMERA_POSITION))
    overhead.set("fovy", str(SENSOR_FOVY_DEG))
    root.find("statistic").set("extent", "3.5")
    root.find("visual/global").set("offwidth", "1600")
    root.find("visual/global").set("offheight", "900")

    drill = _mobile_body("drill_mobile_base", DRILL_START, "paint")
    drill_feed.set("pos", ".36 0 .135")
    drill.append(drill_feed)
    _append(drill, '<geom class="decor" name="mobile_drill_mast" type="box" pos="-.08 .055 .285" size=".025 .028 .25" material="paint"/>')
    _append(drill, '<geom class="decor" name="mobile_drill_arm" type="box" pos=".145 .055 .515" size=".245 .028 .025" material="paint"/>')
    for x in (.295, .425):
        _append(drill, f'<geom class="decor" type="cylinder" pos="{x} .065 .295" size=".007 .21" material="steel"/>')
    _append(drill, '<geom class="decor" name="debris_discharge" type="box" pos=".01 .18 .10" size=".39 .045 .018" material="steel"/>')
    world.append(drill)

    collector = _mobile_body("collector_mobile_base", COLLECTOR_START, "collector_teal")
    lift = ET.fromstring('''<body name="collector_lift">
      <inertial pos="0 0 -.025" mass=".36" diaginertia=".001 .001 .001"/>
      <geom class="decor" name="collector_tool_adapter" type="cylinder" pos="0 0 .027" size=".028 .014" material="dark"/>
      <geom class="decor" name="collector_palm" type="box" pos="0 0 .006" size=".052 .079 .013" material="steel"/>
      <site name="collector_grip_site" pos="0 0 -.055" size=".003" rgba="0 0 0 0"/>
    </body>''')
    for side, label in ((1, "left"), (-1, "right")):
        _append(lift, f'''<body name="collector_finger_{label}" pos="0 {side * .065} 0">
          <joint name="collector_grip_{label}" type="slide" axis="0 {-side} 0" range="0 .057" damping="1"/>
          <inertial pos="0 0 -.035" mass=".08" diaginertia=".00008 .00008 .00008"/>
          <geom class="decor" type="box" pos="0 0 -.025" size=".036 .009 .02" material="collector_teal"/>
          <geom name="collector_pad_{label}" type="box" pos="0 0 -.055" size=".037 .010 .023"
                material="dark" contype="8" conaffinity="4" friction="1.0 .01 .001"/>
        </body>''')
    attach_ur5e(root, collector, lift)
    world.append(collector)
    for name in ("drill_mobile_base", "collector_mobile_base"):
        for axis in ("x", "y"):
            _append(actuators, f'<motor name="{name}_{axis}_motor" joint="{name}_{axis}" ctrllimited="true" ctrlrange="-300 300"/>')
    for side in ("left", "right"):
        _append(actuators, f'<position name="collector_grip_{side}_motor" joint="collector_grip_{side}" kp="180" kv="3" ctrllimited="true" ctrlrange="0 .057" forcelimited="true" forcerange="-5 5"/>')

    equality = ET.SubElement(root, "equality")
    for i in range(DEBRIS_COUNT):
        # Dormant bodies are pinned below the world until actual removal emits
        # them. Releasing an event-created body is the only runtime pose write.
        _append(world, f'''<body name="debris_{i}" pos="{.42 + i * .08} .22 -1">
          <freejoint name="debris_{i}_free"/>
          <geom name="debris_{i}" type="box" size=".027 .023 .023" mass=".16"
                material="debris_material" rgba=".62 .48 .32 0" contype="4" conaffinity="14"
                friction=".9 .01 .001" solref=".012 1"/>
        </body>''')
        _append(equality, f'<weld name="debris_{i}_dormant" body1="debris_{i}" active="true"/>')
        _append(equality, f'<weld name="debris_{i}_grasp" body1="collector_lift" body2="debris_{i}" active="false" solref=".008 1"/>')
    xml = ET.tostring(root, encoding="unicode")
    from ui.cmu_mine import append_mine_assets
    from ui.robot_visuals import decorate_robots
    return append_mine_assets(decorate_robots(add_rail_carts(xml)))


class MiningFleetEpisode(drilling.DrillingEpisode):
    """Actual actuated motion, measured contacts, event-born free-body debris."""

    def __init__(self, seed: int = 101, controller_params: dict | None = None):
        self.seed = int(seed)
        self.controller_params = dict(controller_params or {})
        self.model = mujoco.MjModel.from_xml_string(_scene_xml(self.seed))
        self.data = mujoco.MjData(self.model)
        self._renderer = self._overview_renderer = None
        self._closed = False
        self._feed_dof = int(self.model.jnt_dofadr[self.model.joint("feed").id])
        self._feed_qpos = int(self.model.jnt_qposadr[self.model.joint("feed").id])
        self._spindle_dof = int(self.model.jnt_dofadr[self.model.joint("spindle").id])
        self._tip_site = self.model.site("tool_tip").id
        self._feed_act = self.model.actuator("feed_motor").id
        self._spin_act = self.model.actuator("spindle_motor").id
        self._cutters = {self.model.geom(f"cutter_{i}").id for i in range(4)}
        self._layer_geoms = [self.model.geom(f"rock_layer_{i}").id for i in range(drilling.LAYERS)]
        self._layer_by_geom = {gid: i for i, gid in enumerate(self._layer_geoms)}
        self.layer_work_j = np.zeros(drilling.LAYERS)
        self.layer_threshold_j = np.random.default_rng(self.seed).uniform(3.0, 4.0, drilling.LAYERS)
        self.removed = np.zeros(drilling.LAYERS, dtype=bool)
        self.phase, self.stop_reason = "travelling", ""
        self.peak_force_n = self.max_depth_mm = self._axial_force_n = 0.0
        self._contact_torque_nm = self._cutting_power_w = 0.0
        self._contact_force_world = np.zeros(3)
        self._contact_position = np.array([.42, 0, drilling.ROCK_TOP_M])
        self._contact_count = 0
        self._engaged_layer = self._retracting = self._retracted = False
        self._drill_started_at = None
        self._drill_seen_contact = False
        self._drill_goal_feed = drilling.TOOL_START_Z_M - drilling.ROCK_TOP_M + .089
        self.drill_phase, self.collector_phase = "travelling", "waiting"
        self._collector_state = "waiting"
        self._state_since = 0.0
        self._target_chunk: int | None = None
        self._target_track_id: int | None = None
        self._target_estimate = None
        self._target_slot = 0
        self._planned_releases = 0
        self._last_release_time = -10.0
        self._pickup_misses = 0
        self._pickup_retry_after = {}
        self._grasp_dwell_s = float(np.clip(self.controller_params.get("grasp_dwell_s", .5), .1, 1.5))
        self._held_chunk: int | None = None
        self._produced = np.zeros(DEBRIS_COUNT, bool)
        self._released = np.zeros(DEBRIS_COUNT, bool)
        self._collected = np.zeros(DEBRIS_COUNT, bool)
        self._piece_cart = np.full(DEBRIS_COUNT, -1, int)
        self._settled_for = np.zeros(DEBRIS_COUNT)
        self._born_at = np.full(DEBRIS_COUNT, np.nan)
        self._pickup_times = np.full(DEBRIS_COUNT, np.nan)
        self._grasp_contact_count = 0
        self._collector_contact_force = 0.0
        self._collector_peak_force = 0.0
        self._grasp_events: list[dict] = []
        self._collector_contacts: list[dict] = []
        self._grasp_policy = AcceptedGraspPolicy()
        self._fingers_were_closed = False
        self._observer = FleetObserver(self.controller_params)
        self._corruption = SensorCorruption(self.seed, cam_height_above_table=SENSOR_CAMERA_POSITION[2] - FLOOR_Z)
        self._sensor_frame = -1
        self._tracks = []
        self._last_sensor_rgb = None
        self._last_sensor_depth = None
        self._sensor_occluded = False
        self._cart_load_n = np.zeros(len(CART_CENTERS))
        self._cart_loaded_s = np.zeros(len(CART_CENTERS))
        self._cart_dispatched = np.zeros(len(CART_CENTERS), bool)
        self._base_distance = {"drill": 0.0, "collector": 0.0}
        self._last_base = {"drill": DRILL_START.copy(), "collector": COLLECTOR_START.copy()}
        self._body = {key: self.model.body(name).id for key, name in
                      (("drill", "drill_mobile_base"), ("collector", "collector_mobile_base"), ("grip", "collector_lift"))}
        self._debris_body = [self.model.body(f"debris_{i}").id for i in range(DEBRIS_COUNT)]
        self._debris_geom = [self.model.geom(f"debris_{i}").id for i in range(DEBRIS_COUNT)]
        self._pads = {self.model.geom(f"collector_pad_{side}").id for side in ("left", "right")}
        self._cart_floors = [self.model.geom(f"cart_{i}_floor").id for i in range(len(CART_CENTERS))]
        mujoco.mj_forward(self.model, self.data)
        initial_hand = self.data.xpos[self._body["collector"]] + np.array([GRIP_OFFSET_X, 0, .24])
        self._arm = ArmController(self.model, self.data, initial_hand)

    def _joint(self, name: str) -> tuple[int, int]:
        jid = self.model.joint(name).id
        return int(self.model.jnt_qposadr[jid]), int(self.model.jnt_dofadr[jid])

    def _base_proprio(self, key: str) -> np.ndarray:
        """Planar base forward kinematics from the two own joint encoders."""
        origin = DRILL_START if key == "drill" else COLLECTOR_START
        name = "drill_mobile_base" if key == "drill" else "collector_mobile_base"
        return np.array([origin[0] + self.data.qpos[self._joint(name + "_x")[0]],
                         origin[1] + self.data.qpos[self._joint(name + "_y")[0]], .49])

    def _base_servo(self, key: str, target: np.ndarray, precision: bool = False) -> tuple[float, float]:
        name = "drill_mobile_base" if key == "drill" else "collector_mobile_base"
        position = self._base_proprio(key)[:2]
        speeds = []
        for k, axis in enumerate(("x", "y")):
            _, dof = self._joint(f"{name}_{axis}")
            velocity = self.data.qvel[dof]
            error = target[k] - position[k]
            if precision:
                force = 30000 * error - 2000 * velocity
            else:
                desired = float(np.clip(5.0 * error, -.42, .42))
                force = 500 * (desired - velocity)
            self.data.ctrl[self.model.actuator(f"{name}_{axis}_motor").id] = np.clip(force, -300, 300)
            speeds.append(velocity)
        return float(np.linalg.norm(target - position)), float(np.linalg.norm(speeds))

    def _lift_servo(self, target: float) -> tuple[float, float]:
        base = self._base_proprio("collector")
        desired = np.array([base[0] + GRIP_OFFSET_X, base[1], .61 + target])
        return self._arm.command(desired)

    def _fingers(self, closed: bool) -> None:
        if closed != self._fingers_were_closed:
            self._grasp_policy.reset()
            self._fingers_were_closed = closed
        left, _ = self._joint("collector_grip_left")
        right, _ = self._joint("collector_grip_right")
        measured_gap = max(0.0, .110 - self.data.qpos[left] - self.data.qpos[right])
        target_gap = self._grasp_policy.close_gap_m(measured_gap, float(self.data.time)) if closed else .110
        stroke = float(np.clip((.110 - target_gap) * .5, 0, .057))
        for side in ("left", "right"):
            self.data.ctrl[self.model.actuator(f"collector_grip_{side}_motor").id] = stroke

    def _hold_drill(self) -> None:
        q, v = self.data.qpos[self._feed_qpos], self.data.qvel[self._feed_dof]
        self.data.ctrl[self._feed_act] = np.clip(-1800 * q - 180 * v + self.data.qfrc_bias[self._feed_dof], -150, 150)
        self.data.ctrl[self._spin_act] = 0

    def _control_drill(self) -> None:
        target = DRILL_START if self._retracted else DRILL_WORK
        distance, speed = self._base_servo("drill", target, precision=self._drill_started_at is not None and not self._retracted)
        if self._drill_started_at is None:
            self._hold_drill()
            self.phase = self.drill_phase = "travelling"
            if distance < .0006 and speed < .002:
                self._drill_started_at = float(self.data.time)
        else:
            q = float(self.data.qpos[self._feed_qpos])
            velocity = float(self.data.qvel[self._feed_dof])
            if self._axial_force_n > 1.0:
                self._drill_seen_contact = True
            if not self._retracting and q >= self._drill_goal_feed:
                self._retracting, self.stop_reason = True, "mission_depth"
            if self._retracting:
                if q < .0015 and abs(velocity) < .012:
                    self._retracted = True
                desired_velocity = float(np.clip(-5 * q, -.075, .01))
                self.phase = "complete" if self._retracted else "retracting"
                spin_target = 0.0 if self._retracted else drilling.SPINDLE_TARGET_RAD_S * .45
            elif self.data.time - self._drill_started_at < .35:
                desired_velocity = -4 * q
                spin_target = drilling.SPINDLE_TARGET_RAD_S
                self.phase = "spinup"
            elif not self._drill_seen_contact:
                # Initial surface is a fixed mission coordinate. Subsequent
                # advances use measured force only, never remaining-layer poses.
                gap = float(self.ee_pos()[2] - drilling.ROCK_TOP_M)
                desired_velocity = float(np.clip(10 * (gap - .0001), .006, .040))
                spin_target = drilling.SPINDLE_TARGET_RAD_S
                self.phase = "approach"
            else:
                desired_velocity = .025 if self._axial_force_n > 4 else .018
                spin_target = drilling.SPINDLE_TARGET_RAD_S
                self.phase = "drilling"
            feed_force = 2000 * (desired_velocity - velocity) + self.data.qfrc_bias[self._feed_dof]
            self.data.ctrl[self._feed_act] = float(np.clip(feed_force, -150, 150))
            self.data.ctrl[self._spin_act] = spin_target
            self.drill_phase = "returning" if self._retracted and distance > .015 else self.phase
            if self._retracted and distance <= .015:
                self.drill_phase = "parked"

    def _set_collector_state(self, state: str) -> None:
        self._collector_state = state
        self._state_since = float(self.data.time)

    def _control_collector(self) -> None:
        state = self._collector_state
        if self._target_track_id is None and self._planned_releases < DEBRIS_COUNT:
            previous_cart = self._planned_releases // CART_CAPACITY - 1
            ready = self._planned_releases % CART_CAPACITY != 0 or previous_cart < 0 or self._cart_dispatched[previous_cart]
            candidates = [track for track in self._tracks if track["confidence"] >= .55 and track["age_s"] < .25
                          and self.data.time - track["born_at"] > .15
                          and self.data.time >= self._pickup_retry_after.get(track["track_id"], 0)] if ready else []
            if candidates:
                chosen = min(candidates, key=lambda track: track["track_id"])
                self._target_track_id = int(chosen["track_id"])
                self._target_estimate = np.asarray(chosen["pos"], float)
                self._target_slot = self._planned_releases
                self._set_collector_state("approach")
                state = "approach"
        if self._target_track_id is None:
            distance, speed = self._base_servo("collector", COLLECTOR_START)
            self._lift_servo(.12)
            self._fingers(False)
            self.collector_phase = ("parked" if distance < .015 else "returning") if self._planned_releases >= DEBRIS_COUNT else "searching"
            return
        i = self._target_slot
        if state == "approach":
            observed = next((track for track in self._tracks if track["track_id"] == self._target_track_id), None)
            if observed is not None:
                self._target_estimate = .90 * self._target_estimate + .10 * np.asarray(observed["pos"], float)
        piece = self._target_estimate
        cart = i // CART_CAPACITY
        travel, _ = self._joint(f"cart_{cart}_travel")
        slot = np.asarray(cart_slot_positions(cart, travel_x=float(self.data.qpos[travel]))[i % CART_CAPACITY], float)[:2]
        if state in ("approach", "lower", "grasp"):
            base_target = piece[:2] - np.array([GRIP_OFFSET_X, 0])
        else:
            base_target = slot - np.array([GRIP_OFFSET_X, 0])
        if state == "lift":
            # Keep the base stationary while the actual attached chunk clears
            # the ground. The goal is the measured grasp location, not a pose set.
            base_target = self._grasp_base_xy
        distance, speed = self._base_servo("collector", base_target, precision=state in ("lower", "grasp", "lower_bin", "release"))
        lift_target = .13
        if state in ("lower", "grasp"):
            lift_target = float(np.clip(piece[2] + .055 - .61, -.155, .15))
        elif state in ("lower_bin", "release"):
            lift_target = .015  # chunk center .57, just above the cart rim
        lift_error, lift_speed = self._lift_servo(lift_target)
        self._fingers(state in ("grasp", "lift", "carry", "lower_bin"))
        self.collector_phase = {"approach": "approaching debris", "lower": "lowering gripper",
                                "grasp": "grasping", "lift": "lifting", "carry": "carrying to cart",
                                "lower_bin": "lowering into cart", "release": "releasing"}.get(state, state)
        if state == "approach" and distance < .008 and speed < .04:
            self._set_collector_state("lower")
        elif state == "lower" and distance < .002 and lift_error < .0015 and lift_speed < .015:
            self._set_collector_state("grasp")
        elif state == "lift" and lift_error < .003 and lift_speed < .025:
            self._set_collector_state("carry")
        elif state == "carry" and distance < .004 and speed < .015:
            self._set_collector_state("lower_bin")
        elif state == "lower_bin" and lift_error < .002 and lift_speed < .02:
            held = self._held_chunk
            if held is None:
                raise RuntimeError("Cannot release a grasp that was never established")
            self.data.eq_active[self.model.equality(f"debris_{held}_grasp").id] = False
            self._held_chunk = None
            self._released[held] = True
            self._piece_cart[held] = cart
            self._planned_releases += 1
            self._last_release_time = float(self.data.time)
            self._set_collector_state("release")
        elif state == "release" and self.data.time - self._state_since > .7:
            self._target_chunk = None
            self._target_track_id = None
            self._set_collector_state("waiting")
        elif state == "grasp" and self.data.time - self._state_since > 3.0:
            # A missing bilateral tactile contact is an actual failed pickup.
            self._pickup_misses += 1
            self._pickup_retry_after[self._target_track_id] = float(self.data.time) + 5.0
            self._target_chunk = self._target_track_id = None
            self._set_collector_state("waiting")

    def _spawn_debris(self, index: int) -> None:
        """Initialize newly produced material once, at the moving discharge."""
        body = self._debris_body[index]
        jid = self.model.joint(f"debris_{index}_free").id
        qa, va = int(self.model.jnt_qposadr[jid]), int(self.model.jnt_dofadr[jid])
        # Ejection from the drill's chute is an explicit discrete material
        # creation model, tied only to a completed measured work milestone.
        # Spaced discharge ports keep open gripper pads clear of neighboring
        # pieces. Each free body is born here once at its actual break event.
        position = self.data.xpos[self._body["drill"]] + np.array([.36 - .14 * index, .18, .11])
        self.data.qpos[qa:qa + 3] = position
        self.data.qpos[qa + 3:qa + 7] = (1, 0, 0, 0)
        self.data.qvel[va:va + 6] = (0, 0, .025, 0, 0, 0)
        self.data.eq_active[self.model.equality(f"debris_{index}_dormant").id] = False
        gid = self._debris_geom[index]
        self.model.geom_rgba[gid, 3] = 1
        self.model.geom_contype[gid] = 4
        self.model.geom_conaffinity[gid] = 14
        self._produced[index] = True
        self._born_at[index] = float(self.data.time)

    def _check_grasp(self) -> None:
        if self._collector_state != "grasp" or self._target_track_id is None:
            return
        if self.data.time - self._state_since < self._grasp_dwell_s:
            return
        contacts_by_piece = {}
        for contact in self.data.contact[:self.data.ncon]:
            a, b = int(contact.geom1), int(contact.geom2)
            if contact.efc_address < 0:
                continue
            pad = a if a in self._pads else b if b in self._pads else None
            other = b if pad == a else a
            if pad is not None and other in self._debris_geom:
                contacts_by_piece.setdefault(other, set()).add(pad)
        grasped = next((geom for geom, pads in contacts_by_piece.items() if len(pads) == 2), None)
        if grasped is None:
            return
        i = self._debris_geom.index(grasped)
        touched = contacts_by_piece[grasped]
        self._target_chunk = i
        # Set the weld's relative pose from the actual two contacted bodies.
        # Activating it preserves the measured pose; no body is repositioned.
        eqid = self.model.equality(f"debris_{i}_grasp").id
        hand = self._body["grip"]
        piece = self._debris_body[i]
        rot = self.data.xmat[hand].reshape(3, 3)
        self.model.eq_data[eqid, 3:6] = rot.T @ (self.data.xpos[piece] - self.data.xpos[hand])
        inverse = self.data.xquat[hand].copy()
        inverse[1:] *= -1
        relative = np.zeros(4)
        mujoco.mju_mulQuat(relative, inverse, self.data.xquat[piece])
        self.model.eq_data[eqid, 6:10] = relative
        self.model.eq_data[eqid, 10] = 1
        self.data.eq_active[eqid] = True
        self._held_chunk = i
        self._observer.mark_consumed(self._target_track_id)
        self._pickup_times[i] = float(self.data.time)
        self._grasp_events.append({"t": float(self.data.time), "debris_id": i,
                                   "pad_contacts": len(touched), "contact_force_n": self._collector_contact_force})
        self._grasp_base_xy = self._base_proprio("collector")[:2]
        self._set_collector_state("lift")

    def _measure_gripper(self) -> None:
        force = 0.0
        touched = set()
        wrench = np.zeros(6)
        contacts = []
        for ci in range(self.data.ncon):
            contact = self.data.contact[ci]
            a, b = int(contact.geom1), int(contact.geom2)
            if contact.efc_address < 0:
                continue
            pad = a if a in self._pads else b if b in self._pads else None
            other = b if pad == a else a
            if pad is not None and other in self._debris_geom:
                mujoco.mj_contactForce(self.model, self.data, ci, wrench)
                force += float(np.linalg.norm(wrench[:3]))
                touched.add(pad)
                world_force = contact.frame.reshape(3, 3).T @ wrench[:3]
                if pad == a:
                    world_force *= -1
                contacts.append({"position_m": contact.pos.tolist(), "force_world_n": world_force.tolist()})
        self._collector_contact_force = force
        self._collector_peak_force = max(self._collector_peak_force, force)
        self._grasp_contact_count = len(touched)
        self._collector_contacts = contacts

    def _cart_control(self) -> None:
        for cart in range(len(CART_CENTERS)):
            qpos, _ = self._joint(f"cart_{cart}_travel")
            travel = float(self.data.qpos[qpos])
            if (self._cart_loaded_s[cart] >= .25 and self._planned_releases >= (cart + 1) * CART_CAPACITY
                    and self.data.time - self._last_release_time > .8):
                self._cart_dispatched[cart] = True
            speed = float(np.clip(-2.0 * (travel + 1.05), -.22, 0)) if self._cart_dispatched[cart] else 0.0
            self.data.ctrl[self.model.actuator(f"cart_{cart}_drive").id] = speed

    def _measure_cart_loads(self) -> None:
        loads = np.zeros(len(CART_CENTERS))
        wrench = np.zeros(6)
        for ci in range(self.data.ncon):
            contact = self.data.contact[ci]
            a, b = int(contact.geom1), int(contact.geom2)
            floor = a if a in self._cart_floors else b if b in self._cart_floors else None
            if floor is None or contact.efc_address < 0:
                continue
            mujoco.mj_contactForce(self.model, self.data, ci, wrench)
            world_force = contact.frame.reshape(3, 3).T @ wrench[:3]
            if floor == a:
                world_force *= -1
            loads[self._cart_floors.index(floor)] += max(0.0, -float(world_force[2]))
        self._cart_load_n = .95 * self._cart_load_n + .05 * loads
        stable = (self._cart_load_n > 2.7) & (self._cart_load_n < 4.5)
        self._cart_loaded_s = np.where(stable, self._cart_loaded_s + DT, 0)

    def render_rgbd(self) -> tuple[np.ndarray, np.ndarray]:
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=480, width=640)
        self._renderer.update_scene(self.data, camera="cam_overhead")
        rgb = self._renderer.render().copy()
        self._renderer.enable_depth_rendering()
        try:
            depth = self._renderer.render().copy()
        finally:
            self._renderer.disable_depth_rendering()
        return rgb, depth

    def _observe(self) -> None:
        frame = int(np.floor((self.data.time + 1e-8) / FRAME_INTERVAL_S))
        if frame == self._sensor_frame:
            return
        rgb, depth = self.render_rgbd()
        rgb, depth, occluded = self._corruption.corrupt(rgb, depth, frame)
        self._last_sensor_rgb, self._last_sensor_depth, self._sensor_occluded = rgb, depth, bool(occluded)
        focal = 240 / math.tan(math.radians(SENSOR_FOVY_DEG) / 2)
        calibration = {"K": [[focal, 0, 320], [0, focal, 240], [0, 0, 1]],
                       "cam_pos": list(SENSOR_CAMERA_POSITION), "cam_rot": [[1, 0, 0], [0, -1, 0], [0, 0, -1]],
                       "floor_z": .4, "discharge_roi": [-.35, .62, .10, .30]}
        proprio = {"base_positions": [self._base_proprio(key).tolist() for key in ("drill", "collector")]}
        self._tracks = self._observer.observe(rgb, depth, calibration, proprio, float(self.data.time))
        self._sensor_frame = frame

    def render_camera(self) -> np.ndarray:
        if self._last_sensor_rgb is not None:
            return self._last_sensor_rgb.copy()
        return super().render_camera()

    def step(self) -> dict[str, Any]:
        if self._closed:
            raise RuntimeError("The mining fleet episode is closed")
        self._observe()
        self._control_drill()
        self._control_collector()
        self._cart_control()
        mujoco.mj_step(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        to_remove = self._measure_contacts(accumulate_work=self._drill_started_at is not None)
        if self._axial_force_n >= 1:
            self._engaged_layer = True
        if self._drill_started_at is not None:
            self.max_depth_mm = max(self.max_depth_mm, max(0., drilling.ROCK_TOP_M - self.ee_pos()[2]) * 1000)
        if self._axial_force_n > drilling.FORCE_LIMIT_N and not self._retracting:
            self._retracting, self.stop_reason = True, "force_limit"
            to_remove = []
        for layer in to_remove:
            gid = self._layer_geoms[layer]
            self.removed[layer] = True
            self.model.geom_contype[gid] = self.model.geom_conaffinity[gid] = 0
            self.model.geom_rgba[gid, 3] = 0
        for i in range(DEBRIS_COUNT):
            if self.removed.sum() >= (i + 1) * 2 and not self._produced[i]:
                self._spawn_debris(i)
        if to_remove:
            self._engaged_layer = False
            mujoco.mj_forward(self.model, self.data)
            self._measure_contacts(accumulate_work=False)
        if self._drill_started_at is not None and self.data.time - self._drill_started_at >= 22.0 and not self._retracting:
            self._retracting, self.stop_reason = True, "time_limit"
        self._measure_gripper()
        self._measure_cart_loads()
        self._check_grasp()
        for i in range(DEBRIS_COUNT):
            position = self.data.xpos[self._debris_body[i]]
            _, va = self._joint(f"debris_{i}_free")
            cart = int(self._piece_cart[i])
            if cart < 0:
                continue
            cart_qpos, _ = self._joint(f"cart_{cart}_travel")
            bounds = cart_interior_bounds(cart, travel_x=float(self.data.qpos[cart_qpos]))
            inside = bool(self._released[i] and bounds["x"][0] < position[0] < bounds["x"][1]
                          and bounds["y"][0] < position[1] < bounds["y"][1] and .42 < position[2] < .54)
            settled = inside and np.linalg.norm(self.data.qvel[va:va + 3]) < .10
            self._settled_for[i] = self._settled_for[i] + DT if settled else 0.0
            self._collected[i] = bool(inside and (self._collected[i] or self._settled_for[i] >= .15))
        for key in ("drill", "collector"):
            pos = self.data.xpos[self._body[key], :2]
            self._base_distance[key] += float(np.linalg.norm(pos - self._last_base[key]))
            self._last_base[key] = pos.copy()
        return self.telemetry()

    def telemetry(self) -> dict[str, Any]:
        result = super().telemetry()
        complete = bool(self.removed.all() and self._retracted and self._collected.all()
                        and self.drill_phase == "parked" and self.collector_phase == "parked"
                        and all(self.data.qpos[self._joint(f"cart_{c}_travel")[0]] < -1.0 for c in range(len(CART_CENTERS))))
        result.update({
            "phase": "complete" if complete else ("collecting" if self._retracted else self.phase),
            "mode": "mining_fleet", "duration_s": DURATION_S, "complete": complete,
            "seed": self.seed,
            "autonomy_mode": "experimental sensor-driven fleet; not benchmark-verified",
            "sensor_based": self._sensor_frame >= 0,
            "pickup_observation": "corrupted RGB-D tracks with robot proprioception",
            "sensor_tracks": list(self._tracks), "sensor_occluded": self._sensor_occluded,
            "perception_stats": dict(self._observer.stats),
            "pickup_misses": self._pickup_misses,
            "drilling_complete": bool(self.removed.all() and self._retracted),
            "machine_positions": {"driller": self.data.xpos[self._body["drill"]].tolist(),
                                  "collector": self.data.xpos[self._body["collector"]].tolist()},
            "machine_phases": {"driller": self.drill_phase, "collector": self.collector_phase},
            "machine_distance_m": dict(self._base_distance),
            "debris_produced": int(self._produced.sum()), "debris_collected": int(self._collected.sum()),
            "debris_total": DEBRIS_COUNT, "held_debris": self._held_chunk,
            "cart_counts": [int(np.sum(self._collected & (self._piece_cart == c)))
                            for c in range(len(CART_CENTERS))],
            "cart_capacity": CART_CAPACITY,
            "active_cart": min(int(self._released.sum()) // CART_CAPACITY, len(CART_CENTERS) - 1),
            "cart_centers": [list(center) for center in CART_CENTERS],
            "cart_load_n": self._cart_load_n.tolist(),
            "cart_travel_m": [float(self.data.qpos[self._joint(f"cart_{c}_travel")[0]]) for c in range(len(CART_CENTERS))],
            "cart_departed": [bool(self.data.qpos[self._joint(f"cart_{c}_travel")[0]] < -1.0) for c in range(len(CART_CENTERS))],
            "grasp_contact_count": self._grasp_contact_count,
            "collector_contact_force_n": self._collector_contact_force,
            "collector_peak_force_n": self._collector_peak_force,
            "collector_contacts": list(self._collector_contacts),
            "collector_tcp_m": self.data.site_xpos[self.model.site("collector_grip_site").id].tolist(),
            "grasp_events": list(self._grasp_events),
            "collector_arm": self._arm.telemetry(),
            "grasp_policy": dict(self._grasp_policy.metadata),
            "fleet_planner": "RGB-D pickup tracks, force-controlled drilling, tactile grasps, measured cart loads",
            "grasp_model": "contact-conditioned weld", "drive_model": "actuated planar mobile bases",
            "debris": [{"id": i, "produced": bool(self._produced[i]), "collected": bool(self._collected[i]),
                        "released": bool(self._released[i]), "position": self.data.xpos[self._debris_body[i]].tolist(),
                        "produced_at": float(self._born_at[i]) if self._produced[i] else None,
                        "picked_up_at": float(self._pickup_times[i]) if np.isfinite(self._pickup_times[i]) else None}
                       for i in range(DEBRIS_COUNT)],
        })
        return result


def run_mining_fleet(seed: int = 101, frame_callback: Callable | None = None,
                    controller_params: dict | None = None,
                    frame_interval_s: float = FRAME_INTERVAL_S) -> dict[str, Any]:
    """Run physical mining with an independent presentation callback cadence.

    Sensor updates remain 20 Hz and physics/control remain at DT. Callback
    frames sample the first actual simulation state at or after each requested
    timestamp; a 1/60 interval therefore records distinct physical states.
    """
    frame_interval_s = float(frame_interval_s)
    if not math.isfinite(frame_interval_s) or frame_interval_s < DT:
        raise ValueError(f"frame_interval_s must be finite and at least {DT} seconds")
    ep = MiningFleetEpisode(seed, controller_params=controller_params)
    try:
        if frame_callback is not None:
            frame_callback(ep, ep.telemetry())
        next_frame = frame_interval_s
        for step in range(round(DURATION_S / DT)):
            telemetry = ep.step()
            if frame_callback is not None and ep.data.time + 1e-8 >= next_frame:
                frame_callback(ep, telemetry)
                next_frame += frame_interval_s
        return ep.telemetry()
    finally:
        ep.close()
