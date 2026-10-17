"""Independent MuJoCo drilling demonstration with simulated breakable layers.

This is a discrete removal proxy, not continuum fracture or a rock calibration.
Every reported load and pose comes from MuJoCo. A layer can disappear only after
loaded cutter contact dissipates its prescribed work budget. No renderer is
created until a caller requests an image; CPU physics can run without OpenGL.
"""
from __future__ import annotations

from collections.abc import Callable
import math
from typing import Any

import mujoco
import numpy as np


DT = 0.001
DURATION_S = 12.0
FRAME_INTERVAL_S = 0.05
LAYERS = 12
LAYER_THICKNESS_M = 0.008
ROCK_TOP_M = 0.565
TOOL_START_Z_M = 0.625
FORCE_LIMIT_N = 120.0
MIN_CUTTING_LOAD_N = 12.0
SPINDLE_TARGET_RAD_S = 23.0
_CENTER = np.array([0.42, 0.0])


def _vector(values) -> str:
    return " ".join(f"{float(value):.7f}" for value in values)


def _camera_axes(position, target) -> str:
    backward = np.asarray(position, float) - np.asarray(target, float)
    backward /= np.linalg.norm(backward)
    right = np.cross([0.0, 0.0, 1.0], backward)
    right /= np.linalg.norm(right)
    return _vector(np.concatenate((right, np.cross(backward, right))))


def _scene_xml(seed: int) -> str:
    """All moving presentation geometry belongs to the physical moving bodies."""
    rng = np.random.default_rng(seed)
    rock = []
    for layer in range(LAYERS):
        z = ROCK_TOP_M - (layer + 0.5) * LAYER_THICKNESS_M
        shade = (0.47, 0.38, 0.29) if layer % 3 else (0.34, 0.33, 0.31)
        rock.append(
            f'<geom name="rock_layer_{layer}" type="cylinder" '
            f'pos="0.42 0 {z:.7f}" size="0.039 {LAYER_THICKNESS_M / 2}" '
            f'rgba="{_vector((*shade, 1))}" material="rock" '
            'contype="2" conaffinity="1" friction="0.65 0.001 0.0001"/>'
        )
        # Persistent strata surrounding the bore: four strips and a primitive
        # approximation to the circular wall. These do not participate in wear.
        for dx, dy, sx, sy in ((-.094, 0, .046, .14), (.094, 0, .046, .14),
                              (0, -.094, .048, .046), (0, .094, .048, .046)):
            rock.append(
                f'<geom class="decor" type="box" name="stratum_{layer}_{len(rock)}" '
                f'pos="{.42 + dx} {dy} {z:.7f}" size="{sx} {sy} .00385" '
                f'rgba="{_vector((*shade, 1))}" material="rock"/>'
            )
    for i in range(24):
        a = i * 2 * math.pi / 24
        rock.append(
            f'<geom class="decor" name="bore_wall_{i}" type="box" '
            f'pos="{.42 + .045 * math.cos(a)} {.045 * math.sin(a)} '
            f'{ROCK_TOP_M - LAYERS * LAYER_THICKNESS_M / 2}" '
            f'size=".006 .006 {LAYERS * LAYER_THICKNESS_M / 2}" '
            f'euler="0 0 {a}" material="rock" rgba=".40 .35 .29 1"/>'
        )

    ribs = []
    for strand in range(2):
        for segment in range(30):
            points = []
            for j in (segment, segment + 1):
                a = j * 2 * math.pi / 10 + strand * math.pi
                points.extend((.022 * math.cos(a), .022 * math.sin(a), .025 + j * .007))
            ribs.append(
                f'<geom class="decor" name="flight_{strand}_{segment}" type="capsule" '
                f'fromto="{_vector(points)}" size=".0045" material="steel"/>'
            )
    cutters = []
    for i in range(4):
        a = i * math.pi / 2
        cutters.append(
            f'<geom name="cutter_{i}" type="sphere" '
            f'pos="{.023 * math.cos(a)} {.023 * math.sin(a)} .0075" '
            'size=".0075" material="carbide" contype="1" conaffinity="2" '
            'friction="0.65 0.001 0.0001"/>'
        )
    dressing = []
    for i in range(65):
        x, y = rng.uniform((-.22, -.60), (1.10, .60))
        if abs(x - .42) < .19 and abs(y) < .19:
            continue
        r = rng.uniform(.005, .024)
        dressing.append(
            f'<geom name="scree_{i}" class="decor" type="ellipsoid" '
            f'pos="{x} {y} {.4 + r * .28}" size="{r} {r * .73} {r * .47}" '
            f'euler="0 .1 {rng.uniform(0, 6.28)}" material="rock"/>'
        )
    for x in (.02, .82):
        for y in (-.34, .34):
            dressing.extend((
                f'<geom class="decor" name="foot_{x}_{y}" type="box" '
                f'pos="{x} {y} .425" size=".06 .06 .025" material="dark"/>',
                f'<geom class="decor" name="mast_{x}_{y}" type="box" '
                f'pos="{x} {y} .735" size=".018 .018 .29" material="paint"/>',
            ))
    for y in (-.34, .34):
        dressing.append(
            f'<geom class="decor" type="box" pos=".42 {y} 1.025" '
            'size=".44 .027 .026" material="paint"/>'
        )
    for i in range(12):
        dressing.append(
            f'<geom class="decor" type="box" pos="{.075 + i * .062} -.435 .413" '
            'size=".018 .012 .003" euler="0 0 -.4" material="paint"/>'
        )
    overview = (1.32, -1.26, 1.25)
    axes = _camera_axes(overview, (.42, 0, .66))
    return f'''<mujoco model="deepcore_breakable_layer_drilling">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="{DT}" gravity="0 0 -9.81" integrator="implicitfast"
          cone="elliptic" iterations="60" tolerance="1e-9"/>
  <statistic center=".42 0 .65" extent="1.5"/>
  <visual>
    <global offwidth="1100" offheight="640"/>
    <quality shadowsize="2048" offsamples="4"/>
    <headlight ambient=".19 .18 .16" diffuse=".25 .25 .25" specular=".10 .10 .10"/>
    <map znear=".005" zfar="8"/>
  </visual>
  <asset>
    <texture name="tex_rock" type="2d" builtin="flat" mark="random"
             width="128" height="128" random=".32" rgb1=".72 .66 .56" markrgb=".43 .39 .33"/>
    <texture name="tex_earth" type="2d" builtin="flat" mark="random"
             width="256" height="256" random=".38" rgb1=".50 .42 .32" markrgb=".31 .26 .20"/>
    <material name="rock" texture="tex_rock" texrepeat="3 3" rgba=".53 .46 .36 1" specular=".08"/>
    <material name="earth" texture="tex_earth" texrepeat="5 5" rgba=".54 .47 .37 1" specular=".04"/>
    <material name="paint" rgba=".73 .39 .075 1" specular=".3" shininess=".3"/>
    <material name="steel" rgba=".57 .64 .67 1" specular=".6" shininess=".7"/>
    <material name="dark" rgba=".075 .105 .12 1" specular=".25"/>
    <material name="carbide" rgba=".13 .17 .19 1" specular=".55" shininess=".5"/>
  </asset>
  <default>
    <geom condim="3" solref="-100000 -40" solimp=".95 .99 .001" margin=".00005"/>
    <default class="decor"><geom contype="0" conaffinity="0" group="1" mass="0"/></default>
  </default>
  <worldbody>
    <light name="key" pos=".3 -.6 1.6" dir=".05 .35 -1" diffuse="1 .88 .70" castshadow="true"/>
    <light name="fill" pos="1 .65 1.3" dir="-.3 -.3 -1" diffuse=".45 .52 .62" castshadow="false"/>
    <camera name="overview" pos="{_vector(overview)}" xyaxes="{axes}" fovy="43"/>
    <camera name="cam_overhead" pos=".42 0 1.55" euler="0 0 0" fovy="43"/>
    <geom class="decor" name="world_floor" type="plane" size="3 3 .1" pos="0 0 .36" material="earth"/>
    <geom name="pit_floor" type="box" pos=".42 0 .38" size=".68 .64 .02"
          material="earth" contype="2" conaffinity="1"/>
    <geom class="decor" name="rock_base" type="box" pos=".42 0 .4345" size=".14 .14 .0345" material="rock"/>
    <geom class="decor" name="bore_bottom" type="cylinder" pos=".42 0 .468" size=".039 .001" material="dark"/>
    {''.join(rock)}
    {''.join(dressing)}
    <geom class="decor" name="crossbeam" type="box" pos=".42 0 1.008" size=".035 .37 .022" material="steel"/>
    <geom class="decor" name="guide_left" type="cylinder" pos=".355 .065 .77" size=".009 .21" material="steel"/>
    <geom class="decor" name="guide_right" type="cylinder" pos=".485 .065 .77" size=".009 .21" material="steel"/>
    <geom class="decor" name="power_pack" type="box" pos="-.16 .18 .49" size=".09 .07 .09" material="dark"/>
    <geom class="decor" name="power_pack_lid" type="box" pos="-.16 .18 .586" size=".094 .074 .006" material="paint"/>
    <body name="drill_feed" pos=".42 0 {TOOL_START_Z_M}">
      <joint name="feed" type="slide" axis="0 0 -1" range="-.002 .19" damping="5"/>
      <inertial pos="0 0 .27" mass="2.8" diaginertia=".008 .008 .004"/>
      <geom class="decor" name="motor_housing" type="box" pos="0 0 .285" size=".052 .049 .057" material="paint"/>
      <geom class="decor" name="motor_cap" type="cylinder" pos="0 0 .346" size=".043 .008" material="dark"/>
      <geom class="decor" name="feed_carriage" type="box" pos="0 .06 .27" size=".082 .025 .035" material="dark"/>
      <body name="drill_spindle">
        <joint name="spindle" type="hinge" axis="0 0 1" limited="false" damping=".002" armature=".002"/>
        <inertial pos="0 0 .13" mass=".55" diaginertia=".003 .003 .0002"/>
        <geom class="decor" name="chuck" type="cylinder" pos="0 0 .229" size=".032 .022" material="dark"/>
        <geom class="decor" name="spindle_index" type="box" pos=".033 0 .23" size=".003 .009 .016" material="paint"/>
        <geom class="decor" name="drill_shaft" type="cylinder" pos="0 0 .125" size=".013 .112" material="steel"/>
        <geom class="decor" name="cutter_carrier" type="cylinder" pos="0 0 .019" size=".029 .008" material="steel"/>
        {''.join(ribs)}
        {''.join(cutters)}
        <site name="tool_tip" pos="0 0 0" size=".001" rgba="0 0 0 0"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <motor name="feed_motor" joint="feed" gear="1" ctrllimited="true" ctrlrange="-150 150"/>
    <velocity name="spindle_motor" joint="spindle" kv=".20" ctrllimited="true"
              ctrlrange="0 30" forcelimited="true" forcerange="-2 2"/>
  </actuator>
  <sensor>
    <actuatorfrc name="feed_drive_force" actuator="feed_motor"/>
    <actuatorfrc name="spindle_drive_torque" actuator="spindle_motor"/>
    <jointvel name="spindle_velocity" joint="spindle"/>
  </sensor>
</mujoco>'''


class DrillingEpisode:
    """One actuator-driven drilling cycle; call step() or use run_drilling()."""

    def __init__(self, seed: int = 101):
        self.seed = int(seed)
        self.model = mujoco.MjModel.from_xml_string(_scene_xml(self.seed))
        self.data = mujoco.MjData(self.model)
        self._renderer = None
        self._overview_renderer = None
        self._closed = False
        self._feed_dof = int(self.model.jnt_dofadr[self.model.joint("feed").id])
        self._feed_qpos = int(self.model.jnt_qposadr[self.model.joint("feed").id])
        self._spindle_dof = int(self.model.jnt_dofadr[self.model.joint("spindle").id])
        self._tip_site = self.model.site("tool_tip").id
        self._feed_act = self.model.actuator("feed_motor").id
        self._spin_act = self.model.actuator("spindle_motor").id
        self._cutters = {self.model.geom(f"cutter_{i}").id for i in range(4)}
        self._layer_geoms = [self.model.geom(f"rock_layer_{i}").id for i in range(LAYERS)]
        self._layer_by_geom = {gid: i for i, gid in enumerate(self._layer_geoms)}
        self.layer_work_j = np.zeros(LAYERS)
        self.layer_threshold_j = np.random.default_rng(self.seed).uniform(3.0, 4.0, LAYERS)
        self.removed = np.zeros(LAYERS, dtype=bool)
        self.phase = "spinup"
        self.stop_reason = ""
        self.peak_force_n = 0.0
        self.max_depth_mm = 0.0
        self._axial_force_n = 0.0
        self._contact_torque_nm = 0.0
        self._cutting_power_w = 0.0
        self._contact_force_world = np.zeros(3)
        self._contact_position = np.array([.42, 0, ROCK_TOP_M])
        self._contact_count = 0
        self._engaged_layer = False
        self._retracting = False
        self._retracted = False
        mujoco.mj_forward(self.model, self.data)

    def ee_pos(self) -> np.ndarray:
        """Measured cutting-tip center, in world coordinates (meters)."""
        return self.data.site_xpos[self._tip_site].copy()

    def _measure_contacts(self, accumulate_work: bool) -> list[int]:
        d = self.data
        force = np.zeros(3)
        torque_z = 0.0
        layer_load = np.zeros(LAYERS)
        layer_power = np.zeros(LAYERS)
        weighted_position = np.zeros(3)
        position_weight = 0.0
        count = 0
        wrench = np.zeros(6)
        tip = self.ee_pos()
        omega = float(d.qvel[self._spindle_dof])
        downward_speed = float(d.qvel[self._feed_dof])
        for ci in range(d.ncon):
            contact = d.contact[ci]
            g1, g2 = int(contact.geom1), int(contact.geom2)
            if contact.efc_address < 0 or (g1 not in self._cutters and g2 not in self._cutters):
                continue
            # MuJoCo's contact-frame force acts on geom2. Rows of frame are
            # world-space contact axes; transpose maps contact to world.
            sign = 1.0 if g2 in self._cutters else -1.0
            other = g1 if sign > 0 else g2
            mujoco.mj_contactForce(self.model, d, ci, wrench)
            frame = contact.frame.reshape(3, 3)
            f_world = sign * (frame.T @ wrench[:3])
            t_world = sign * (frame.T @ wrench[3:])
            arm = contact.pos - tip
            moment = np.cross(arm, f_world) + t_world
            force += f_world
            torque_z += float(moment[2])
            load = max(0.0, float(f_world[2]))
            weighted_position += contact.pos * load
            position_weight += load
            count += 1
            layer = self._layer_by_geom.get(other)
            if layer is not None and not self.removed[layer]:
                # Exact rigid-tool contact-point velocity for these two DOFs.
                velocity = np.cross([0.0, 0.0, omega], arm)
                velocity[2] -= downward_speed
                power = max(0.0, -float(np.dot(f_world, velocity) + t_world[2] * omega))
                layer_load[layer] += load
                layer_power[layer] += power
        self._axial_force_n = max(0.0, float(force[2]))
        self._contact_torque_nm = abs(torque_z)
        self._contact_force_world = force
        self._contact_count = count
        self._cutting_power_w = float(layer_power.sum())
        if position_weight > 0:
            self._contact_position = weighted_position / position_weight
        self.peak_force_n = max(self.peak_force_n, self._axial_force_n)
        break_layers = []
        if accumulate_work and not self._retracting:
            for layer in range(LAYERS):
                if layer_load[layer] >= MIN_CUTTING_LOAD_N and abs(omega) >= 1.0:
                    self.layer_work_j[layer] += layer_power[layer] * DT
                    if self.layer_work_j[layer] >= self.layer_threshold_j[layer] and not self.removed[layer]:
                        break_layers.append(layer)
        return break_layers

    def _control(self) -> None:
        d = self.data
        q = float(d.qpos[self._feed_qpos])
        velocity = float(d.qvel[self._feed_dof])
        if self._retracting:
            if q < .0015 and abs(velocity) < .012:
                self._retracted = True
            desired_velocity = float(np.clip(-5.0 * q, -.075, .01))
            self.phase = ("complete" if self.removed.all() else "stopped") if self._retracted else (
                "safety_retract" if self.stop_reason == "force_limit" else "retracting")
            spin_target = 0.0 if self._retracted else SPINDLE_TARGET_RAD_S * .45
        elif d.time < .35:
            self.phase = "spinup"
            desired_velocity = -4 * q
            spin_target = SPINDLE_TARGET_RAD_S
        else:
            self.phase = "drilling" if self._engaged_layer else "approach"
            remaining = np.flatnonzero(~self.removed)
            surface_z = ROCK_TOP_M - (int(remaining[0]) if len(remaining) else LAYERS) * LAYER_THICKNESS_M
            gap = float(self.ee_pos()[2] - surface_z)
            # Decelerate before each remaining surface; otherwise first contact
            # at the free-travel speed creates a genuine unsafe impact impulse.
            approach_speed = float(np.clip(15.0 * (gap - .0001), .004, .045))
            desired_velocity = .028 if self.phase == "drilling" else approach_speed
            spin_target = SPINDLE_TARGET_RAD_S
        # Bias compensation counters actual modeled gravity. The velocity servo
        # advances under load; no qpos or qvel is assigned by the controller.
        feed_force = 2000.0 * (desired_velocity - velocity) + d.qfrc_bias[self._feed_dof]
        d.ctrl[self._feed_act] = float(np.clip(feed_force, -150, 150))
        d.ctrl[self._spin_act] = spin_target

    def step(self) -> dict[str, Any]:
        """Advance one physical millisecond and return current measurements."""
        if self._closed:
            raise RuntimeError("The drilling episode is closed")
        self._control()
        mujoco.mj_step(self.model, self.data)
        # Refresh poses and forces at the integrated state before publishing.
        mujoco.mj_forward(self.model, self.data)
        to_remove = self._measure_contacts(accumulate_work=True)
        if self._axial_force_n >= 1.0:
            self._engaged_layer = True
        self.max_depth_mm = max(self.max_depth_mm, max(0.0, ROCK_TOP_M - self.ee_pos()[2]) * 1000)
        if self._axial_force_n > FORCE_LIMIT_N and not self._retracting:
            self._retracting = True
            self.stop_reason = "force_limit"
            to_remove = []
        for layer in to_remove:
            gid = self._layer_geoms[layer]
            self.removed[layer] = True
            self.model.geom_contype[gid] = 0
            self.model.geom_conaffinity[gid] = 0
            self.model.geom_rgba[gid, 3] = 0.0
        if to_remove:
            self._engaged_layer = False
            # Clear removed contacts before displaying a zero-contact state.
            mujoco.mj_forward(self.model, self.data)
            self._measure_contacts(accumulate_work=False)
        if self.removed.all() and not self._retracting:
            self._retracting = True
            self.stop_reason = "layers_removed"
        elif self.data.time >= 9.5 and not self._retracting:
            self._retracting = True
            self.stop_reason = "time_limit"
        return self.telemetry()

    def telemetry(self) -> dict[str, Any]:
        d = self.data
        remaining = np.flatnonzero(~self.removed)
        active = int(remaining[0]) if len(remaining) else None
        return {
            "t": float(d.time), "phase": self.phase,
            "axial_force_n": float(self._axial_force_n),
            "torque_nm": abs(float(d.actuator_force[self._spin_act])),
            "depth_mm": float(max(0.0, ROCK_TOP_M - self.ee_pos()[2]) * 1000),
            "rpm": float(d.qvel[self._spindle_dof] * 60 / (2 * math.pi)),
            "layers_removed": int(self.removed.sum()), "layers_total": LAYERS,
            "peak_force_n": float(self.peak_force_n), "limit_force_n": FORCE_LIMIT_N,
            "complete": bool(self.removed.all() and self._retracted),
            "material_model": "breakable layers", "stop_reason": self.stop_reason,
            "max_depth_mm": float(self.max_depth_mm),
            "feed_mm": float(d.qpos[self._feed_qpos] * 1000),
            "contact_torque_nm": float(self._contact_torque_nm),
            "feed_actuator_force_n": float(d.actuator_force[self._feed_act]),
            "cutting_power_w": float(self._cutting_power_w),
            "contact_count": self._contact_count,
            "contact_force_world_n": self._contact_force_world.tolist(),
            "contact_position_m": self._contact_position.tolist(),
            "active_layer": active,
            "layer_work_j": float(self.layer_work_j[active]) if active is not None else 0.0,
            "layer_threshold_j": float(self.layer_threshold_j[active]) if active is not None else 0.0,
            "total_cutting_work_j": float(self.layer_work_j.sum()),
        }

    def render_camera(self) -> np.ndarray:
        """Return the measured overhead RGB camera image, 640 by 480 pixels."""
        if self._closed:
            raise RuntimeError("The drilling episode is closed")
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=480, width=640)
        self._renderer.update_scene(self.data, camera="cam_overhead")
        return self._renderer.render().copy()

    def render_overview(self) -> np.ndarray:
        """Return native MuJoCo overview RGB, 1100 by 640 pixels."""
        if self._closed:
            raise RuntimeError("The drilling episode is closed")
        if self._overview_renderer is None:
            self._overview_renderer = mujoco.Renderer(self.model, height=640, width=1100)
        self._overview_renderer.update_scene(self.data, camera="overview")
        return self._overview_renderer.render().copy()

    def close(self) -> None:
        """Release optional camera contexts; safe to call more than once."""
        for name in ("_renderer", "_overview_renderer"):
            renderer = getattr(self, name, None)
            if renderer is not None:
                renderer.close()
                setattr(self, name, None)
        self._closed = True


def run_drilling(seed: int = 101, frame_callback: Callable | None = None) -> dict[str, Any]:
    """Run twelve simulated seconds, reporting live model/data at exactly 20 Hz.

    Callback signature is ``frame_callback(episode, telemetry)``. A callback
    exception propagates after camera cleanup, allowing a UI to cancel a run.
    The callback controls wall-clock playback pacing; no sleep changes physics.
    """
    ep = DrillingEpisode(seed)
    steps_per_frame = round(FRAME_INTERVAL_S / DT)
    try:
        if frame_callback is not None:
            frame_callback(ep, ep.telemetry())
        for step in range(round(DURATION_S / DT)):
            telemetry = ep.step()
            if frame_callback is not None and (step + 1) % steps_per_frame == 0:
                frame_callback(ep, telemetry)
        return ep.telemetry()
    finally:
        ep.close()
