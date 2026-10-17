"""Official Menagerie UR5e assembly and actuator-driven task-space control.

The source geometry, inertias, joints, and torque bounds come from the licensed
UR5e asset. IK operates only on an independent planning MjData. Live joint
positions are assigned once for the initial condition, never for robot motion.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np


ASSET_DIR = Path(__file__).resolve().parents[1] / "assets/robot_reference/universal_robots_ur5e"
JOINTS = ("shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
          "wrist_1_joint", "wrist_2_joint", "wrist_3_joint")
ACTUATORS = ("shoulder_pan", "shoulder_lift", "elbow", "wrist_1", "wrist_2", "wrist_3")


def attach_ur5e(root: ET.Element, mobile_base: ET.Element, gripper: ET.Element) -> None:
    """Append the six-joint upstream arm and attach the physical gripper."""
    source = ET.parse(ASSET_DIR / "ur5e.xml").getroot()
    classes = {element.get("class") for element in source.findall(".//default") if element.get("class")}
    materials = {element.get("name") for element in source.findall("asset/material")}
    for element in source.iter():
        for attr in ("class", "childclass"):
            if element.get(attr) in classes:
                element.set(attr, "ur5e_" + element.get(attr))
        if element.get("material") in materials:
            element.set("material", "ur5e_" + element.get("material"))
    for child in source.find("default"):
        root.find("default").append(deepcopy(child))
    # Robot collision volumes have a dedicated bit; source visual meshes are
    # already noncolliding. Ground and self contact remain physically enabled.
    collision = root.find("default").find(".//default[@class='ur5e_collision']/geom")
    collision.set("contype", "16")
    collision.set("conaffinity", "18")
    collision.set("solref", ".015 1")
    for element in source.find("asset"):
        element = deepcopy(element)
        if element.tag == "material":
            element.set("name", "ur5e_" + element.get("name"))
        elif element.tag == "mesh":
            filename = element.get("file")
            element.set("name", "ur5e_" + Path(filename).stem)
            element.set("file", str((ASSET_DIR / "assets" / filename).resolve()))
        root.find("asset").append(element)
    arm = deepcopy(source.find("worldbody/body"))
    arm.set("name", "ur5e_base")
    arm.set("pos", "0 0 .075")
    for body in arm.findall(".//body"):
        body.set("name", "ur5e_" + body.get("name"))
    for index, geom in enumerate(arm.findall(".//geom")):
        if geom.get("mesh"):
            geom.set("mesh", "ur5e_" + geom.get("mesh"))
            geom.set("name", "ur5e_visual_" + geom.get("mesh")[5:])
        else:
            geom.set("name", f"ur5e_collision_{index}")
    wrist = arm.find(".//body[@name='ur5e_wrist_3_link']")
    attachment = wrist.find("site[@name='attachment_site']")
    gripper.set("pos", attachment.get("pos"))
    # The upstream flange's +Z points out of the wrist. Our fingers extend
    # along local -Z, so rotate the adapter by pi around flange X; otherwise
    # a downward gripper target would place the wrist below the floor.
    gripper.set("quat", "1 1 0 0")
    gripper.set("childclass", "decor")
    ET.SubElement(gripper, "site", name="ur5e_control_site", pos="0 0 0", size=".002", rgba="0 0 0 0")
    wrist.append(gripper)
    mobile_base.append(arm)
    for index, actuator in enumerate(source.find("actuator")):
        actuator = deepcopy(actuator)
        actuator.set("name", "ur5e_" + actuator.get("name"))
        kp, kd = (2400, 120) if index < 3 else (1200, 40)
        actuator.set("gainprm", str(kp))
        actuator.set("biasprm", f"0 {-kp} {-kd}")
        root.find("actuator").append(actuator)


class ArmController:
    """Damped Jacobian IK and gravity-compensated position actuator targets."""

    def __init__(self, model, data, initial_target: np.ndarray):
        self.model, self.data = model, data
        self.plan = mujoco.MjData(model)
        self.jids = np.array([model.joint(name).id for name in JOINTS])
        self.qids = model.jnt_qposadr[self.jids].copy()
        self.base_qids = np.array([model.jnt_qposadr[model.joint("collector_mobile_base_" + axis).id]
                                  for axis in ("x", "y")])
        self.dofs = model.jnt_dofadr[self.jids].copy()
        self.aids = np.array([model.actuator("ur5e_" + name).id for name in ACTUATORS])
        self.site = model.site("ur5e_control_site").id
        self.hand = model.body("collector_lift").id
        self.base = model.body("collector_mobile_base").id
        self.last_base = data.xpos[self.base].copy()
        self.jacp = np.zeros((3, model.nv))
        self.jacr = np.zeros((3, model.nv))
        self.goal = np.asarray(initial_target, float).copy()
        self.last_goal = self.goal.copy()
        self.last_update = -.01
        self.goal_q = np.array([-1.5708, -1.5708, 1.5708, -1.5708, -1.5708, 0.0])
        self.goal_velocity = np.zeros(6)
        self.ik_error_m = float("inf")
        self.orientation_error_rad = float("inf")
        self._solve(self.goal, iterations=160)
        if self.ik_error_m > .003 or self.orientation_error_rad > .03:
            raise RuntimeError(f"Initial UR5e IK failed: {self.ik_error_m:.4f} m, {self.orientation_error_rad:.4f} rad")
        if float(data.time) != 0:
            raise RuntimeError("Initial arm pose may only be assigned at reset")
        data.qpos[self.qids] = self.goal_q
        mujoco.mj_forward(model, data)
        self._actuate()

    def _kinematics(self):
        mujoco.mj_kinematics(self.model, self.plan)
        mujoco.mj_comPos(self.model, self.plan)

    def _error(self, target):
        position = target - self.plan.site_xpos[self.site]
        current = np.zeros(4)
        mujoco.mju_mat2Quat(current, self.plan.site_xmat[self.site])
        # Desired hand orientation is world identity. Its quaternion error is
        # inverse(current), expressed in world coordinates.
        current[1:] *= -1
        rotation = np.zeros(3)
        mujoco.mju_quat2Vel(rotation, current, 1.0)
        return position, rotation

    def _solve(self, target: np.ndarray, iterations: int = 6):
        # Forward kinematics needs only this robot's joint encoders. Debris,
        # carts and the other rover are outside the arm planning observation.
        self.plan.qpos[self.base_qids] = self.data.qpos[self.base_qids]
        self.plan.qpos[self.qids] = self.goal_q
        for _ in range(iterations):
            self._kinematics()
            p, r = self._error(target)
            if np.linalg.norm(p) < 3e-5 and np.linalg.norm(r) < 1e-4:
                break
            mujoco.mj_jacSite(self.model, self.plan, self.jacp, self.jacr, self.site)
            jac = np.vstack((self.jacp[:, self.dofs], .35 * self.jacr[:, self.dofs]))
            error = np.concatenate((p, .35 * r))
            delta = jac.T @ np.linalg.solve(jac @ jac.T + np.eye(6) * .000025, error)
            delta *= min(1.0, .20 / max(np.linalg.norm(delta), 1e-12))
            self.plan.qpos[self.qids] += delta
            bounds = self.model.jnt_range[self.jids]
            self.plan.qpos[self.qids] = np.clip(self.plan.qpos[self.qids], bounds[:, 0] + .01, bounds[:, 1] - .01)
        self._kinematics()
        p, r = self._error(target)
        self.ik_error_m = float(np.linalg.norm(p))
        self.orientation_error_rad = float(np.linalg.norm(r))
        self.goal_q = self.plan.qpos[self.qids].copy()

    def _actuate(self):
        kp = self.model.actuator_gainprm[self.aids, 0]
        kd = -self.model.actuator_biasprm[self.aids, 2]
        # Position-target offset supplies gravity and desired velocity feed
        # forward through the actuators, preserving their physical force bounds.
        targets = self.goal_q + (kd * self.goal_velocity + self.data.qfrc_bias[self.dofs]) / kp
        limits = self.model.actuator_ctrlrange[self.aids]
        self.data.ctrl[self.aids] = np.clip(targets, limits[:, 0], limits[:, 1])

    def command(self, desired_world: np.ndarray) -> tuple[float, float]:
        now = float(self.data.time)
        base = self.data.xpos[self.base]
        self.goal += base - self.last_base
        self.last_base = base.copy()
        if now - self.last_update >= .009999:
            dt = .01 if self.last_update < 0 else now - self.last_update
            delta = np.asarray(desired_world) - self.goal
            length = np.linalg.norm(delta)
            self.goal += delta * min(1., .26 * dt / max(length, 1e-12))
            previous_q = self.goal_q.copy()
            self._solve(self.goal)
            self.goal_velocity = np.clip((self.goal_q - previous_q) / dt, -1.8, 1.8)
            self.last_update = now
        self._actuate()
        position = self.data.site_xpos[self.site]
        mujoco.mj_jacSite(self.model, self.data, self.jacp, self.jacr, self.site)
        speed = np.linalg.norm(self.jacp @ self.data.qvel)
        return float(np.linalg.norm(np.asarray(desired_world) - position)), float(speed)

    def telemetry(self) -> dict:
        return {"model": "Universal Robots UR5e (MuJoCo Menagerie)",
                "joint_positions_rad": self.data.qpos[self.qids].tolist(),
                "joint_target_rad": self.goal_q.tolist(),
                "joint_torque_nm": self.data.actuator_force[self.aids].tolist(),
                "ik_error_m": self.ik_error_m,
                "tool_position_m": self.data.site_xpos[self.site].tolist()}
