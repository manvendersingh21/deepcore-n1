"""Native MuJoCo presentation view of the real excavation episode.

Only the presentation scene and its private visual model are dressed. Collision
geometry, dynamics, sensor views, and original model/data arrays remain unchanged.
The frame consumer must call close_render(ep) in its success/abort cleanup path.
"""
from __future__ import annotations

from dataclasses import dataclass
from copy import copy
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import cv2

WIDTH, HEIGHT = 1100, 640
_AMBER = (0.78, 0.43, 0.09, 1.0)
_DARK = (0.12, 0.15, 0.16, 1.0)
_STEEL = (0.50, 0.57, 0.60, 1.0)
_LIGHT = (1.0, 0.86, 0.62, 1.0)


@dataclass
class _Decoration:
    kind: int
    size: np.ndarray
    pos: np.ndarray
    mat: np.ndarray
    color: tuple
    emission: float = 0.0
    metal: bool = False


def _box(size, pos, color=_DARK, angle=0.0, metal=True):
    c, s = np.cos(angle), np.sin(angle)
    return _Decoration(mujoco.mjtGeom.mjGEOM_BOX, np.asarray(size, dtype=float),
                       np.asarray(pos, dtype=float), np.array([c, -s, 0, s, c, 0, 0, 0, 1]), color, metal=metal)


def _beam(a, b, radius, color=_STEEL):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    axis = b - a
    length = np.linalg.norm(axis)
    axis /= length
    across = np.cross(axis, [0, 0, 1] if abs(axis[2]) < 0.95 else [0, 1, 0])
    across /= np.linalg.norm(across)
    basis = np.column_stack((across, np.cross(axis, across), axis))
    return _Decoration(mujoco.mjtGeom.mjGEOM_CYLINDER, np.array([radius, length / 2, 0.0]),
                       (a + b) / 2, basis.reshape(-1), color, metal=True)


def _site_dressing():
    items = []
    # Portal feet stand beyond the physical basin, on its existing outer rim.
    for x in (-0.035, 0.935):
        for y in (-0.555, 0.555):
            items.extend([
                _box((0.045, 0.045, 0.012), (x, y, 0.635), _STEEL),
                _box((0.018, 0.018, 0.22), (x, y, 0.875), _AMBER),
                _box((0.021, 0.021, 0.035), (x, y, 0.693)),
            ])
            for dx in (-0.029, 0.029):
                for dy in (-0.029, 0.029):
                    items.append(_box((0.004, 0.004, 0.004), (x + dx, y + dy, 0.65)))
    for y in (-0.555, 0.555):
        items.extend([
            _box((0.54, 0.025, 0.027), (0.45, y, 1.092), _AMBER),
            _box((0.54, 0.008, 0.004), (0.45, y, 1.123), _STEEL),
        ])
    for x, end in ((-0.035, 0.14), (0.935, 0.76)):
        items.append(_beam((x, 0.555, 0.72), (end, 0.555, 1.07), 0.008, _AMBER))
    # Painted curb strips sit on the physical rim, clear of the pit and objects.
    for y in (-0.507, 0.507):
        items.append(_box((0.408, 0.012, 0.003), (0.45, y, 0.625)))
        for i in range(20):
            items.append(_box((0.010, 0.011, 0.0035), (0.068 + i * 0.04, y, 0.626), _AMBER, -0.35))
    for x in (-0.18, 1.08):
        y = 0.68
        items.extend([
            _box((0.055, 0.055, 0.012), (x, y, 0.634)),
            _beam((x, y, 0.65), (x, y, 1.07), 0.008, _DARK),
            _box((0.045, 0.018, 0.026), (x, y, 1.075)),
        ])
        lamp = _box((0.037, 0.002, 0.018), (x, y - 0.02, 1.075), _LIGHT)
        lamp.emission = 0.7
        items.append(lamp)
    # Site equipment and conduit give the gantry a tangible scale and setting.
    items.extend([
        _box((0.09, 0.06, 0.085), (-0.25, 0.19, 0.71)),
        _box((0.093, 0.063, 0.009), (-0.25, 0.19, 0.802), _AMBER),
        _box((0.026, 0.003, 0.018), (-0.25, 0.127, 0.752), _STEEL),
    ])
    for i in range(5):
        items.append(_box((0.055, 0.002, 0.003), (-0.25, 0.128, 0.654 + i * 0.014), _STEEL))
    cable = [(-0.25, 0.25, 0.638), (-0.26, 0.37, 0.638), (-0.22, 0.48, 0.638),
             (-0.13, 0.60, 0.638), (-0.035, 0.61, 0.638), (-0.035, 0.56, 0.70)]
    items.extend(_beam(a, b, 0.006, _DARK) for a, b in zip(cable, cable[1:]))
    # Seeded scree outside the basin. These are visual objects with no contacts.
    rng = np.random.default_rng(6281)
    for _ in range(105):
        while True:
            x, y = rng.uniform((-0.72, -1.08), (1.63, 1.2))
            if not (-0.33 < x < 1.22 and -0.66 < y < 0.80):
                break
        radius = rng.uniform(0.012, 0.064)
        size = np.array([radius, radius * rng.uniform(0.6, 1.2), radius * rng.uniform(0.3, 0.65)])
        color = float(rng.uniform(0.21, 0.38))
        angle = rng.uniform(-np.pi, np.pi)
        rock = _box(size, (x, y, 0.62 + size[2] * 0.5),
                    (color * 1.15, color, color * 0.83, 1.0), angle, metal=False)
        # A mix of fractured slabs and rounded weathered fragments.
        if rng.random() < 0.6:
            rock.kind = mujoco.mjtGeom.mjGEOM_ELLIPSOID
        items.append(rock)
    return items


def _add(scene, item):
    if scene.ngeom >= scene.maxgeom:
        return
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(geom, item.kind, item.size, item.pos, item.mat, np.asarray(item.color, dtype=np.float32))
    geom.category = mujoco.mjtCatBit.mjCAT_DECOR
    geom.objtype = mujoco.mjtObj.mjOBJ_UNKNOWN
    geom.objid = -1
    geom.segid = -1
    geom.emission = item.emission
    geom.specular = 0.32 if item.metal else 0.06
    geom.shininess = 0.35 if item.metal else 0.04
    geom.reflectance = 0.08 if item.metal else 0.0
    scene.ngeom += 1


def _context(ep):
    state = getattr(ep, '_mine_presentation', None)
    if state is not None:
        return state
    # The presentation context has its own visual model so photographic texture
    # uploads cannot alter the overhead sensor's model or GPU texture resources.
    from ui.ar_overlays import VisualModel
    visual = VisualModel(ep.model)
    visual_model = visual.model
    assets = Path(__file__).resolve().parents[1] / 'assets'
    for texture_name, source in (
        ('tex_earth', 'soil_ambientcg_Ground112/Ground112_1K-JPG_Color.jpg'),
        ('tex_wall', 'rock_ambientcg_Rock064/Rock064_1K-JPG_Color.jpg'),
        ('tex_rock', 'rock_ambientcg_Rock064/Rock064_1K-JPG_Color.jpg'),
    ):
        path = assets / source
        tid = mujoco.mj_name2id(visual_model, mujoco.mjtObj.mjOBJ_TEXTURE, texture_name)
        if tid < 0 or not path.is_file() or visual_model.tex_nchannel[tid] != 3:
            continue
        image = cv2.imread(str(path))
        if image is None:
            continue
        width, height = int(visual_model.tex_width[tid]), int(visual_model.tex_height[tid])
        image = cv2.cvtColor(cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        adr = int(visual_model.tex_adr[tid])
        visual_model.tex_data[adr:adr + width * height * 3] = image.reshape(-1)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = (0.45, 0.015, 0.68)
    camera.distance = 1.88
    camera.azimuth = 133
    camera.elevation = -47
    # Independent option object avoids changes to the episode sensor renderer.
    option = mujoco.MjvOption()
    option.sitegroup[:] = 0
    option.sitegroup[0] = 1
    names = {i: mujoco.mj_id2name(visual_model, mujoco.mjtObj.mjOBJ_GEOM, i) or ''
             for i in range(visual_model.ngeom)}
    renderer = mujoco.Renderer(visual_model, HEIGHT, WIDTH, max_geom=6500)
    state = {'renderer': renderer, 'camera': camera, 'option': option,
             'dressing': [], 'names': names, 'visual': visual, 'sensor_packet': None, 'sensor_at': -1.}
    ep._mine_presentation = state
    return state


def render_presentation(ep: Any, rgb=None, depth=None, out=None, rec=None) -> np.ndarray:
    """Return a 1100x640 RGB native rendering of the current measured state."""
    state = _context(ep)
    renderer = state['renderer']
    visual_data = state['visual'].sync(ep)
    renderer.update_scene(visual_data, camera=state['camera'], scene_option=state['option'])
    scene = renderer.scene
    scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = True
    scene.flags[mujoco.mjtRndFlag.mjRND_FOG] = False
    scene.flags[mujoco.mjtRndFlag.mjRND_HAZE] = False
    # Change presentation material appearance only, after MuJoCo builds this scene.
    earth_texid = next((geom.texid for geom in scene.geoms[:scene.ngeom]
                        if geom.objtype == mujoco.mjtObj.mjOBJ_GEOM
                        and state['names'].get(geom.objid) == 'pit_floor'), -1)
    rock_texid = next((geom.texid for geom in scene.geoms[:scene.ngeom]
                       if geom.objtype == mujoco.mjtObj.mjOBJ_GEOM
                       and state['names'].get(geom.objid) == 'boulder0'), -1)
    for geom in scene.geoms[:scene.ngeom]:
        if geom.objtype != mujoco.mjtObj.mjOBJ_GEOM:
            continue
        name = state['names'].get(geom.objid, '')
        if name in ('rubble_a', 'rubble_b'):
            geom.rgba[:] = (0.33, 0.36, 0.38, 1.0)
            geom.specular = 0.08
        elif name == 'core':
            geom.rgba[:] = (0.13, 0.82, 0.87, 1.0)
            geom.emission = 0.55
            geom.specular = 0.4
        elif name in ('palm', 'wrist_cyl'):
            geom.rgba[:] = (0.77, 0.48, 0.17, 1.0)
        elif name.startswith('rim_') or name == 'world_floor':
            geom.rgba[:] = (0.3, 0.32, 0.33, 1.0)
            geom.texid = earth_texid
            geom.texrepeat[:] = (5, 5)
            geom.texuniform = 1
        elif name.startswith('wall_'):
            geom.rgba[:] = (0.35, 0.37, 0.38, 1.0)
            geom.texid = rock_texid
            geom.texrepeat[:] = (3, 3)
            geom.texuniform = 1
        elif name == 'pit_floor':
            geom.rgba[:] = (0.31, 0.33, 0.34, 1.0)
    # The default scene spotlight is too narrow for the wider presentation camera.
    if scene.nlight:
        light = scene.lights[0]
        light.diffuse[:] = (0.4, 0.45, 0.5)
        light.ambient[:] = (0.035, 0.04, 0.045)
        light.cutoff = 70
        light.exponent = 2
    if scene.nlight > 1:
        scene.lights[1].diffuse[:] = (0.23, 0.28, 0.34)
        scene.lights[1].ambient[:] = (0.025, 0.03, 0.035)
    for item in state['dressing']:
        _add(scene, item)
        if not item.metal:
            scene.geoms[scene.ngeom - 1].texid = rock_texid
    from ui.ar_overlays import add_headlamp, add_sensor_overlays, add_force_arrow
    add_headlamp(scene, ep)
    if rgb is not None and depth is not None:
        now = float(ep.data.time)
        if now-state['sensor_at']>=.199 or state['sensor_packet'] is None:
            from ui.sensor_overlay import camera_packet
            state['sensor_packet'] = camera_packet(rgb, depth, ep.camera_matrix, ep.cam_pos, ep.cam_rot)
            state['sensor_at'] = now
    estimates = (out or {}).get('estimated_objects', {})
    command = (out or {}).get('action', {}).get('target_pos')
    gate = getattr(rec, 'core_gate_force', []) if rec is not None else []
    core_force = max(gate, default=float(getattr(ep,'core_force_now',0)))
    add_sensor_overlays(scene, state['sensor_packet'], estimates, command, ep.ee_pos(), core_force)
    add_force_arrow(scene, ep.ee_pos(), ep.wrist_force_world(), 18.)
    return renderer.render().copy()


def close_render(ep: Any) -> None:
    """Release presentation resources; call in the frame consumer's finally block.

    This is idempotent and never closes either of Episode's sensor renderers.
    """
    state = getattr(ep, '_mine_presentation', None)
    if state is None:
        return
    try:
        state['renderer'].close()
    finally:
        del ep._mine_presentation


def presentation_camera(ep: Any):
    """Expose the free camera for projection of native HUD force annotations."""
    return _context(ep)['camera']
