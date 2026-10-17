"""Actuated, collision-aware open rail carts for mine-debris collection scenes."""
from __future__ import annotations

import math
import xml.etree.ElementTree as ET


CART_CENTERS = ((-0.35, 0.65), (0.70, 0.65), (1.75, 0.65))
CART_CAPACITY = 2
CART_FLOOR_CENTER_Z = 0.413
CART_FLOOR_HALF_THICKNESS = 0.013
CART_FLOOR_TOP_Z = CART_FLOOR_CENTER_Z + CART_FLOOR_HALF_THICKNESS  # 0.426 m
CART_RIM_TOP_Z = 0.56
CART_INTERIOR_HALF_X = 0.18
CART_INTERIOR_HALF_Y = 0.18
CART_SLOT_Y_OFFSETS = (-0.085, 0.085)
DEFAULT_CHUNK_HALF_HEIGHT_M = 0.023
CART_TRAVEL_RANGE_M = (-3.0, 0.0)
CART_TRAVEL_VALIDATION_TOLERANCE_M = 0.05
CART_TRAVEL_SPEED_RANGE_MPS = (-0.4, 0.0)

_PHYSICAL_COLLISION = {"contype": "2", "conaffinity": "4", "group": "0"}
_VISUAL_ONLY = {"contype": "0", "conaffinity": "0", "group": "2"}
_MATERIALS = {
    "rail_cart_steel": {"rgba": ".30 .35 .38 1", "specular": ".55", "shininess": ".35"},
    "rail_cart_interior": {"rgba": ".23 .28 .30 1", "specular": ".25"},
    "rail_cart_safety_yellow": {"rgba": ".96 .56 .07 1", "specular": ".3"},
    "rail_cart_dark": {"rgba": ".09 .12 .13 1", "specular": ".25"},
    "rail_track_steel": {"rgba": ".19 .22 .24 1", "specular": ".55"},
    "rail_sleeper": {"rgba": ".22 .20 .17 1", "specular": ".15"},
}


def cart_slot_positions(index: int, chunk_half_height_m: float = DEFAULT_CHUNK_HALF_HEIGHT_M,
                        travel_x: float = 0.0
                         ) -> tuple[tuple[float, float, float], ...]:
    """Return two fragment center poses inside cart ``index`` at its travel offset."""
    _check_cart_index(index)
    half_height = float(chunk_half_height_m)
    if not math.isfinite(half_height) or half_height < 0:
        raise ValueError("chunk_half_height_m must be a finite nonnegative number")
    travel = _check_travel_x(travel_x)
    cx, cy = CART_CENTERS[index]
    z = CART_FLOOR_TOP_Z + half_height
    return tuple((cx + travel, cy + offset, z) for offset in CART_SLOT_Y_OFFSETS)


def cart_interior_bounds(index: int, travel_x: float = 0.0) -> dict[str, object]:
    """Return absolute interior bounds and floor/rim heights at travel offset."""
    _check_cart_index(index)
    cx, cy = CART_CENTERS[index]
    cx += _check_travel_x(travel_x)
    return {
        "x": (cx - CART_INTERIOR_HALF_X, cx + CART_INTERIOR_HALF_X),
        "y": (cy - CART_INTERIOR_HALF_Y, cy + CART_INTERIOR_HALF_Y),
        "floor_top_z": CART_FLOOR_TOP_Z,
        "rim_top_z": CART_RIM_TOP_Z,
    }


def cart_layout_metadata() -> list[dict[str, object]]:
    """Return a fresh JSON-friendly layout summary for telemetry and the HUD."""
    return [
        {
            "index": i,
            "name": f"cart_{i}",
            "center_xy": list(center),
            "capacity": CART_CAPACITY,
            "interior_bounds": cart_interior_bounds(i),
            "slot_positions": [list(position) for position in cart_slot_positions(i)],
            "joint": f"cart_{i}_travel",
            "actuator": f"cart_{i}_drive",
            "travel_range_m": list(CART_TRAVEL_RANGE_M),
            "travel_speed_range_mps": list(CART_TRAVEL_SPEED_RANGE_MPS),
            "initially_parked": True,
        }
        for i, center in enumerate(CART_CENTERS)
    ]


def _check_cart_index(index: int) -> None:
    if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(CART_CENTERS):
        raise IndexError(f"cart index must be an integer from 0 to {len(CART_CENTERS) - 1}")


def _check_travel_x(travel_x: float) -> float:
    travel = float(travel_x)
    lower = CART_TRAVEL_RANGE_M[0] - CART_TRAVEL_VALIDATION_TOLERANCE_M
    upper = CART_TRAVEL_RANGE_M[1] + CART_TRAVEL_VALIDATION_TOLERANCE_M
    if not math.isfinite(travel) or not lower <= travel <= upper:
        raise ValueError(f"travel_x must be inside [{lower}, {upper}] meters")
    return travel


def _add_materials(root: ET.Element) -> None:
    asset = root.find("asset")
    if asset is None:
        asset = ET.Element("asset")
        compiler = root.find("compiler")
        if compiler is None:
            root.insert(0, asset)
        else:
            root.insert(list(root).index(compiler) + 1, asset)
    existing = {material.get("name") for material in asset.findall("material")}
    for name, attributes in _MATERIALS.items():
        if name not in existing:
            ET.SubElement(asset, "material", name=name, **attributes)


def _geom(parent: ET.Element, name: str, kind: str, pos: str, size: str,
          material: str, visual_only: bool = False, **extra: str) -> ET.Element:
    attributes = {"name": name, "type": kind, "pos": pos, "size": size,
                  "material": material}
    attributes.update(_VISUAL_ONLY if visual_only else _PHYSICAL_COLLISION)
    attributes.update(extra)
    return ET.SubElement(parent, "geom", **attributes)


def _add_cart(worldbody: ET.Element, index: int) -> None:
    cx, cy = CART_CENTERS[index]
    cart = ET.SubElement(worldbody, "body", name=f"cart_{index}",
                         pos=f"{cx:.4f} {cy:.4f} 0")
    ET.SubElement(cart, "joint", name=f"cart_{index}_travel", type="slide",
                  axis="1 0 0", range=f"{CART_TRAVEL_RANGE_M[0]:.1f} {CART_TRAVEL_RANGE_M[1]:.1f}",
                  damping="15", armature=".3")
    ET.SubElement(cart, "inertial", pos="0 0 .22", mass="36",
                  diaginertia="1.4 1.7 1.2")

    # The open tub is the only colliding part: its floor supports loose chunks,
    # while its four walls keep them within the physical capacity area.
    _geom(cart, f"cart_{index}_floor", "box", f"0 0 {CART_FLOOR_CENTER_Z:.4f}",
          f".200 .200 {CART_FLOOR_HALF_THICKNESS:.3f}", "rail_cart_interior",
          friction=".9 .01 .001", solref=".012 1")
    rim_center_z = (CART_FLOOR_TOP_Z + CART_RIM_TOP_Z) / 2
    rim_half_z = (CART_RIM_TOP_Z - CART_FLOOR_TOP_Z) / 2
    for end, x in (("front", -.20), ("rear", .20)):
        _geom(cart, f"cart_{index}_{end}_wall", "box", f"{x:.3f} 0 {rim_center_z:.4f}",
              f".020 .200 {rim_half_z:.4f}", "rail_cart_steel",
              friction=".9 .01 .001", solref=".012 1")
    for side, y in (("left", -.20), ("right", .20)):
        _geom(cart, f"cart_{index}_{side}_wall", "box", f"0 {y:.3f} {rim_center_z:.4f}",
              f".180 .020 {rim_half_z:.4f}", "rail_cart_steel",
              friction=".9 .01 .001", solref=".012 1")

    # Chassis beams and bolted corner posts make the thin open wagon read as a
    # fabricated industrial cart. These details are visual only.
    for side, y in (("left", -.145), ("right", .145)):
        _geom(cart, f"cart_{index}_underframe_{side}", "box", f"0 {y} .393",
              ".205 .018 .012", "rail_cart_dark", visual_only=True)
    for x in (-.185, .185):
        for y in (-.185, .185):
            _geom(cart, f"cart_{index}_corner_post_{x:+.3f}_{y:+.3f}", "box",
                  f"{x:.3f} {y:.3f} .477", ".018 .018 .083",
                  "rail_cart_steel", visual_only=True)

    # Yellow and black service markings on the upper outer ends; boltheads and
    # rim bars are simple primitive presentation geometry, not moving parts.
    for end, x in (("front", -.222), ("rear", .222)):
        for stripe in range(5):
            y = -.16 + stripe * .08
            material = "rail_cart_safety_yellow" if stripe % 2 == 0 else "rail_cart_dark"
            _geom(cart, f"cart_{index}_{end}_hazard_{stripe}", "box",
                  f"{x:.3f} {y:.3f} .538", ".003 .029 .018", material,
                  visual_only=True)
    for side, y in (("left", -.222), ("right", .222)):
        _geom(cart, f"cart_{index}_{side}_rim_bar", "box", f"0 {y:.3f} .555",
              ".185 .008 .005", "rail_cart_safety_yellow", visual_only=True)

    # Axles, wheels, and buffers are fixed to the sliding cart body. They travel
    # with the wagon but do not spin or collide with debris.
    for axle_i, x in enumerate((-.14, .14)):
        _geom(cart, f"cart_{index}_axle_{axle_i}", "cylinder",
              f"{x:.3f} 0 .425", ".008 .177", "rail_cart_dark",
              visual_only=True, euler=f"{math.pi / 2:.6f} 0 0")
        for side_i, y in enumerate((-.15, .15)):
            _geom(cart, f"cart_{index}_wheel_{axle_i}_{side_i}", "cylinder",
                  f"{x:.3f} {y:.3f} .425", ".025 .020", "rail_cart_dark",
                  visual_only=True, euler=f"{math.pi / 2:.6f} 0 0")
            _geom(cart, f"cart_{index}_wheel_hub_{axle_i}_{side_i}", "cylinder",
                  f"{x:.3f} {y + (0.020 if y > 0 else -0.020):.3f} .425",
                  ".010 .003", "rail_cart_steel", visual_only=True,
                  euler=f"{math.pi / 2:.6f} 0 0")
    for end, x in (("front", -.25), ("rear", .25)):
        _geom(cart, f"cart_{index}_{end}_buffer", "box", f"{x:.3f} 0 .445",
              ".025 .055 .020", "rail_cart_dark", visual_only=True)


def add_rail_carts(xml: str) -> str:
    """Append three actuated open mine carts and their visual track to MJCF.

    Physical cart floors and walls use collision group 0, ``contype=2`` and
    ``conaffinity=4`` so event-created debris can settle in the tubs. Each body
    starts parked at its position default of zero; the native controller may
    drive its `cart_i_drive` velocity actuator negative after loading two
    chunks. Wheels, axles, rails, sleepers, and cosmetic hardware never collide.
    """
    if not isinstance(xml, str) or not xml.strip():
        raise ValueError("xml must be a nonempty MJCF string")
    try:
        root = ET.fromstring(xml)
    except ET.ParseError as exc:
        raise ValueError(f"invalid MJCF XML: {exc}") from exc
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("MJCF must contain worldbody")
    existing = {geom.get("name") for geom in worldbody.findall(".//geom")}
    floor_markers = {f"cart_{i}_floor" for i in range(len(CART_CENTERS))}
    rails_present = {name for name in ("mine_track_rail_left", "mine_track_rail_right") if name in existing}
    if floor_markers.issubset(existing) and len(rails_present) == 2:
        return xml
    if floor_markers.intersection(existing) or rails_present:
        raise ValueError("partial rail-cart layout already exists in MJCF")

    _add_materials(root)
    for side, y in (("left", .50), ("right", .80)):
        _geom(worldbody, f"mine_track_rail_{side}", "box", f"-.700 {y:.3f} .405",
              "3.300 .016 .005", "rail_track_steel", visual_only=True)
    sleeper_i = 0
    x = -3.90
    while x <= 2.61:
        _geom(worldbody, f"mine_track_sleeper_{sleeper_i}", "box", f"{x:.3f} .650 .388",
              ".055 .205 .012", "rail_sleeper", visual_only=True)
        sleeper_i += 1
        x += .25
    for index in range(len(CART_CENTERS)):
        _add_cart(worldbody, index)
    actuator = root.find("actuator")
    if actuator is None:
        actuator = ET.Element("actuator")
        root.insert(list(root).index(worldbody), actuator)
    for index in range(len(CART_CENTERS)):
        ET.SubElement(actuator, "velocity", name=f"cart_{index}_drive",
                      joint=f"cart_{index}_travel", kv="200",
                      ctrllimited="true",
                      ctrlrange=f"{CART_TRAVEL_SPEED_RANGE_MPS[0]:.1f} {CART_TRAVEL_SPEED_RANGE_MPS[1]:.1f}",
                      forcelimited="true", forcerange="-160 160")
    return ET.tostring(root, encoding="unicode")
