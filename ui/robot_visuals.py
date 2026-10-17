"""Optional inertialess visual detail attached to the two actual mobile bases."""
from __future__ import annotations
import xml.etree.ElementTree as ET


def decorate_robots(xml: str) -> str:
    """Add named housings, tread details and cable runs; preserve all dynamics.

    Geoms have zero mass and no contacts. No bodies/joints/controllers are added.
    Call after the physical fleet and industrial arm have been assembled.
    """
    root = ET.fromstring(xml)
    for name, color in [('drill_mobile_base', '.78 .42 .09 1'),
                        ('collector_mobile_base', '.055 .40 .39 1')]:
        body = root.find(f".//body[@name='{name}']")
        if body is None or body.find(f"geom[@name='{name}_visual_engine_cover']") is not None:
            continue
        def geom(suffix, kind, **attributes):
            ET.SubElement(body, 'geom', name=f'{name}_visual_{suffix}', type=kind,
                          contype='0', conaffinity='0', mass='0', density='0', group='2', **attributes)
        geom('engine_cover', 'box', pos='-.095 0 .091', size='.105 .10 .026', rgba=color)
        geom('cover_cap', 'box', pos='-.095 0 .122', size='.10 .096 .005', rgba='.13 .15 .16 1')
        for side in [-1, 1]:
            y = side * .18
            geom(f'track_guard_{side}', 'box', pos=f'0 {side * .142} .008',
                 size='.205 .041 .014', rgba=color)
            for i in range(13):
                geom(f'tread_top_{side}_{i}', 'box', pos=f'{-.186 + i * .031} {side * .145} -.001',
                     size='.005 .035 .005', rgba='.10 .115 .12 1')
            for i in range(5):
                geom(f'hub_{side}_{i}', 'cylinder', pos=f'{-.14 + i * .07} {y} -.025',
                     quat='.7071068 .7071068 0 0', size='.015 .005', rgba='.28 .30 .31 1')
            for i in range(6):
                geom(f'vent_{side}_{i}', 'box', pos=f'{-.16 + i * .026} {side * .102} .09',
                     size='.008 .002 .016', rgba='.075 .085 .09 1')
            geom(f'headlamp_case_{side}', 'box', pos=f'.196 {side * .09} .038',
                 size='.013 .027 .017', rgba='.09 .11 .12 1')
            geom(f'headlamp_lens_{side}', 'box', pos=f'.210 {side * .09} .038',
                 size='.001 .022 .012', rgba='.93 .88 .68 1')
        # Short service cable segments follow the chassis, outside the tool's contact path.
        cable_points = [(-.18, .075, .13), (-.12, .075, .16), (-.04, .075, .17), (.035, .065, .13)]
        for i, (a, b) in enumerate(zip(cable_points, cable_points[1:])):
            geom(f'service_hose_{i}', 'capsule', fromto=' '.join(str(v) for v in (*a, *b)),
                 size='.006', rgba='.055 .065 .07 1')
        geom('service_coupling', 'cylinder', pos='.035 .065 .13', size='.012 .012', rgba='.48 .50 .51 1')
    return ET.tostring(root, encoding='unicode')
