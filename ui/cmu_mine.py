"""Render-only downloaded CMU survey geometry for native fleet presentation.

All source transforms and cutaway details are recorded in derived/metadata.json.
This helper adds no bodies, joints, floor changes, forces, or contacts.
"""
from __future__ import annotations
import json
from pathlib import Path
import warnings
import xml.etree.ElementTree as ET

DERIVED = Path(__file__).resolve().parents[1] / 'assets/cmu_mine_reference/derived'


def append_mine_assets(xml: str) -> str:
    """Append optional source-derived CMU mesh assets/noncolliding world geoms.

    Geom and mesh names are ``cmu_tunnel_000`` through the metadata chunk count.
    Missing assets leave the original XML untouched. Call before model compile.
    """
    try:
        metadata = json.loads((DERIVED / 'metadata.json').read_text())
        chunks = metadata['chunks']
        paths = [(chunk['name'], (DERIVED / chunk['obj']).resolve()) for chunk in chunks]
        if not paths or any(not path.is_file() for _, path in paths):
            raise FileNotFoundError('CMU derived mesh chunks unavailable')
    except (OSError, ValueError, KeyError, TypeError) as error:
        warnings.warn(f'CMU survey presentation unavailable: {error}', RuntimeWarning)
        return xml
    root = ET.fromstring(xml)
    worldbody = root.find('worldbody')
    if worldbody is None:
        raise ValueError('Native mine presentation requires worldbody')
    if worldbody.find("geom[@name='cmu_tunnel_000']") is not None:
        return xml
    asset = root.find('asset')
    if asset is None:
        asset = ET.SubElement(root, 'asset')
    rock_material = asset.find("material[@name='rock']")
    texture_material = next((material for material in asset.findall('material')
                             if material.get('texture') == 'tex_rock'), None)
    material = rock_material if rock_material is not None else texture_material
    appearance = ({'material': material.get('name'), 'rgba': '1 1 1 1'}
                  if material is not None else {'rgba': '.46 .40 .33 1'})
    for name, path in paths:
        mesh_name = f'cmu_{name}'
        ET.SubElement(asset, 'mesh', name=mesh_name, file=str(path), maxhullvert='32')
        ET.SubElement(worldbody, 'geom', name=mesh_name, type='mesh', mesh=mesh_name,
                      contype='0', conaffinity='0', group='2', density='0', **appearance)
    return ET.tostring(root, encoding='unicode')
