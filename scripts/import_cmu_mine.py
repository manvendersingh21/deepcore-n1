"""Derive a display-normalized cutaway from the downloaded CMU VRML1 survey.

This importer deliberately supports the source's verified flat VRML1 profile;
it rejects other node arrangements instead of silently dropping transforms.
No survey coordinates are smoothed, procedurally replaced, or nonuniformly scaled.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import re
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'assets/cmu_mine_reference/mine1-mod.vrml'
OUTPUT = ROOT / 'assets/cmu_mine_reference/derived'


def import_mine(source: Path = SOURCE, output: Path = OUTPUT) -> dict:
    raw = source.read_bytes()
    text = raw.decode('ascii')
    if not text.startswith('#VRML V1.0 ascii'):
        raise ValueError('Expected downloaded VRML V1.0 ASCII source')
    nodes = re.findall(r'\b([A-Za-z][A-Za-z0-9]*)\s*\{', text)
    if nodes != ['Material', 'Rotation', 'Coordinate3', 'IndexedFaceSet']:
        raise ValueError(f'Unsupported VRML1 scene profile: {nodes}')
    rotation = np.fromstring(re.search(r'Rotation\s*\{\s*rotation\s*([^}]+)', text)[1], sep=' ')
    if rotation.shape != (4,) or not np.allclose(rotation, [1, 0, 0, -1.5708]):
        raise ValueError('Source rotation changed; review import basis')
    coords = re.search(r'Coordinate3\s*\{\s*point\s*\[(.*?)\]', text, re.S)[1]
    vertices = np.fromstring(coords.replace(',', ' '), sep=' ').reshape(-1, 3)
    indices = re.search(r'IndexedFaceSet\s*\{\s*coordIndex\s*\[(.*?)\]', text, re.S)[1]
    indices = np.fromstring(indices.replace(',', ' '), dtype=np.int64, sep=' ')
    # This source is triangulated already. Fail if polygon boundaries differ.
    faces = indices.reshape(-1, 4)
    if not np.all(faces[:, 3] == -1) or np.any(faces[:, :3] < 0) or np.any(faces[:, :3] >= len(vertices)):
        raise ValueError('Expected valid triangulated IndexedFaceSet')
    faces = faces[:, :3]
    if not np.isfinite(vertices).all():
        raise ValueError('Nonfinite source coordinates')
    # VRML's authored X rotation puts Z-up survey points into Y-up viewer space.
    # Undo that viewer orientation to choose survey Z as the workcell vertical,
    # then rigidly align the tunnel direction with workcell X.
    scale = 0.006
    rigid = np.array([[0., 1., 0.], [-1., 0., 0.], [0., 0., 1.]])
    translation = np.array([4.32, 0., .4])
    world = vertices @ rigid.T * scale + translation
    tri_source = vertices[faces]
    # A real continuous opening near the beginning of the survey, not the huge network.
    inside = ((tri_source[:, :, 1] >= -1000) & (tri_source[:, :, 1] <= -300)
              & (np.abs(tri_source[:, :, 0]) <= 380) & (tri_source[:, :, 2] <= 330)).all(axis=1)
    centers = world[faces].mean(axis=1)
    # Remove near-side wall and forward roof to expose the machinery to the camera.
    visible = ((centers[:, 1] >= 0.0) | (centers[:, 2] <= .65)) & (centers[:, 2] <= 1.58)
    # Leave the existing physical workcell's plane exposed, avoiding coincident floor faces.
    floor_overlap = ((centers[:, 0] > -.9) & (centers[:, 0] < 1.8)
                     & (np.abs(centers[:, 1]) < 1.05) & (centers[:, 2] <= .43))
    visible &= ~floor_overlap
    selected = faces[inside & visible]
    if not len(selected):
        raise ValueError('Selected survey cutaway is empty')
    output.mkdir(parents=True, exist_ok=True)
    chunks = []
    for start in range(0, len(selected), 2400):
        face_chunk = selected[start:start + 2400]
        used, inverse = np.unique(face_chunk, return_inverse=True)
        points = world[used]
        local_faces = inverse.reshape(-1, 3)
        name = f'tunnel_{start // 2400:03d}'
        # MuJoCo culls backfaces. Duplicate positions with reversed winding for
        # the second side; separate indices prevent opposing normals averaging.
        obj = ''.join('v %.7f %.7f %.7f\n' % tuple(v) for v in np.vstack([points, points]))
        obj += ''.join('f %d %d %d\n' % tuple(f + 1) for f in local_faces)
        obj += ''.join('f %d %d %d\n' % tuple(f[::-1] + len(points) + 1) for f in local_faces)
        (output / f'{name}.obj').write_text(obj)
        chunks.append({'name': name, 'obj': f'{name}.obj', 'vertices': points.astype(np.float32).round(7).tolist(),
                       'indices': local_faces.ravel().tolist()})
    browser = {'chunks': chunks}
    current_objs = {chunk['obj'] for chunk in chunks}
    for stale in output.glob('tunnel_*.obj'):
        if stale.name not in current_objs:
            stale.unlink()
    (output / 'tunnel.json').write_text(json.dumps(browser, separators=(',', ':')) + '\n')
    used_world = world[np.unique(selected)]
    matrix = np.eye(4)
    matrix[:3, :3] = scale * rigid
    matrix[:3, 3] = translation
    metadata = {
        'source': 'CMU mine-mapping / downloaded mine1-mod.vrml (VRML V1.0 ASCII)',
        'source_local_path': str(source.relative_to(ROOT)), 'source_sha256': hashlib.sha256(raw).hexdigest(),
        'source_units': 'unknown; no metric survey calibration claimed',
        'normalization': 'Uniform display scale 0.006 workcell units per source unit; source survey Z used as vertical. Authored VRML viewer rotation is canceled by basis choice.',
        'authored_vrml_axis_angle': rotation.tolist(),
        'source_to_workcell_matrix_row_major': matrix.tolist(),
        'selection_source_bounds': {'x': [-380, 380], 'y': [-1000, -300], 'z': [0, 330]},
        'cutaway': 'Keep complete source triangles within selection; omit near-side wall where world centroid y<0 and z>0.65; omit all roof triangles above centroid z1.58. Remove floor triangles overlapping workcell x(-0.9,1.8), abs(y)<1.05,z<=0.43 to avoid coplanar physical floor. No generated replacement walls.',
        'source_vertices': len(vertices), 'source_triangles': len(faces),
        'selected_triangles': len(selected), 'chunks': [{'name': c['name'], 'obj': c['obj']} for c in chunks],
        'native_backfaces': 'OBJ duplicates unchanged source positions with reversed face winding and separate indices for double-sided native rendering; browser uses DoubleSide material.',
        'world_bounds': [used_world.min(axis=0).tolist(), used_world.max(axis=0).tolist()],
        'camera': {'position': [3.4, -4.2, 3.1], 'target': [.42, .1, .95]},
        'render_only': True,
        'rights_note': 'Downloaded research reference; no license is asserted by this conversion. Consult source manifest before redistribution.',
    }
    (output / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    return metadata


if __name__ == '__main__':
    result = import_mine()
    print(json.dumps({k: result[k] for k in ['selected_triangles', 'world_bounds', 'chunks']}, indent=2))
