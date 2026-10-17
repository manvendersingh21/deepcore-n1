# Downloaded CMU mine integration

Source: `assets/cmu_mine_reference/mine1-mod.vrml`, a downloaded CMU mine-mapping
VRML V1.0 ASCII survey mesh. Source downloads and manifest remain untouched.
The importer reads its actual Coordinate3 vertices and IndexedFaceSet faces;
it does not use the Three.js VRML2 loader or substitute procedural tunnels.
Source rights are not asserted by conversion; consult the download manifest.

The source contains 382,680 vertices and 743,918 triangles. The strict importer
accepts this source's Material / Rotation / Coordinate3 / IndexedFaceSet profile
and rejects unexpected node structures or nontriangular faces. Its source hash,
bounds, orientation, selection and transform are saved in derived/metadata.json.

The selected continuous tunnel opening spans source Y -1000..-300 and X within
±380. Its raw coordinate Z is vertical; the source's VRML rotation was a viewer
orientation into VRML Y-up coordinates. We choose the original survey Z-up basis.
The exact rigid alignment and uniform scale are:

```
world_x = .42 + .006 * (source_y + 650)
world_y = -.006 * source_x
world_z = .4 + .006 * source_z
```

Source units are unknown. This is a display normalization to the workcell, not
metric calibration or a claim that the simulation replicates survey dimensions.
No nonuniform scale, vertex smoothing, displacement or invented walls are used.
Triangles are selected whole. A cutaway omits the near wall above world Z .65
and all roof triangles with centroid above Z 1.58, allowing overhead and orbit
views of both machines. Survey floor triangles in the central physical floor
footprint X(-.9,1.8), abs(Y)<1.05, Z<=.43 are omitted to avoid coplanar flicker.
The retained mesh has 15,181 triangles in 7 chunks, approximately 1.2 MB total
including browser JSON and native OBJ assets. All chunks have 3D extent.
Final bounds: X[-1.68,2.52], Y[-1.77,2.27], Z[.4,1.70].

Browser integration:

```js
import { loadCMUMine } from './cmu_mine.js';
const survey = await loadCMUMine(parentGroup);
// survey.group, survey.metadata, survey.dispose()
// metadata.camera suggests position [3.4,-4.2,3.1], target [.42,.1,.95]
```

The helper imports the existing `three` module. It loads derived JSON, preserves
survey normals through computed vertex normals, and adds double-sided rough
stone surfaces. Missing/invalid assets return an empty detached group with
metadata:null and an error; simulation presentation can continue. A caller that
is disposed while awaiting this function should immediately dispose its result.

Native integration:

```python
from ui.cmu_mine import append_mine_assets
xml = append_mine_assets(xml)
# Then compile the physical model as usual.
```

This adds seven noncolliding world mesh geoms `cmu_tunnel_000` through
`cmu_tunnel_006`, each group2, density0, contype0, conaffinity0. It adds no body,
joint, contact, force or floor change. Browser scene streaming should skip
`cmu_*` geoms because browser loads these source-derived meshes independently.
Missing files leave the original XML unchanged with a warning. OBJ asset paths
are absolute, resolved from the installed repository root.

Conversion command: `.venv/bin/python -m scripts.import_cmu_mine`.
CPU conversion and mesh bounds inspection only; no tests or GPU runs performed.
