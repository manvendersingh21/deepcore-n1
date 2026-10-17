# Synthetic mine presentation environment

Generated deterministically by `.venv/bin/python -m ui.mine_environment`.
This is synthetic geological/industrial scenery, distinct from the separately
loaded actual CMU survey. It is not claimed as surveyed or calibrated geology.

`environment.json` shares exact displaced/faceted tunnel triangles, support
profiles, service runs and light positions across native/browser views. OBJ
meshes include explicit flat normals and double-sided face indices. A front
operator cutaway exposes the actual workline; full end vaults extend tunnel depth.
Floor datum .4; tunnel axis X, extent X[-5.4,4.2], sides about±2.12, roof about2.7.

Native: `ui.mine_environment.append_presentation_environment(xml)` adds fixed
noncolliding group2 geoms/assets/lights to a **separate presentation model only**.
No body, joint or equality is added. The caller must allocate its own MjData,
copy actual state, and preserve source alpha by geom name. Never modify the
observation model: the robot's corrupted RGB-D inputs must remain unchanged.
Actual named core/rubble/rock_base/debris_N geoms get bounded replacement visual
meshes without moving their body/geom transforms. Unit mesh templates remain
within the original half-size envelope; cyan crystal is used only for actual core.

Browser: import `/assets/mine_environment/environment.js`, then await
`loadMineEnvironment(parent)` -> `{group,metadata,dispose}`. Unit object templates
are not placed independently; root attaches its actual object visuals to real
poses. Texture/loading failure keeps a plain material or reports an empty group.
Dispose releases geometry/material/texture resources.

`rock_color.png` is a PNG conversion of the existing ambientCG Rock064 CC0 color
texture (see `assets/LICENSES.md`); native fallback uses it because MJCF's image
loader does not accept this JPG through the current file path. Browser references
the existing CC0 JPG/normal maps. No new external downloads.

CPU model compilation succeeded (a minimal model retained its original joint).
No GPU rendering, 60FPS claim or completed visual assessment has been made here.
