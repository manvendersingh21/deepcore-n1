# Third-party assets: sources and licenses

Total on disk: 26,735,549 bytes (about 26.7 MB). Cap was 60 MB.

| Local path | Type | Source | License | Size on disk |
|---|---|---|---|---|
| `soil_ambientcg_Ground112/` | PBR texture set (soil/ground), 1K JPG. Maps: Color, NormalGL, NormalDX, Roughness, AmbientOcclusion, Displacement. Also .mtlx, .tres, .usdc, .blend, preview PNG. | https://ambientcg.com/a/Ground112 (zip: https://ambientcg.com/get?file=Ground112_1K-JPG.zip) | CC0 1.0 (ambientCG site: "CC0 Public Domain license") | 9,762,840 B (zip was 9,765,076 B) |
| `rock_ambientcg_Rock064/` | PBR texture set (rock), 1K JPG. Maps: Color, NormalGL, NormalDX, Roughness, AmbientOcclusion, Displacement. Also .mtlx, .tres, .usdc, .blend, preview PNG. | https://ambientcg.com/a/Rock064 (zip: https://ambientcg.com/get?file=Rock064_1K-JPG.zip) | CC0 1.0 (ambientCG site) | 8,203,490 B (zip was 8,205,462 B) |
| `hdri_polyhaven/kloofendal_48d_partly_cloudy_puresky_1k.hdr` | HDRI, 1K Radiance .hdr (partly cloudy sky) | https://polyhaven.com/a/kloofendal_48d_partly_cloudy_puresky | CC0 1.0 (Poly Haven license page) | 1,435,119 B |
| `mesh_polyhaven_overhead_crane/overhead_crane_1k.gltf` + `overhead_crane.bin` + `textures/` (6 JPGs, 1K) | Industrial mesh: overhead gantry crane, glTF 2.0 | https://polyhaven.com/a/overhead_crane (1K files under https://dl.polyhaven.org/file/ph-assets/Models/gltf/1k/overhead_crane/ and https://dl.polyhaven.org/file/ph-assets/Models/jpg/1k/overhead_crane/) | CC0 1.0 (Poly Haven license page) | 7,331,814 B |

Notes:
- The soil and rock zips also contain .blend and .usdc files (about 2.4 MB combined). Kept because they came in the zip. Delete them if only the JPG maps are needed.
- No excavator model was found in the CC0 sources checked (see below).

## Rejected

- **Robotiq 2F-85 gripper (MuJoCo Menagerie, `robotiq_2f85/`)**: BSD-3-Clause (ROS-Industrial). Outside the CC0 / public-domain / MIT / Apache whitelist. Downloaded, then deleted.
- **Ground108 (ambientCG)**: same kind of soil set as Ground112. Not needed.
- **2K / 4K / 8K texture variants and PNG zips** (for example Ground112 8K JPG 424 MB, PNG zips about 20 MB each): too large for this scope.
- **Robotiq 2f85.png render (1.1 MB) and Menagerie scene.xml**: not needed.
- **Poly Haven rusted_spade_01 (CC0, tools)**: an alternative dig-site prop (about 2.4 MB total). Not used because the brief asked for one industrial mesh.
- **Excavator search**: none of the CC0 catalogs checked had an excavator. Poly Haven's 521 models were searched for excavator, backhoe, digger, loader, tractor and similar terms; no match. ambientCG is textures only. MuJoCo Menagerie's gripper models are BSD. Not searched beyond these sources.

## License notes

- CC0 1.0 (ambientCG and Poly Haven items): no attribution required. Kept as-is.
- Attribution is not legally required for CC0. Listed here for traceability.

## Scope

- Nothing under `core/`, `agents/`, any `scene.xml`, or `core/LOCK.json` was touched.
