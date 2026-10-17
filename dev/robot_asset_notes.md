# Industrial robot source investigation and ready assets

Requested model: AVRcontent, **Animated Industrial Manipulator - Robotic Arm**,
https://sketchfab.com/3d-models/animated-industrial-manipulator-robotic-arm-6d77e1ce2f1640249412773ff893ff95

The public official endpoint
https://api.sketchfab.com/v3/models/6d77e1ce2f1640249412773ff893ff95
was fetched successfully on 2026-10-17. Raw public metadata is retained in
`assets/robot_reference/sketchfab_model_api.txt`; its public HTML page is retained
in `sketchfab_page.txt`. The response identifies AVRcontent and reports:

- `isDownloadable: false`
- `license.label: Standard`, `license.slug: st`
- `price: 1500` (raw API field; no currency interpretation asserted)
- 88,385 faces, 46 textures, one animation.

The requested asset is not offered as a public downloadable open-source model.
Its viewer embed grants no source download or redistribution right. No protected
viewer geometry, textures or animations were extracted. The exact model needs
an authorized source archive and its applicable purchase/creator license before
it can be converted into articulated MuJoCo bodies.

## Legitimate downloaded fallback

Actual Universal Robots UR5e from Google DeepMind MuJoCo Menagerie:
https://github.com/google-deepmind/mujoco_menagerie/tree/main/universal_robots_ur5e

Pinned commit: `62ab00f2bc44f6f20c1c9b7a13f3fb1dd41a4e35`.
Downloaded unmodified into `assets/robot_reference/universal_robots_ur5e/`:
MJCF `ur5e.xml`, scene, README, changelog, model LICENSE and 20 OBJ mesh files.
Total 25 files, 30,994,086 bytes. Git blob hashes and individual byte sizes are
retained in `assets/robot_reference/menagerie_manifest.json`.

The **model-specific BSD-3-Clause license**, copyright 2018 ROS Industrial
Consortium, is included intact. The repository root license does not replace
this model-specific license. Preserve it and attribution when redistributing.
This fallback is distinct from AVRcontent's model and must not be labelled as it.

`ur5e.xml` contains actual mass/inertia, a six-joint articulated body tree,
position actuators, collision geometry and detailed per-link visual OBJ meshes.
Joint names: `shoulder_pan_joint`, `shoulder_lift_joint`, `elbow_joint`,
`wrist_1_joint`, `wrist_2_joint`, `wrist_3_joint`. Terminal body `wrist_3_link`
contains `attachment_site`, local position `0 0.1 0`, quaternion `-1 1 0 0`.
Upstream meshdir is `assets`; integrations must resolve this when merging MJCF.
It is suitable for actual simulated articulation rather than prerecorded animation.

Optional `ui.robot_visuals.decorate_robots(xml)->xml` adds inertialess, noncontact
tracked-base panel/hub/hose/lamp details to existing `drill_mobile_base` and
`collector_mobile_base`. It creates no joints or controls and does not change
survey geometry. Call after physical/industrial-arm assembly; root owns wiring.

Native CMU backfaces were addressed separately in the source importer: native
OBJ assets duplicate unchanged survey positions and reversed face indices, so
both sides render without fabricating thickness or moving survey vertices.
Browser DoubleSide geometry remains unchanged.

No tests, simulated episodes or GPU rendering were run during asset preparation.
