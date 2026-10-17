# User-provided robotic_arm.glb inspection

Source: `dev/robotic_arm.glb`, 55,136,408 bytes.
SHA-256: `f532a105fe55403dcc0ddb46fcc2207bd2de0f2bab40b418a595f106146bf12d`.

The embedded asset metadata identifies **ROBOTIC ARM** by
**Stefan_Spasevski_Art**, source
https://sketchfab.com/3d-models/robotic-arm-bd6a77c81eb943fb891f17af6ef01223.
It is a different asset from the previously linked AVRcontent animated arm.
The embedded attribution/license string is preserved verbatim in metadata.json;
user supplied and authorized this local reference for the task.

## Actual structure and conversion

GLB2 contains **283 nodes, 138 meshes, five materials, zero skins and zero
animations**. Meshes are rigid separate parts with generic polySurface names.
Most parts are siblings under RootNode; their vertices are already arranged in
the authored pose. There are no named shoulder/elbow/wrist link frames, joint
axes, limits, kinematic chain, semantic six-link segmentation or rigged joint
pivots. A few node translations group fingers/details, but do not establish a
six-degree-of-freedom robot rig. No baked animation is used or claimed.

`scripts/import_robotic_arm.py` parses GLB buffers/accessors directly (including
byte strides), composes authored node transforms, and exports **138 OBJ parts**
with authored world coordinates. Normals and UVs are retained; original five
base-color textures are extracted without modification. MTL preserves base
color/opacity and roughness/metalness extensions. Full original PBR material
metadata remains in metadata.json; non-base-color texture maps remain in the
original GLB rather than being invented or repainted.

Each part's source node/mesh/material, original composed transform, vertex/face
counts and bounds are recorded. No uniform scale or link mapping is guessed.
Authored world bounds:
X[-3.14889,3.14802], Y[-.04743,13.50530], Z[-6.79515,10.13529].
These large authored dimensions are not treated as a calibrated mine robot.

A CPU-only actual-triangle orthographic inspection is saved at
`assets/robot_reference/user_arm/inspection.png`; it shows the recognizable
industrial arm, gripper, hoses and base in its static authored pose. The source
looks visually useful, but its shape is not a drop-in visual equivalent of the
UR5e body chain.

## Integration assessment

Keep the working **articulated six-joint UR5e** during the current fleet work.
Replacing its visual link meshes with these posed parts without a verified
manual part-to-link assignment, pivot placement and coordinate alignment would
make parts detach, stretch or rotate around the wrong axes. A single rigid
whole-arm attachment would misrepresent articulation and is not integrated.

Prepared OBJ/material metadata can support deliberate future manual rigging.
It does not provide a validated MuJoCo jointed model. No working robot physics,
controller, scene or accepted benchmark source was modified by this inspection.
Native integration agent was notified immediately so sensor-based fleet work
could continue without disruption.

Reproduce: `.venv/bin/python -m scripts.import_robotic_arm`.
CPU conversion and image inspection completed; no GPU rendering or physics
rollouts were run for this task.
