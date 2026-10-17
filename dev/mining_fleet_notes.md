# Sensor-driven mining fleet (experimental)

`MiningFleetEpisode(seed, controller_params=None)` exposes the MuJoCo model,
data, `step()`, `telemetry()`, measured tool position, corrupted overhead RGB,
RGB-D rendering, overview rendering, and deterministic resource cleanup.
`run_mining_fleet` returns actual final telemetry and emits 20 Hz callbacks.
This scene is separate from the immutable excavation benchmark; transferring
one accepted gripper method does not certify the fleet task.

## Physical scene

Both rovers move through force actuators on planar X/Y joints. The drill uses
the measured cutter contact-work model from `ui.drilling_scene`. Its controller
uses a fixed mission depth, own joint encoders, and measured axial force; it
does not inspect remaining rock-layer poses or removed-layer flags.

The collector carries the official Menagerie UR5e articulated mesh and six
physical joints. Jacobian IK runs in independent planning data. Position
actuators, gravity compensation, and upstream torque limits move the live arm.
Only the reset initial condition assigns robot joint positions directly.

Six 0.16 kg free bodies are born at discharge ports on successive two-layer
removal events. This is a discrete material creation boundary condition, not
continuum fracture. No existing debris is teleported during pickup or loading.
A retained-grasp weld is enabled only after both physical pads contact the
same fragment, preserving the measured relative pose. It is released over a
cart. The physical finger targets call the accepted v4 gripper closure method
through `AcceptedGraspPolicy`; this is a partial controller transfer, not the
full excavation vision or policy pipeline.

Three carts each accept two distinct fragments. They have physical floors,
walls, X slide joints, and velocity actuators. After two release actions and a
stable measured floor load, a cart drives left 1.05 m. The verifier counts
released fragments only after 150 ms of settling inside the measured moving
cart bounds. Completion additionally requires all six pieces retained, all
three carts beyond 1 m travel, all twelve layers removed, tool retracted, and
both rovers parked. Counts are not mesh visibility or commanded deliveries.

## Sensor and control boundary

The overhead camera is at (.42, .52, 1.65) m, points down, and renders 640×480
RGB and metric depth with 50° vertical field of view. Every control image
passes through the unchanged seeded `core.sensor_corruption.SensorCorruption`:
RGB variation/glare, depth noise/dropout/spikes, and a moving occlusion band.
The observer accepts only these corrupted arrays, fixed calibration, and own
base positions computed from the two joint encoders. It never receives body
IDs, debris positions, birth flags, contact IDs, or physics collection counts.

Temporal and spatial depth medians suppress noise. Brown raised foreground
components are projected using calibration and tracked. The detector requires
at least 40 actual valid samples by default and at least 20 mm extent in each
horizontal direction. The size threshold is bounded to 20–100 samples for
fleet parameter learning. Actual corrupted close-camera samples showed real
pieces with 174–257 samples versus false floor islands with 6–17 samples.
The planner targets these tracks, freezes a visual estimate during lowering,
and uses bilateral tactile contacts for grasp confirmation. A failed contact
attempt causes a measured miss and a five-second retry cooldown.

Physics event generation and independent outcome verification can inspect
simulator state. Controller target selection cannot inspect debris ground
truth. Fleet learning evaluates fresh sensor-based runs in its own ledger;
this environment's development runs do not themselves demonstrate learning.

## Measurements and evidence

Telemetry includes actual thrust, torque, depth, RPM, cutter force vector,
collector pad forces and contact positions, measured TCP position, physical
cart load and travel, robot travel, tactile grasp events, observer tracks,
misses, and actual per-piece outcome data for display/verifier use.
`collector_contacts` vectors are force **on the pad**, with the sign converted
from MuJoCo's contact frame. The native/browser overlays use measured vectors.

The original three-piece linear-gripper prototype succeeded and its historical
receipt remains `dev/mining_fleet_summary.json`; that result does not represent
the current six-piece sensor-driven cart scene. The first current sensor run
completed all twelve layers but collected 0/6 pieces with seven misses because
the distant camera produced false floor tracks. Its preserved receipt is
`dev/fleet_sensor_baseline_summary.json` and its trace is
`dev/fleet_sensor_baseline_trace.json`. The closer-camera run achieved a real
grasp and cart release, then exposed a helper validation error on tiny positive
cart travel permitted by MuJoCo soft joint limits. The helper now accepts the
measured soft-limit tolerance without changing the nominal physical range.
Current corrected-cycle evidence is written under `dev/fleet_sensor_robust_*`.

Imported CMU mine meshes are rendered by `ui.cmu_mine`; they have no collision
masks and do not alter the work-cell dynamics. The floor is at z=.4 m. All
rendering/development cycles use GPU 0 exclusively when allocated by the root
agent. GPU 1 is never used.

## Completed sensor-driven cart cycle

The corrected seed-101 episode ran all 80 simulated seconds in 240.09 wall
seconds on GPU 0. Final receipt: `dev/fleet_sensor_row_summary.json`; 1 Hz
measured trace: `dev/fleet_sensor_row_trace.json`; corrupted RGB-D examples:
`dev/fleet_sensor_row_20.npz`, `_35.npz`, and `_65.npz`.

- `complete=true`; twelve layers removed; six distinct fragments collected.
- Cart counts `[2, 2, 2]`; each cart actually traveled 1.05053 m and retained
  both fragments; final measured floor loads are 3.1392 N per cart.
- Both machines parked. The 1 Hz trace first reports completion at 76 s.
- No failed pickup attempts. Six bilateral-contact grasp events refer to
  actual physical IDs in visible-track order `[5, 0, 2, 4, 3, 1]`.
- Peak drill thrust 83.2355 N; maximum penetration 89.0123 mm; peak sum of
  collector pad contact-force magnitudes 10.6682 N.

Discharge ports now form a horizontal row: x=.42,.28,.14,0,-.14,-.28 at y=.18.
This leaves physical clearance between the UR5e pickup posture and the parked
cart row. A broad discharge chute initially occludes fragments; the camera
acquires them after the drill returns, with no production-flag gating. Its
fixed search ROI is x[-.35,.62], y[.10,.30]. The earlier orthogonal layout's
real arm/cart collision is retained in `dev/fleet_sensor_cart_collision_trace.json`.
Arm planning data now copies only the collector's own base and arm encoders,
never whole-world joint state. Following the successful cycle, only displayed
phase wording changed from “bin” to “cart” and from “searching” to “returning”
after all release actions. Physics and perception were frozen for subsequent
fresh multi-seed fleet-learning evaluation. This single successful development
seed is not a multi-seed fleet validation claim.

`run_mining_fleet(..., frame_interval_s=.05)` separates presentation cadence
from the fixed 20 Hz sensor and 1 kHz physics/control cadence. Native capture
can request `frame_interval_s=1/60`; each callback receives the first actual
simulation state at or after its scheduled timestamp, with no frame duplication
or pose interpolation. The default remains 20 Hz. This presentation-only API
was added after the successful physical run for the subsequent final capture.
