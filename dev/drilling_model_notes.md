# Drilling demonstration model

`ui/drilling_scene.py` is independent of the locked excavation benchmark. It
creates its own MuJoCo model and data; all geoms are MuJoCo primitives. The
drill feed is a downward slide joint and the spindle is a driven hinge joint.
The four carbide cutter spheres make physical frictional contact with the rock.
The helical flights, shaft, and motor casing are presentation geometry attached
to the physical joints. The flight capsules do not cut or collide.

The material model must be labeled **simulated breakable layers**. It is a
discrete removal proxy, not continuum fracture, a calibrated drilling prediction,
or a demonstration of evolved policy performance. Twelve 8 mm disks occupy the
bore inside a persistent rock block. A seeded work threshold of 3–4 J is assigned
to each disk. A layer accumulates work only while cutter contact carries at least
12 N of upward load and the measured spindle speed exceeds 1 rad/s. Work comes
from negative contact-force power at the moving cutter contact point, integrated
over physical simulation steps. When its threshold is reached, that disk's
collision masks and visibility are disabled. No simulation clock or authored
penetration curve removes material. The disks are removed rather than converted
to conserved chips; surrounding rock is visual geometry.

The contact force uses `mj_contactForce`, transformed from the contact frame into
world coordinates with the sign for the cutter body. MuJoCo's force convention
points from geom1 toward geom2; see the official
[contact documentation](https://mujoco.readthedocs.io/en/stable/computation/).
Axial load is the positive vertical component of the summed tool contact force.
The contact moment includes each force's lever arm about the spindle axis.
Published spindle torque is the actual velocity-actuator force with unit gear,
in N m; this includes acceleration and bearing drag even without rock contact.
The separate contact torque and axial contact load are zero when contacts vanish.
Feed actuator force includes gravity compensation and is reported separately.
Depth is current measured cutting-tip depth below the original rock top, clamped
at zero outside the rock; `max_depth_mm` preserves maximum measured depth after
retraction. The final disk disappears while the cutter is at that disk's top,
so maximum tip depth is about 88 mm although the twelve removed disks span 96 mm.
RPM comes from the spindle joint velocity. Contact readings are instantaneous,
without smoothing or substituted forces; brief contact losses can read zero even
while the controller remains in its drilling stage.

The feed controller commands actuator force to regulate downward speed, with
gravity compensation from MuJoCo's bias force. It never writes poses or joint
velocities. Approach speed decreases with the measured gap to the next layer.
Once contact engages that layer, feed stays engaged until removal. The contact
solver uses direct `solref="-100000 -40"` stiffness/damping and an elliptic friction
cone; these are demonstration model parameters, not measured rock properties.
A measured axial load above 120 N requests safety retraction. Removing
all disks requests ordinary retraction; otherwise the 9.5 second deadline starts
retraction. The cycle ends at 12 simulated seconds. `complete` requires all disks
removed and the tool returned to its clearance position; an incomplete or safety
stopped cycle remains an explicit unsuccessful result.

## Integration

`DrillingEpisode(seed=101)` exposes `model`, `data`, `ee_pos()`, `step()`,
`telemetry()`, `render_camera()`, `render_overview()`, and idempotent `close()`.
`run_drilling(seed=101, frame_callback=None)` returns final telemetry and invokes
`frame_callback(episode, telemetry)` initially and every 0.05 simulated seconds.
The consumer supplies playback pacing and may cancel by raising an exception.
Renderers are lazy, so model construction and physics require no GPU. The named
`overview` camera renders at 1100 × 640; `cam_overhead` renders at 640 × 480.

Telemetry includes the agreed time, phase, axial force, actuator torque, current
depth, RPM, removed/total layer counts, peak/limit force, and completion fields.
Additional fields expose contact torque, world force vector, contact position,
contact count, actual feed actuator force, cutting power, active-layer work and
threshold, cumulative cutting work, maximum depth, and stop reason. Browser
clients must update per-geom alpha/visibility after disk removal as well as poses;
hidden disks retain their MuJoCo geom IDs. The native renderer reads the same
model alpha and data directly.

## Observed physics run

The authorized CPU run on `cis-a6000`, using seed 101, produced 241 callbacks
(initial state plus 20 Hz for twelve simulated seconds). All 12 layers had been
removed in the 8.50 s frame; the 10.75 s frame reported completed retraction.
The final result was `complete: true`, with maximum measured tip depth 87.93 mm,
peak axial load 114.05 N below the 120 N limit, and accumulated cutting work
43.97 J. At callback sampling times, maximum motor torque was 0.962 N m and
maximum contact torque was 1.705 N m. Sampled contact began at 1.90 s. The final
unloaded state had zero axial and contact torque, zero current depth, and the
spindle had stopped. No rendering context or GPU work was used in this check.

An earlier configuration's initial contact exceeded the limit and actually
triggered safety retraction with zero layers removed and `complete: false`.
This exposed excessive friction damping in the original contact settings and
prompted the current contact and approach changes; that aborted result was not
counted as successful drilling.
