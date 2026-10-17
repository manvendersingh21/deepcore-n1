# Native drilling presentation

`ui.drilling_hud.DrillingFrameBuilder(seed)` is callable as
`builder(episode, telemetry)` and returns a 1600 x 900 BGR frame. Use it as a
context manager or call `close()` in a `finally` block. It owns an independent
MuJoCo renderer and model copy, including photographic texture uploads. Each
frame copies the simulation's current geom colors so removed layers disappear
at the actual physics event. It never changes the episode model or sensor
renderer.

The almost full-frame scene shows the real moving drill and layer state.
Additional quarry rocks are static presentation scenery. The tiny robot-camera
inset is the actual RGB image from `episode.render_camera()`. Four compact
readouts display axial contact thrust, measured spindle motor torque, drill
depth, and measured RPM. The force arrow projects the measured contact location
and contact-force vector through the actual presentation camera. Its length is
scaled for legibility and capped at 0.18 m. If only the minimal telemetry contract
is supplied, it shows the measured axial reaction at the bit along world +Z.

The title is **Simulated rock layers**. No core extraction scores, invented
learning events, material calibration claims, or continuum fracture claims are
introduced.

Native window:

```sh
python -m ui.drilling_hud --seed 101
```

`run(seed=101)` drives the actual simulation and returns its summary; Q, Escape,
or closing the window aborts cleanly. After normal completion the measured final
frame remains visible until the window is closed.

`capture(seed=101, jpeg_quality=94)` returns `(jpeg_frames, summary)` from the
simulation's 20 Hz callbacks. JPEGs are byte strings ready for `np.frombuffer`
and `cv2.imdecode`; `summary` is the unmodified result of `run_drilling`.
Capture does not add hold frames or change the source cadence. The recorder can
resample these frames and append a hold using its own video output cadence.

An actual seed-101 capture completed on cis-a6000 GPU 0, with the required EGL
vendor file at `$HOME/.local/share/glvnd/egl_vendor.d/10_nvidia.json`. GPU 1 was
not used. The 241 measured source frames are recorded at 20 fps in
`dev/demo_drilling_native_raw.mp4` (1600 x 900, 12.05 seconds, MPEG-4 Part 2).
The final source summary is in `dev/drilling_native_summary.json`.

Observed result: all 12 layers removed; complete at the end of the 12-second
cycle; maximum penetration 87.93 mm; peak axial contact force 114.05 N versus a
120 N force limit; integrated cutting work 43.97 J. Current depth and RPM return
to zero after the tool retracts and stops. The UI keeps the measured maximum
depth in the small depth caption so the completed result remains visible.

The final recording's contact (3.05 s), mid-cycle (4.20 s), and retracted
(12.00 s) frames are saved as `dev/drilling_native_contact.jpg`,
`dev/drilling_native_mid.jpg`, and `dev/drilling_native_final.jpg`. Contact and
final images were visually inspected. The revised view exposes the contact
and final bore, uses brighter lighting, and shows photographic quarry scenery.
The separate sensor image retains its simulation camera appearance.
