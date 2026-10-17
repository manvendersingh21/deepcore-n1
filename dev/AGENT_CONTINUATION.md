# DeepCore-N1: current agent handoff

Updated 2026-10-17. Read this before older checkpoints in HANDOFF.md.
This checkpoint was written before resuming implementation at the user's request.

## Latest user scope — narrowed after this checkpoint

The user subsequently requested: "just make the robots look 3-d realistic ...
for now and have another agent give me the demo." Deliver robot appearance and
an actual fleet demo first. The broader requirements below remain backlog, not
requirements to delay this narrower delivery.

Active continuation: root improved browser robot PBR materials and beveled
housings; /root/native_finish added isolated native presentation casing details,
vents, bolts and wheel hubs, improved lighting/framing, and started an actual
80-second fleet capture on GPU0. /root/capture_finish owns final H.264 packaging
and inspection. /root/claude_review engaged the actual Claude CLI under HACP.
Recordings preserved before any server restart at dev/replays/{baseline,accepted}.json
with checksums/provenance in dev/replays/manifest.json.

Preview: dev/native_enclosed_smoke_v2.jpg. This is an appearance preview, not
proof of a completed episode. Full recording completion is still pending here.
Native preliminary warm frame timings were approximately 22.6 ms median and
24.2 ms p95 (not real-time 60 FPS). Browser source syntax checks passed.

## Original objective and completion criteria

Finish an enclosed industrial mine visualizer in browser and native MuJoCo:
faceted basalt and emissive crystalline core; complete articulated robot with
wrist spotlight/cone; actual RGB-D cloud/frustum, estimated pose brackets, issued
trajectory and measured force overlays; 1/2/3 camera/sensor keys and 0/N actual
baseline/accepted replay; verify performance and export submission_demo.mp4.
The fleet must visibly mine, collect six fragments, fill and dispatch three carts.
Run and document genuine fleet learning separately from the locked benchmark.
Do not call the project complete until the recording and verification exist.

## Verified versus pending

- **Verified fleet episode:** dev/fleet_sensor_row_summary.json and corresponding
  trace: seed 101, 80 simulated seconds, 12/12 layers, 6/6 fragments, zero missed
  pickups, cart counts [2,2,2], all departed approximately 1.05053 m, both robots
  parked. Peak drill 83.2355 N; collector 10.6682 N; depth 89.0123 mm.
- **Verified benchmark:** dev/final_verification.json, created 21:42 UTC, passed,
  STOP_DONE; fresh optimization 81.92 and holdout 85.56, six successful extractions,
  no shatter, core and ledger source hashes matched. Its fleet scope description
  predates the successful sensor fleet episode; preserve historical evidence.
- Accepted v4 is a **curated fallback**, not a successful Nemotron-authored
  promotion. Actual Nemotron v3 was rejected. The >88 score on every seed is unmet.
- Fresh browser accepted seed 101 completed: extraction true, two rubble cleared,
  no shatter, score 85.97, peak core 4.87 N, wrist 5.65 N.
- Fresh browser baseline seed 101 failed extraction **without shattering**.
  Never fabricate the requested v0 shatter effect; trigger only from actual events.
- **Unfinished:** final visual polish, native GPU smoke/performance measurement,
  steady 60 FPS proof, submission video, actual fleet learning trial, side-by-side
  comparison, and rerun of the newly hardened finalization workflow.
- Initial browser sampling was roughly 42 FPS including load; this is neither a
  steady benchmark nor proof of 60 FPS. No final video is verified.

## Safety, integrity, and scope

Never edit or relock core/scene.xml, core/env_runner.py, core/sensor_corruption.py,
core/verifier_harness.py or core/LOCK.json. Do not inspect holdout diagnostics
during candidate mutation. Keep historical ledgers and failed runs intact.
Never expose .env. Never invent sensor points, force readings, policy provenance,
future plans, learning outcomes or hardware readiness.

Fleet approximations: actuated planar bases, discrete contact-work rock removal,
contact-conditioned weld grasp. It is not validated for physical robots. Sensor
policy uses corrupted RGB-D tracks and proprioception; ground truth belongs only
in rendering/evaluation. New aesthetic rock geometry is presentation only and
must not silently alter the policy camera or physics.

## Code map and integration status

| Area | Files | Status / remaining work |
|---|---|---|
| Browser scene | ui/web/scene.js | Faceted rock/core, wrist light/cone, full mesh links, force arrows, interpolated actual poses, environment loader implemented. Old quarry still visible in core mode; gantry pillar occludes default camera. Fix composition and dark basalt palette. |
| Browser AR | ui/web/ar_telemetry.js | Actual cloud/frustum, estimated brackets/axes, current command segment, shatter ring. Curve represents only issued target, not a complete future approach/dump plan. |
| Sensor packets | ui/sensor_overlay.py | Corrupted RGB-D backprojection and segmentation-stage replay; 10 CPU tests reported passed. Never substitute ideal geometry. |
| Backend | ui/web_server.py | Actual scene/frame streaming, fleet learning endpoint, in-memory baseline/accepted recording cache. Final core callback omits latest sensor fields; preserve last valid packet for final-frame UI. |
| UI | ui/web/app.js, index.html, style.css | Scene-first layout, Details panel, 1/2/3/0/N keys. Fix replay policy/seed metadata, replay version overwritten by startReplay, initial operation selector sync. Side-by-side not implemented. |
| Enclosure | ui/mine_environment.py, assets/mine_environment/ | Deterministic synthetic faceted surround, steel arches, cables, lights. CMU survey assets remain separately attributed; do not call synthetic enclosure the measured scan. |
| Native presentation | ui/ar_overlays.py, drilling_hud.py, mining_hud.py, mine_visuals.py, live_hud.py | Agent implemented separate presentation model/state copying and AR; CPU checks only. Review, sync remote, GPU render and measure. |
| Fleet | ui/mining_fleet.py, ui/fleet_perception.py, ui/industrial_arm.py | Successful sensor episode; keep stable. Optional frame_interval_s supports 1/60 actual physics callbacks; sensors remain 20 Hz. |
| Learning | agents/fleet_learning.py, scripts/learn_fleet.py | Actual baseline/candidate evaluation and separate hash-checked ledger ready; no genuine trial yet. Optimization-only fleet scope. |
| Integrity | shared_brain/memory_store.py, agents/mission_contract.py, agents/contract_orchestrator.py, scripts/finalize_verification.py | Persistent mission anchors, DB/JSON comparison, campaign lock, atomic fail-closed reports, private source snapshots implemented and CPU tested. Existing mission_anchor() returned None; inspect explicit legacy migration before rerun, never rewrite history. |
| Capture | scripts/verify_visualizer.mjs, scripts/record_submission.mjs | Prepared, help only checked. Inspect and execute; recorder claims timestamp-preserving encoding and failure below target. Do not assume scripts prove performance until run. |

Accepted source: workspace/versions/v4.py and workspace/candidate_pipeline.py,
SHA-256 601d8825913e741e2069f49fd1036198001ecfa432e244ac34b5bb8f3d0af97b.

## Runtime and tools

- Workspace /Users/manubaba/Documents/deepcore-n1; local Python .venv/bin/python.
- Local server http://127.0.0.1:8080 was live and idle/complete at checkpoint.
  Started with DEEPCORE_GPU_HOST=cis-a6000 .venv/bin/python -u main.py run-web --port 8080.
- **Before restarting:** save /api/recording?version=baseline and version=accepted
  to files. Cached recordings are in process memory only.
- /api/state?compact=1 provides status/result without the giant scene payload.
- Remote: ssh cis-a6000, repo ~/deepcore-n1, Python
  ~/perception-loop-n1/.venv/bin/python. GPU0 ONLY; GPU1 belongs to another user.
- GPU environment: MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 CUDA_VISIBLE_DEVICES=0
  __EGL_VENDOR_LIBRARY_FILENAMES=$HOME/.local/share/glvnd/egl_vendor.d/10_nvidia.json.
- Native changes above need synchronization to remote; do not blindly overwrite
  newer remote evidence or databases. Serialize GPU jobs and ledger writers.
- Playwright /tmp/deepcore-web-deps/node_modules/playwright; Chrome
  /Applications/Google Chrome.app/Contents/MacOS/Google Chrome;
  ffmpeg /opt/homebrew/bin/ffmpeg.

## Agent state and delegation

User explicitly authorized internal and external parallel agents and smaller
agents for small tasks. Prior Claude/Cursor/AGY reviews are real and preserved in
dev/reviews/fleet_claude, fleet_cursor, fleet_agy. HACP contracts were settled.
native_drilling completed successfully and released GPU0. native_recording and
completion_audit ended with quota errors after writing their work; do not report
them as actively running. Check live agent status before assigning new ownership.
Separate file ownership: scene/browser polish; native rendering; capture/audit.
Avoid concurrent edits to shared files and concurrent GPU simulations.

## Resume order

1. Preserve cached actual baseline/accepted replay payloads before server restart.
2. Fix core viewport occlusion/old quarry and replay metadata/sensor persistence.
   Inspect actual screenshots of both core and final articulated fleet scenes.
3. Review and sync native presentation; short GPU smoke and measured frame timings.
4. Execute browser verification on cached replays, measure steady frame times,
   optimize and report actual achieved FPS. Implement missing comparison behavior.
5. Run genuine fleet learning with stable physics/perception source hashes:
   python -m scripts.learn_fleet. Three baseline plus three candidate runs may take
   approximately 24 minutes. Rejection is a valid outcome; never fabricate promotion.
6. Record actual native/browser sequences and submission_demo.mp4; inspect output,
   dimensions, duration, cadence, visual quality and provenance. 60 FPS export
   metadata alone does not establish 60 distinct rendered frames per second.
7. Rerun hardened final verification with explicit legacy mission handling;
   preserve old reports. Update README and this handoff with exact evidence paths,
   remaining limitations and final measured performance.

Latest visual inspection: dev/ar_baseline_live.png exposes the core camera
occlusion and missing final sensor PIP. Earlier preview images and demo_full.mp4
are historical, not final proof. Real robot transfer needs hardware calibration,
real drives/grasp models, local force/safety control and supervised bench trials.
