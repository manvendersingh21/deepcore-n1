# Current handoff — read first

The authoritative continuation checkpoint is [dev/AGENT_CONTINUATION.md](dev/AGENT_CONTINUATION.md),
updated 2026-10-17 before further implementation at the user's request. It records
the successful sensor-driven fleet episode, remaining visual/performance work,
agent quota interruptions, runtime details, integrity rules and ordered next steps.
The checkpoints below are historical and include superseded pending/failure states.

# Historical checkpoint — 2026-10-17

## Current user priority

Finish a realistic mine scene with two actuated robots, measured contact forces,
rail-cart loading/transport, and actual sensor-based learning in this same fleet.
The locked core-recovery benchmark is a separate verified track. User explicitly
authorized implementation, asset use and verification; continue without new
permission requests. Never invent success, poses, forces, scores or promotions.

## Verified evidence

- `dev/final_verification.json` completed **STOP_DONE / passed=true** at2026-10-17T21:42Z: fresh opt81.92 / holdout85.56, all six cores extracted, both rubble cleared, no shatter. Core hashes and all recorded source hashes matched; evidence/ledger unchanged.
- Accepted v4 SHA601d8825913e741e2069f49fd1036198001ecfa432e244ac34b5bb8f3d0af97b; historical accepted82.02/85.54 preserved. Prior accepted comparatorv1=30.59/32.31. >88everyseed remains unmet.
- Fresh real architect/CV/control review consumed latest saved guidancev2. Vision draft static-valid but UNVERIFIED; control reply rejected for JSON extra data. No draft promoted. Historical LLMv3 rejected; accepted improvements are curated fallbacks.
- Production bounded memory, transaction retention, prompt consumption and frozen mission wiring implemented; latest focused CPU suite53passed. External Claude findings remain under remediation by completion_audit; do not claim final hardening finished.

## Fleet and assets: honest current state

- Native/browser use the downloaded CMU survey cutaway (source units unknown, uniform display normalization). Actual six-joint UR5e is actuator-driven with per-link meshes; rail-cart and force/load UI/backend wiring is in place.
- First actual sensor/cart baseline FAILED: six floor false tracks appeared before debris creation. native_drilling is changing camera placement and adding temporal/noise-robust perception. No verified sensor-based collection success yet. Earlier privileged/scripted results do not validate this new policy.
- `ui/fleet_perception.py` observer and `agents/fleet_learning.py` learning mechanism exist. `scripts/learn_fleet.py` exposes `run_learning(event_fn=...,use_llm=True)` and `load_active_parameters()`. Fresh baseline/candidate evaluation, source/parameter hashes and a SEPARATE fleet ledger are prepared; no real fleet learning trial/promotion has run yet. Wait for stable physics/perception before learning.
- User local`dev/robotic_arm.glb` is a different static asset (Stefan_Spasevski_Art),283nodes/138meshes,no skins/animations/joint rig. Source-preserving OBJ/material conversion and preview are underassets/robot_reference/user_arm; no guessed UR5 link mapping. Working UR5 remains.
- New final demo is NOT recorded. Keep old artifacts, but do not label old baseline or privileged demos as final autonomous fleet proof.

## Active ownership / compute

- native_drilling owns fleet/camera/observer and actual sensor/cart physics debugging; exclusiveGPU0 until release. No parallel rendering jobs. GPU1 belongs to someone else.
- Root owns browser/backend/UI wiring, cart load/force display, actual fresh verification/proposal workflow and final recording coordination.
- completion_audit is remediating external Claude findings and reviewing final integrity. native_recording most recently owns learnerCLI/fleetlearning tests and current documentation; observer ownership is native_drilling.
- Remotehostcis-a6000, repo~/deepcore-n1, interpreter~/perception-loop-n1/.venv/bin/python. EGL GPU0 environment as in historical compute notes; never change/relock core.
- Current local server port62001 has an old preview; restart after syncing stable files. Final native/browser recording pending.

## Next actions

Resolve real false-floor detections using actual corrupted sensor captures; verify
pickup, cart loading and transport with measured contacts/outcomes. Then run a
genuine separate fleet baseline/candidate learning trial, preserve failures,
complete external-review hardening, refresh the browser/native scene and record
the final demo. Keep benchmark STOP_DONE evidence distinct from fleet readiness.

# Historical checkpoint and handoff (superseded; preserved verbatim)

# Active checkpoint — 2026-10-17 (latest; overrides historical notes below)

## Current user priority
Implement the downloaded CMU 3D mine with TWO moving machines: one drills/mines, the other collects physical debris. Both browser and native MuJoCo views must look clean and show measured loads. Latest request explicitly authorized download and implementation; do not ask again.

## Completed
- CMU model and 3 animations downloaded into `assets/cmu_mine_reference/`, 132,758,969 bytes total. Source URLs, SHA256, sizes, unspecified license in manifest/README. VRML1 model has 382,680 vertices / 743,918 triangles. One original AVI has malformed audio metadata; identical retry confirms source issue.
- Independent drilling `ui/drilling_scene.py`: actual contact-work removal, 12/12 layers, max depth87.93mm, peak thrust114.05N; videos `dev/demo_drilling_browser.mp4`, `dev/demo_drilling_native_raw.mp4`.
- Evolution FINISHED. Old PID2418796 is terminal (never resume). New bounded model call path works (Nemotron enable_thinking=False). v3 actual LLM candidate rejected for regression; v4 fallback forceguard accepted, STOP_DONE. opt82.02 / holdout85.54, allsix extraction,2rubblecleared,no shatter. This is accepted curated fallback, not successful LLM-authored improvement. Local shared_brain and workspace synced from remote after writer finished. >88 everyseed remains unmet.
- Browser actualscene/poses, HDR and bloom, measuredforcearrow; new fleet controls/wiring in progress.
- Locked core unchanged, prior local+remote hashchecks matched; never relock.

## Active work / ownership
- `/root/native_drilling`: NEW `ui/mining_fleet.py`, `dev/mining_fleet_notes.md`. Two actuated mobilebases, physical freebody debris from contact-work milestones, contact-conditioned grasp, bin placement. GPU0 allocated exclusively to this agent until it releases. Marker geom/body `drill_mobile_base`, `collector_mobile_base`. floor.4,workaxis(.42,0),bin(1.05,.35),~35s.
- `/root/native_recording`: CMU importer `scripts/import_cmu_mine.py`, `ui/web/cmu_mine.js`, `ui/cmu_mine.py`, derivedassets and integrationnotes. Source-derived cutaway tunnel, uniformscale+rigidtransform documented (source unitsunknown). Browser `loadCMUMine(parent)->{group,metadata,dispose}`. Native `append_mine_assets(xml)->xml`, noncolliding cmu_* geoms.
- `/root/completion_audit`: downloads finished; now adding fleet operation to browser recorder and auditing accepted ledger metrics read-only.
- Root owns web_server/app/index/style/scene wiring, native `ui/mining_hud.py` + fleetcamera adjustment in drilling_hud, CLIrun-mine. Need actualrun+capture after integration. No formaltests requested/run.

## Compute / run
- `ssh cis-a6000`, GPU0 ONLY. GPU1 belongs to someoneelse.
- remote repo `~/deepcore-n1`; Python `~/perception-loop-n1/.venv/bin/python`.
- env `MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 CUDA_VISIBLE_DEVICES=0 __EGL_VENDOR_LIBRARY_FILENAMES=$HOME/.local/share/glvnd/egl_vendor.d/10_nvidia.json`.
- localserver http://127.0.0.1:8080, execsession18976, loaded pre-fleetbackend; needs graceful restart after syncingnewfiles; no liveworker currently.
- Native record command when ready: `main.py run-mine --out dev/demo_mining_fleet_raw.mp4 --seed101` (use normal spaced CLI flags).
- Final60sbaseline/learning/acceptedrecovery `scripts/record_demo.py` ready but not recorded; update maybeinclude fleet instead of isolateddrill once done. Old `dev/demo_full.mp4` is baselinefailure. Do not call it finaldemo.

## Outstanding
FinishCMU+fleet integration, inspect/captureactualbrowser+nativeepisodes, truthfulcount/force/UI, refresh README and thischeckpoint. Then final60srecording and acceptedbrowserrecovery; >88all6target stillopen. No fabricated ledger,force,poses or success. Holdouts neverautopsied. Use smallcontext agents forsmalltasks.

# Historical handoff (older snapshots; superseded above)

# DeepCore-N1 — handoff

Read the current status first. Historical measurements below are preserved from the earlier
2026-10-17 ~13:20 local snapshot and are not the latest results. Everything here is measured on
the real MuJoCo physics unless marked otherwise.

## Current status (2026-10-17)

- Adopted `agents/patches/evolved_pipeline.py` from `controller_yawnear.py`, SHA-256
  `6ea1b6cd95255908782968d029e862d59a3265985d289e01319a6a732593075a`. The six-seed gate was
  met: all runs were safe and holdout seeds 101–103 extracted. Seed 33 was safe but did not
  extract; the >88 target on every seed remains unmet. Local scores: `dev/scratch/controller_yawnear_scores.jsonl`.
- Current `core/LOCK.json` hashes pass locally and on the remote host. Do not relock or change
  `core/`.
- Native presentation proof: `dev/native_hud_preview.jpg`. The browser preview
  `dev/demo_control_room.mp4` is an experimental single-seed comparison of the old candidate on
  holdout 101 (baseline 14.38, candidate 86.15); it is not an accepted result.
- The current orchestration run is **RUNNING**. Its outcome is pending; do not report it complete
  or accepted until the main agent supplies the result.
- Both the browser control room and native MuJoCo HUD are supported. The browser visualizes the
  real scene and streamed poses; native `ui/mine_visuals.py` additions are render-only dressing.

## 1. What the project is
RGB-D excavation controller. The arm must extract a fragile cyan core from under two rubble
boulders into a hopper without shattering it (wrist-force contact gate > 18 N = shattered).
Score = `core/verifier_harness.py` composite (locked math, do not change).

## 2. Where things live
- Physics (LOCKED, do not edit): `core/scene.xml`, `core/env_runner.py`, `core/sensor_corruption.py`,
  `core/verifier_harness.py`. Hashes pinned in `core/LOCK.json` (NOT YET WRITTEN for the current
  physics fix — see section 6).
- Controller under development: `agents/patches/evolved_pipeline.py` (the template).
  `agents/fallback_library.py` materialises v1/v2/v3 from it (FEATURES dict per version).
- Scratch experiments: `dev/scratch/` (evolved_pipeline_*.py variants, diag_seed*.py).
- HUD: `ui/live_hud.py` (honest states: AWAITING RUN / NOT STARTED / IN PROGRESS / CORE SHATTERED /
  NOT EXTRACTED / CORE EXTRACTED). Tests: `tests/test_ui_unit.py` (7 pass).
- Recorder: `scripts/record_demo.py`. Assets (CC0): `assets/` with LICENSES.md.
- Verification log: `dev/verify_report.md`.
- Delegate prompts: `dev/prompts/`.

## 3. Compute
- GPU host: `cis-a6000` (ssh). Use GPU0 only: `export MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0
  CUDA_VISIBLE_DEVICES=0 __EGL_VENDOR_LIBRARY_FILENAMES=$HOME/.local/share/glvnd/egl_vendor.d/10_nvidia.json`
  Interpreter: `~/perception-loop-n1/.venv/bin/python`. Repo copy: `~/deepcore-n1`.
- GPU1 is a different user's job. Do not touch.
- Scoring pattern (one fresh process per seed; verify_integrity patched out for scoring only):
  `import core.integrity as ci; ci.verify_integrity=lambda: {}` then
  `vh.evaluate(path, (seed,))["per_seed"][0]` and print composite, shattered, extracted, rubble, peak N.
- Mac-air / HACP: OUT OF SCOPE per user. Do not use.

## 4. Current measured results (template as of md5 0cd40225 + core_grasp fix)
| seed | composite | shattered | extracted | rubble | peak core N |
|---|---|---|---|---|---|
| 11  | 35.7 | YES | no  | 1 | 18.03 |
| 22  | 80.5 | no  | yes | 2 | 14.93 |
| 33  | 28.7 | YES | no  | 1 | 18.92 |
| 101 | 85.8 | no  | yes | 2 | 4.54 |
| 102 | 86.9 | no  | yes | 2 | 11.34 |
| 103 | 82.2 | no  | yes | 2 | 12.26 |

Gate: no seed may shatter; holdout 101-103 must extract. Spec target: >88 on all seeds.

## 5. Root causes (diagnosed)
- Time budget: the rubble cycles use up the 11 s episode. Fixed by a 6.5 s cutoff that sends the
  controller straight to the core (in template under `core_grasp`).
- Seed 11: right finger pad lands on rubble_a (sits on top of core at ~(0.387,-0.025,0.463)) while
  grasping rubble_b; the load goes through rubble_a into the core. Reordering stacked-first fixed 11
  but broke 101/103/33 (rejected).
- Seed 33: re-grasps a boulder already >12 cm from core, wasting time; rubble_a loses its slot.
  "Skip-cleared" fix (scratch: `dev/scratch/evolved_pipeline_skipcleared.py`) gives 33=44.7 safe,
  101-103 hold, but seed 11 still shatters (35.8) — NOT adopted yet.
- Release-timing hypothesis: TESTED AND REJECTED (no change in composite). See verify_report.md.

## 6. What is NOT done
- No accepted version beyond v0 (ledger has none), so the recorded demo replays v0 (a failure).
  Demo `dev/demo_full.mp4` (60 s, real physics) is an honest v0 baseline failure, NOT a win.
- `core/LOCK.json` exists but was written against the pre-fix physics. Re-lock only when the
  controller passes all six seeds, and re-check hashes on both machines first.
- Track 2 (shared brain token budget, efficiency gain) and Track 3 (evolved prompts, rejected
  attempt, holdout before/after table) are not complete.
- Spec target >88 on all six seeds is NOT met.

## 7. Next actions (in order)
1. Get a controller change that makes seed 11 not shatter while keeping 22, 33, 101-103 passing.
   Opus diagnosis agent is working on this (dev/scratch on cis). Verify any candidate on all six.
2. Only if all six pass: copy the winning change into agents/patches/evolved_pipeline.py, check
   md5 on both machines, regenerate v1-v3 via agents/fallback_library.py, re-score v0-v3 on holdout.
3. Accept versions into the ledger through the orchestrator (not by hand), then record the real
   evolution demo with scripts/record_demo.py.
4. Re-lock core/LOCK.json only after the final physics/hashes are confirmed.

## 8. Rules that must hold
- Do not change core/ physics, the verifier, the scoring math, or any threshold (18 N, 28 N, 25 ms,
  9 cm, 12 cm scorer distances).
- No faked demos, scores, or state. Every frame and number comes from run_episode.
- One change at a time; score all six seeds before adopting anything.

## 9. Latest measurements (appended after the initial handoff)

### Rim-first rubble ordering (dev/scratch/evolved_pipeline_seed11.py) — REJECTED
Sorts a boulder within 5 cm of the core first. Six-seed result, GPU0:
| seed | composite | shattered | extracted | rubble | peak N |
|---|---|---|---|---|---|
| 11 | 35.59 | YES | no | 1 | 18.03 |
| 22 | 80.46 | no | yes | 2 | 14.93 |
| 33 | 28.72 | YES | no | 1 | 18.92 |
| 101 | 85.80 | no | yes | 2 | 4.54 |
| 102 | 86.80 | no | yes | 2 | 11.34 |
| 103 | 82.31 | no | yes | 2 | 12.26 |
No change on seed 11. Do not adopt.

### Skip-cleared (dev/scratch/evolved_pipeline_skipcleared.py) — PARTIAL
Six-seed: 11=35.8 SHATTERED, 22=80.8, 33=44.7 safe, 101=86.1, 102=87.0, 103=82.6.
Fixes seed 33, not seed 11. Not adopted; candidate for combining with a seed-11 fix.

### Opus seed-11 diagnosis agent — STALLED
Launched on opus; it made no candidate and its last log entry was an unanswered request.
Nothing from it is adopted. Treat its output as absent.

## 10. Open problem, in one line
Seed 11: right pad lands on rubble_a (on top of core) while grasping rubble_b. Neither
ordering change (stacked-first or rim-first) fixes it on its own, and the time/order rules
that fix 33 do not touch it. Next candidate should target the grasp itself, not the order:
e.g. stop descending on rubble_b when the pad's contact is with rubble_a, or lift the wrist
before closing when another boulder is within the pad footprint. Verify on all six seeds.

## 11. Next agent: start here
1. Read sections 4, 5, 9, 10.
2. Build one candidate on a copy in dev/scratch/ (never the template first).
3. Score all six seeds (section 3 pattern). Adopt only if no seed shatters and 101-103 extract.
4. Then follow section 7 steps 2-4.
