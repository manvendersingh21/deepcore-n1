# Visualizer capture review (peer b, read-only)

Scope: `scripts/verify_visualizer.mjs`, `scripts/record_submission.mjs`, `ui/web/app.js`.
I only read the code. Nothing was run: no browser, no recorder, no GPU. Live server check: `/api/state?compact=1` returned status complete, run_id 3, seed 101, pipeline accepted.

## Findings

### B1. BLOCKER: `startReplay` deletes the replay version, so both scripts time out
- `app.js:50` sets `window.__deepcoreReplay.version=version`. Then `app.js:51` calls `startReplay()`, and `app.js:92` replaces the whole object with `{active,frame,total}`, which has no version.
- `record_submission.mjs:151` and `verify_visualizer.mjs:100` wait for `__deepcoreReplay.version === v`. That never becomes true, so both fail after 15 s.
- In verify, the same problem shows up again later. Scrubbing (`:137`) and clicking replay (`:139`) call `startReplay` again, so `verify_visualizer.mjs:150` would throw "failed to render" even after the wait was fixed.
- Fix: `window.__deepcoreReplay={...window.__deepcoreReplay,active:true,frame:replayIndex,total:replayFrames.length};`. Also add `run_id`/`seed` there so the scripts can check the replay against the preflight recording.

### B2. HIGH: the 200 ms poll overwrites what the replay says about its source
- `app.js:250`: `poll()` rewrites `#integrity` every 200 ms. Within one tick it replaces `Recorded seed … · policy_source` from `app.js:49`, so the video never shows where the replay came from.
- `app.js:122`: `#phase-detail` during replay uses the live `state.seed`/`state.pipeline`. The server is currently `pipeline: accepted`, so the **v0 baseline replay is labelled "Accepted controller"**. That label is false on screen.
- `app.js:200`: `insight-source` has the same problem: `SEED ${state.seed} · ${state.pipeline}` is shown on the replay result.
- Fix: keep a `replayRecording` object (seed, policy_source, version, run_id). While a replay is loaded (`replayRecording && run-mode==='MEASURED REPLAY'`), have poll skip `#integrity`, and have `renderFrame`/`renderResult` take seed and pipeline from the recording instead of `state`.

### B3. HIGH: the `>= 60` average check fails on a real 60 Hz display, and averages hide stalls
- `record_submission.mjs:187,200` and `verify_visualizer.mjs:172` compare `measured_fps` against 60. `captureStream(60)` caps at 60, and rAF on a 60 Hz panel averages about 59.9–60.0, so a perfect run will usually fail. A run with hitches can still pass on the average. p95, p99 and `over_20ms` are recorded but never checked.
- Fix: pass only if `measured_fps >= 0.98*target` **and** p95 interval ≤ about 1.25×(1000/target) ms **and** the `over_20ms` share is small (for example under 1%). Write the thresholds into the evidence file.

### B4. HIGH: the recording measures encoded frames but presents them as the render rate
- `record_submission.mjs:58,186-205`: the pass/fail result comes only from WebM/MP4 frame timestamps. That shows how many frames were encoded. It does not show that the renderer drew 60 distinct frames per second (AGENT_CONTINUATION:113-114). MediaRecorder can drop frames under VP9 load at 1600x900. Also, `captureStream` emits a frame on every canvas draw, including draws that didn't change the scene.
- Fix: in the page, collect every renderer frame interval from start to stop (`__deepcoreRenderStats` only keeps the last 600). Report renderer cadence and encoded cadence separately, and pass only when both meet the target. Record the WebGL renderer name (`WEBGL_debug_renderer_info` UNMASKED_RENDERER) and the headless/headed mode. Fail or flag the run if it rendered on SwiftShader, because headless Chrome often falls back to software GL.
- Smaller issue (`:70-75`): `frame_count` includes frames with duplicate timestamps while `deltas` drops them. Use only unique timestamps when computing fps.

### B5. MEDIUM: the recorder can start while the server is busy, and a 0/N key press can then do nothing
- `app.js:29`: `switchVersion` returns silently if `busy`. The preflight (`record_submission.mjs:92-108`, `verify_visualizer.mjs:63-87`) never checks `/api/state` status. If a run is in progress, the key press does nothing and the script just times out. If the run_id changes during a replay, `app.js:252` also stops the replay.
- Fix: before starting, require `/api/state?compact=1` status to be `complete` or `ready`, and check again after the capture.

### B6. MEDIUM: nothing ties the replay the UI played to the replay the script checked
- The UI fetches its own copy of `/api/recording` (`app.js:33`), separate from the preflight copy. The evidence only stores run_id/seed and not a hash of the payload.
- Fix: store the sha256 of each preflight recording body in evidence. After each key press, check that `__deepcoreReplay.run_id/seed/total` match it, which needs B1's extra fields.

### B7. LOW
- `verify_visualizer.mjs:164`: the check that keys 1 and 2 produce different views compares screenshot hashes. AR animation (`ar.tick`) can change the image on its own, so this check is weak. Compare the camera pose or mode instead.
- `verify_visualizer.mjs:171`: `camera_views_differ = true` is hardcoded. Set it from the actual check.
- `app.js:42`: `sceneRun=state.run_id` throws if a key is pressed before the first poll arrives. The scripts wait long enough that this doesn't hit them, but the UI is fragile.
- `record_submission.mjs:179`: any console error, such as an HDR load warning logged as an error, fails the run only after the full roughly 2×11 s capture. That's acceptable, but listing the error text in the evidence first would make the cause clearer.
- `record_submission.mjs:194-195`: assigning `mp4_bytes` twice is odd but works. A simple `await stat()` would be cleaner.

## OK as written
- Blocking POST in both scripts prevents any computation from starting. The preflight requires actual RGB-D, estimates and target frames. MP4 conversion uses `-fps_mode passthrough` and checks that the frame count is unchanged. A failing capture is kept and marked failed. Replay speed defaults to 1×.

## Priority for the parent's fixes
B1 first: nothing runs without it. Then B2 (the false baseline label), then B3/B4 before claiming 60 FPS.
