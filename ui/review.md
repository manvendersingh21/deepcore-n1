# Review: DeepCore browser control room (peer b)

Scope: `web_server.py`, `web/app.js`, `web/scene.js`, `web/index.html`, `web/style.css`.
Method: read-only source review against `core/env_runner.py`, `core/verifier_harness.py`,
`agents/contract_orchestrator.py`; syntax checks; GET `/` and `/api/state` on the running server.

## Verdict: acceptable. No fake data found. Minor issues listed below.

## Real-data wiring (verified)
- Run: `run_episode(load_pipeline_class(path), seed, frame_callback=capture)`. The callback signature
  `(ep, rgb, depth, out, rec)` matches. Poses come from `d.geom_xpos/geom_xmat`, force from
  `ep.core_force_now` and `wrist_force_world()`, peak from `rec.core_gate_force`, p99 latency from
  `rec.latency_ms`. The camera image is the controller's real corrupted RGB from `observe()`.
- Result: `score_record(rec)` with `_`-prefixed keys removed, which is the same rule `_public()` uses.
- Learning loop: `ContractOrchestrator(..., event_fn=...)` events are streamed as they are emitted;
  the ledger comes from `brain.load_ledger()`. Nothing is synthesized.
- `verify_integrity()` runs before any worker. Seeds are whitelisted to the 6 evaluation seeds,
  and controller names are whitelisted. POST requests check the origin. SSH host is set at launch only.
- The UI shows "—" or "Awaiting" when it has no data. The 3D site is built from the scene geoms the
  server streams. Its only procedural content is seeded scree outside the workspace, which is
  scenery and not data.
- UI constants match core: 2200 steps / VISION_EVERY 10 = 220 frames (scrubber max 219),
  SIM_DT 0.005 = 200 Hz, 11.0 s, FRACTURE_N 18 N, force contract 28 N, latency 25 ms.
- `python -m py_compile`-equivalent parse passes. `node --check` passes for app.js and scene.js.
  `/` returns 200. `/api/state` returned ready, compute cis-a6000, integrity verified, 117 geoms.

## Issues (non-blocking)
1. Holdout exposure: the default seed is 101 (holdout), and `score_record` output for holdout seeds
   includes `pipeline_errors`, which `evaluate()` strips for holdout. app.js shows them in a notice.
   Peer a said they would strip these. Also recommend defaulting to seed 11.
2. The ledger only loads after an evolve run finishes. On server start the Learning tab shows empty
   even when `shared_brain` already has entries. Recommend loading `SharedBrain.load_ledger()` at startup.
3. `/api/state` is polled every 200 ms and returns the whole state, including ~117 scene geoms, each
   time. `/api/replay` returns every frame with a base64 JPEG. This works locally but is heavy over SSH.
   Recommend dropping `scene` from the polled state after the first load.
4. Agent-card mapping falls back to "verifier" for any agent name it does not recognize. The current
   names (Root Architect / CV Specialist / Excavation Control Specialist / Independent Verifier) map correctly.
5. `phaseNames` lacks `SURVEY` and `TRANSPORT`, which the evolved pipeline uses. The fallback
   title-cases them, so this is cosmetic only.
6. The "RECOVERY TRAY" label position is hard-coded and does not come from the hopper geom.
   Cosmetic only.
