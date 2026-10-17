# DeepCore-N1 — Frozen Interfaces (read before touching anything)

## Ownership
- `core/` is LOCKED after Phase 1 (SHA-256 in `core/LOCK.json`, `core.integrity.verify_integrity()` runs before every evaluation). Never edit it.
- Recursive agents may only READ `shared_brain/` and WRITE `workspace/candidate_pipeline.py`.
- Delegates: only edit the files you are assigned. Import from `core` only through the public API below.

## World (core/scene.xml + core/env_runner.py)
- Physics 200 Hz (`dt=0.005`). Episode = `EPISODE_STEPS=1400` `mj_step` calls (7.0 s). Before t=0 the world is pre-settled (not counted).
- Vision + `pipeline.step()` at 20 Hz (every `VISION_EVERY=10` physics steps) → 140 calls/episode; action held in between.
- Pit floor z=0.400. Objects (`OBJECTS = ("core", "rubble_a", "rubble_b")`):
  - `core` cyan box half-size 0.022³ (mass 0.06), centre z≈0.422, spawn (0.42±0.045, 0±0.045), yaw ±30°.
  - `rubble_a` ochre box 0.036×0.024×0.020 (0.11 kg) resting on top-left of the core (centre z≈0.465).
  - `rubble_b` magenta box 0.032×0.024×0.020 (0.10 kg) leaning on the core's right edge (centre z≈0.455).
- Zones (`env_runner.DUMP_ZONE`, `env_runner.HOPPER_ZONE`): rubble dump (0.30, -0.16, 0.42); extraction hopper (0.52, +0.16, 0.42).
- Cameras: `cam_overhead` (0.42, 0, 1.05) straight down, fovy 45°, 640×480. `cam_perspective` 3rd-person for the HUD.

## Candidate contract (`workspace/candidate_pipeline.py`)
```python
class ExcavationPipeline:
    def __init__(self, cam_intrinsics: np.ndarray, cam_pos: np.ndarray, cam_rot: np.ndarray): ...
    def step(self, t_sec: float, rgb: np.ndarray, depth: np.ndarray, proprio: dict) -> dict: ...
```
- `cam_intrinsics` 3×3 K (fx=fy≈579.4, cx=320, cy=240). `cam_pos` (3,), `cam_rot` 3×3 OpenCV convention (X right, Y down, Z forward):
  `p_world = cam_pos + cam_rot @ [(u-cx)/fx*d, (v-cy)/fy*d, d]`, `d = depth[v,u]` metres (z-depth).
- `rgb` uint8 (480,640,3) corrupted; `depth` float32 (480,640), with noise σ 3 mm, ~10% NaN edge dropouts, 3% spikes (0.25/1.80 m), a dust-plume band during frames 40..70.
- `proprio`: `{"ee_pos": (3,), "ee_yaw": float, "gripper": float (0 closed..1 open, measured opening), "wrist_force": (3,) N world frame, "wrist_torque": (3,)}`
- Return:
```python
{"estimated_objects": {"core"|"rubble_a"|"rubble_b": {"pos": np.ndarray(3,), "yaw": float}},
 "action": {"target_pos": np.ndarray(3,), "target_yaw": float, "gripper": float},   # gripper 1.0 = 90 mm open, 0.0 closed
 "debug_overlay": {"bboxes": [(name, x, y, w, h, yaw_deg), ...], "phase": str}}
```
  `bboxes` = rotated rect in pixels (centre x,y, width, height, angle) like `cv2.minAreaRect`.
- `target_pos` = grip centre (between finger pads), clipped to the workspace; velocity/accel-limited setpoint + Jacobian IK.
- Forbidden in candidates (AST guard `core.verifier_harness.static_check`): mujoco/core/os/sys/subprocess/io/… imports, open/eval/exec/getattr, dunder escapes, file IO.

## Verifier public API (`core/verifier_harness.py`)
- `OPTIMIZATION_SEEDS = (11, 22, 33)`, `HOLDOUT_SEEDS = (101, 102, 103)`, `FRACTURE_N = 18.0`, `LATENCY_CONTRACT_MS = 25.0`.
- `evaluate(pipeline_path, seeds, diagnostics=True) -> dict`:
```python
{"aggregate": {"composite_score", "vision_3d_rmse_mm", "rubble_cleared_count", "peak_core_contact_force_n",
               "core_shattered", "core_extracted", "extraction_rate", "vision_latency_p99_ms", "fps"},
 "per_seed": [{"seed", <same metric keys>, "score_breakdown": {...}}],
 "autopsy": {seed: {...}}}        # ONLY for optimization seeds; {} for holdout
```
  Aggregate: means over seeds; `core_shattered` = any; `core_extracted` = all; force/latency = max.
- `check_contract(opt, holdout, best_opt, best_holdout) -> (ok: bool, reasons: list[str])` (Law 5).
- `static_check(source) -> list[str]`, `load_pipeline_class(path)`.
- CLI: `python -m core.verifier_harness workspace/candidate_pipeline.py --seeds 11 22 33`.

## Env public API (`core/env_runner.py`) — HUD / demo only
- `run_episode(pipeline_cls, seed, frame_callback=None, occluder_source=None) -> EpisodeRecord`
- `frame_callback(ep, rgb, depth, out, rec)` called after each vision step. `ep.render_perspective()` → RGB (480,640,3);
  `ep.ground_truth()` → `{name: (pos, yaw)}`; `ep.core_force_now` (float, N, gated core-contact force this step);
  `ep.wrist_force_world()` (3,); `rec.force_t / rec.force_vec / rec.phase / rec.latency_ms` lists.
- `occluder_source() -> (mask_bool HxW | None, rgb HxWx3 | None)` (live webcam hand).
- `core.sensor_corruption.WebcamOccluder(device=0)` with `.read()` and `.close()`.

## Ledger / shared brain
- `shared_brain/ledger.json`: JSON list of
  `{"version": int, "source": "baseline" | "llm:<model>" | "fallback_patch:<name>", "contract_id": str,
    "opt": <aggregate>, "holdout": <aggregate>, "accepted": bool, "reasons": [str], "timestamp": iso, "sha256": str,
    "path": "workspace/versions/v<N>.py"}`
- Accepted versions are archived at `workspace/versions/v<N>.py`. HUD/demo read numbers ONLY from ledger.json.
- `shared_brain/playbook.md` stays under 1,500 tokens (token estimate = `ceil(len(text)/4)`), 4 fixed sections.
