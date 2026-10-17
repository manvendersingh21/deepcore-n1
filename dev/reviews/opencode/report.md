# opencode review — controller_yawlock.py & evolved_pipeline.py (peer b)

Scope: static review of `dev/scratch/controller_yawlock.py` and `agents/patches/evolved_pipeline.py`, plus
measured data in `dev/scratch/controller_yawlock_scores.jsonl` and `dev/scratch/controller_*seed33*.json`.
No GPU simulation was run; no controller files were edited. Scoring semantics read from `core/verifier_harness.py`.

## 1. Why seed33 scores lower (72.01 vs 87–90 on other seeds)

Seed33 keeps full clearance (25.0) and extraction (30.0); the entire deficit is:

- **vision 13.39** (RMSE 22.15 mm; seed102: 10.42 mm → ~13.4 pt gap)
- **safety 0.54** (peak gated core contact 17.46 N vs ~5 N on seeds 11/101/102 → ~9.5 pt gap)

### 1a. Vision deficit is perception-level, not control-level

`controller_yawlock.py:79` (`_pick_blob`):

```python
score = -float(np.abs(col[0] - centre[0])) - 0.02 * float(area)
```

Every cyan pixel is charged ~2 % of the blob's area against hue purity. The core's RGB-Yaw metrics
(22.15 mm RMSE, 13.39 vision) are **bit-identical across yawlock (72.01), yawnear (35.25), yawsoft (29.08)
and relief (45.49)** — control mutations change nothing — so the error is fixed at detection time.
Mechanism: the initial cyan mask at seed33 is fragmented (shadow/lean geometry minAreaRect on a partial
blob); the area penalty then rewards an interior fragment over the complete core, and `_measure`
(`controller_yawlock.py:100-104`) takes the **median** of the back-projected mask pixels. The median of a
partial mask's pixels is displaced toward the visible side → a persistent xy bias that survives the IQR
depth filter. Since vision RMSE is only scored over the first 40 vision frames (`verifier_harness.py:28`,
"before contact moves the objects"), this bias alone sets vision.

**Minimal fix (generic, geometry-based, not per-seed):** for `name == "core"` no dump-marker confounder
exists (the dump marker is ochre; only rubble_a competes), so select the **largest** valid connected
component for the core (or equivalently: restrict `_pick_blob` to rubble). Keep the existing hue-based
selection for rubble_a/rubble_b. Optional hardening of `_measure`: replace the median xy of the mask
pixels with the `cv2.moments` centroid back-projected at the median depth, which stays unbiased when only
part of the core is visible. Both changes are layout-agnostic and apply to every seed.

Expected effect: seed33 RMSE ~22 → ~14–16 mm direction (seed102/seed-median levels), vision ≈ 17–19,
seed33 composite ≈ 78–81.

### 1b. Safety deficit: 17.46 N spike at initial closure on rubble_a

`dev/controller_progress.md` ("Opt33 diagnostic", "Force-control followups") localises the peak to the
first closure on rubble_a, and the previous full sweeps (yawnear seed22 15.38 N, yawlock seed22 15.38 N,
seed33 11.7–17.5 N in every variant) show this is a **two-tick target-staleness artefact shared by all
committable candidates**, worsened when the rubble estimate is biased inward:

- `DESCEND` computes the creep target `max(gz, ee[2] - 0.012)` (`controller_yawlock.py:261`) **before**
  the `fz > TOUCH_N` transition to GRASP (`:266`) — so the tick that detects contact still commands
  another 12 mm of downward travel, held by the runner between 20 Hz vision ticks (`env_runner.py:9-13`).
- The EQOM side: the 200 Hz compliance reflex (`env_runner.py:373`) clamps downward Z only above 11 N,
  and pad *closure* is not Z-limited at all; pads closing on an ~13 mm off-axis face of rubble_a wedge
  against the core stack → 15–17 N.

**Minimal fix (generic):** when the DESCEND→GRASP transition fires on the force trigger, overwrite the
action z for that tick with the current `ee[2]` (one line after `_goto(t_sec, "GRASP")` in the DESCEND
branch). This stops the extra downward push on the contact tick. This is deliberately weaker than the
relief/unload experiment that regressed seed22 to 19.15 N (progress: "Tested immediate 6 mm unloading…"),
so it should not reopen seed22, while removing the shared contribution to the 15–17 N spikes.

Note the vision fix in 1a also contributes here: with the core/mask centroid unbiased, the inward-offset
estimate that wedges the pads is reduced.

### 1c. Why extraction still succeeded at seed33 under yawlock but not under yawnear

Under yawnear the core remains at its stale pre-handling position and the gripper closes empty at 8.05 s
and 10.05 s (progress record); retry budget (`t_sec < 7.0 ? 3 : 2`) is then exhausted before REDS/class
detection can reconnect. yawlock extracts because `_update_tracks` re-acquires after 3 consistent
re-detections >6 cm jumps (`:129-143`) plus the "dropped but already clear of the core" early-done rule
(`:219-225`). Recommendation stands from the prior sweep: **keep the near-only yaw commit
(`controller_yawnear.py`, all six seeds safety-side) as the shape of the committed-pose feature**, and fix
seed33 via 1a; the full always-lock of yawlock is what shattered seed103 at 25.01 N
(`controller_yawlock_seed103.json`), so adopting raw `controller_yawlock.py` is *not* advised despite its
5/6 laydown. This is the classic safety/extraction trade: whichever candidate is adopted, the vision fix
(1a) is the one lever that raises seed33 without touching opposing seeds.

## 2. Generic controller issues found

1. **`agents/patches/evolved_pipeline.py` has drifted from the measured candidate.** `diff` shows
   `dev/scratch/controller_yawlock.py` contains two blocks absent from the shipped patch source:
   - the dropped-in-flight/already-clear-of-core skip (`controller_yawlock.py:219-225`);
   - the recorded-lock yaw retention on re-lock (`:240-241`) and its core_grasp gate in
     `evolved_pipeline.py`'s sibling `controller_yawlock_gated.py:240-242`.
   Consequence: any fallback stage materialised from `evolved_pipeline.py` replays the pre-fix behaviour
   whose known failure modes are documented (`controller_progress.md` guesswork list: seed11 inward
   estimation, empty closures). Suggest root syncs the measured candidate into `agents/patches/` before
   the next fallback-library build.

2. **SURVEY can dispatch with partially-detected objects** (`:211`): at `t_sec >= 1.8` the force_guard
   fallback starts APPROACH even if a track is still `None`; the nominal-site fallback
   (`:237`, `p = [0.42, 0, 0.44]`) then drives an empty GRASP at a scene-independent site. On yawnear's
   seed33 run this is one of two sources of empty closures, since track[core] went None after handling
   shifted the core. Minimal generic hardening: if the current queue head's track is None at APPROACH
   entry, insert a short second SURVEY dwell (e.g. re-enter SURVEY until the head is tracked again or
   t > 6.5 s) instead of reaching to the nominal site.

3. **`_track_grip` rate-limits only downward** (`:194`): `self.grip_cmd = max(target, self.grip_cmd - 0.12)`
   jumps upward instantly. Harmless for closure, but on RELEASE→RETREAT handoff `grip` passes through
   `SOFT_OPEN` outside `_track_grip`, so no bew applies; fine. Low priority; note for a future pass.

4. **`budget = 3 if t_sec < 7.0 else 2`** (`:284`, `:297`) is time-based and composite-friendly, but a
   repeated-empty-retry loop is bounded only by `tries`; combined with issue 2 the episode can end with
   the queue drained but the core unhandled. Bounds are explicit and safe; no change needed for the
   11 s horizon.

## 3. Recommended minimal change set (ordered, no per-seed hardcoding)

| # | Change | Where | Expected seed impact |
|---|--------|-------|----------------------|
| 1 | Core: largest-valid-component selection instead of area-penalised hue score | `agents/patches/evolved_pipeline.py` `_pick_blob` | seed33 vision 13.39→~18, seeds 11/22/102 mostly unchanged (verified laydown identical when core mask is unfragmented) |
| 2 | `_measure` xy from component centroid at median depth (stay median in z) | `_measure` | reduces residual xy bias on all seeds (supports issue 1a) |
| 3 | Hold ee z on the DESCEND→GRASP contact tick | `step` DESCEND branch | seed22/33 closure spikes drop toward the ≤ 13.83 N observed relief level without the extraction loss of full relief |
| 4 | Sync the two yawlock-only blocks from `dev/scratch/controller_yawlock.py` into `agents/patches/evolved_pipeline.py` (rule 1 of the diff above) | patch source | keeps the shipped fallback library consistent with the measured candidate |

Items 1–3 are one-line-to-few-lines diffs, seed-agnostic, and are consistent with the prior "waiting-on
GPU0" constraint; they do not introduce per-seed thresholds.

## 4. Sources

* `dev/scratch/controller_yawlock.py` (candidate under review, in-repo measured candidate)
* `agents/patches/evolved_pipeline.py` (shipped fallback source; diffed above)
* `dev/scratch/controller_yawlock_scores.jsonl` (yawlock 6-seed sweep; seed103 written from the 46.63
  re-run per `controller_progress.md` processes note)
* `dev/scratch/controller_{yawsoft,yawnear,relief}_seed33.json`, `controller_yawnear_scores.jsonl`,
  `controller_lock_scores.jsonl` (same-vision/control-controlled comparisons)
* `dev/controller_progress.md` (shared measured diagnostics: 13 mm inward estimate, DESCEND tick staleness,
  seed33 core drift (0.394,−0.035)→(0.351,−0.030), stale grasps at 8.05/10.05 s)
* `core/verifier_harness.py` (scoring contract: 40-frame vision window, gate forces, thresholds
  18 N fracture / 28 N contract / 25 ms latency)
