# Controller experiments — 2026-10-17

All candidates are scratch copies. No core, verifier, template, threshold, or ledger edits.
Measurements use the intact verifier and current integrity lock on `cis-a6000`, GPU0, fresh process per seed.

## Evidence

Seed11 diagnosis on existing skip-cleared controller: first rubble_b descent stops at wrist z≈0.499 because right pad contacts rubble_a. At t2.075, while first closure command is0.88, gated force reaches18.03N. Estimated grasp is about13mm inward of actual rubble_b center. Later rubble_a has fallen onto floor, but the unconditional nearby-core grasp-height clamp keeps pads above its top; repeated empty grasps consume the episode.

## New candidates

| Candidate | Change | Seed11 score | Peak N | Shattered | Extracted | Cleared |
| --- | --- | ---: | ---: | --- | --- | ---: |
| controller_preshape.py | Skip-cleared + rubble descent opening0.80 |24.43|16.67|no|no|0|
| controller_creep.py | Skip-cleared + descent increments6mm |35.90|18.03|yes|no|1|
| controller_offset.py | Skip-cleared + rubble target10mm outward from estimated core |38.31|15.07|no|no|1|
| controller_floorgrasp.py | Offset candidate + permit lower grip on floor-settled rubble |38.23|15.07|no|no|1|
| controller_lock.py | Skip-cleared + freeze grasp position/yaw during approach |83.82|12.01|no|yes|2|

Preshape and creep candidates rejected. Offset safe on seed11, full gate unproven. No candidate adopted.

## Breakthrough: committed grasp pose

`controller_lock.py` snapshots the grasp target/yaw once APPROACH starts, and retains it through the reach. The original code updated lock every frame while the hand obscured the target, moving/rotating the grasp into neighboring rock. Lock still resets between retries and objects, so re-detection remains available. Seed11 now extracts at9.65s and completes at9.9s. Six-seed fresh-process sweep running; not adopted until all pass.

## Full committed-pose sweep — rejected

|Seed|Score|Shattered|Extracted|Peak N|Rubble|
|---|---:|---|---|---:|---:|
|11|84.09|False|True|12.01|2|
|22|47.64|True|False|20.09|2|
|33|47.93|False|False|11.44|2|
|101|85.95|False|True|4.46|2|
|102|90.24|False|True|4.46|2|
|103|34.11|True|False|23.7|1|

Candidate SHA256 a06f4a1f2978ba9aa9dfcd1110d5b15b219b743e4e33947dbabd55bf3f7fb506. Seed22 and103 regress; do not adopt. Next: yaw-only lock, softer closure.

## Yaw-only committed grasp sweep

`controller_yawlock.py` keeps the initial finger yaw during each approach, while position follows vision. The first five seeds all extracted safely:11=87.84/4.45N,22=80.16/15.38N,33=72.01/17.46N,101=86.10/5.27N,102=90.21/4.45N. Sweep shell2377765 stopped by SIGSTOP after102 to reserve GPU0 for UI recording; child2379414 confirmed terminal zombie. Resume shell with SIGCONT for103; do not restart sweep.

`controller_yawlock_gated.py` adds the existing core_grasp feature gate plus comments, so earlier fallback stages retain their existing behavior. All v3 features are enabled, making its v3 control behavior identical to measured candidate.

Sweep resumption note: after SIGCONT, shell2377765 became terminal. Local exec session6258 was missing following usage interruption, and103 stdout was lost (JSONL remained5 rows). Both process handle and remote process confirmed absent before restarting only103, writing remote `dev/scratch/controller_yawlock_seed103.json` for durable result.

## Yaw-only final result — rejected

103 scored46.63, shattered at25.01N (2 rubble). Five prior seeds extract safely, but all-six gate fails. Result saved `controller_yawlock_seed103.json`.

## Force-control followups

Opt33 diagnostic:17.46N peak occurs during initial closure on rubble_a; DESCEND changes phase but leaves the old downward target in action until next vision tick. Tested immediate6mm unloading at touchdown plus relief while fz>5.5:11 extracts80.70/15.48N,22 fails19.15N,33 safe45.49/13.83N without extraction. Rejected. Slower closure0.08/tick on yawlock also regresses33 to18.70N. Rejected.

`controller_yawnear.py` tests yaw latching only within55mm of target so distant approaches can correct stale orientation after previous object movement. Optimization11/22/33 running. No template changes.

## Near-only yaw commit — safety gate PASS

All six safe; all three holdouts extracted. Opt33 still does not extract. Candidate `controller_yawnear.py`.

|Seed|Score|Shattered|Extracted|Peak N|Rubble|
|---|---:|---|---|---:|---:|
|11|87.93|False|True|4.41|2|
|22|80.3|False|True|15.38|2|
|33|35.25|False|False|11.76|1|
|101|85.89|False|True|4.87|2|
|102|90.21|False|True|4.5|2|
|103|80.45|False|True|14.05|2|

Interpretation: commit yaw only once the hand is within55mm of target. Distant approaches still correct stale orientation after prior rock movement; nearby reaches resist hand occlusion. Existing core_grasp gate preserves prior fallback stages. No template adoption by this agent. Full extraction and >88 all seeds remain unmet.

## Remaining opt33 diagnosis

On yawnear the cyan core moves from(0.394,-0.035) to(0.351,-0.030) during rock handling. Repeated grasps still target(0.399,-0.048) at8.05s and10.05s and close empty. Candidate blob selection penalizes every area pixel by0.02, so small cyan fragments may beat the complete core. `controller_largestcore.py` prefers largest valid cyan component (no cyan dump marker), while preserving existing selection for rubble and existing feature stages. This hypothesis is unverified; GPU0 reserved for parent orchestration.

Remote candidate SHA256 matches local:6ea1b6cd95255908782968d029e862d59a3265985d289e01319a6a732593075a. `core.integrity.verify_integrity()` passes on GPU host without bypass. Parent informed and adopting safety-gated version through normal orchestrator.

Prepared isolated followup `controller_orange_highlight.py`: raise orange mask V ceiling220→255; existing threshold rejects highlighted top surfaces, which can bias estimated center toward the remaining edge. Pending opt33 verification. `controller_perception_trace.py` explicitly restricts diagnostic seeds to optimization set.
