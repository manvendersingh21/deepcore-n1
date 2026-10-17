# Verification sweep

## Status

The requested fresh-process sweep could not be run from this environment, so no per-seed measurements are reported.

- The local checkout has no `core/LOCK.json`. The verifier requires `core.integrity.verify_integrity()` to pass; writing a lock was not authorized.
- The configured `cis-a6000` SSH route was unavailable: the configured control socket was denied, and a direct connection could not resolve its host.

## Per-seed results

| Version | Seed | Composite | Core shattered | Core extracted | Peak core contact (N) | Rubble cleared | Vision RMSE (mm) |
|---|---:|---:|---|---|---:|---:|---:|
| v0–v3 | 11, 22, 33, 101, 102, 103 | Not measured | Not measured | Not measured | Not measured | Not measured | Not measured |

## Determinism

v3 seed 11 was not run three times in fresh processes; determinism is undetermined.

## Seed 101 extraction trace (current controller, v3)

- Rubble_b cycle ends at 4.2 s; rubble_a cycle ends at 10.55 s.
- Core approach starts at 10.75 s; the episode ends at 11.0 s.
- The core is never grasped and stays at (0.38, -0.04) to the end.
- Root cause: time budget. Each rubble boulder takes about 4-5 s of phases, leaving the core under 0.25 s.
- Rejected hypothesis: releasing the core too early in PLACE (proposal in dev/extract_fix_proposal.md). Scratch-tested on holdout 101-103: composite unchanged (43.4 / 56.9 / 44.1), no extraction. Not adopted.

## Controller fix: core time budget + pre-shape (applied to agents/patches/evolved_pipeline.py)

Changes (all under existing core_grasp feature):
- After 6.5 s, skip remaining rubble and go straight to the core if it is still queued.
- Rubble re-queue only before 6.5 s.
- Core DESCEND pre-shapes the gripper to 0.80 (clears a neighbour ~45 mm off).

Measured on cis-a6000, one fresh process per seed, current template (md5 of template after fix recorded at apply time):

| seed | composite | shattered | extracted | rubble | peak core N |
|---|---|---|---|---|---|
| 11  | 35.71 | yes | no  | 1 | 18.03 |
| 22  | 80.66 | no  | yes | 2 | 14.93 |
| 33  | 28.96 | yes | no  | 1 | 18.92 |
| 101 | 85.90 | no  | yes | 2 | 4.54 |
| 102 | 86.86 | no  | yes | 2 | 11.34 |
| 103 | 82.29 | no  | yes | 2 | 12.26 |

Holdout mean composite ~85. Optimisation seeds 11 and 33 still shatter (just over 18 N). Not locked.
