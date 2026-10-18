# DeepCore-N1 — measured results

Every number below comes from a saved result file in this repo. Nothing here is
estimated or live. Sources are listed at the end of each section.

## 1. Locked benchmark: fresh verification (`dev/final_verification.json`)

Verifier: `core/verifier_harness.py`, locked by SHA-256 in `core/LOCK.json`.
Status: `passed = true`, `evidence_unchanged = true`.

| Split | Seeds | Composite (/100) | Extraction rate | Shattered |
|---|---|---|---|---|
| Optimization | 11, 22, 33 | **81.92** | 3 / 3 | none |
| Holdout | 101, 102, 103 | **85.56** | 3 / 3 | none |

Per seed:

| Seed | Split | Composite | Peak core force (N) | Extracted | Shattered |
|---|---|---|---|---|---|
| 11 | opt | 87.76 | 4.41 | yes | no |
| 22 | opt | 80.25 | 15.38 | yes | no |
| 33 | opt | 77.76 | 11.76 | yes | no |
| 101 | holdout | 85.89 | 4.87 | yes | no |
| 102 | holdout | 90.25 | 4.50 | yes | no |
| 103 | holdout | 80.55 | 14.05 | yes | no |

**Target not met:** the goal of composite > 88 on every seed is still open.
Seeds 22, 33, 101 and 103 score below 88.

Controller: accepted v4, SHA-256
`601d8825913e741e2069f49fd1036198001ecfa432e244ac34b5bb8f3d0af97b`.
This is a **curated fallback** from the patch library, not a model-written
improvement. The earlier LLM-authored candidate (v3) was rejected.

## 2. Recorded replays, seed 101 (`dev/replays/manifest.json`)

| Version | Composite | Core extracted | Core shattered | Rubble cleared | Peak core force (N) |
|---|---|---|---|---|---|
| Baseline (v0) | 14.33 | no | no | 0 | 5.05 |
| Accepted (v4) | 85.97 | yes | no | 2 | 4.87 |

Each replay is 221 frames of real MuJoCo physics. Checksums are in the manifest.

## 3. Mining crew, sensor-driven (`dev/native_realistic_fleet.json`)

Run: seed 101, native MuJoCo, 80 s simulated time, 60 fps encoded video
(4801 frames). Autonomy: experimental, sensor-driven, **not benchmark-verified**.

| Metric | Value |
|---|---|
| Drilling layers removed | 12 / 12 |
| Max drill depth | 89.0 mm |
| Peak drill (axial) force | 83.2 N (limit 120 N) |
| Debris fragments collected | 6 / 6 |
| Missed pickups | 0 |
| Rail carts loaded | [2, 2, 2] (capacity 2 each) |
| Carts dispatched | 3 / 3 |
| Collector peak contact force | 10.7 N |
| Perception input | corrupted RGB-D tracks plus joint encoders |

Only one seed has been run for the mining crew.

## 4. Performance — honest numbers

| Measurement | Value | Source |
|---|---|---|
| Native frame build, p50 | 26.3 ms | `dev/native_realistic_fleet.json` |
| Native frame build, p95 | 32.4 ms | same |
| Real-time 60 fps at p95 | **not met** (`native_60fps_p95_met: false`) | same |

The 60 fps value in the video is the encoding rate. It does not mean 60 frames
were rendered per second.

## 5. Not yet done

- Fleet learning: the learner is written and tested on CPU (53 tests), but no
  genuine learning run or promotion has happened.
- More than one seed for the mining crew.
- Real-time 60 fps rendering.
- Any physical robot. Everything is simulation.

## Reproduce

```bash
.venv/bin/python main.py run-web --port 8080   # browser control room
.venv/bin/python main.py run-mine --seed 101   # native mining-crew view
```
