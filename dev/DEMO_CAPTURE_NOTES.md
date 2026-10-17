# Demo capture evidence

## Current scope

The requested final deliverable is an actual fleet episode showing articulated
robots drilling, collecting debris, and filling and dispatching rail carts in an
enclosed mine. Final encoding and inspection are pending the native recording.
No final video or 60 FPS performance claim is established by this checkpoint.

## Preserved benchmark recordings

`replays/baseline.json` and `replays/accepted.json` were copied from the running
web server before restart could discard its in-memory cache. Both contain 221
actual recorded MuJoCo frames for seed 101. `replays/manifest.json` records their
SHA-256 hashes and original results/provenance.

- Baseline: failed extraction, zero rubble cleared, no shattered core.
- Accepted: extracted intact, two rubble cleared, score 85.97.
- Accepted policy source is `fallback_patch:v3_force_guard_gentle_place`.
  It is not a Nemotron-authored promotion.

## Browser capture tooling

`scripts/verify_visualizer.mjs` now blocks POST requests, warms up the renderer,
measures while replay moves, and collects both versions before judging cadence.
`scripts/record_submission.mjs` blocks POST requests, warms up the renderer and
preserves actual partial WebM evidence on failure. Both pass Node syntax checks.
A full invocation remains deferred while fleet visuals and recording finish.

A read-only 1600×900 browser screenshot is `capture_current_inspection.png`.
Its five-second initial render sample averaged 57.27 FPS including load. This is
not a steady rendering benchmark. The image shows the old core benchmark gantry,
not the final articulated fleet deliverable.
