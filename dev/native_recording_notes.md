# Native recording

The default 60 second sequence now opens with 12 seconds of actual drilling,
followed by 10 seconds baseline recovery, 12 seconds recorded ledger recap, and
26 seconds accepted recovery. Drilling runs its own genuine material model;
the recovery benchmark episodes are independent simulations. This sequence does
not represent continuous physics from drilling through extraction. Source frames
are repeated/skipped to fit output segments. `--no-drilling` restores the original
20/15/25 second comparison; existing positional Python `main` arguments remain
compatible, with `include_drilling=True` appended after `fast`.

The matching `.json` sidecar records the actual drilling summary, seed, selected
accepted version (null if none is recorded), actual recovery version, output
segment durations/frame counts and source callback cadence. It does not create
benchmark scores. Drilling capture uses `ui.drilling_hud.capture(seed=holdout_seed)`.

The recorder runs actual seeded MuJoCo episodes through `FrameBuilder`, using
`ui.mine_visuals.render_presentation` for the robot view and the actual RGB feed
for the camera view. It preserves final verifier scoring via `finalize_state`.

The middle segment replays recorded ledger entries. An empty ledger explicitly
reports that no learning history is recorded. It does not create agent events or
claim an accepted improvement. The final episode uses the best accepted holdout
entry, falling back to the newest accepted version, and then to baseline v0.

`capture_episode` now closes the presentation renderer in a `finally` block.
It also closes the episode's sensor renderers if a callback raises or recording
is interrupted, because `run_episode` only closes these on normal completion.
Raw RGB camera and presentation arrays remain the recap backdrop.

The evolution recap streams one rendered frame at a time into the video writer,
avoiding retention of all 450 full resolution recap frames. The loop still
emits exactly its requested segment frame count. Episode capture retains the
settled `finalize_state` frame and its outcome hold frames.

Prepared full native recording command (run after evolution finishes and GPU is
available):

```sh
.venv/bin/python -m scripts.record_demo --out dev/demo_native_mining.mp4 --seconds 60 --fps 30 --holdout-seed 101
```

Keep `dev/demo_full.mp4` until the new artifact is recorded and inspected. At
preparation time the ledger was empty; no GPU workload has been started here.
