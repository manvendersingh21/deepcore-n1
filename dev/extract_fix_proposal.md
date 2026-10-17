# Extraction placement proposal

## Proposed one-line diff

In `agents/patches/evolved_pipeline.py`, inside the `PLACE` phase:

```diff
-                    if ee[2] <= place_z + 0.001 or (FEATURES["force_guard"] and fz > 4.5) or dt > 1.5:
+                    if ee[2] <= place_z + 0.001 or (FEATURES["force_guard"] and fz > 4.5) or (name != "core" and dt > 1.5):
```

## Reasoning

The core's destination is already `HOPPER` in `TRANSPORT`, `PLACE`, and `RELEASE`, so this proposal leaves its XY target unchanged. In `PLACE`, the controller descends toward `PLACE_Z["core"]` in 12 mm steps, but the unconditional 1.5-second timeout can transition to `RELEASE` before the end-effector reaches that height. Since release then holds its current height while opening, the core may be released above the intended placement point. Restricting the timeout to rubble preserves the core's existing height and force release triggers and lets its commanded descent finish before release.

This is a code-reading hypothesis based on the described v3 failure. It has not been run or validated in simulation.
