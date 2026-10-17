# Accepted-policy reuse in the mining fleet

`agents.mining_policies.AcceptedGraspPolicy` dynamically loads the newest
accepted pipeline recorded in `shared_brain/ledger.json`. It checks the
source SHA-256 against the ledger before importing it, then directly calls
that source's `ExcavationPipeline._track_grip` method.

The adapter reuses only the gripper closure law. It passes a measured pad gap
to the accepted controller every 50 ms, caches the returned target between
updates, and converts the normalized command back to meters. Call
`reset()` at the start of a grasp and after release. The adapter constructs the
accepted pipeline with dummy camera calibration arrays; it does not invoke a
vision step, simulate perception, or run the accepted excavation pipeline.

`metadata` reports the accepted version, ledger source path and hash, the
source `TOUCH_N` value, and the evaluation scope. The ledger acceptance was
measured on core recovery. Transfer to fleet debris grasping has not been
benchmark-verified. This is controller reuse, not reinforcement learning or
a new training run.

The adapter fails explicitly when the ledger has no accepted policy, when its
source is missing or differs from the recorded hash, or when the accepted
module cannot provide the expected controller and constants. It has no
fallback policy.
