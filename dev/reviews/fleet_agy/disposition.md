# Peer A disposition: fleet control and safety review

The independent report is accepted as a useful static review, with the following root-reviewed dispositions. No simulation or test was run for these decisions.

| Report item | Disposition | Remediation/status |
|---|---|---|
| Premature completion / fixed-duration loop | Not accepted as a defect. The quoted `complete` predicate requires all carts to travel below `-1.0`, and the fixed 80 s run is intentional evidence capture. | No control-flow change planned from this finding. Keep completion claims tied to the telemetry predicate. |
| Abrupt precision-servo force transition | Accepted as an actionable control concern. | Native drilling owns smoothing or slew-limiting the force transition. Implementation and runtime verification are pending. |
| Fixed-seed learner overfitting | Scope correction. Seeds 11/22/33 are explicitly optimization-only; that alone does not establish a false benchmark claim. | Native recording owns learner hardening. Add a distinct holdout gate before describing candidates as generalizing beyond optimization seeds. Pending. |
| Simulator chassis pose used in perception filtering | Accepted as a proprioception-modeling gap. | Native drilling will replace direct `data.xpos` input with joint/encoder-derived chassis position. Pending. |
| Direct qpos/qvel writes when spawning debris | Not accepted as policy leakage. This is an event-driven material-boundary condition, not object carry control. | Preserve the documented simulation abstraction; clarify its scope where needed. No physical-fracture claim is implied. |

These dispositions record root review, not completed fixes. The two accepted code changes and any holdout-gate work remain pending; this static review did not verify them at runtime.
