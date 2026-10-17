# Sensor-based fleet and genuine parameter learning

## Implemented observer; integration must be verified

`ui.fleet_perception.FleetObserver(params=None).observe(rgb,depth,calibration,
robot_proprio,t)` accepts only aligned corrupted RGB-D, camera calibration and
own-robot proprioception. It accepts no MuJoCo object, object IDs, contact state
or ideal object positions. Calibration keys: K, cam_pos, cam_rot (optical right,
down, forward -> world), optional floor_z and discharge_roi. The fixed camera
must observe a first pre-discharge background frame.

Foreground depth changes, measured height, mineral RGB ratios and component
size identify fragments in the known discharge workspace. Pixel backprojection,
robust percentile/median estimates and a known nominal fragment half-height
produce position estimates. A 3x3 closing operation only bridges component
labels through dropout gaps; positions use actual valid measured pixels.
No noise repair uses ideal simulator depth. Nearest spatial tracking retains up
to six tracks briefly through occlusion. mark_collected/mark_consumed removes
an observed completed pickup track from future discharge targeting.

Default ROI X[-.1,.65], Y[.1,1.1], pixel Z(.412,.46) at floor .4. This excludes
known cart-loading positions/heights but must be checked against actual noisy
sensor images. Each return includes track_id, pos, confidence, last_seen,
age_s, visible and provenance=corrupted_rgbd. Track IDs are generated visual
identities, never simulator body IDs.

The physical controller integration belongs to the fleet agent: it must render
and corrupt actual RGB/depth at 20 Hz, provide static camera calibration, and
replace debris ground-truth targeting with track positions. Body IDs/contacts
may establish physical grasp constraints or verification, but cannot determine
the policy's target estimate. `sensor_based=True` must only be emitted after
that actual wiring is used. Merely importing the observer does not establish
sensor-based autonomy. Cart dispatch/load sensing requires the same distinction.

Local CPU sanity checks used a synthetic isolated depth rectangle to check
backprojection/tracking/removal and invalid parameter/provenance rejection.
They passed, and are not evidence of a successful actual fleet run. Actual
corrupted sensor capture and physics pickup verification remain necessary.

## Separate measured fleet parameter learning

`agents.fleet_learning.FleetLearner(evaluator,root='shared_brain/fleet_learning',
client=None).run_iteration(seeds=(11,22,33))` performs fresh baseline and candidate
rollouts through evaluator(parameters,seed). No core recovery ledger, DoD,
controller or benchmark score is modified. Fleet evidence is appended atomically
to its separate ledger; no prefilled success or invented baseline is used.

Bounded knobs: detect_min_pixels [3,30], detection_height_min_m [.006,.025],
track_memory_s [.3,2], grasp_dwell_s [.3,1.5]. Native integration must apply all
proposed parameters. Optional real Crusoe/Nemotron proposes only bounded JSON
parameters; malformed/unsafe proposals fail. Without a client, actual measured
misses or incomplete delivery trigger bounded tracking/dwell adjustments.

Every evaluator result must attest sensor_based=True and provide its actual
seed, debris_collected, debris_total, peak_force_n, collector_peak_force_n and
pickup_misses; missing metrics fail rather than being filled in. No holdout
seed is accepted by this optimization loop. Its declared fleet-only objective
is 100*mean delivered fraction - .05*peak drill force - 2*mean pickup misses.
Candidate promotion requires fresh score improvement, nondecreasing delivery,
drill peak below120N and collector peak below25N. Failure/no-change is recorded
honestly; rejected parameters are not selected as the next starting controller.
Fresh baseline/candidate raw summary metrics, parameter hashes, source/rationale
and real model usage (when applicable) remain in the fleet ledger.

This is parameter adaptation based on measured outcomes, not a claim that an
LLM learned full mining behavior or that core benchmark acceptance transfers to
fleet autonomy. At preparation time no genuine fleet learning rollout or
promotion has yet been run by this task.
