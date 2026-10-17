# Independent Fleet Control and Safety Review

**Reviewer:** Peer B  
**Contract ID:** `c-8f1bdd6d88ea4422be3a43b5512cf044`  
**Revision Digest:** `318350ad1e4558a66c1d6d7cff31d93dd3cf632d67ba959e29c8e8c006f4d8a1`  
**Status:** Frozen / Executing Contract Submission  
**Scope:** `ui/mining_fleet.py`, `ui/industrial_arm.py`, `ui/fleet_perception.py`, `agents/fleet_learning.py`

---

## Executive Summary

An independent static safety, state-integrity, and control review of the cooperative mining fleet pipeline was conducted. The codebase exhibits rigorous structural separation in several key subsystems:
- Perception in [`FleetObserver`](file:///Users/manubaba/Documents/deepcore-n1/ui/fleet_perception.py#L12) processes only corrupted RGB-D point clouds and nominal static calibration, without direct simulator body handles or ground-truth poses.
- Mobile base and arm controls operate through actuator torques, damped IK, and motor position references rather than kinematic teleportation.
- Debris generation is coupled to measured cutter work milestones, and arm pickup relies on bilateral tactile contact before grasping.

However, several critical and high-severity issues were identified:
1. **Critical:** In [`MiningFleetEpisode.telemetry`](file:///Users/manubaba/Documents/deepcore-n1/ui/mining_fleet.py#L602-L604), the mission `complete` boolean evaluates completion before cart dispatch finishes and before debris has settled inside cart boundaries, while the main run loop [`run_mining_fleet`](file:///Users/manubaba/Documents/deepcore-n1/ui/mining_fleet.py#L647-L661) ignores `telemetry["complete"]` entirely, running for a fixed 80-second duration without early exit on completion or unrecoverable failure.
2. **High:** In [`MiningFleetEpisode._base_servo`](file:///Users/manubaba/Documents/deepcore-n1/ui/mining_fleet.py#L244-L249), `precision=True` employs an unbounded high-gain PD controller ($K_p = 30000, K_d = 2000$) whose output is clipped only at the control limits, creating instantaneous actuator force saturation and abrupt acceleration changes when switching states.
3. **High:** In [`FleetLearner._append`](file:///Users/manubaba/Documents/deepcore-n1/agents/fleet_learning.py#L116-L129), ledger writes use process-local `fcntl.flock` on an arbitrary `ledger.lock` file while loading and atomic rewriting are unversioned; furthermore, [`FleetLearner.run_iteration`](file:///Users/manubaba/Documents/deepcore-n1/agents/fleet_learning.py#L201-L206) evaluates candidates on the identical fixed seed set `(11, 22, 33)` without out-of-distribution holdout evaluation, creating an over-claim regarding general fleet policy improvement.
4. **Medium:** In [`MiningFleetEpisode._spawn_debris`](file:///Users/manubaba/Documents/deepcore-n1/ui/mining_fleet.py#L409-L411), runtime state write directly mutates MuJoCo `qpos` and `qvel` upon debris ejection. While documented as an event-driven discrete material ejection model, it constitutes a privileged state write during active physics simulation.
5. **Medium:** In [`FleetObserver.observe`](file:///Users/manubaba/Documents/deepcore-n1/ui/fleet_perception.py#L68-L71), robot chassis filtering relies on exact simulator root body coordinates `data.xpos` passed through `robot_proprio`, which is true ground-truth world position rather than noisy wheel/odometry estimates.

---

## Confirmed Actionable Findings

### 1. Premature Completion Condition and Dead Run-Loop Control
- **Severity:** High
- **Location:** [`ui/mining_fleet.py:602-604`](file:///Users/manubaba/Documents/deepcore-n1/ui/mining_fleet.py#L602-L604) and [`ui/mining_fleet.py:655-659`](file:///Users/manubaba/Documents/deepcore-n1/ui/mining_fleet.py#L655-L659)
- **Evidence:**
  In [`MiningFleetEpisode.telemetry`](file:///Users/manubaba/Documents/deepcore-n1/ui/mining_fleet.py#L602-L604):
  ```python
  complete = bool(self.removed.all() and self._retracted and self._collected.all()
                  and self.drill_phase == "parked" and self.collector_phase == "parked"
                  and all(self.data.qpos[self._joint(f"cart_{c}_travel")[0]] < -1.0 for c in range(len(CART_CENTERS))))
  ```
  However, `self._collected[i]` at line 593 can become true if `self._settled_for[i] >= .15` while the cart is still stationary. The final cart check requires all carts `travel < -1.0`, but carts are dispatched sequentially (`self._cart_dispatched[cart] = True` at line 493). If cart dispatch fails or halts, `complete` remains false. More critically, in [`run_mining_fleet`](file:///Users/manubaba/Documents/deepcore-n1/ui/mining_fleet.py#L655-L659):
  ```python
  for step in range(round(DURATION_S / DT)):
      telemetry = ep.step()
      if frame_callback is not None and (step + 1) % interval == 0:
          frame_callback(ep, telemetry)
  return ep.telemetry()
  ```
  The simulation unconditionally runs for all 80,000 steps ($80.0\text{ s}$) regardless of whether `telemetry["complete"]` is achieved or whether an unrecoverable failure state (e.g. repeated missed pickups or drill stall) has occurred.
- **Impact:** Benchmark and rollout execution wastes compute cycles, and downstream callers cannot rely on `run_mining_fleet` terminating upon task success or aborting safely upon mission failure.
- **Minimal Fix:**
  In [`ui/mining_fleet.py:655`](file:///Users/manubaba/Documents/deepcore-n1/ui/mining_fleet.py#L655), check `telemetry.get("complete")` and break early when completion criteria are met:
  ```python
  for step in range(round(DURATION_S / DT)):
      telemetry = ep.step()
      if frame_callback is not None and (step + 1) % interval == 0:
          frame_callback(ep, telemetry)
      if telemetry.get("complete"):
          break
  ```

---

### 2. Discontinuous Actuator Force Transition in Base Servo
- **Severity:** High
- **Location:** [`ui/mining_fleet.py:244-249`](file:///Users/manubaba/Documents/deepcore-n1/ui/mining_fleet.py#L244-L249)
- **Evidence:**
  In `_base_servo`:
  ```python
  if precision:
      force = 30000 * error - 2000 * velocity
  else:
      desired = float(np.clip(5.0 * error, -.42, .42))
      force = 500 * (desired - velocity)
  self.data.ctrl[self.model.actuator(f"{name}_{axis}_motor").id] = np.clip(force, -300, 300)
  ```
  When the collector robot transitions from `approach` (`precision=False`) to `lower` (`precision=True`), or drill transitions to precision at lines 277 & 359, the proportional gain instantly jumps by a factor of 60 ($500 \to 30000$). For a position error of just $5\text{ mm}$ ($0.005\text{ m}$), $30000 \times 0.005 = 150\text{ N}$, immediately commanding $50\%$ of maximum actuator limit in a single simulation timestep ($1\text{ ms}$).
- **Impact:** High jerk ($dF/dt$), potential slip/instability in contact dynamics, and actuator force saturation bang-bang chatter during mode switching.
- **Minimal Fix:**
  Apply rate-limiting or ramp the control force between control intervals, or clamp the error entering the precision feedback:
  ```python
  if precision:
      error_clamped = np.clip(error, -0.01, 0.01)
      force = 3000 * error_clamped - 400 * velocity
  ```

---

### 3. Fixed Seed Overfitting and Inadequate Generalization Gate in Fleet Learning
- **Severity:** Medium
- **Location:** [`agents/fleet_learning.py:172-175`](file:///Users/manubaba/Documents/deepcore-n1/agents/fleet_learning.py#L172-L175) and [`agents/fleet_learning.py:200-206`](file:///Users/manubaba/Documents/deepcore-n1/agents/fleet_learning.py#L200-L206)
- **Evidence:**
  [`FleetLearner.run_iteration`](file:///Users/manubaba/Documents/deepcore-n1/agents/fleet_learning.py#L172) strictly enforces:
  ```python
  if not seeds or len(set(seeds)) != len(seeds) or any(seed not in (11, 22, 33) for seed in seeds):
      raise ValueError('Fleet learning uses unique optimization seeds 11/22/33 only')
  ```
  Candidate promotion decisions (`decision = 'PROMOTE'`) compare candidate scores against baseline scores exclusively on these exact three seeds `(11, 22, 33)`. There is zero out-of-distribution or holdout evaluation before promotion.
- **Impact:** Parameter optimization can easily overfit to artifacts specific to seeds 11, 22, and 33 (e.g. particular debris fragment trajectories or camera dropouts), while claiming general improvement in fleet ledger records.
- **Minimal Fix:**
  Add a validation step on distinct holdout seeds (e.g., 44, 55) before promoting parameters to `PROMOTE`.

---

### 4. Privilege Leak: Ideal Simulator Poses Used for Chassis Filtering
- **Severity:** Medium
- **Location:** [`ui/mining_fleet.py:538`](file:///Users/manubaba/Documents/deepcore-n1/ui/mining_fleet.py#L538) and [`ui/fleet_perception.py:68-71`](file:///Users/manubaba/Documents/deepcore-n1/ui/fleet_perception.py#L68-L71)
- **Evidence:**
  In [`MiningFleetEpisode._observe`](file:///Users/manubaba/Documents/deepcore-n1/ui/mining_fleet.py#L538):
  ```python
  proprio = {"base_positions": [self.data.xpos[self._body[key]].tolist() for key in ("drill", "collector")]}
  ```
  `self.data.xpos` is the exact MuJoCo Cartesian coordinate of the body origin in the world frame. In [`FleetObserver.observe`](file:///Users/manubaba/Documents/deepcore-n1/ui/fleet_perception.py#L71), this is used to exclude points within $0.20\text{ m}$ of the chassis:
  ```python
  keep &= np.linalg.norm(world[:, :2] - position[:2], axis=1) > .20
  ```
  While labeled as proprioception, reading world `xpos` directly bypasses sensor modeling (odometry noise, drift, wheel slip).
- **Impact:** Slight privileged state leak into the perception pipeline; if realistic odometry drift were present, chassis points would fail rejection and be falsely classified as debris fragments.
- **Minimal Fix:**
  Simulate noisy proprioception by adding an odometry accumulator or adding realistic Gaussian noise to base position coordinates passed in `proprio`.

---

### 5. Runtime Kinematic State Mutation via `qpos`/`qvel` Assignment
- **Severity:** Low
- **Location:** [`ui/mining_fleet.py:409-411`](file:///Users/manubaba/Documents/deepcore-n1/ui/mining_fleet.py#L409-L411)
- **Evidence:**
  In `_spawn_debris`:
  ```python
  self.data.qpos[qa:qa + 3] = position
  self.data.qpos[qa + 3:qa + 7] = (1, 0, 0, 0)
  self.data.qvel[va:va + 6] = (0, 0, .025, 0, 0, 0)
  self.data.eq_active[self.model.equality(f"debris_{index}_dormant").id] = False
  ```
  Directly writing to `qpos` and `qvel` during simulation bypasses physical conservation laws.
- **Impact:** Acceptable as a discrete material generation proxy given the domain, but represents a discontinuity in simulator momentum and state.
- **Minimal Fix:**
  Ensure docstrings and benchmark claims explicitly document that debris introduction is a discrete event-driven boundary condition rather than continuum fracture physics.

---

## Limits and Scope of Review

1. Static review was performed strictly on the four requested inputs (`mining_fleet.py`, `industrial_arm.py`, `fleet_perception.py`, `fleet_learning.py`) and referenced assets.
2. In accordance with read-only review rules, live simulation runs, video rendering, test suites, and GPU invocations were not performed.
3. No malicious backdoors or unconstrained remote execution paths were detected.
