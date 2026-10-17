## 1. Camera Geometry & Back-Projection Equations
- p_world = cam_pos + cam_rot @ [(u-cx)/fx*d, (v-cy)/fy*d, d], where d = depth[v,u] metres; K has fx=fy≈579.4, cx=320, cy=240.
## 2. Verified 3D Vision Filters (Dust/Glare/Occlusion Invariant)
- Depth contains NaN/0 dropouts and 0.25/1.80 m spikes; dust plume occurs during frames 40..70.
- v1 (fallback_patch:v1_hsv_iqr_z_ordering): opt RMSE 124.6->18.7 mm, composite 12.570->30.590, holdout composite 32.310
- HSV bands + morph open/close + largest blob beat raw RGB thresholds under glare; use IQR-median depth over the mask, never a single pixel (NaN/0.25/1.80 m spikes).
## 3. Verified Excavation Ordering & Force-Guarded Control Laws
- The core is buried under rubble_a, and rubble_b leans on its right edge; stop contact at the 18 N fracture threshold.
- v1: cleared 0.7 rubble, peak core force 14.9 N (<18.0), latency p99 9.8 ms, extraction 0.00
- Excavate overburden first: sort rubble by descending Z, move each to the dump zone, then the core.
- v4: cleared 2.0 rubble, peak core force 10.5 N (<18.0), latency p99 10.3 ms, extraction 1.00
- Force-guarded touchdown: fast to 2 cm above the grasp height, then creep and stop on |Fz|>6.5 N; release the core low (z~0.4245) in the hopper.
## 4. Graveyard of Rejected Hypotheses (DO NOT REPEAT)
- No rejected hypotheses are recorded yet.
- g2 fallback_patch:v2_yaw_tracker (g2-19eea010): rejected - opt composite 27.42 not > 30.59; core shattered
- g3 llm:nvidia/NVIDIA-Nemotron-3-Super-120B-A12B (g3-30c70db3): rejected - opt composite 30.57 not > 30.59; holdout composite 32.28 < 32.31
