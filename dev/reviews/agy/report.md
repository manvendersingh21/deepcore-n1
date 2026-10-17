# Independent Code Review: Native MuJoCo Visuals & Live HUD

**Target Modules:** `ui/mine_visuals.py`, `ui/live_hud.py`  
**Review Reference:** `dev/native_hud_preview.jpg`  
**Reviewer:** HACP Peer B (`agy review`)  
**Role Scope:** Advisory review only (Root performs all code modifications)

---

## Executive Summary

The native MuJoCo presentation renderer (`ui/mine_visuals.py`) and cinematic HUD (`ui/live_hud.py`) successfully replace synthetic/mock interfaces with a high-fidelity, physically anchored 3D visualization. The mine basin dressing, lighting adjustments, gantry crane superstructure, and HUD styling achieve strong visual realism and professional polish (as demonstrated in `dev/native_hud_preview.jpg`).

However, four key areas require attention before final lock:
1. **Camera Projection Discrepancy:** The wrist force vector is projected using the XML camera `"cam_perspective"` instead of the active free presentation camera (`mjCAMERA_FREE`), leading to misaligned or invalid 3D-to-2D screen projections.
2. **Renderer Lifecycle & Slot Hijacking:** `mine_visuals.py` overwrites `ep.persp_renderer` with a 1200x750 renderer, altering expectations for any callers relying on `Episode.render_perspective()` (640x480).
3. **Dead Code & Wasted Per-Frame Compute:** Over 450 lines of obsolete renderers (`_draw_camera`, `_draw_persp`, `_draw_rail`, `_draw_tree_panel`, `_draw_scoreboard`) remain in `ui/live_hud.py`. Furthermore, expensive per-frame computations like `depth_confidence` are calculated on every timestep but never rendered.
4. **Dropped Visual Overlays:** Perception bounding boxes (`bboxes`) and webcam occlusion indicators are omitted from the robot vision picture-in-picture panel.

---

## Detailed Findings

### 1. Physics & Sensor Isolation

* **Isolated Visual Model (Positive):**  
  In `ui/mine_visuals.py` (`_context()`), `visual_model = copy(ep.model)` creates an independent `MjModel`. Uploading photographic textures (`tex_earth`, `tex_wall`, `tex_rock`) into `visual_model.tex_data` does not mutate `ep.model.tex_data`. Overhead RGB-D perception camera arrays and ground-truth physics calculations remain completely isolated.
* **Decoration Geometry Isolation (Positive):**  
  All dressing objects (portal frames, crane rails, lamps, conduit, outer scree) are injected into `scene.geoms` with `category = mjCAT_DECOR`, `objtype = mjOBJ_UNKNOWN`, and `objid = -1`. No dynamic bodies, contacts, joints, or sensor buffers are touched.
* **`persp_renderer` Slot Collision (Issue):**  
  `mine_visuals.py` overwrites `ep.persp_renderer = mujoco.Renderer(visual_model, HEIGHT, WIDTH, max_geom=2000)`.  
  In `core/env_runner.py`, `ep.persp_renderer` is allocated at 640x480 for `ep.render_perspective()`. Replacing this attribute causes callers of `ep.render_perspective()` to receive a 1200x750 image with presentation materials and lighting instead of the baseline sensor view.  
  *Recommendation:* Isolate presentation rendering entirely under `ep._mine_presentation['renderer']` or a dedicated attribute (`ep._mine_renderer`) without overwriting `ep.persp_renderer`.

---

### 2. Correctness & 3D Projections

* **Wrist Force Arrow Camera Mismatch:**  
  In `ui/live_hud.py` (`FrameBuilder.__call__`, lines 1135–1138):
  ```python
  uv = project_world_points(ep.model, ep.data, self.persp_camera,
                            np.stack([ee, ee + 0.01 * wf]), persp.shape[1], persp.shape[0])
  ```
  `self.persp_camera` defaults to `"cam_perspective"` (the fixed XML camera at `pos="1.05 -0.62 0.95"`, `fovy="50"`).  
  However, `render_presentation(ep)` renders using `state['camera']`, an interactive `mjvCamera` (`mjCAMERA_FREE`, `lookat=(0.45, 0.015, 0.68)`, `distance=1.88`, `azimuth=133`, `elevation=-47`).  
  `mine_visuals.py` provides `presentation_camera(ep)`, but it is unused. Any projected screen-space vector is misaligned relative to the end-effector visible in the perspective frame.
* **Aspect Ratio & Letterboxing:**  
  `mine_visuals.py` renders at `WIDTH, HEIGHT = 1200, 750` (16:10 / 1.6 aspect ratio). `CARD["persp"]` in `live_hud.py` is `1100x640` (~1.72 aspect ratio). `_native_image` letterboxes the rendering horizontally with 38 px margins on both sides.  
  *Recommendation:* Align `WIDTH, HEIGHT` in `mine_visuals.py` to match `1100, 640` (or integer multiple), avoiding letterbox scaling and saving GPU/rasterizer fill-rate.
* **Gantry Bridge Kinematics Tracking (Positive):**  
  The gantry bridge and trolley decor track `ep.ee_pos()[:2]` directly. Because the gantry has Cartesian prismatic slide joints along X and Y, the visual bridge aligns with the physical carriage position without artificial lag or fictitious trajectory generation.

---

### 3. Visual Legibility & Operator UX

* **Legibility & Hierarchy (Positive):**  
  As seen in `dev/native_hud_preview.jpg`:
  - Dark slate background (`#0B1116`) provides excellent contrast against `#EDEDF2` typography.
  - The extraction site view is prominent, with crisp framing and clean title/status overlays.
  - Metric cards at the bottom (`CORE CONTACT`, `WRIST FORCE`, `VISION LATENCY`, `POSITION ERROR`) provide instant status readability with unit badges and sparklines.
  - The right rail clearly presents the recursive learning agent hierarchy and holdout verification ledger.
* **Missing Overlays on Robot Vision Thumbnail:**  
  The overhead camera thumbnail (`CARD["cam"]`, 160x120) displays raw camera RGB, but the bounding box predictions (`s.bboxes`) computed by perception are completely omitted in `render_frame()`. Showing rotated bounding boxes on this thumbnail allows the operator to verify vision model tracking against the raw feed.
* **Missing Webcam Occlusion Indicator:**  
  When webcam occlusion is toggled via `W`, `s.webcam_on` is updated, but `render_frame()` provides no persistent visual indicator on the camera panel (unlike earlier versions that displayed `WEBCAM OCCLUDER ON`).
* **Hotkey Strip Contrast:**  
  The footer control instructions at y=884 (`text(img, "0-3 Version ...", 20, 884, .36, C["muted"])`) are rendered as plain unaccented muted gray text. Adding subtle keycaps or bracket contrast would significantly improve discoverability during live demonstrations.

---

### 4. Code Cleanup & Maintenance

* **Extensive Legacy Dead Code in `ui/live_hud.py`:**  
  `render_frame()` implements the new native layout directly, rendering ~450 lines of prior UI code unused:
  - `_chrome()` (lines 374–388) and global `_CHROME`
  - `_draw_camera()` (lines 401–466)
  - `badge_text()` (lines 468–476)
  - `_draw_persp()` (lines 478–543)
  - `_segmented()` (lines 544–566)
  - `_draw_rail()` (lines 568–636)
  - `_draw_tree_panel()` (lines 649–735)
  - `_metric_rows()` (lines 736–747)
  - `_draw_scoreboard()` (lines 753–865)
  - Dead primitive helpers: `glow()`, `soft_shadow()`, `_Offset`, `pill()`, `check_glyph()`, `cross_glyph()`.
* **Wasted Per-Frame Depth Masking:**  
  In `FrameBuilder.__call__` (line 1149), `depth_confidence(depth, bboxes)` performs `cv2.fillPoly` and boolean array slicing on every single simulation step. The resulting `s.depth_conf` is never displayed in `render_frame()`.
* **Test Suite Alignment:**  
  `tests/test_ui_unit.py` tests `hud.badge_text()`, assuming it drives the main status badge, but `render_frame()` implements its own inline outcome logic. Tests should cover the actual `render_frame()` status branches.

---

## Prioritized Action Items for Root

1. **[High] Fix or clean up force arrow projection:** Either adapt `project_world_points` to use `presentation_camera(ep)` camera extrinsics/intrinsics, or remove the unused `force_arrow_px` calculation from `FrameBuilder`.
2. **[Medium] Avoid mutating `ep.persp_renderer`:** Maintain the presentation renderer in `ep._mine_presentation['renderer']` to protect standard 640x480 `ep.render_perspective()` callers.
3. **[Medium] Prune dead code and per-frame compute in `live_hud.py`:** Remove legacy `_draw_*` functions and omit `depth_confidence` calculation if depth confidence bars are not part of the active design.
4. **[Low] Add bounding box overlays to Robot Vision:** Draw scaled bounding boxes and class colors onto `CARD["cam"]` thumbnail so perception accuracy can be visually inspected live.
