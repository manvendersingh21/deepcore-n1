# Independent native mining UI review (Peer B)

**Scope:** Cart poses/labels, replay/telemetry, mesh transforms/loading, force/contact annotations in the listed mining UI sources. Static read-only review; no server, browser, or simulation runs.

**Verdict:** Two confirmed presentation bugs in the web fleet view. No confirmed cart pose-stream or mesh-transform defects in the Python→JSON→`scene.js` path from code inspection alone.

---

## Confirmed findings

### Medium — Fleet “DEBRIS BIN” label anchored at wrong world pose

**Location:** `ui/web/scene.js:234`

**Evidence:** In fleet mode, `load()` adds a fixed label at `(1.05, 0.35, 0.53)`. Measured collection targets rail-cart slots on the line at `y = 0.65` (`ui/rail_carts.py:8`, `ui/rail_carts.py:33-44`). The collector carry/release logic uses `cart_slot_positions` / `cart_interior_bounds` (`ui/mining_fleet.py:348-362`, `ui/mining_fleet.py:586-588`), not `(1.05, 0.35)`. `BIN_CENTER` in `ui/mining_fleet.py:36` is unused for placement; it only appears in cart-drive math (`ui/mining_fleet.py:494`).

**Impact:** Operators see a “DEBRIS BIN” tag offset from the actual open carts where debris is deposited (~0.3 m in Y), undermining trust in cart-world labeling.

**Minimal fix:** Anchor the label to a cart-centric pose (e.g. active cart center from `frame.fleet.cart_centers` + `cart_travel_m`, or a fixed offset over `CART_CENTERS[1]`), or rename/repoint the legend to “rail cart” and place at `(0.70, 0.65, ~0.55)`.

---

### Medium — Live timeline scrubber max stuck at 11 s while fleet/drill episodes are longer

**Locations:** `ui/web/index.html:31` (`max="219"`), `ui/web/app.js:85`, `ui/web/app.js:120`, `ui/web/app.js:213-214`

**Evidence:** During live streaming, scrubber value is set to `Math.round(frame.t * 20)` (20 Hz indexing) in both excavation and drilling render paths. HTML default `max` is `219` (~11 s). `max` is only updated after completion when replay buffers load (`app.js:213-214`). Fleet episodes advertise `duration_s: 80.0` (`ui/mining_fleet.py:30`, `ui/mining_fleet.py:605`); drilling HUD uses `duration` up to 12 s (`ui/web/app.js:106`, `ui/web_server.py:76`).

**Impact:** For live fleet (~1600 index units) or drill (~240), the range control stays capped at 219, so the thumb and timeline do not reflect true progress until the run finishes and replay is enabled.

**Minimal fix:** On job/mode start (or first fleet/drill frame), set `scrubber.max` from `Math.round((frame.duration || 12) * 20)` or from `replayFrames.length - 1` once known; keep HTML default for excavation-only if desired.

---

## Concerns (not confirmed as bugs)

1. **Cart motion in 3D** — Carts are actuated in MJCF (`ui/rail_carts.py:116-122`, `ui/mining_fleet.py:485-493`) and cart-body geoms should be streamed as dynamic poses (`ui/web_server.py:64`, `ui/web_server.py:72-78`). No cart-specific telemetry is bound in `app.js`, but pose updates should still move cart meshes. Not verified without a running episode.

2. **CMU survey alignment** — `ui/web/cmu_mine.js` places `tunnel.json` vertices at group identity; `metadata.json` documents a workcell transform, but sampled vertices already sit in workcell-scale coordinates. Misalignment with the fleet cell was not confirmed statically.

3. **Drill thrust/torque screen labels** — Fleet HUD labels use `frame.ee` (tool tip from `ee_pos()`, `ui/drilling_scene.py:239-241`) as anchor (`ui/web/scene.js:293-295`). Readable but may float at the cutter rather than the driller chassis.

4. **Grip contact arrows** — `gripArrows` are created in `update()` and not torn down on `load()` (`ui/web/scene.js:299-315`, `ui/web/scene.js:203-208`). Likely harmless (hidden when empty); possible minor scene clutter across mode reloads.

---

## Areas reviewed without confirmed defects

- **Mesh payload:** MuJoCo mesh geoms embed local vertices once and update via `geom_xpos` / `geom_xmat` (`ui/web_server.py:65-70`, `ui/web/scene.js:194-201`, `ui/web/scene.js:266-270`). CMU MJCF geoms are skipped in physics load and loaded separately (`ui/cmu_mine.py:48-49`, `ui/web/scene.js:216-217`, `ui/web/scene.js:250-259`).
- **Force/contact annotations:** Tool load arrow uses `contact_force_world_n` / `contact_position_m` from drilling telemetry (`ui/drilling_scene.py:388-389`, `ui/web/scene.js:277-289`). Collector contacts use `collector_contacts` and `collector_tcp_m` (`ui/mining_fleet.py:629-630`, `ui/web/scene.js:294-315`). Native HUD mirrors grip contacts (`ui/mining_hud.py:38-48`).
- **Replay after complete:** Buffer fetch and scrubber `max` reset look correct (`ui/web_server.py:303-305`, `ui/web/app.js:211-214`, `ui/web/app.js:46-62`).
- **Decor / robot visuals:** Non-colliding visuals attach to mobile bodies (`ui/robot_visuals.py:19-20`); should follow the same pose stream as physical bases.

---

## Limits

- No runtime WebGL, MuJoCo, or HTTP verification.
- `ui/web/cmu_mine.js` was consulted only via `scene.js` import for mesh loading.
- Excavation (non-fleet) paths were skimmed for regressions but were not the primary focus.
