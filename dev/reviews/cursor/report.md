# DeepCore N1 — Browser UI interaction review (peer b)

**Scope:** `ui/web` control room at `http://127.0.0.1:8080`  
**Method:** Playwright (system Chrome) at desktop (1440×900), tablet (768×1024), mobile (390×844, 320×568). Did **not** trigger `Run excavation` or learning-loop jobs (GPU/sim owned elsewhere).  
**Session note:** Initial review hit a server with `status: complete`, 221 replay frames, and 2 runs in `/api/state`. The web process was restarted once during testing; afterward the app correctly showed `status: ready`, empty run history, and disabled replay until a new run completes.

## Executive summary

The mission-control UI is generally solid: tabs switch reliably, the WebGL site loads, camera presets update active state, and replay worked end-to-end when a completed episode was present. No blocking interaction failures were found on desktop or mobile for the flows exercised. The highest-impact issues are **mobile run-history readability** (wide data table) and a few **replay / copy inconsistencies** in the metrics panel.

## Blocking

_None identified in scoped interaction testing._

## Major

1. **Run history table is not mobile-friendly**  
   At `max-width: 600px`, `.data-table` sets `min-width: 650px` (`ui/web/style.css`). When `/api/state` includes runs, the history tab renders a six-column table that exceeds typical phone widths (~390px). The panel can scroll horizontally, but users must discover sideways scrolling to read outcomes—easy to miss and feels broken on first use.  
   **Suggestion:** Card-style rows on narrow breakpoints, fewer columns, or `min-width: 0` with wrapped cells.

## Minor

1. **Rubble cleared resets during replay**  
   After a completed run, metrics show final rubble count (e.g. `2 / 2`). Starting replay sets `#rubble` back to `—` (`renderFrame` replay branch in `ui/web/app.js`). Force/latency replay correctly; rubble looks like data loss.  
   **Suggestion:** Keep last measured rubble until replay ends, or label as “N/A during replay”.

2. **Phase detail mislabels non-baseline controllers**  
   `phase-detail` uses only `baseline` vs “Experimental rollout” for any other `pipeline` value (`candidate`, `accepted`). Selecting “Latest accepted” still shows experimental copy during live frames and replay.

3. **Connection / compute strip hidden on smaller desktops**  
   Below 1200px width, `.connection { display: none }` removes “SYSTEM CONNECTED” and compute node from the header. Tabs still work; this is a product choice but reduces diagnosability on laptops.

## Verified working

| Area | Result |
|------|--------|
| **Tabs** | Mission control, Learning lab, Run history show/hide panels; nav active state updates (all breakpoints). |
| **History empty CTA** | “Open mission control” (`data-tab="mission"`) switches back to mission panel. |
| **3D viewport** | WebGL canvas mounts; scene loading overlay clears when geometry is available. |
| **Camera presets** | Orbit / Close-up / Top-down toggle `.active` class; selection survives tab round-trip. |
| **Replay (completed run)** | Play enables with frames loaded; `window.__deepcoreReplay.active` becomes true; scrubber enables; playback advances and pauses at end. |
| **Layout** | No page-level horizontal overflow at 1440, 768, 390, 320 widths in automated checks; mission bar wraps on narrow screens. |
| **Fresh server state** | Replay and scrubber correctly disabled when no completed episode; timeline copy “Ready · select a controller and run”. |

## Not tested (out of scope)

- `Run excavation`, `Run learning loop`, and evolve flows (explicitly avoided).  
- Fullscreen control, orbit drag gestures, and multi-minute replay at 2× speed.  
- Behavior with populated learning ledger / agent event stream (no evolve run in this session).

## Files referenced

- `ui/web/index.html` — structure, tabs, transport, camera feed  
- `ui/web/app.js` — tab/replay/state polling logic  
- `ui/web/style.css` — responsive rules including `.data-table` min-width  
