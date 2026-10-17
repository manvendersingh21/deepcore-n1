# Browser demo capture

The browser recorder captures the real control room at 1600 × 1000. Each segment
replays the server's recorded episode poses and measurements. It does not inject
simulation events, learning progress, or scores.

Dependencies: Node.js, Playwright with Chromium, and ffmpeg for MP4 output.
The recorder finds Playwright in the project, at `PLAYWRIGHT_MODULE`, or at
`/tmp/deepcore-web-deps/node_modules/playwright`.
On macOS, the existing Google Chrome app is used when bundled Chromium is
unavailable. Set `PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH` to select another browser.

Start the web server with the GPU host before recording:

```sh
DEEPCORE_GPU_HOST=cis-a6000 python -m ui.web_server
```

Capture the current completed run without starting simulation:

```sh
node scripts/record_web_demo.mjs
```

Run baseline and candidate sequentially on holdout seed 101, then record each
complete replay:

```sh
node scripts/record_web_demo.mjs --run --seed 101
```

The default output is `dev/demo_control_room.mp4`. Raw WebM segments, first and
last screenshots, and an `evidence.json` file containing the actual result and
frame count are retained in a timestamped `dev/web_capture_*` directory.
Camera presets change every seven seconds while replay is active. The recorder
holds the real final result for three seconds. It exits unsuccessfully if the
worker, browser, replay, or video conversion fails; the error and partial video
remain available for diagnosis. It never restarts a worker on timeout.

The frontend must expose replay activity either as
`window.__deepcoreReplay.active` or as `#replay-btn[data-replaying="true"]`.
Recording live evolution is deliberately separate: only start evolution when
the real model service is configured, and display the server's actual events.
