#!/usr/bin/env node
/** Inspect the two cached controller replays, interactive views, AR inputs, and render cadence. */
import { createRequire } from 'node:module';
import { existsSync } from 'node:fs';
import { mkdir, writeFile } from 'node:fs/promises';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { createHash } from 'node:crypto';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const argv = process.argv.slice(2);
function option(name, fallback) {
  const i = argv.indexOf(name);
  if (i < 0) return fallback;
  if (!argv[i + 1] || argv[i + 1].startsWith('--')) throw new Error(`${name} needs a value`);
  return argv[i + 1];
}
if (argv.includes('--help')) {
  console.log(`Verify the baseline and accepted interactive visualizer using cached real recordings only.
  node scripts/verify_visualizer.mjs [--url http://127.0.0.1:8080]
      [--out dev/visualizer_verification] [--sample-seconds 4] [--min-fps 60] [--headed]

This reads /api/recording?version=baseline|accepted, tests hotkeys 0/N/1/2/3,
checks the recorded RGB-D/estimate/command inputs, saves screenshots and measures
browser render cadence. It never starts a simulation. It fails when either cached
recording is missing or the measured target is not met.`);
  process.exit(0);
}
const url = option('--url', 'http://127.0.0.1:8080');
const output = resolve(ROOT, option('--out', 'dev/visualizer_verification'));
const sampleSeconds = Number(option('--sample-seconds', '4'));
const minFps = Number(option('--min-fps', '60'));
if (!Number.isFinite(sampleSeconds) || sampleSeconds < 1 || sampleSeconds > 30) throw new Error('--sample-seconds must be 1..30');
if (!Number.isFinite(minFps) || minFps <= 0 || minFps > 240) throw new Error('--min-fps must be 0..240');

const require = createRequire(import.meta.url);
let chromium;
for (const module of [process.env.PLAYWRIGHT_MODULE, 'playwright', '/tmp/deepcore-web-deps/node_modules/playwright'].filter(Boolean)) {
  try { ({ chromium } = require(module)); break; } catch { /* Try the next available Playwright install. */ }
}
if (!chromium) throw new Error('Playwright is missing. Install it or set PLAYWRIGHT_MODULE.');

const evidence = { url, started_at: new Date().toISOString(), target_fps: minFps, versions: {}, browser_errors: [] };
const checks = [];
const sha = data => createHash('sha256').update(data).digest('hex');
let browser;
try {
  await mkdir(output, { recursive: true });
  const executablePath = [process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH,
    '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome', chromium.executablePath()]
    .find(path => path && existsSync(path));
  browser = await chromium.launch({ executablePath, headless: !argv.includes('--headed'), args: ['--autoplay-policy=no-user-gesture-required'] });
  const context = await browser.newContext({ viewport: { width: 1600, height: 900 }, deviceScaleFactor: 1 });
  const page = await context.newPage();
  await page.route('**/api/**', async route => {
    if (route.request().method() === 'POST') return route.abort('blockedbyclient');
    return route.continue();
  });
  page.on('pageerror', error => evidence.browser_errors.push(error.message));
  page.on('console', message => { if (message.type() === 'error') evidence.browser_errors.push(message.text()); });

  const recordings = {};
  for (const version of ['baseline', 'accepted']) {
    const response = await context.request.get(`${url}/api/recording?version=${version}`);
    if (!response.ok()) throw new Error(`${version} recording unavailable: HTTP ${response.status()} (no compute was started)`);
    const recording = await response.json();
    if (recording.provenance !== 'actual recorded MuJoCo rollout') throw new Error(`${version} recording lacks actual MuJoCo provenance`);
    if (!Array.isArray(recording.frames) || recording.frames.length < 2 || !Array.isArray(recording.scene)) throw new Error(`${version} recording has no usable scene/frames`);
    const frames = recording.frames;
    const sensor = frames.filter(frame => frame.sensor?.provenance === 'actual_corrupted_rgbd').length;
    const estimated = frames.filter(frame => frame.estimates && Object.keys(frame.estimates).length > 0).length;
    const command = frames.filter(frame => Array.isArray(frame.planned_target) && frame.planned_target.length === 3 && frame.planned_target.every(Number.isFinite)).length;
    const camera = frames.filter(frame => typeof frame.camera === 'string' && frame.camera.length > 0).length;
    const cadence = frames.map(frame => frame.t).filter(Number.isFinite);
    if (!sensor) throw new Error(`${version} recording contains no actual corrupted RGB-D frames`);
    if (!estimated) throw new Error(`${version} recording contains no measured perception estimates`);
    if (!command) throw new Error(`${version} recording contains no recorded controller target commands`);
    recordings[version] = recording;
    evidence.versions[version] = {
      seed: recording.seed, run_id: recording.run_id, policy_source: recording.policy_source,
      provenance: recording.provenance, frame_count: frames.length, scene_geom_count: recording.scene.length,
      actual_corrupted_rgbd_frames: sensor, estimate_frames: estimated, planned_target_frames: command,
      camera_image_frames: camera,
      replay_duration_s: cadence.length > 1 ? cadence.at(-1) - cadence[0] : null,
    };
    checks.push(`${version}: actual recording, RGB-D, estimates, and target commands present`);
  }

  await page.goto(url, { waitUntil: 'domcontentloaded' });
  await page.locator('#scene canvas').waitFor({ state: 'visible', timeout: 30000 });
  await page.waitForFunction(() => window.__deepcoreRenderStats?.samples > 10, null, { timeout: 30000 });
  // Settle mesh/image decoding and shader compilation before measuring steady rendering.
  await page.waitForTimeout(4000);
  const body = page.locator('body');
  await body.click({ position: { x: 5, y: 5 } });
  await page.evaluate(() => document.activeElement?.blur());

  async function selectVersion(version) {
    await page.keyboard.press(version === 'baseline' ? '0' : 'n');
    await page.waitForFunction(v => window.__deepcoreReplay?.version === v && window.__deepcoreReplay?.total > 1,
      version, { timeout: 15000 });
    await page.waitForTimeout(250);
    if (await page.evaluate(() => Boolean(window.__deepcoreReplay?.active))) await page.locator('#replay-btn').click();
    await page.waitForFunction(() => window.__deepcoreReplay?.active === false);
  }
  async function sampleRender() {
    return page.evaluate(async durationMs => {
      const intervals = [];
      let prior = null, started = null;
      await new Promise(resolve => {
        const tick = now => {
          if (started === null) started = now;
          if (prior !== null) intervals.push(now - prior);
          prior = now;
          if (now - started >= durationMs) resolve(); else requestAnimationFrame(tick);
        };
        requestAnimationFrame(tick);
      });
      const sorted = [...intervals].sort((a, b) => a - b);
      const percentile = q => sorted[Math.min(sorted.length - 1, Math.floor((sorted.length - 1) * q))] ?? null;
      const mean = intervals.reduce((sum, value) => sum + value, 0) / Math.max(1, intervals.length);
      return { raf_frames: intervals.length, elapsed_ms: started === null ? 0 : prior - started,
        measured_fps: intervals.length * 1000 / Math.max(1, prior - started), mean_frame_ms: mean,
        median_frame_ms: percentile(.5), p95_frame_ms: percentile(.95), p99_frame_ms: percentile(.99),
        over_20ms: intervals.filter(value => value > 20).length,
        canvas: { width: document.querySelector('#scene canvas')?.width, height: document.querySelector('#scene canvas')?.height },
        render_stats: window.__deepcoreRenderStats };
    }, sampleSeconds * 1000);
  }

  for (const version of ['baseline', 'accepted']) {
    await selectVersion(version);
    const frames = recordings[version].frames;
    const arFrames = frames.filter(frame => frame.sensor?.provenance === 'actual_corrupted_rgbd' && frame.estimates && Object.keys(frame.estimates).length);
    const firstAr = frames.findIndex(frame => arFrames.includes(frame));
    if (firstAr < 0) throw new Error(`${version} has no frame pairing actual RGB-D with perception estimates`);
    await page.locator('#scrubber').evaluate((input, index) => { input.value = String(index); input.dispatchEvent(new Event('input', { bubbles: true })); }, firstAr);
    await page.waitForTimeout(1000);
    await page.locator('#replay-btn').click();
    const stats = await sampleRender();
    if (await page.evaluate(() => Boolean(window.__deepcoreReplay?.active))) await page.locator('#replay-btn').click();
    stats.measurement_scope = 'steady interactive rendering while replaying measured controller states';
    const rendered = await page.evaluate(() => ({
      version: window.__deepcoreReplay?.version,
      total: window.__deepcoreReplay?.total,
      canvasReady: Boolean(document.querySelector('#scene canvas')?.width && document.querySelector('#scene canvas')?.height),
      sensorImage: (() => { const image = document.querySelector('#sensor-image'); return Boolean(image && image.complete && image.naturalWidth > 0); })(),
      sensorPanelExpanded: document.querySelector('#sensor-panel')?.classList.contains('sensor-expanded') || false,
    }));
    if (rendered.version !== version || !rendered.canvasReady) throw new Error(`${version} replay failed to render in the interactive canvas`);
    evidence.versions[version].render = stats;
    evidence.versions[version].sensor_image_visible = rendered.sensorImage;
    evidence.versions[version].paired_sensor_estimate_frame = firstAr;
    await page.screenshot({ path: resolve(output, `${version}_orbit.png`) });

    await page.keyboard.press('1');
    await page.waitForTimeout(300);
    await page.screenshot({ path: resolve(output, `${version}_chase.png`) });
    const chasePng = await page.screenshot();
    await page.keyboard.press('2');
    await page.waitForTimeout(300);
    await page.screenshot({ path: resolve(output, `${version}_orbit_key.png`) });
    const orbitPng = await page.screenshot();
    if (sha(chasePng) === sha(orbitPng)) throw new Error(`${version}: hotkeys 1 and 2 produced identical views`);
    await page.keyboard.press('3');
    await page.waitForFunction(() => document.querySelector('#sensor-panel')?.classList.contains('sensor-expanded'));
    await page.screenshot({ path: resolve(output, `${version}_sensor.png`) });
    await page.keyboard.press('3');

    evidence.versions[version].view_keys = ['1: chase', '2: orbit', '3: sensor panel'];
    evidence.versions[version].camera_views_differ = true;
    evidence.versions[version].render_target_met = stats.measured_fps >= minFps;
    checks.push(`${version}: hotkeys 0/N and 1/2/3; ${stats.measured_fps.toFixed(2)} measured render FPS`);

  }
  const belowTarget = Object.entries(evidence.versions).filter(([, value]) => !value.render_target_met);
  if (belowTarget.length) throw new Error(belowTarget.map(([version, value]) => `${version}: ${value.render?.measured_fps.toFixed(2)} FPS below ${minFps}`).join('; '));
  if (evidence.browser_errors.length) throw new Error(`Browser errors: ${evidence.browser_errors.join('; ')}`);
  evidence.passed = true;
  evidence.checks = checks;
  console.log(`Verified baseline and accepted views at target ${minFps} FPS. Evidence: ${output}`);
} catch (error) {
  evidence.passed = false;
  evidence.error = error.stack || String(error);
  evidence.checks = checks;
  console.error(evidence.error);
  process.exitCode = 1;
} finally {
  evidence.finished_at = new Date().toISOString();
  await mkdir(output, { recursive: true }).catch(() => {});
  await writeFile(resolve(output, 'evidence.json'), JSON.stringify(evidence, null, 2) + '\n').catch(() => {});
  await browser?.close().catch(() => {});
}
