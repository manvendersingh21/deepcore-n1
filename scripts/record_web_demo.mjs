#!/usr/bin/env node
/** Record the real control room. No metrics, poses, or agent events are injected. */
import { createRequire } from 'node:module';
import { existsSync, readdirSync } from 'node:fs';
import { mkdir, writeFile } from 'node:fs/promises';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawnSync } from 'node:child_process';

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..');
const argv = process.argv.slice(2);
function option(name, fallback) {
  const i = argv.indexOf(name);
  if (i < 0) return fallback;
  if (!argv[i + 1] || argv[i + 1].startsWith('--')) throw new Error(`${name} needs a value`);
  return argv[i + 1];
}
if (argv.includes('--help')) {
  console.log(`Record the real DeepCore browser UI.
  node scripts/record_web_demo.mjs [--run] [--seed 101]
      [--url http://127.0.0.1:8080] [--out dev/demo_control_room.mp4]
      [--pipelines baseline,candidate] [--operation excavate|drill|fleet] [--timeout 900] [--headed]

Default: replay the current completed run; never starts compute.
--run: execute the selected pipelines sequentially using the browser Run button,
       wait for each real result, and record each complete replay.
Install Playwright in the project or set PLAYWRIGHT_MODULE. The temporary install
at /tmp/deepcore-web-deps/node_modules/playwright is also supported.
MP4 conversion requires ffmpeg; raw WebM segments are retained in all cases.`);
  process.exit(0);
}
const url = option('--url', 'http://127.0.0.1:8080');
const output = resolve(ROOT, option('--out', 'dev/demo_control_room.mp4'));
const seed = Number(option('--seed', '101'));
const timeout = Number(option('--timeout', '900')) * 1000;
const operation = option('--operation', 'excavate');
if (!['excavate', 'drill', 'fleet'].includes(operation)) throw new Error('Invalid operation');
const pipelines = operation === 'excavate' ? option('--pipelines', 'baseline,candidate').split(',') : ['candidate'];
if (!Number.isInteger(seed) || ![11, 22, 33, 101, 102, 103].includes(seed)) throw new Error('Invalid evaluation seed');
if (!Number.isFinite(timeout) || timeout <= 0) throw new Error('--timeout must be positive');
if (pipelines.some(p => !['baseline', 'candidate', 'accepted'].includes(p))) throw new Error('Invalid pipeline');
const require = createRequire(import.meta.url);
let chromium;
for (const module of [process.env.PLAYWRIGHT_MODULE, 'playwright', '/tmp/deepcore-web-deps/node_modules/playwright'].filter(Boolean)) {
  try { ({ chromium } = require(module)); break; } catch { /* Try the next installation. */ }
}
if (!chromium) throw new Error('Playwright is missing. Install it or set PLAYWRIGHT_MODULE.');

const stamp = new Date().toISOString().replace(/[:.]/g, '-');
const artifactDir = resolve(dirname(output), `web_capture_${stamp}`);
await mkdir(artifactDir, { recursive: true });
const browserErrors = [];
const evidence = { url, started_at: new Date().toISOString(), mode: argv.includes('--run') ? 'run-and-replay' : 'existing-replay', captures: [], browser_errors: browserErrors };
const segments = [];
const executablePath = [process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH,
  '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome', chromium.executablePath()]
  .find(path => path && existsSync(path));
const browser = await chromium.launch({ executablePath, headless: !argv.includes('--headed'), args: ['--autoplay-policy=no-user-gesture-required'] });
const control = await browser.newContext({ viewport: { width: 1600, height: 1000 }, deviceScaleFactor: 1 });
const page = await control.newPage();

function hasEncoder(ffmpeg, name) {
  const result = spawnSync(ffmpeg, ['-hide_banner', '-encoders'], { encoding: 'utf8' });
  return !result.error && result.status === 0 && new RegExp(`\\b${name}\\b`).test(result.stdout || '');
}

function findFfmpeg({ requireH264 = false } = {}) {
  const candidates = [process.env.DEEPCORE_FFMPEG_PATH, process.env.FFMPEG_PATH].filter(path => path && existsSync(path));
  const system = spawnSync('ffmpeg', ['-version'], { encoding: 'utf8' });
  if (!system.error && system.status === 0) candidates.push('ffmpeg');

  // Playwright keeps a small ffmpeg build beside its browser cache. Use that
  // binary for WebM assembly when the host does not have ffmpeg on PATH.
  let cacheRoot = dirname(chromium.executablePath());
  for (let i = 0; i < 5; i++) cacheRoot = dirname(cacheRoot);
  const binary = process.platform === 'darwin' ? 'ffmpeg-mac'
    : process.platform === 'win32' ? 'ffmpeg-win32.exe' : 'ffmpeg-linux';
  try {
    candidates.push(...readdirSync(cacheRoot, { withFileTypes: true })
      .filter(entry => entry.isDirectory() && /^ffmpeg-\d+$/.test(entry.name))
      .map(entry => resolve(cacheRoot, entry.name, binary))
      .filter(path => existsSync(path))
      .sort((a, b) => Number(b.match(/ffmpeg-(\d+)/)?.[1] || 0) - Number(a.match(/ffmpeg-(\d+)/)?.[1] || 0)));
  } catch { /* No Playwright ffmpeg cache on this host. */ }
  return candidates.find(path => !requireH264 || hasEncoder(path, 'libx264')) || null;
}

function monitor(target) {
  target.on('pageerror', error => browserErrors.push(error.message));
  target.on('console', message => { if (message.type() === 'error') browserErrors.push(message.text()); });
}
monitor(page);
async function state() {
  const response = await control.request.get(`${url}/api/state`);
  if (!response.ok()) throw new Error(`State request failed: HTTP ${response.status()}`);
  return response.json();
}
async function waitForResult(runId = null) {
  const deadline = Date.now() + timeout;
  let lastStatus;
  while (Date.now() < deadline) {
    const current = await state();
    if (current.status !== lastStatus) {
      console.log(`Worker: ${current.status} | ${current.pipeline} seed ${current.seed}`);
      lastStatus = current.status;
    }
    if (current.status === 'error') throw new Error(current.error || 'Simulation failed');
    if (current.status === 'complete' && current.result && (runId === null || current.run_id > runId)) return current;
    await page.waitForTimeout(500);
  }
  throw new Error(`No completed episode within ${timeout / 1000}s. The worker was not restarted.`);
}
async function waitUntilIdle() {
  const deadline = Date.now() + timeout;
  while (Date.now() < deadline) {
    const current = await state();
    if (current.status === 'error') throw new Error(current.error || 'Worker failed');
    if (['idle', 'ready', 'complete'].includes(current.status)) return current;
    await page.waitForTimeout(500);
  }
  throw new Error('Worker is still busy; no second simulation was started.');
}
async function capture(current) {
  const captureOperation = current.job === 'fleet' ? 'fleet' : current.job === 'drill' ? 'drill' : 'excavate';
  const label = `${captureOperation === 'excavate' ? current.pipeline : captureOperation}_seed${current.seed}`;
  const replayResponse = await control.request.get(`${url}/api/replay`);
  if (!replayResponse.ok()) throw new Error(`Replay request failed: HTTP ${replayResponse.status()}`);
  const replay = await replayResponse.json();
  if (!Array.isArray(replay.frames) || replay.frames.length < 2) throw new Error('Completed run has no replay frames');
  const context = await browser.newContext({ viewport: { width: 1600, height: 1000 }, deviceScaleFactor: 1, recordVideo: { dir: artifactDir, size: { width: 1600, height: 1000 } } });
  const target = await context.newPage();
  monitor(target);
  const video = target.video();
  try {
    await target.goto(url, { waitUntil: 'domcontentloaded' });
    await target.locator('#replay-btn').waitFor();
    await target.waitForFunction(() => !document.querySelector('#replay-btn')?.disabled);
    await target.locator('[data-camera="overview"]').click();
    await target.waitForTimeout(1000);
    await target.locator('#replay-btn').click();
    await target.waitForFunction(() => window.__deepcoreReplay?.active === true || document.querySelector('#replay-btn')?.dataset.replaying === 'true', { }, { timeout: 10000 });
    await target.screenshot({ path: resolve(artifactDir, `${label}_start.png`) });
    const replayDeadline = Date.now() + timeout;
    let angle = 0;
    let nextAngleAt = Date.now() + 7000;
    while (true) {
      const playing = await target.evaluate(() => window.__deepcoreReplay?.active ?? (document.querySelector('#replay-btn')?.dataset.replaying === 'true'));
      if (!playing) break;
      if (Date.now() > replayDeadline) throw new Error('Replay did not finish before timeout');
      if (Date.now() >= nextAngleAt) {
        angle++;
        const camera = ['overview', 'close', 'top'][angle % 3];
        await target.locator(`[data-camera="${camera}"]`).click();
        nextAngleAt = Date.now() + 7000;
      }
      await target.waitForTimeout(200);
    }
    await target.locator('[data-camera="overview"]').click();
    await target.waitForTimeout(3000);
    await target.screenshot({ path: resolve(artifactDir, `${label}_result.png`) });
    evidence.captures.push({ operation: captureOperation, pipeline: current.pipeline, seed: current.seed, run_id: current.run_id, frame_count: replay.frames.length, result: current.result, runs: current.runs });
  } catch (error) {
    await target.screenshot({ path: resolve(artifactDir, `${label}_error.png`) }).catch(() => {});
    throw error;
  } finally {
    await context.close();
    const segment = resolve(artifactDir, `${label}.webm`);
    await video.saveAs(segment);
    segments.push(segment);
  }
  console.log(`Recorded ${label}: ${replay.frames.length} real frames`);
}
try {
  await page.goto(url, { waitUntil: 'domcontentloaded' });
  if (argv.includes('--run')) {
    for (const pipeline of pipelines) {
      const before = await waitUntilIdle();
      await page.locator('#operation').selectOption(operation);
      await page.locator('#seed').selectOption(String(seed));
      if (operation === 'excavate') await page.locator('#pipeline').selectOption(pipeline);
      await page.locator('#run-btn').click();
      await capture(await waitForResult(before.run_id));
    }
  } else {
    const current = await state();
    if (!['starting', 'running', 'complete'].includes(current.status)) throw new Error('No completed or running episode to replay. Use --run to authorize baseline and candidate runs.');
    await capture(await waitForResult());
  }
  if (browserErrors.length) throw new Error(`Browser reported errors: ${browserErrors.join('; ')}`);
  const list = resolve(artifactDir, 'segments.txt');
  await writeFile(list, segments.map(path => `file '${path.replaceAll("'", "'\\''")}'`).join('\n') + '\n');
  if (output.endsWith('.mp4')) {
    const ffmpeg = findFfmpeg({ requireH264: true });
    if (!ffmpeg) throw new Error(`An ffmpeg build with libx264 is required for MP4 output; install one or set FFMPEG_PATH (WebM segments remain in ${artifactDir})`);
    const conversion = spawnSync(ffmpeg, ['-hide_banner', '-loglevel', 'error', '-y', '-f', 'concat', '-safe', '0', '-i', list, '-c:v', 'libx264', '-preset', 'medium', '-crf', '18', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', '-an', output], { encoding: 'utf8' });
    if (conversion.error || conversion.status !== 0) throw new Error(`MP4 conversion failed; WebM segments remain in ${artifactDir}: ${conversion.error?.message || conversion.stderr}`);
  } else if (segments.length === 1) {
    await videoCopy(segments[0], output);
  } else {
    const ffmpeg = findFfmpeg();
    if (!ffmpeg) throw new Error(`ffmpeg is missing; install it or set FFMPEG_PATH (WebM segments remain in ${artifactDir})`);
    const conversion = spawnSync(ffmpeg, ['-hide_banner', '-loglevel', 'error', '-y', '-f', 'concat', '-safe', '0', '-i', list, '-c', 'copy', output], { encoding: 'utf8' });
    if (conversion.error || conversion.status !== 0) throw new Error(`WebM assembly failed: ${conversion.error?.message || conversion.stderr}`);
  }
  evidence.output = output;
  console.log(`Saved ${output}`);
} catch (error) {
  evidence.error = error.stack || String(error);
  console.error(evidence.error);
  process.exitCode = 1;
} finally {
  evidence.finished_at = new Date().toISOString();
  await writeFile(resolve(artifactDir, 'evidence.json'), JSON.stringify(evidence, null, 2) + '\n');
  await browser.close();
}
async function videoCopy(source, target) {
  const { copyFile } = await import('node:fs/promises');
  await copyFile(source, target);
}
