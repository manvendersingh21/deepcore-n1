#!/usr/bin/env node
/** Record the actual cached baseline then accepted replay from the live WebGL canvas. */
import { createRequire } from 'node:module';
import { existsSync } from 'node:fs';
import { mkdir, writeFile } from 'node:fs/promises';
import { resolve, dirname, extname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { createHash } from 'node:crypto';
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
  console.log(`Record the real WebGL canvas at a requested 60 FPS while replaying cached real runs.
  node scripts/record_submission.mjs [--url http://127.0.0.1:8080]
      [--out submission_demo.mp4] [--timeout 900] [--headed] [--min-fps 60]

The script preflights both /api/recording entries before opening the browser, then
uses the UI hotkeys 0 and N to replay baseline followed by accepted. It never starts
compute and never duplicates frames to meet a frame-rate target. A WebM source and
capture evidence JSON are retained beside the output. MP4 conversion preserves the
captured timestamps; a below-target recording is still retained but exits as failed.`);
  process.exit(0);
}
const url = option('--url', 'http://127.0.0.1:8080');
const output = resolve(ROOT, option('--out', 'submission_demo.mp4'));
const timeoutMs = Number(option('--timeout', '900')) * 1000;
const minFps = Number(option('--min-fps', '60'));
if (!Number.isFinite(timeoutMs) || timeoutMs <= 0) throw new Error('--timeout must be positive');
if (!Number.isFinite(minFps) || minFps <= 0 || minFps > 240) throw new Error('--min-fps must be 0..240');

const require = createRequire(import.meta.url);
let chromium;
for (const module of [process.env.PLAYWRIGHT_MODULE, 'playwright', '/tmp/deepcore-web-deps/node_modules/playwright'].filter(Boolean)) {
  try { ({ chromium } = require(module)); break; } catch { /* Try the next available Playwright install. */ }
}
if (!chromium) throw new Error('Playwright is missing. Install it or set PLAYWRIGHT_MODULE.');

function findBinary(envNames, fixedPath, command) {
  const candidates = envNames.map(name => process.env[name]).filter(path => path && existsSync(path));
  if (fixedPath && existsSync(fixedPath)) candidates.push(fixedPath);
  const system = spawnSync(command, ['-version'], { encoding: 'utf8' });
  if (!system.error && system.status === 0) candidates.push(command);
  return candidates[0] || null;
}
const ffmpeg = findBinary(['DEEPCORE_FFMPEG_PATH', 'FFMPEG_PATH'], '/opt/homebrew/bin/ffmpeg', 'ffmpeg');
const ffprobe = findBinary(['DEEPCORE_FFPROBE_PATH', 'FFPROBE_PATH'], '/opt/homebrew/bin/ffprobe', 'ffprobe');
const sourceWebm = output.replace(/\.mp4$/i, '.source.webm');
const evidencePath = output.replace(/\.[^.]+$/, '.capture.json');
const artifactDir = resolve(dirname(output), `${output.split('/').at(-1).replace(/\.[^.]+$/, '')}_capture`);
const evidence = { url, output, source_webm: sourceWebm, started_at: new Date().toISOString(),
  capture_method: 'HTMLCanvasElement.captureStream(60) + MediaRecorder; captured renderer frames only',
  target_fps: minFps, browser_errors: [], recordings: [], no_compute_started: true };
let browser;
let page;
const chunks = [];
const sha = data => createHash('sha256').update(data).digest('hex');

async function inspectVideo(path) {
  if (!ffprobe) throw new Error('ffprobe is required to validate recorded timestamps and frame counts');
  const result = spawnSync(ffprobe, ['-v', 'error', '-select_streams', 'v:0', '-show_frames',
    '-show_entries', 'frame=best_effort_timestamp_time', '-of', 'json', path], { encoding: 'utf8', maxBuffer: 64 * 1024 * 1024 });
  if (result.error || result.status !== 0) throw new Error(`ffprobe failed for ${path}: ${result.error?.message || result.stderr}`);
  const times = (JSON.parse(result.stdout).frames || []).map(frame => Number(frame.best_effort_timestamp_time)).filter(Number.isFinite);
  const deltas = times.slice(1).map((time, i) => time - times[i]).filter(delta => delta > 0).sort((a, b) => a - b);
  const p = q => deltas[Math.min(deltas.length - 1, Math.floor((deltas.length - 1) * q))] ?? null;
  const duration = times.length > 1 ? times.at(-1) - times[0] : 0;
  return { frame_count: times.length, timestamp_duration_s: duration,
    measured_fps: duration > 0 ? (times.length - 1) / duration : 0,
    median_frame_interval_ms: p(.5) === null ? null : p(.5) * 1000,
    p95_frame_interval_ms: p(.95) === null ? null : p(.95) * 1000,
    p99_frame_interval_ms: p(.99) === null ? null : p(.99) * 1000,
    source_sha256: sha(await import('node:fs/promises').then(fs => fs.readFile(path))) };
}

try {
  await mkdir(dirname(output), { recursive: true });
  await mkdir(artifactDir, { recursive: true });
  const executablePath = [process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH,
    '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome', chromium.executablePath()]
    .find(path => path && existsSync(path));
  browser = await chromium.launch({ executablePath, headless: !argv.includes('--headed'), args: ['--autoplay-policy=no-user-gesture-required'] });
  const context = await browser.newContext({ viewport: { width: 1600, height: 900 }, deviceScaleFactor: 1 });

  // Check both recordings before navigating. Missing cached evidence must never trigger the UI's run fallback.
  for (const version of ['baseline', 'accepted']) {
    const response = await context.request.get(`${url}/api/recording?version=${version}`);
    if (!response.ok()) throw new Error(`${version} recording unavailable: HTTP ${response.status()} (no compute was started)`);
    const recording = await response.json();
    if (recording.provenance !== 'actual recorded MuJoCo rollout' || !Array.isArray(recording.frames) || recording.frames.length < 2 || !Array.isArray(recording.scene)) {
      throw new Error(`${version} cached recording is missing actual scene/replay evidence`);
    }
    const frames = recording.frames;
    const sensorFrames = frames.filter(frame => frame.sensor?.provenance === 'actual_corrupted_rgbd').length;
    const estimateFrames = frames.filter(frame => frame.estimates && Object.keys(frame.estimates).length > 0).length;
    const commandFrames = frames.filter(frame => Array.isArray(frame.planned_target) && frame.planned_target.length === 3 && frame.planned_target.every(Number.isFinite)).length;
    if (!sensorFrames || !estimateFrames || !commandFrames) throw new Error(`${version} cached frames lack actual RGB-D, estimates, or target commands`);
    evidence.recordings.push({ version, run_id: recording.run_id, seed: recording.seed, policy_source: recording.policy_source,
      provenance: recording.provenance, frames: frames.length, actual_corrupted_rgbd_frames: sensorFrames,
      estimate_frames: estimateFrames, planned_target_frames: commandFrames,
      replay_duration_s: frames.at(-1).t - frames[0].t });
  }

  page = await context.newPage();
  await page.route('**/api/**', async route => {
    if (route.request().method() === 'POST') return route.abort('blockedbyclient');
    return route.continue();
  });
  page.on('pageerror', error => evidence.browser_errors.push(error.message));
  page.on('console', message => { if (message.type() === 'error') evidence.browser_errors.push(message.text()); });
  await page.goto(url, { waitUntil: 'domcontentloaded' });
  await page.locator('#scene canvas').waitFor({ state: 'visible', timeout: 30000 });
  await page.waitForFunction(() => window.__deepcoreRenderStats?.samples > 10, null, { timeout: 30000 });
  await page.waitForTimeout(4000);
  await page.evaluate(() => document.activeElement?.blur());
  await page.exposeFunction('__deepcoreStoreSubmissionChunk', base64 => { chunks.push(Buffer.from(base64, 'base64')); });
  evidence.canvas = await page.evaluate(async () => {
    const canvas = document.querySelector('#scene canvas');
    if (!canvas || typeof canvas.captureStream !== 'function') throw new Error('WebGL canvas captureStream is unavailable');
    const stream = canvas.captureStream(60);
    const track = stream.getVideoTracks()[0];
    const supported = ['video/webm;codecs=vp9', 'video/webm;codecs=vp8', 'video/webm'].find(type => MediaRecorder.isTypeSupported(type));
    if (!supported) throw new Error('No WebM MediaRecorder codec is supported');
    const recorder = new MediaRecorder(stream, { mimeType: supported, videoBitsPerSecond: 10000000 });
    let chain = Promise.resolve();
    recorder.addEventListener('dataavailable', event => {
      if (!event.data?.size) return;
      chain = chain.then(async () => {
        const buffer = await event.data.arrayBuffer();
        let binary = '';
        const bytes = new Uint8Array(buffer);
        for (let offset = 0; offset < bytes.length; offset += 0x8000) binary += String.fromCharCode(...bytes.subarray(offset, offset + 0x8000));
        await window.__deepcoreStoreSubmissionChunk(btoa(binary));
      });
    });
    window.__deepcoreSubmission = { recorder, stream, chain: () => chain, mimeType: supported, frames: [] };
    recorder.start(1000);
    return { width: canvas.width, height: canvas.height, mime_type: supported, track_settings: track.getSettings(),
      capture_request_fps: 60, started_at: new Date().toISOString() };
  });

  async function replay(version) {
    const started = Date.now();
    await page.keyboard.press(version === 'baseline' ? '0' : 'n');
    await page.waitForFunction(v => window.__deepcoreReplay?.version === v && window.__deepcoreReplay?.active === true,
      version, { timeout: 15000 });
    await page.screenshot({ path: resolve(artifactDir, `${version}_replay.png`) });
    const end = Date.now() + timeoutMs;
    while (Date.now() < end) {
      const status = await page.evaluate(() => ({ active: Boolean(window.__deepcoreReplay?.active),
        frame: window.__deepcoreReplay?.frame, total: window.__deepcoreReplay?.total }));
      if (!status.active && status.total > 1 && status.frame >= status.total - 1) {
        return { wall_duration_s: (Date.now() - started) / 1000, replay_frames: status.total, final_frame: status.frame };
      }
      await page.waitForTimeout(100);
    }
    throw new Error(`${version} replay did not finish before timeout; partial WebM chunks will be retained`);
  }
  evidence.recordings[0].playback = await replay('baseline');
  evidence.recordings[1].playback = await replay('accepted');
  await page.waitForTimeout(500);
  await page.evaluate(async () => {
    const recording = window.__deepcoreSubmission?.recorder;
    if (!recording || recording.state !== 'recording') throw new Error('Canvas recorder stopped unexpectedly');
    await new Promise((resolve, reject) => {
      recording.addEventListener('stop', resolve, { once: true });
      recording.addEventListener('error', event => reject(event.error || new Error('MediaRecorder error')), { once: true });
      recording.stop();
    });
    await window.__deepcoreSubmission.chain();
    window.__deepcoreSubmission.stream.getTracks().forEach(track => track.stop());
  });
  if (evidence.browser_errors.length) throw new Error(`Browser errors: ${evidence.browser_errors.join('; ')}`);
  if (!chunks.length) throw new Error('MediaRecorder produced no canvas frames');

  const source = Buffer.concat(chunks);
  await writeFile(sourceWebm, source);
  evidence.source_bytes = source.length;
  evidence.source_sha256 = sha(source);
  evidence.source_webm_timing = await inspectVideo(sourceWebm);
  evidence.target_met_in_webm = evidence.source_webm_timing.measured_fps >= minFps;
  if (extname(output).toLowerCase() === '.mp4') {
    if (!ffmpeg) throw new Error(`ffmpeg is missing; actual canvas capture remains at ${sourceWebm}`);
    const conversion = spawnSync(ffmpeg, ['-hide_banner', '-loglevel', 'error', '-y', '-i', sourceWebm,
      '-map', '0:v:0', '-an', '-c:v', 'libx264', '-preset', 'medium', '-crf', '18', '-pix_fmt', 'yuv420p',
      '-fps_mode', 'passthrough', '-movflags', '+faststart', output], { encoding: 'utf8', maxBuffer: 8 * 1024 * 1024 });
    if (conversion.error || conversion.status !== 0) throw new Error(`MP4 conversion failed; source WebM remains at ${sourceWebm}: ${conversion.error?.message || conversion.stderr}`);
    evidence.mp4_bytes = (await import('node:fs/promises')).stat(output).then(stat => stat.size);
    evidence.mp4_bytes = await evidence.mp4_bytes;
    evidence.mp4_timing = await inspectVideo(output);
    if (evidence.mp4_timing.frame_count !== evidence.source_webm_timing.frame_count) {
      throw new Error(`MP4 conversion changed frame count (${evidence.source_webm_timing.frame_count} -> ${evidence.mp4_timing.frame_count}); cadence evidence is not accepted`);
    }
    evidence.target_met_in_mp4 = evidence.mp4_timing.measured_fps >= minFps;
  } else {
    await writeFile(output, source);
    evidence.target_met_in_mp4 = evidence.target_met_in_webm;
  }
  evidence.passed = evidence.target_met_in_mp4;
  evidence.finished_at = new Date().toISOString();
  await writeFile(evidencePath, JSON.stringify(evidence, null, 2) + '\n');
  console.log(`Recorded baseline then accepted canvas replay: ${output}`);
  console.log(`Measured cadence: ${evidence.mp4_timing?.measured_fps ?? evidence.source_webm_timing.measured_fps.toFixed(3)} FPS; target ${minFps} FPS`);
  if (!evidence.passed) {
    console.error(`Recorded source cadence is below ${minFps} FPS; no duplicated frames were added.`);
    process.exitCode = 1;
  }
} catch (error) {
  evidence.passed = false;
  evidence.error = error.stack || String(error);
  evidence.finished_at = new Date().toISOString();
  console.error(evidence.error);
  process.exitCode = 1;
} finally {
  // Preserve actual partial evidence even if a hotkey, browser, or timeout failed.
  if (chunks.length && !evidence.source_bytes) {
    await page?.evaluate(async () => {
      const capture = window.__deepcoreSubmission;
      if (capture?.recorder?.state === 'recording') {
        await new Promise(resolve => { capture.recorder.addEventListener('stop', resolve, { once: true }); capture.recorder.stop(); });
        await capture.chain();
        capture.stream.getTracks().forEach(track => track.stop());
      }
    }).catch(() => {});
    const partial = Buffer.concat(chunks);
    await writeFile(sourceWebm, partial).catch(() => {});
    evidence.partial_source_bytes = partial.length;
    evidence.partial_source_sha256 = sha(partial);
  }
  await writeFile(evidencePath, JSON.stringify(evidence, null, 2) + '\n').catch(() => {});
  await browser?.close().catch(() => {});
}
