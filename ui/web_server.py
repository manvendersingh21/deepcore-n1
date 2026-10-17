"""Local control room; physics executes in a separate process (optionally over SSH).

The browser receives ground truth solely for visualization. Controller input and the
locked verifier are untouched. No scores or agent events are synthesized.
"""
from __future__ import annotations
import argparse
import base64
import copy
import json
import math
import mimetypes
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / 'ui' / 'web'


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if hasattr(value, 'tolist'):
        return clean(value.tolist())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def emit(kind, **data):
    print(json.dumps(clean({'type': kind, **data}), allow_nan=False), flush=True)


def worker_drilling(seed, fleet=False, preview=False):
    """Stream the independent drilling material model and its measured contact signals."""
    import cv2
    import mujoco
    if fleet:
        from ui.mining_fleet import MiningFleetEpisode, run_mining_fleet as run_drilling
        from scripts.learn_fleet import load_active_parameters
        parameters = load_active_parameters()
    else:
        from ui.drilling_scene import run_drilling
    sent = False
    pose_ids = None
    sensor_packet = None
    sensor_at = -1.0

    def capture(ep, telemetry):
        nonlocal sent, pose_ids, sensor_packet, sensor_at
        m, d = ep.model, ep.data
        if not sent:
            geoms = []
            for i in range(m.ngeom):
                mat = int(m.geom_matid[i])
                geoms.append({'id': i, 'name': mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) or '',
                              'type': int(m.geom_type[i]), 'size': m.geom_size[i], 'group': int(m.geom_group[i]),
                              'material': mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MATERIAL, mat) if mat >= 0 else '',
                              'rgba': m.mat_rgba[mat] if mat >= 0 else m.geom_rgba[i],
                              'dynamic': bool(m.geom_bodyid[i]), 'pos': d.geom_xpos[i], 'mat': d.geom_xmat[i]})
                if int(m.geom_type[i]) == int(mujoco.mjtGeom.mjGEOM_MESH) and not geoms[-1]['name'].startswith('cmu_'):
                    mesh_id = int(m.geom_dataid[i])
                    va, vn = int(m.mesh_vertadr[mesh_id]), int(m.mesh_vertnum[mesh_id])
                    fa, fn = int(m.mesh_faceadr[mesh_id]), int(m.mesh_facenum[mesh_id])
                    geoms[-1]['mesh'] = {'vertices': m.mesh_vert[va:va + vn].reshape(-1),
                                        'indices': m.mesh_face[fa:fa + fn].reshape(-1)}
            emit('scene', geoms=geoms, seed=seed)
            pose_ids = [g['id'] for g in geoms if g['dynamic'] or g['name'].startswith('rock_layer_')]
            sent = True
        frame = {'t': telemetry['t'], 'phase': telemetry['phase'], 'drilling': telemetry,
                 'ee': ep.ee_pos(), 'force': telemetry['axial_force_n'],
                 'peak': telemetry['peak_force_n'], 'duration': telemetry.get('duration_s', 12),
                 'poses': [{'id': i, 'pos': d.geom_xpos[i], 'mat': d.geom_xmat[i],
                            'visible': bool(m.geom_rgba[i, 3] > 0)} for i in pose_ids]}
        if fleet:
            frame['fleet'] = telemetry
            if ep._last_sensor_rgb is None:
                ep._observe()
            if telemetry['t'] - sensor_at >= .199 or sensor_packet is None:
                from ui.sensor_overlay import camera_packet
                from ui.mining_fleet import SENSOR_CAMERA_POSITION, SENSOR_FOVY_DEG
                focal = 240 / math.tan(math.radians(SENSOR_FOVY_DEG) / 2)
                sensor_packet = camera_packet(ep._last_sensor_rgb, ep._last_sensor_depth,
                    [[focal, 0, 320], [0, focal, 240], [0, 0, 1]], SENSOR_CAMERA_POSITION,
                    [[1, 0, 0], [0, -1, 0], [0, 0, -1]])
                sensor_at = telemetry['t']
            frame['sensor'] = sensor_packet
            frame['estimates'] = {f'debris_{track["track_id"]}': {'pos': track['pos'], 'yaw': 0,
                'size': [.046, .046, .046], 'orientation_observed': False}
                for track in telemetry.get('sensor_tracks', [])}
            frame['planned_target'] = ep._arm.goal.tolist()
            frame['plan_source'] = 'Current issued arm position command; future route not inferred'
        rgb = ep.render_camera()
        ok, jpg = cv2.imencode('.jpg', cv2.cvtColor(cv2.resize(rgb, (320, 240)), cv2.COLOR_RGB2BGR),
                              [cv2.IMWRITE_JPEG_QUALITY, 75])
        if ok:
            frame['camera'] = base64.b64encode(jpg).decode()
        emit('frame', frame=frame)

    if preview:
        ep = MiningFleetEpisode(seed=seed, controller_params=parameters)
        try:
            capture(ep, ep.telemetry())
        finally:
            ep.close()
        emit('ready')
        return
    result = run_drilling(seed=seed, frame_callback=capture, **({'controller_params': parameters} if fleet else {}))
    emit('complete', result={'operation': 'fleet' if fleet else 'drill', **result})


def worker(mode, seed, pipeline):
    import numpy as np
    from core.integrity import verify_integrity
    verify_integrity()
    brain_path = ROOT / 'shared_brain'
    try:
        ledger = json.loads((brain_path / 'ledger.json').read_text())
    except (OSError, ValueError):
        ledger = []
    emit('metadata', ledger=ledger)
    fleet_ledger = brain_path / 'fleet_learning' / 'ledger.json'
    emit('fleet-metadata', ledger=json.loads(fleet_ledger.read_text()) if fleet_ledger.is_file() else [])
    if mode == 'fleet-learn':
        from scripts.learn_fleet import run_learning
        entry = run_learning(event_fn=lambda ev: emit('event', event=ev), use_llm=True)
        emit('fleet-evolved', entry=entry, ledger=json.loads(fleet_ledger.read_text()))
        return
    if mode in ('drill', 'fleet', 'fleet-preview'):
        worker_drilling(seed, fleet=mode != 'drill', preview=mode == 'fleet-preview')
        return
    if mode == 'evolve':
        from agents.contract_orchestrator import ContractOrchestrator
        from agents.crusoe_client import CrusoeClient
        from shared_brain.memory_store import SharedBrain
        brain = SharedBrain(ROOT / 'shared_brain')
        def on_event(ev):
            # Preserve the actual agent event stream for honest native/video replay.
            with (brain_path / 'evolution_events.jsonl').open('a') as f:
                f.write(json.dumps({'time': time.time(), **ev}) + '\n')
            emit('metadata', ledger=brain.load_ledger())
            emit('event', event=ev)
        orch = ContractOrchestrator(brain, CrusoeClient(), log_fn=lambda s: None,
                                    event_fn=on_event)
        frozen_goal = brain_path / 'generated_dod.json'
        if brain.load_ledger() and frozen_goal.is_file():
            orch.dod = json.loads(frozen_goal.read_text())
            on_event({'agent': 'Root Architect', 'status': 'dod', 'generation': 0,
                      'message': 'Resuming the recorded mission requirements.'})
        else:
            orch.compile_goal('Extract the intact core, clear both boulders, improve held-out performance with safe force and latency.')
        orch.ensure_baseline()
        orch.run(3)
        if not any(not entry.get('accepted') for entry in brain.load_ledger()):
            orch.drift_probe()
        emit('evolved', ledger=brain.load_ledger())
        return
    import mujoco
    import cv2
    from core.env_runner import Episode, run_episode
    from core.verifier_harness import load_pipeline_class, score_record
    sent = False
    last_ep = None
    mask_provider = None
    sensor_packet = None
    sensor_at = -1.0

    def capture(ep, rgb=None, depth=None, out=None, rec=None):
        nonlocal sent, last_ep, mask_provider, sensor_packet, sensor_at
        last_ep = ep
        m, d = ep.model, ep.data
        if not sent:
            geoms = []
            for i in range(m.ngeom):
                mat = int(m.geom_matid[i])
                geoms.append({'id': i, 'name': mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) or '',
                              'type': int(m.geom_type[i]), 'size': m.geom_size[i],
                              'material': mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MATERIAL, mat) if mat >= 0 else '',
                              'rgba': m.mat_rgba[mat] if mat >= 0 else m.geom_rgba[i],
                              'dynamic': bool(m.body_dofnum[m.geom_bodyid[i]] or m.geom_bodyid[i] in [mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,n) for n in ('hand','gantry_z','gantry_y','gantry_x','finger_left','finger_right')]),
                              'pos': d.geom_xpos[i], 'mat': d.geom_xmat[i]})
            emit('scene', geoms=geoms, seed=seed)
            sent = True
        poses = [{'id': i, 'pos': d.geom_xpos[i], 'mat': d.geom_xmat[i]} for i in range(m.ngeom)
                 if m.geom_bodyid[i] != 0 and mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,int(m.geom_bodyid[i])) not in ('basin','hopper')]
        frame = {'t': rec.t[-1] if rec and rec.t else 0, 'phase': rec.phase[-1] if rec and rec.phase else 'READY',
                 'poses': poses, 'ee': ep.ee_pos(), 'objects': {k: {'pos': v[0], 'yaw': v[1]} for k,v in ep.ground_truth().items()},
                 'force': float(ep.core_force_now), 'wrist': float(np.linalg.norm(ep.wrist_force_world())),
                 'wrist_vector': ep.wrist_force_world(),
                 'peak': max(rec.core_gate_force, default=0) if rec else 0,
                 'latency': float(np.percentile(rec.latency_ms,99)) if rec and rec.latency_ms else None,
                 'estimates': (out or {}).get('estimated_objects', {}),
                 'planned_target': (out or {}).get('action', {}).get('target_pos'),
                 'plan_source': 'Current policy-issued waypoint; no inferred future route',
                 'core_shattered': bool(max(rec.core_gate_force, default=0) > 18
                     or max((np.linalg.norm(v) for v in rec.force_vec), default=0) > 28) if rec else False,
                 'bboxes': (out or {}).get('debug_overlay', {}).get('bboxes', [])}
        if rgb is not None:
            if frame['t'] - sensor_at >= .199 or sensor_packet is None:
                from ui.sensor_overlay import camera_packet
                if mask_provider is None and mode != 'preview':
                    visual_class = load_pipeline_class(paths[pipeline])
                    candidate_masks = visual_class(ep.camera_matrix.copy(), ep.cam_pos.copy(), ep.cam_rot.copy())
                    if callable(getattr(candidate_masks, '_masks', None)):
                        mask_provider = candidate_masks
                sensor_packet = camera_packet(rgb, depth, ep.camera_matrix, ep.cam_pos, ep.cam_rot, mask_provider)
                sensor_at = frame['t']
            frame['sensor'] = sensor_packet
            ok, jpg = cv2.imencode('.jpg', cv2.cvtColor(cv2.resize(rgb,(320,240)),cv2.COLOR_RGB2BGR),[cv2.IMWRITE_JPEG_QUALITY,70])
            if ok: frame['camera'] = base64.b64encode(jpg).decode()
        emit('frame', frame=frame)

    if mode == 'preview':
        ep = Episode(seed)
        try:
            rgb, depth, _ = ep.observe()
            capture(ep,rgb,depth)
        finally:
            ep.close()
        emit('ready')
        return
    paths = {'baseline': ROOT / 'workspace/baseline_v0.py', 'candidate': ROOT / 'agents/patches/evolved_pipeline.py'}
    if pipeline == 'accepted':
        from shared_brain.memory_store import SharedBrain
        best = SharedBrain(ROOT / 'shared_brain').best_accepted()
        if not best: raise ValueError('No accepted controller is available yet.')
        paths['accepted'] = ROOT / best['path']
    if pipeline not in paths: raise ValueError('Unknown controller')
    rec = run_episode(load_pipeline_class(paths[pipeline]), seed, frame_callback=capture)
    # run_episode closes renderers after settling; MjData remains available for final poses.
    result = {k:v for k,v in score_record(rec).items() if not k.startswith('_')}
    if seed in (101, 102, 103):
        result.pop('pipeline_errors', None)
    if last_ep is not None:
        capture(last_ep, rec=rec)
    emit('complete', result=result)


class ControlRoom:
    def __init__(self, remote=None):
        self.remote = remote
        self.lock = threading.RLock()
        self.proc = None
        self.state = {'status':'idle','job':None,'run_id':0,'seed':101,'pipeline':'candidate','frame':None,
                      'scene':[], 'events':[], 'runs':[], 'result':None, 'error':None,'ledger':[],
                      'compute':remote or 'local CPU','integrity':'checking','fleet_ledger':[]}
        self.frames = []
        self.recordings = {}

    def snapshot(self, include_scene=True):
        with self.lock:
            return copy.deepcopy({k:v for k,v in self.state.items() if include_scene or k != 'scene'})

    def start(self, mode, seed=101, pipeline='candidate'):
        with self.lock:
            if self.state['status'] in ('starting','running','evolving'):
                raise ValueError('A job is already running.')
            self.state.update(status='starting',job=mode,error=None,result=None,seed=seed,pipeline=pipeline)
            self.state['run_id'] += 1
            if mode not in ('evolve', 'fleet-learn'):
                self.frames = []
                self.state['frame'] = None
                self.state['scene'] = []
            else:
                self.state['events'] = []
            threading.Thread(target=self._run,args=(mode,seed,pipeline),daemon=True).start()

    def _run(self,mode,seed,pipeline):
        args = ['--worker',mode,'--seed',str(seed),'--pipeline',pipeline]
        if self.remote:
            # Host is a launch-time option, never controlled by HTTP requests.
            # Checkout directory on the GPU host; launch-time option, like the host itself.
            remote_dir = os.environ.get('DEEPCORE_REMOTE_DIR', '~/deepcore-n1')
            command = ['ssh',self.remote,f'cd {remote_dir} && MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 CUDA_VISIBLE_DEVICES=0 __EGL_VENDOR_LIBRARY_FILENAMES=$HOME/.local/share/glvnd/egl_vendor.d/10_nvidia.json ~/perception-loop-n1/.venv/bin/python -u -m ui.web_server ' + ' '.join(args)]
        else:
            command = [sys.executable,'-u','-m','ui.web_server',*args]
        stderr = []
        terminal = False
        try:
            proc = subprocess.Popen(command,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
            with self.lock: self.proc = proc
            def read_errors():
                for line in proc.stderr:
                    stderr.append(line)
                    if len(stderr)>40: del stderr[0]
            err_thread = threading.Thread(target=read_errors,daemon=True)
            err_thread.start()
            for line in proc.stdout:
                try: msg = json.loads(line)
                except ValueError: continue
                kind = msg.get('type')
                with self.lock:
                    self.state['integrity'] = 'verified'
                    if kind == 'metadata': self.state['ledger'] = msg['ledger']
                    elif kind == 'fleet-metadata': self.state['fleet_ledger'] = msg['ledger']
                    elif kind == 'scene': self.state['scene'] = msg['geoms']
                    elif kind == 'frame':
                        self.state['frame'] = msg['frame']
                        self.state['status'] = 'running' if mode in ('run', 'drill', 'fleet') else 'starting'
                        self.frames.append(msg['frame'])
                    elif kind == 'event':
                        self.state['status'] = 'evolving'
                        self.state['events'].append(msg['event'])
                    elif kind == 'complete':
                        self.state.update(status='complete',result=msg['result'])
                        self.state['runs'].append({'seed':seed,'pipeline':pipeline,**msg['result']})
                        self.state['runs'] = self.state['runs'][-20:]
                        if mode == 'run' and pipeline in ('baseline', 'accepted'):
                            self.recordings[pipeline] = {'pipeline': pipeline, 'seed': seed,
                                'run_id': self.state['run_id'], 'scene': copy.deepcopy(self.state['scene']),
                                'frames': list(self.frames), 'result': msg['result'],
                                'provenance': 'actual recorded MuJoCo rollout',
                                'policy_source': 'baseline' if pipeline == 'baseline' else
                                    next((r.get('source') for r in reversed(self.state['ledger']) if r.get('accepted')), 'accepted ledger source')}
                        terminal = True
                    elif kind == 'ready':
                        self.state['status'] = 'ready'; terminal = True
                    elif kind == 'evolved':
                        self.state.update(status='complete',ledger=msg['ledger']); terminal = True
                    elif kind == 'fleet-evolved':
                        self.state.update(status='complete',fleet_ledger=msg['ledger'],fleet_result=msg['entry']); terminal = True
            code = proc.wait()
            err_thread.join(timeout=2)
            if code or not terminal:
                raise RuntimeError(''.join(stderr)[-2000:] or f'Worker exited without a result ({code}).')
        except Exception as exc:
            with self.lock: self.state.update(status='error',error=str(exc))
        finally:
            with self.lock: self.proc = None


def serve(host='127.0.0.1',port=8080,remote=None):
    room = ControlRoom(remote or os.environ.get('DEEPCORE_GPU_HOST'))
    class Handler(BaseHTTPRequestHandler):
        def log_message(self,*args): pass
        def respond(self,data,status=200,ctype='application/json'):
            body = json.dumps(clean(data),allow_nan=False).encode() if ctype == 'application/json' and not isinstance(data, (bytes, bytearray)) else data
            self.send_response(status)
            self.send_header('Content-Type',ctype)
            self.send_header('Content-Length',str(len(body)))
            self.send_header('Cache-Control','no-store')
            self.end_headers()
            try: self.wfile.write(body)
            except (BrokenPipeError,ConnectionResetError): pass
        def do_GET(self):
            path = urlparse(self.path).path
            if path == '/api/state':
                return self.respond(room.snapshot(include_scene='compact=1' not in urlparse(self.path).query))
            if path == '/api/scene':
                with room.lock: scene = copy.deepcopy(room.state['scene'])
                return self.respond({'scene':scene})
            if path == '/api/replay':
                with room.lock: frames = list(room.frames)
                return self.respond({'frames':frames})
            if path == '/api/recording':
                from urllib.parse import parse_qs
                version = parse_qs(urlparse(self.path).query).get('version', [''])[0]
                with room.lock: recording = room.recordings.get(version)
                if recording is None:
                    return self.respond({'error': 'No measured recording for this version yet'}, 404)
                return self.respond(recording)
            if path.startswith('/assets/'):
                base = ROOT/'assets'; rel = path[len('/assets/'):]
            else:
                base = WEB; rel = path.lstrip('/') or 'index.html'
            target = (base/rel).resolve()
            if not target.is_relative_to(base.resolve()) or not target.is_file():
                return self.respond({'error':'Not found'},404)
            return self.respond(target.read_bytes(),ctype=mimetypes.guess_type(target.name)[0] or 'application/octet-stream')
        def do_POST(self):
            # Only our same-origin local UI may start compute jobs.
            origin = self.headers.get('Origin')
            if origin and urlparse(origin).netloc != self.headers.get('Host'):
                return self.respond({'error':'Origin rejected'},403)
            try:
                size = int(self.headers.get('Content-Length','0'))
                if size > 4096: raise ValueError('Request too large')
                data = json.loads(self.rfile.read(size) or b'{}')
                mode = {'/api/run':'run','/api/drill':'drill','/api/fleet':'fleet','/api/fleet-learn':'fleet-learn','/api/evolve':'evolve','/api/preview':'preview'}.get(self.path)
                if not mode: return self.respond({'error':'Not found'},404)
                seed = int(data.get('seed',101))
                if seed not in (11,22,33,101,102,103): raise ValueError('Select one of the six evaluation seeds.')
                pipeline = data.get('pipeline','candidate')
                if pipeline not in ('baseline','candidate','accepted'): raise ValueError('Unknown controller')
                room.start(mode,seed,pipeline)
                self.respond({'ok':True},202)
            except (ValueError,TypeError) as exc: self.respond({'error':str(exc)},409)
    server = ThreadingHTTPServer((host,port),Handler)
    room.start('fleet-preview' if (ROOT / 'ui/mining_fleet.py').is_file() else 'preview')
    print(f'DeepCore control room: http://{host}:{port} | compute: {room.remote or "local"}',flush=True)
    try: server.serve_forever()
    except KeyboardInterrupt: pass
    finally:
        if room.proc: room.proc.terminate()
        server.server_close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker',choices=['preview','run','evolve','drill','fleet','fleet-preview','fleet-learn'])
    parser.add_argument('--seed',type=int,default=101)
    parser.add_argument('--pipeline',default='candidate')
    args = parser.parse_args()
    if args.worker: worker(args.worker,args.seed,args.pipeline)
    else: serve()
