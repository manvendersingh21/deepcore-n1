import { MineScene } from './scene.js';
const $ = id => document.getElementById(id);
const fmt = (n, digits=1) => typeof n === 'number' && Number.isFinite(n) ? n.toFixed(digits) : '—';
const esc = s => String(s ?? '').replace(/[&<>"']/g, c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
 let mine;
try { mine = new MineScene($('scene')); } catch(e) { notice(`3D view unavailable: ${e.message}. Try a browser with WebGL enabled.`); }
let state, sceneRun = -1, previousRun = -1, replayFrames = [], replayActive = false, replayIndex=0, lastTime=0, replayClock=0;
let forceHistory=[], seenTime=-1, completedRun=-1, busy=false, lastEvents='', lastRuns='', lastLedger='';
let replayResult=null, replayMetadata=null, pendingVersion=null;
const versionRecordings=new Map();
window.__deepcoreReplay = {active:false,frame:0,total:0};
const phaseNames = {
 READY:'Ready for excavation', SCAN:'Surveying the dig site', APPROACH:'Aligning the gripper', DESCEND:'Descending toward the target',
 GRASP:'Securing the grasp', LIFT:'Lifting the material', TRANSFER:'Transporting material', PLACE:'Placing into the recovery zone',
 RELEASE:'Releasing the material', RETREAT:'Withdrawing the gripper', DONE:'Excavation complete',
 LUNGE_STRAIGHT_DOWN:'Baseline: descending into the rubble', CARRY_TO_HOPPER:'Moving toward the recovery hopper',
};
function notice(message) { $('notice').textContent=message; $('notice').hidden=!message; }
$('scene').addEventListener('mine-load-error', e=>notice(`Mine geometry unavailable: ${e.detail}`));
function tab(name) { document.querySelectorAll('.tab-panel').forEach(el=>el.hidden=el.id!==name); document.querySelectorAll('.nav').forEach(el=>el.classList.toggle('active',el.dataset.tab===name)); if(name==='mission') mine?.resize(); }
document.querySelectorAll('[data-tab]').forEach(b=>b.addEventListener('click',()=>tab(b.dataset.tab)));
document.querySelectorAll('[data-camera]').forEach(b=>b.addEventListener('click',()=>{mine?.cameraMode(b.dataset.camera);document.querySelectorAll('[data-camera]').forEach(x=>x.classList.toggle('active',x===b));}));
$('fullscreen-btn').onclick=async()=>{try{if(document.fullscreenElement)await document.exitFullscreen();else await $('viewport').requestFullscreen();}catch(e){notice(e.message);}};
$('inspector-btn').onclick=()=>{
 const open=document.querySelector('.workspace').classList.toggle('show-inspector');
 $('inspector-btn').setAttribute('aria-expanded',String(open));
};
async function switchVersion(version) {
 if(busy){notice('Wait for the current physics run to finish before switching versions.');return;}
 notice('');
 let recording=versionRecordings.get(version);
 if(!recording) {
  const response=await fetch(`/api/recording?version=${version}`);
  if(response.ok){recording=await response.json();versionRecordings.set(version,recording);}
 }
 if(!recording) {
  pendingVersion=version;$('operation').value='excavate';$('pipeline').value=version;
  notice(`Recording a real ${version==='baseline'?'v0':'accepted controller'} rollout. The version key will replay it instantly once recorded.`);
  await run('run');return;
 }
 stopReplay();replayFrames=recording.frames;replayResult=recording.result;replayMetadata={seed:recording.seed,pipeline:version,source:recording.policy_source};replayIndex=0;
 mine?.load(recording.scene);sceneRun=state.run_id;
 document.body.classList.remove('fleet-mode');$('fleet-workflow').hidden=true;
 $('site-name').textContent=version==='baseline'?'v0 · Baseline':'vN · Accepted controller';
 $('activity-heading').textContent='CONTROLLER EVIDENCE';
 $('scene-legend').innerHTML='<span><i class="swatch cyan"></i>RGB-D / estimated pose</span><span><i class="swatch orange"></i>Measured force</span>';
 $('scrubber').max=String(replayFrames.length-1);$('scrubber').disabled=false;$('replay-btn').disabled=false;
 $('operation').value='excavate';$('pipeline').value=version;setBusy(false);
 $('integrity').textContent=`Recorded seed ${recording.seed} · ${recording.policy_source}`;
 window.__deepcoreReplay.version=version;
 renderFrame(replayFrames[0],true);startReplay();
}
document.addEventListener('keydown',event=>{
 if(event.target.closest('input,select,textarea')||event.ctrlKey||event.metaKey||event.altKey)return;
 const key=event.key.toLowerCase();
 if(key==='1'){mine?.cameraMode('chase');event.preventDefault();}
 else if(key==='2'){mine?.cameraMode('overview');event.preventDefault();}
 else if(key==='3'){$('sensor-panel').classList.toggle('sensor-expanded');event.preventDefault();}
 else if(key==='0'||key==='n'){switchVersion(key==='0'?'baseline':'accepted').catch(e=>notice(e.message));event.preventDefault();}
});
async function request(path,body) {
 const res=await fetch(path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
 const data=await res.json(); if(!res.ok) throw new Error(data.error || 'Request failed'); return data;
}
async function run(mode) {
 if(busy)return;
 busy=true; setBusy(true); notice(''); stopReplay(); replayFrames=[];
 try { await request(`/api/${mode}`,{seed:Number($('seed').value),pipeline:$('pipeline').value}); }
 catch(e){notice(e.message);busy=false;setBusy(false);}
}
$('run-btn').onclick=()=>run(({drill:'drill',fleet:'fleet'})[$('operation')?.value]||'run');
$('operation').onchange=()=>setBusy(busy);
$('evolve-btn').onclick=()=>run('evolve');
$('learning-run-btn').onclick=()=>{tab('learning');run('evolve');};
$('fleet-learn-btn').onclick=()=>run('fleet-learn');
function setBusy(value) {
 // Polled every 200 ms: only touch the DOM when a value changes, otherwise an open
 // scenario dropdown is rebuilt under the user's cursor and appears to glitch.
 const setIfChanged=(el,prop,next)=>{if(el[prop]!==next)el[prop]=next;};
 for(const id of ['run-btn','evolve-btn','learning-run-btn','fleet-learn-btn','seed','pipeline','operation']) setIfChanged($(id),'disabled',value);
 const operation=$('operation').value, drill=operation!=='excavate';
 setIfChanged($('pipeline'),'disabled',value||drill);
 setIfChanged($('pipeline').parentElement,'hidden',drill);
 for(const option of $('seed').options){
  const label=drill?`Scenario ${option.value}`:`${option.value} · ${Number(option.value)>=100?'Holdout':'Optimization'}`;
  setIfChanged(option,'textContent',label);
 }
 setIfChanged($('evolve-btn'),'disabled',value||drill);
 $('run-btn').innerHTML=value?'<span class="loader" style="width:12px;height:12px"></span> Running':`<span>▶</span> ${operation==='fleet'?'Start mining crew':drill?'Start drilling':'Recover core'} <span>↗</span>`;
}
function stopReplay() {replayActive=false;$('replay-btn').textContent='▶';$('replay-btn').setAttribute('aria-label','Replay measured episode');$('replay-btn').dataset.replaying='false';window.__deepcoreReplay.active=false;}
function startReplay() {
 if(!replayFrames.length)return;
 if(replayIndex>=replayFrames.length-1)replayIndex=0;
 replayClock=replayFrames[replayIndex].t;replayActive=true;lastTime=performance.now();
 $('replay-btn').textContent='Ⅱ';$('replay-btn').setAttribute('aria-label','Pause replay');$('replay-btn').dataset.replaying='true';
 $('run-mode').textContent='MEASURED REPLAY';$('timeline-label').textContent='Replaying the recorded physics rollout';
 window.__deepcoreReplay={...window.__deepcoreReplay,active:true,frame:replayIndex,total:replayFrames.length};
}
 $('replay-btn').onclick=()=>replayActive?stopReplay():startReplay();
$('scrubber').oninput=e=>{stopReplay();replayIndex=Number(e.target.value);renderFrame(replayFrames[replayIndex],true);$('run-mode').textContent='MEASURED REPLAY';};
function animate(now) {
 if(replayActive&&replayFrames.length){
  replayClock+=(now-lastTime)/1000*Number($('speed').value);
  while(replayIndex<replayFrames.length-1&&replayFrames[replayIndex+1].t<=replayClock)replayIndex++;
  renderFrame(replayFrames[replayIndex],true);
  window.__deepcoreReplay.frame=replayIndex;
  if(replayIndex===replayFrames.length-1){stopReplay();if(replayResult||state?.result)renderResult(replayResult||state.result);}
 }
 lastTime=now;requestAnimationFrame(animate);
}
requestAnimationFrame(animate);
function renderFrame(frame,replay=false) {
 if(!frame)return;
 if(frame.sensor){
  if(frame.sensor.depth_image)$('depth-image').src=`data:image/jpeg;base64,${frame.sensor.depth_image}`;
  if(frame.sensor.mask_image){$('mask-image').src=`data:image/jpeg;base64,${frame.sensor.mask_image}`;$('mask-caption').textContent='Controller segmentation stage';}
  else{$('mask-image').removeAttribute('src');$('mask-caption').textContent='Segmentation image not exposed';}
 }
 if(!replay) $('scrubber').max=String(Math.round((frame.duration||11)*20));
 mine?.update(frame);
 $('fleet-status').hidden=!frame.fleet;
 $('excavation-metrics').hidden=Boolean(frame.drilling);$('drilling-metrics').hidden=!frame.drilling;
 if(frame.drilling){renderDrilling(frame,replay);return;}
 const p=String(frame.phase||'READY'), upper=p.toUpperCase();
 $('phase').textContent=phaseNames[upper] || p.toLowerCase().replace(/[_:]+/g,' ').replace(/^./,c=>c.toUpperCase());
 $('phase-label').textContent=replay?'MEASURED EPISODE REPLAY':state?.job==='preview'?'MISSION STATUS':'ROBOT ACTIVITY';
 $('phase-detail').textContent=upper==='READY'?'Select a controller and run the excavation.':`Seed ${(replayMetadata||state)?.seed ?? '—'} · ${{baseline:'Baseline controller',candidate:'Experimental rollout',accepted:'Accepted controller'}[(replayMetadata||state)?.pipeline]||'Controller rollout'} · Real robot and object poses`;
 $('time').textContent=`${fmt(frame.t,2)} / 11.00 s`;
 $('force').textContent=fmt(frame.force);$('peak-force').textContent=`Peak ${fmt(frame.peak)} N`;
 $('latency').textContent=fmt(frame.latency);
 $('force-meter').style.width=`${Math.min(100,frame.force/18*100)}%`;
 $('force-meter').style.background=frame.force>=18?'var(--red)':'var(--cyan)';
 $('latency-meter').style.width=`${Math.min(100,(frame.latency||0)/25*100)}%`;
 $('latency-meter').style.background=frame.latency>=25?'var(--red)':'var(--cyan)';
 if(replay){$('scrubber').value=String(replayIndex);forceHistory=replayFrames.slice(Math.max(0,replayIndex-99),replayIndex+1).map(f=>f.force||0);}
 else if(frame.t!==seenTime){forceHistory.push(frame.force||0);forceHistory=forceHistory.slice(-100);seenTime=frame.t;$('scrubber').value=String(Math.round(frame.t*20));}
 const scale=Math.max(18,...forceHistory);
 $('force-line').setAttribute('d',forceHistory.map((f,i)=>`${i?'L':'M'}${i/(Math.max(1,forceHistory.length-1))*150},${34-f/scale*31}`).join(' '));
 if(frame.camera){$('sensor-image').src=`data:image/jpeg;base64,${frame.camera}`;$('sensor-image').style.display='block';$('sensor-empty').hidden=true;}
 if(replay){$('rubble').textContent='—';$('rubble-detail').textContent='Final clearance shown when replay ends';$('outcome').textContent=frame.peak>18?'Core force exceeded':'Replay in progress';$('outcome').className=frame.peak>18?'bad':'';$('outcome-detail').textContent='Final outcome is measured after settling';}
}
function renderDrilling(frame,replay=false) {
 const d=frame.drilling;
 $('drill-thrust').textContent=fmt(d.axial_force_n);
 $('drill-torque').textContent=fmt(d.torque_nm,3);
 $('drill-depth').textContent=fmt(d.depth_mm);
 $('drill-rpm').textContent=frame.fleet?fmt(d.collector_contact_force_n):fmt(d.rpm,0);
 $('drill-fourth-label').textContent=frame.fleet?'COLLECTOR GRIP CONTACT':'SPINDLE SPEED';
 $('drill-fourth-unit').textContent=frame.fleet?'N':'rpm';
 $('drill-force-detail').textContent=`Peak ${fmt(d.peak_force_n)} N · limit ${fmt(d.limit_force_n,0)} N`;
 $('drill-depth-detail').textContent=`${d.layers_removed} / ${d.layers_total} layers removed`;
 $('drill-status').textContent=String(d.phase||'Ready').replaceAll('_',' ');
 $('drill-material').textContent='Simulated rock layers';
 $('phase').textContent=String(d.phase||'Ready').toLowerCase().replaceAll('_',' ').replace(/^./,c=>c.toUpperCase());
 $('phase-label').textContent=replay?'MEASURED DRILL REPLAY':'DRILL OPERATION';
 $('phase-detail').textContent=`${fmt(d.axial_force_n)} N thrust · ${fmt(d.torque_nm,3)} N·m torque · ${fmt(d.depth_mm)} mm depth`;
 $('time').textContent=`${fmt(frame.t,2)} / ${fmt(frame.duration||12,2)} s`;
 if(frame.fleet){
  const f=frame.fleet, phases=f.machine_phases||{};
  $('phase-label').textContent=replay?'MINING CREW · REPLAY':'MINING CREW';
  $('phase-detail').textContent=`${f.debris_collected??0} / ${f.debris_produced??0} fragments collected · ${fmt(d.axial_force_n)} N tool load`;
  $('driller-phase').textContent=String(phases.driller||phases.drill||f.driller_phase||d.phase||'Ready').replaceAll('_',' ');
  $('collector-phase').textContent=String(phases.collector||f.collector_phase||'Waiting for debris').replaceAll('_',' ');
  const counts=f.cart_counts||[], capacity=f.cart_capacity||2;
  const departed=(f.cart_departed||[]).filter(Boolean).length;
  $('debris-count').textContent=`${counts.filter(n=>n>=capacity).length} / ${counts.length} · ${departed} dispatched`;
  $('crew-drill-state').textContent=$('driller-phase').textContent;
  $('crew-collector-state').textContent=$('collector-phase').textContent;
  $('crew-count').innerHTML=`${Number(f.debris_collected)||0} <small>fragments</small>`;
  $('crew-progress-detail').textContent=`${f.debris_collected??0} / ${f.debris_total??0} loaded · ${d.layers_removed} / ${d.layers_total} rock layers cut`;
  $('cart-loads').innerHTML=counts.map((count,i)=>`<div class="cart-load"><span>Cart ${i+1} · ${count}/${capacity}</span><small>${f.cart_departed?.[i]?'Dispatched':(f.cart_travel_m?.[i]??0)<-.03?'Departing':'At loading station'}</small><div class="load-meter"><i style="width:${Math.min(100,100*count/capacity)}%"></i></div></div>`).join('');
 }
 if(replay){$('scrubber').value=String(replayIndex);forceHistory=replayFrames.slice(Math.max(0,replayIndex-99),replayIndex+1).map(f=>f.drilling?.axial_force_n||0);}
 else if(frame.t!==seenTime){forceHistory.push(d.axial_force_n||0);forceHistory=forceHistory.slice(-100);seenTime=frame.t;$('scrubber').value=String(Math.round(frame.t*20));}
 const scale=Math.max(d.limit_force_n||1,...forceHistory);
 $('drill-force-path').setAttribute('d',forceHistory.map((f,i)=>`${i?'L':'M'}${i/Math.max(1,forceHistory.length-1)*150},${34-f/scale*31}`).join(' '));
 if(frame.camera){$('sensor-image').src=`data:image/jpeg;base64,${frame.camera}`;$('sensor-image').style.display='block';$('sensor-empty').hidden=true;}
}
function renderResult(r) {
 if(r.operation==='fleet'){
  $('phase').textContent=r.complete?'Mining and collection complete':'Mining cycle ended';
  $('phase-label').textContent='MINING CREW · RESULT';
  $('phase-detail').textContent=`${r.layers_removed} / ${r.layers_total} rock layers removed · ${r.debris_collected??0} / ${r.debris_produced??0} fragments collected · Peak ${fmt(r.peak_force_n)} N`;
  $('drill-status').textContent=r.complete?'Crew finished':'Cycle ended';
  return;
 }
 if(r.operation==='drill'){
  $('phase').textContent=r.complete?'Bore complete':r.phase==='OVERLOAD'?'Drill load limit reached':'Drilling cycle ended';
  $('phase-label').textContent='MEASURED DRILL OUTCOME';
  $('phase-detail').textContent=`${r.layers_removed} / ${r.layers_total} layers removed · Max depth ${fmt(r.max_depth_mm??r.depth_mm)} mm · Peak ${fmt(r.peak_force_n)} N`;
  $('drill-status').textContent=r.complete?'Bore complete':'Cycle ended';
  return;
 }
 $('rubble').textContent=r.rubble_cleared_count??'—';
 $('rubble-detail').textContent='Final position after settling';
 const text=r.core_shattered?'Core shattered':r.core_extracted?'Core recovered':'Not extracted';
 $('outcome').textContent=text;$('outcome').className=r.core_extracted?'good':r.core_shattered?'bad':'warn';
 $('outcome-detail').textContent=`Composite ${fmt(r.composite_score)} / 100 · single seed`;
 $('phase').textContent=r.core_extracted?'Core recovered intact':r.core_shattered?'Run ended · core fractured':'Run ended · extraction incomplete';
 $('phase-label').textContent='MEASURED OUTCOME';
 $('phase-detail').textContent=`${r.rubble_cleared_count}/2 boulders cleared · Peak core force ${fmt(r.peak_core_contact_force_n)} N · Score ${fmt(r.composite_score)}`;
 $('peak-force').textContent=`Peak ${fmt(r.peak_core_contact_force_n)} N`;
 $('latency').textContent=fmt(r.vision_latency_p99_ms);
 $('insight-text').textContent=r.core_shattered?'Contact force exceeded the safety gate. This run is a failure; the controller needs a safer grasp or approach.':r.core_extracted?'The core reached the hopper intact and at rest. This single-seed result still needs verification across all optimization and holdout seeds.':'The episode ended before the core reached a stable position in the hopper. The learning loop can use optimization-seed failures to revise perception and control.';
 $('insight-source').textContent=`LOCKED VERIFIER · SEED ${(replayMetadata||state).seed} · ${(replayMetadata||state).pipeline.toUpperCase()}`;
 if(r.pipeline_errors?.length)notice(`Controller errors: ${r.pipeline_errors.join('; ')}`);
}
function renderEvents(events) {
 const serialized=JSON.stringify(events);if(serialized===lastEvents)return;lastEvents=serialized;
 $('event-count').textContent=`${events.length} events`;
 document.querySelectorAll('.agent').forEach(el=>{el.classList.remove('active','done','error');el.querySelector('i').textContent='WAITING';});
 if(!events.length)return;
 $('event-log').innerHTML=events.map(e=>`<div class="event"><div><b>${esc(e.agent)}</b><small>GEN ${esc(e.generation)} · ${esc(e.status)}</small></div><p>${esc(e.message)}</p></div>`).join('');
 $('event-log').scrollTop=$('event-log').scrollHeight;
 for(const e of events){
  const name=String(e.agent).toLowerCase();const key=name.includes('architect')?'architect':name.includes('cv')||name.includes('vision')?'cv':name.includes('control')?'control':'verifier';
  const el=document.querySelector(`[data-agent="${key}"]`);
  el.classList.remove('active','done','error');el.classList.add(e.status==='running'||e.status==='retry'?'active':['failed','rejected','error'].includes(e.status)?'error':'done');
  el.querySelector('i').textContent=String(e.status).toUpperCase();
 }
 const e=events.at(-1);$('generation').textContent=`GEN ${e.generation}`;
 $('insight-text').textContent=e.message;$('insight-source').textContent=`${e.agent} · ${e.status}`;
}
function table(headers,rows) {
 const container=document.createElement('div');
 container.innerHTML=`<table class="data-table"><thead><tr>${headers.map(h=>`<th>${h}</th>`).join('')}</tr></thead><tbody>${rows.join('')}</tbody></table>`;
 container.querySelectorAll('tbody tr').forEach(row=>row.querySelectorAll('td').forEach((cell,i)=>cell.dataset.label=headers[i]));
 return container.innerHTML;
}
function renderRuns(runs) {
 const serialized=JSON.stringify(runs);if(serialized===lastRuns)return;lastRuns=serialized;if(!runs.length)return;
 $('history-table').innerHTML=table(['OPERATION','SEED','SCORE / DEPTH','MATERIAL CLEARED','PEAK FORCE','OUTCOME'],[...runs].reverse().map(r=>r.operation==='fleet'?`<tr><td>Mining crew<small>Driller + collector</small></td><td>${r.seed}</td><td>${fmt(r.max_depth_mm??r.depth_mm)} mm</td><td>${r.debris_collected??0} / ${r.debris_produced??0} fragments</td><td>${fmt(r.peak_force_n)} N</td><td class="${r.complete?'good':'warn'}">${r.complete?'Collected':'Cycle ended'}</td></tr>`:r.operation==='drill'?`<tr><td>Drilling<small>Simulated rock layers</small></td><td>${r.seed}</td><td>${fmt(r.max_depth_mm??r.depth_mm)} mm</td><td>${r.layers_removed} / ${r.layers_total} layers</td><td>${fmt(r.peak_force_n)} N</td><td class="${r.complete?'good':'warn'}">${r.complete?'Bore complete':'Cycle ended'}</td></tr>`:`<tr><td>${esc(r.pipeline)}<small>Core recovery · single seed</small></td><td>${r.seed}</td><td>${fmt(r.composite_score)}</td><td>${r.rubble_cleared_count} / 2 boulders</td><td>${fmt(r.peak_core_contact_force_n)} N</td><td class="${r.core_extracted?'good':r.core_shattered?'bad':'warn'}">${r.core_extracted?'Recovered intact':r.core_shattered?'Shattered':'Not extracted'}</td></tr>`));
}
function renderLedger(ledger) {
 const serialized=JSON.stringify(ledger);if(serialized===lastLedger)return;lastLedger=serialized;
 const accepted=ledger.filter(e=>e.accepted&&e.version>0);
 $('accepted-count').innerHTML=`${accepted.length}<small> ${accepted.length?'stored in shared brain':'awaiting verification'}</small>`;
 if(!ledger.length)return;
 $('ledger-table').className='';$('ledger-table').innerHTML=table(['VERSION / SOURCE','OPT SCORE','HOLDOUT SCORE','DECISION'],[...ledger].reverse().map(e=>`<tr><td>v${esc(e.version)}<small>${esc(e.source)}</small></td><td>${fmt(e.opt?.composite_score)}</td><td>${fmt(e.holdout?.composite_score)}</td><td class="${e.accepted?'good':'bad'}">${e.accepted?'Accepted':'Rejected'}<small>${esc((e.reasons||[]).join('; '))}</small></td></tr>`));
}
function renderFleetLearning(rows) {
 const entry=rows.at(-1), accepted=rows.filter(row=>row.accepted).length;
 const learning=state?.job==='fleet-learn'&&busy;
 $('fleet-learning-status').textContent=learning?'Evaluating fresh physics runs':entry?`${entry.decision} · ${accepted} accepted changes`:'Awaiting measured comparison';
 const baseline=entry?.baseline_fresh, candidate=entry?.candidate_fresh;
 $('fleet-learning-detail').textContent=learning?(state.events?.at(-1)?.message||'Comparing sensor and grasp parameters on optimization scenarios.'):
   entry?`Fleet score ${fmt(baseline?.score)} → ${fmt(candidate?.score)}. ${entry.accepted?'Improvement passed the fleet force gate.':(entry.reasons||[]).join(' ')}`:
   'Fresh runs compare camera and grasp parameters before a change is accepted.';
}
async function poll() {
 try{
 const res=await fetch('/api/state?compact=1');if(!res.ok)throw new Error(`Server returned ${res.status}`);const next=await res.json();state=next;
  $('connection-label').textContent='SYSTEM CONNECTED';$('connection-dot').style.background='var(--green)';
  $('compute').textContent=state.compute==='cis-a6000'?'cis-a6000 · GPU 0':state.compute;
  $('integrity').textContent=replayMetadata?`Recorded seed ${replayMetadata.seed} · ${replayMetadata.source}`:state.job?.startsWith('fleet')?(state.frame?.fleet?.sensor_based?'Experimental RGB-D fleet · measured physics':'Fleet simulation · measured physics'):state.job==='drill'?'MuJoCo · simulated rock layers':state.integrity==='verified'?'● Physics integrity verified':'Checking simulation integrity';
  busy=['starting','running','evolving'].includes(state.status);setBusy(busy);
  if(state.run_id!==previousRun){previousRun=state.run_id;forceHistory=[];seenTime=-1;stopReplay();replayFrames=[];replayResult=null;replayMetadata=null;$('operation').value=state.job?.startsWith('fleet')?'fleet':state.job==='drill'?'drill':'excavate';completedRun=-1;$('scrubber').disabled=true;$('replay-btn').disabled=true;$('rubble').textContent='—';$('outcome').textContent='Awaiting result';$('outcome').className='';$('outcome-detail').textContent='Measured after the simulation settles';}
  if(sceneRun!==state.run_id){
    const geometry=await(await fetch('/api/scene')).json();
    if(geometry.scene.length){mine?.load(geometry.scene);sceneRun=state.run_id;$('scene-loading').hidden=true;
     const fleet=state.job?.startsWith('fleet');
     document.body.classList.toggle('fleet-mode',fleet);
     $('fleet-workflow').hidden=!fleet;
     $('activity-heading').textContent=fleet?'MINING CREW':'LEARNING LOOP';
     $('site-name').textContent=fleet?'CMU · Bruceton mine':'Basin N1';
     $('zone-label').textContent=fleet?'DRILLING + DEBRIS RECOVERY':'PRECISION EXTRACTION ZONE';
     $('scene-legend').innerHTML=fleet?'<span><i class="swatch orange"></i>Driller</span><span><i class="swatch cyan"></i>Collector</span><span><i class="swatch green"></i>Rail carts</span>':'<span><i class="swatch cyan"></i>Fragile core</span><span><i class="swatch orange"></i>Overburden</span><span><i class="swatch green"></i>Recovery hopper</span>';
     $('physics-rate').textContent=state.job==='drill'||fleet?'1000 Hz':'200 Hz';
    }
  }
  if(!replayActive&&$('run-mode').textContent!=='MEASURED REPLAY') renderFrame(state.frame);
  if(busy){$('run-mode').textContent=state.status==='evolving'?'LEARNING IN PROGRESS':state.job==='preview'?'PREPARING SITE':'LIVE PHYSICS';$('timeline-label').textContent=state.status==='evolving'?'Generating and verifying controller changes':'Streaming real simulation poses';}
  else if(state.status==='complete'&&state.job==='fleet-learn'){$('run-mode').textContent='LEARNING EVALUATION COMPLETE';$('timeline-label').textContent=state.fleet_result?.decision||'Fleet comparison recorded';}
  else if(state.status==='ready'){$('run-mode').textContent='PHYSICS SIMULATION';$('timeline-label').textContent='Ready · select a controller and run';}
  else if(state.status==='complete'&&state.result&&completedRun!==state.run_id){
    completedRun=state.run_id;renderFrame(state.frame);renderResult(state.result);$('run-mode').textContent='RUN COMPLETE';$('timeline-label').textContent='Measured episode · replay or inspect the timeline';
    const replay=await(await fetch('/api/replay')).json();replayFrames=replay.frames;replayIndex=replayFrames.length-1;
    replayResult=state.result;
    $('scrubber').max=String(replayFrames.length-1);$('scrubber').value=String(replayIndex);$('scrubber').disabled=false;$('replay-btn').disabled=!replayFrames.length;
    if(pendingVersion){const version=pendingVersion;pendingVersion=null;await switchVersion(version);}
  }else if(state.status==='complete'&&state.result&&!replayActive&&$('run-mode').textContent!=='MEASURED REPLAY')renderResult(state.result);
  if(state.status==='error'){notice(state.error);$('scene-loading').hidden=true;$('phase').textContent='Simulation needs attention';$('phase-detail').textContent='See the error above. You can retry when the issue is resolved.';}
  renderEvents(state.events||[]);renderRuns(state.runs||[]);renderLedger(state.ledger||[]);renderFleetLearning(state.fleet_ledger||[]);
 }catch(e){$('connection-label').textContent='RECONNECTING';$('connection-dot').style.background='var(--amber)';notice(`Connection interrupted: ${e.message}. Retrying automatically.`);}
 setTimeout(poll,200);
}
poll();
