// Static (GitHub Pages) mode for the control room. There is no simulation
// backend here: the API is answered from recorded MuJoCo episodes in
// replays/*.json, and run/learn requests are refused instead of faked.
const realFetch = window.fetch.bind(window);
const recordings = new Map();
const json = (body, status = 200) => new Response(JSON.stringify(body), {status, headers: {'Content-Type': 'application/json'}});

function recording(version) {
  if (!recordings.has(version)) recordings.set(version, realFetch(`replays/${version}.json`).then(r => {
    if (!r.ok) throw new Error(`replays/${version}.json returned ${r.status}`);
    return r.json();
  }));
  return recordings.get(version);
}

const staticState = {
  status: 'idle', job: 'run', run_id: 0, compute: 'Recorded replay · no live backend', integrity: 'verified',
  frame: null, events: [], runs: [], ledger: [], fleet_ledger: [],
};

window.fetch = async (input, init = {}) => {
  const url = new URL(typeof input === 'string' ? input : input.url, window.location.href);
  const api = url.pathname.match(/\/api\/([\w-]+)$/);
  if (!api) return realFetch(input, init);
  if ((init.method || 'GET').toUpperCase() === 'POST')
    return json({error: 'This public page replays recorded physics only. Clone the repo and run `python main.py run-web` for live MuJoCo runs.'}, 403);
  if (api[1] === 'state') return json(staticState);
  if (api[1] === 'scene') return json({scene: (await recording('accepted')).scene});
  if (api[1] === 'recording') return json(await recording(url.searchParams.get('version') || 'accepted'));
  if (api[1] === 'replay') return json({frames: []});
  return json({error: 'not available in static mode'}, 404);
};

window.addEventListener('load', () => {
  document.body.classList.add('static-demo');
  // Start on the accepted controller replay; 0 / N switch baseline / accepted.
  setTimeout(() => document.body.dispatchEvent(new KeyboardEvent('keydown', {key: 'n', bubbles: true})), 1200);
});
