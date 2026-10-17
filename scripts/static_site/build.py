"""Build the static GitHub Pages replay site into ./site.

The page is the browser control room driven by recorded MuJoCo episodes
(dev/replays/*.json) instead of a live simulation backend, plus the native
fleet recording. Usage: .venv/bin/python -m scripts.static_site.build
"""
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'site'
HERE = Path(__file__).parent

BANNER = """<div class="static-banner">
<strong>Public replay.</strong> Every pose and force below was recorded from real MuJoCo runs;
nothing is live. Press <kbd>0</kbd> baseline · <kbd>N</kbd> accepted controller.
<a href="fleet_demo.mp4" target="_blank" rel="noopener">▶ Watch the 80 s mining-fleet demo</a>
<a href="https://github.com/manvendersingh21/deepcore-n1" target="_blank" rel="noopener">Source</a>
</div>
"""

STYLE = """<style>
.static-banner{position:relative;z-index:50;padding:8px 16px;background:#14232a;color:#d8e6ea;
 font:13px/1.5 system-ui,sans-serif;border-bottom:1px solid #2a4650;display:flex;gap:14px;flex-wrap:wrap;align-items:center}
.static-banner a{color:#5fe0d0}.static-banner kbd{background:#22343b;border-radius:3px;padding:0 4px}
body.static-demo #run-btn,body.static-demo #evolve-btn,body.static-demo #learning-run-btn,body.static-demo #fleet-learn-btn{opacity:.4;pointer-events:none}
</style>
"""


def rewrite(path: Path, pairs):
    text = path.read_text()
    for old, new in pairs:
        text = text.replace(old, new)
    path.write_text(text)


def main():
    if OUT.exists():
        shutil.rmtree(OUT)
    shutil.copytree(ROOT / 'ui/web', OUT)
    for name in ('hdri_polyhaven', 'mine_environment', 'rock_ambientcg_Rock064', 'soil_ambientcg_Ground112'):
        shutil.copytree(ROOT / 'assets' / name, OUT / 'assets' / name)
    shutil.copy(ROOT / 'assets/LICENSES.md', OUT / 'assets/LICENSES.md')
    (OUT / 'replays').mkdir()
    for name in ('baseline.json', 'accepted.json', 'manifest.json'):
        shutil.copy(ROOT / 'dev/replays' / name, OUT / 'replays' / name)
    shutil.copy(HERE / 'static_shim.js', OUT / 'static_shim.js')

    # Project Pages live under /<repo>/, so root-absolute URLs must become relative.
    rewrite(OUT / 'index.html', [
        ('href="/style.css"', 'href="style.css"'),
        ('"three":"/vendor/three.module.js"', '"three":"./vendor/three.module.js"'),
        ('<script type="module" src="/app.js">', '<script src="static_shim.js"></script>\n<script type="module" src="app.js">'),
        ('href="/" aria-label', 'href="./" aria-label'),
        ('</head>', STYLE + '</head>'),
        ('<body>', '<body>\n' + BANNER),
    ])
    rewrite(OUT / 'scene.js', [("'/assets/", "'./assets/"), ('`/assets/', '`./assets/')])
    rewrite(OUT / 'assets/mine_environment/environment.js', [
        ("fetch('/assets/mine_environment/environment.json')", "fetch(new URL('./environment.json', import.meta.url))"),
        ('`/assets/rock_ambientcg_Rock064/', '`${new URL("../rock_ambientcg_Rock064/", import.meta.url)}'),
    ])

    video = ROOT / 'dev/native_realistic_fleet.mp4'
    if video.exists():
        subprocess.run(['ffmpeg', '-loglevel', 'error', '-y', '-i', str(video), '-vf', 'scale=1280:-2,fps=30',
                        '-c:v', 'libx264', '-crf', '28', '-preset', 'slow', '-pix_fmt', 'yuv420p',
                        '-movflags', '+faststart', '-an', str(OUT / 'fleet_demo.mp4')], check=True)
    (OUT / '.nojekyll').write_text('')
    print(f'Built {OUT}')


if __name__ == '__main__':
    main()
