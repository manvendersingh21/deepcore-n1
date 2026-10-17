# Perception scratch experiments — 2026-10-17

Candidates derive from adopted controller_yawnear.py. No core, template, threshold, generated patch, or ledger edits. Remote GPU0 fresh-process intact verifier; no active verifier harness child before each launch. OPT seed33 only.

|Candidate|Score|Extracted|Shattered|Peak core N|Rubble|Vision RMSE mm|
|---|---:|---|---|---:|---:|---:|
|controller_largestcore.py|35.66|false|false|11.76|1|22.15|
|controller_orange_highlight.py|47.94|false|false|11.75|2|22.06|
|controller_combinedcolor.py|48.17|false|false|11.75|2|22.06|

Orange V maximum255 improves rubble clearance but does not resolve core extraction. Largest cyan component alone has no measured task benefit. No candidate ready for adoption or six-seed gate.

Durable remote outputs: `dev/scratch/largestcore_33.json`, `orange_highlight_33.json`, `combinedcolor_33.json`; stderr alongside. Completed local execution handles86041,46395,91945. OPT33-only perception_frames.py snapshot diagnostic currently session88853.

## OPT33 mask diagnosis

Snapshot diagnostic completed (session88853). At6.5/8.5/10.5s raw cyan measurement is null, so tracker retention is expected. True core settled at(0.35076,-0.02958), old estimate(0.39834,-0.04863). At8.5s visible cyan core RGB is heavily illuminated: sampled H90,V255, saturation19–98 (median46). Existing cyan S>=90 removes most of exposed core. This is perception color filtering, independent of verifier thresholds. `controller_cyansat.py` changes only candidate cyan saturation minimum90→25, baseline yawnear otherwise; isolated opt33 session70319 started after evaluator absence check. Remote durable cyansat_33.json/.err.

`controller_cyansat.py` opt33 result: **77.78, core extracted intact,2 rubble,11.76N peak**,22.15mm vision RMSE,9.27ms p99. This resolves missing late core detection. No all-six acceptance claim; parent requested exclusive GPU coordination before that sweep. Combined cyanS25+orangeV255 isolated opt33 currently session21043, durable cyansat_orange_33.json/.err.

Combined cyanS25+orangeV255 opt33:77.86, extracted safe,2 rubble,11.75N peak,22.06mm RMSE. Parent granted exclusive all-six window and paused main evolution PID2418796. CyanS25-only sweep local55278 remote2433189, durable cyansat_all6.jsonl/.err; each seed creates independent evaluate environment within one Python process, so fresh-process latency acceptance remains separate. Parent informed. No new broader-orange experiment started.

## Cyan saturation all-six safety/extraction sweep PASS

Completed session55278, remote2433189 exited. Parent informed GPU0 released immediately to resume main evolution.

|Seed|Score|Extracted|Shattered|Peak N|Rubble|
|---|---:|---|---|---:|---:|
|11|87.75|true|false|4.41|2|
|22|80.41|true|false|15.38|2|
|33|77.92|true|false|11.76|2|
|101|85.84|true|false|4.87|2|
|102|90.19|true|false|4.50|2|
|103|80.56|true|false|14.05|2|

Candidate `dev/scratch/controller_cyansat.py`: only cyan HSV saturation floor90→25 relative to adopted yawnear. Core integrity passed, no holdout autopsy. Six fresh environments in one interpreter; p99 latency8.51–10.67ms, timing is experimental and not final acceptance. Scores>88 all six remain unmet. Durable local and remote `dev/scratch/cyansat_all6.jsonl`. Parent owns template/adoption decision.

## Broader orange rejected

Parent adopted cyanS25 and resumed main. Followup `controller_orange_broad.py` (cyanS25 + orange H8..32,S25..255,V120..255), opt33 fresh process10429:14.83, shattered76.29N,1 rubble, vision62.36mm. No adoption. Broad hue/saturation admits competing scene blobs; current selector's hue/area score has no spatial association. No additional GPU run started following failure.

## Narrow orange highlight band OPT33 improvement

Broad mask includes dark earth-tone texture, confirmed CPU HSV analysis on saved OPT33 frame. `controller_orange_highband.py` instead preserves saturated band and adds only H23..32,S>=25,V>=230 highlight pixels, retaining cyanS25. Fresh opt33 session68311 result80.93, intact extraction,2 rubble,11.34N peak,18.18mm vision RMSE,10.44ms p99. Improvement over cyanS25 opt33 score77.78,22.15mm RMSE; no all-six claim. OPT11 run pending session below.

Narrow highlight additional OPT gates (fresh processes):11=90.73/4.47N/RMSE9.93mm;22=80.32/15.38N/RMSE14.00mm. All three optimization seeds safe, extracted,2 rubble. Parent informed candidate awaits all-six exclusive safety gate; GPU0 released. Opt22 full autopsy saved local/remote orange_highband_22_full.json. Mean xyz biases mm: core[9.0,-12.5,0.0], rubble_a[-4.0,2.4,0.0], rubble_b[2.7,16.0,5.5]. Geometry errors mostly xy from partial masks; z concern is rubble_b5.5mm. No holdout diagnosis.

Parent requested no new GPU starts while main Control g1 nears evaluator. Prepared **untested** `controller_core_completion.py` from highband: when observed cyan top is a narrow rectangle with a long side>30mm and higher nearby rubble, complete the short dimension toward that occluder using existing core width44mm. Targets observed core xy bias from partial visibility, leaves fully visible squares untouched. No simulation or adoption; waiting main GPU coordination and highband all-six gate first.

## Narrow highlight all-six gate REJECTED

Exclusive window session99841 complete; parent immediately notified GPU released to resume2418796. All safe, but seed103 aggregate failed extraction (no autopsy):36.96,1 rubble,15.74N. Other scores11=90.54,22=80.13,33=80.95,101=86.02,102=90.09, all extracted2 rubble. Integrity passed. Durable local/remote highband_all6.jsonl/.err. Keep adopted cyanS25 candidate.

Parent authorized exploratory core-completion opt22 after main resumes network. Started official OPT22 full diagnostic after verifier absence check; remote core_completion_22_full.json/.log. Candidate based rejected highband, so even OPT success requires transplant onto adopted cyanS25 before any adoption gate.

## Partial core square completion experiments

Corecompletion OPT22 nearest-overburden selection regressed core bias to[9.2,-20,0]mm, totalRMSE16.19,score78.55; safe/extracted15.38N. Selecting highest nearby rubble instead (`controller_core_topcomplete.py`) improved core bias[8.9,-5.0,0],core mean10.4mm, totalRMSE12.30,score81.46; safe/extracted15.38N. Both based rejected highband. Transplanted only highest-overburden completion onto adopted cyanS25 as `controller_core_infer.py`, OPT22 official diagnostic pending. No allsix/adoption claim. Uses existing OpenCV minAreaRect/boxPoints primitive fitting; no extra imports. Convex hull alone cannot recover an invisible rectangle edge, so bounded known-width completion supplies that missing constraint.

`controller_core_infer.py` completion-only (cyanS25 baseline) opt22 reproduced RMSE12.30,81.45, extracted safe15.38N. OPT11 extracted safe4.41N,88.02,RMSE13.26 (small gain over13.49). OPT33 now pending; no allsix claim. Main verifier absence checked between trials.

## Fitted rectangle centre OPT22 promising

`controller_rectcentre.py` changes only xy from median surface pixels to existing OpenCV minAreaRect centre backprojected at same robust median depth; z unchanged. OPT22 fresh process93802:84.90, intact extraction,2 rubble,12.31N peak (baseline15.38),11.76mm RMSE (baseline14). Bias core[7,-12.2,0], orange[-3.7,1.7,0], magenta[3,9.5,5.4]mm. Testing other OPT gates before any allsix request.

Core-infer OPT33 safe/extracted77.84/11.82N, RMSE22.15 unchanged; no33 vision benefit.

Rectcentre allOPT gates safe/extracted2 rubble:11=88.98/4.45N/RMSE11.84,22=84.90/12.31N/RMSE11.76,33=79.25/12.48N/RMSE19.39. Best isolated perception candidate across OPT so far. Requested parent coordination for later exclusive allsix gate; GPU0 released. No template edits and no holdout autopsy.
