# Final contract audit

Audit date: 2026-10-17. This is an implementation/evidence audit of the locked
RGB-D excavation benchmark and agent/memory tracks. The new mobile mining fleet
is a separate presentation/physics operation; privileged fleet control does not
establish sensor-only autonomous mining or benchmark acceptance.

## CPU checks executed (initial audit)

- `.venv/bin/python -m pytest -q tests/test_shared_brain_unit.py tests/test_agents_unit.py tests/test_tracks_unit.py`: **31 passed**, 0.77 s.
- `.venv/bin/python -m pytest -q tests/test_pipeline_e2e.py::test_sha256_integrity_and_no_privileged_leak`: **1 passed**, 0.10 s.
- Read-only `verify_integrity()` matched all four locked files: verifier_harness.py, scene.xml, env_runner.py, sensor_corruption.py.
- Read-only source SHA-256 comparison matched every ledger entry's recorded path/hash, including both rejections.

No physics rollouts, remote benchmark runs, LLM calls, ledger changes or locked
core edits were performed. The other end-to-end tests require real physics and
were deliberately excluded; root is independently running six-seed verification.

Coverage limitation: `tests/test_shared_brain_unit.py` imports
`shared_brain.alt.memory_store`, not production `shared_brain.memory_store`.
Those passing pruning/concurrency tests validate the alternate implementation.
Agent/track tests do instantiate the production store with isolated temporary
roots, but do not cover every stricter requirement below.

## Sensor and holdout boundary

**Implemented for the excavation benchmark.** `core/env_runner.py:435` constructs
the candidate with copied camera intrinsics/extrinsics. Its only step inputs at
line 449 are simulated time, copied corrupted RGB/depth and `ep.proprio()`.
`observe()` calls sensor corruption after clean rendering; proprio includes only
end-effector position/yaw, gripper opening and wrist force/torque. Ground truth
is recorded separately for scoring and supplied to privileged visualization
callbacks, not passed to the pipeline step.

`core/verifier_harness.py:233` checks core integrity before evaluation, refuses
requested diagnostics outside optimization seeds and strips pipeline errors from
nondiagnostic per-seed results. `ContractOrchestrator._evaluate` evaluates
optimization seeds with diagnostics=True and holdout with diagnostics=False.
Architect/specialist prompts contain optimization metrics/autopsies. Actual
ledger autopsy keys are exclusively 11, 22 and 33 for every entry; no holdout
101/102/103 autopsy was observed. Holdout aggregate and per-seed metric numbers
remain available for acceptance, which is different from exposing autopsies.

**Limit:** this is a verified call boundary and AST blacklist, not proof of a
security sandbox against every hostile Python program. `sandbox_exec.py` uses
NemoClaw only when installed, otherwise a resource-limited subprocess. The
locked AST guard, passed candidate static check and absence of privileged input
support the implemented pipeline's sensor-only claim; do not extrapolate that
to the separate privileged fleet controller.

## Definition of done, decisions and hash guard

**Implemented:** goal compilation emits machine-readable whitelisted assertions
before evaluation; formal decisions include CONTINUE, REPLAN, ROLLBACK_DRIFT,
STOP_DONE and PING_HUMAN. The actual v4 entry is STOP_DONE. Locked core hashes
currently match. Law 5 requires strictly increasing optimization composite,
nondecreasing holdout composite, wrist force below 28 N, latency below 25 ms and
no shattered cores on either split. The per-generation RMSE sub-contract target
is advisory: `_verify` adds a note when it is missed but does not reject on that
basis. Actual v4 has such a note (16.6 mm vs 9.3 mm target).

**Gaps requiring work before claiming strict frozen contracts:**

1. `compile_goal` stores mutable `self.dod` and can overwrite the JSON on another
   call; no frozen snapshot/hash or mutation check is enforced during the run.
2. Mandatory assertion insertion checks only `(seeds, metric, op)`, ignoring the
   threshold/value. An LLM assertion such as holdout composite > 0 suppresses
   insertion of the required > 85 assertion. Actual recorded DoD does contain
   > 85 and safe required values; this is a general enforcement defect rather
   than evidence that this completed run used weakened mandatory values.
3. PING_HUMAN is enumerated but no `decide` branch returns it.

The core lock guards four files against their checked-in manifest; the manifest
and guard itself are outside their own protected hash set. This is tamper
recognition relative to that manifest, not cryptographically anchored provenance.

## Shared memory and operation lifecycle

**Implemented:** production playbook updates deduplicate bullets and prune
sections 1–3 before the graveyard until `ceil(characters/4) < 1500`; file writes
use a lock, temporary file and atomic rename. SQLite uses WAL/busy timeout and
stores read/write transactions, version diffs, and split/seed metric rows.
Conflict probes and logs exist. Actual playbook estimate is **399 tokens**;
actual SQLite snapshot has **33 transactions, 5 versions, 30 seed metric rows**.

**Gaps:**

1. `read_context(agent_id,max_tokens)` logs the budget but returns the whole file
   without bounding it. Normal writes are bounded, but oversized external content
   or a smaller requested budget is not enforced on reads.
2. `run_generation` performs one `read_playbook` as Root Architect, then passes
   that shared string into specialists. It does not perform per-agent
   `read_context` immediately before every operation and `atomic_update` after
   every operation. Goal compilation also lacks that lifecycle. Actual SQLite
   playbook reads are only Root Architect (4 reads), confirming this gap.
3. Production SQLite has no automatic retention/pruning policy; DELETE statements
   exist only in explicit `reset`. Playbook pruning is not SQLite log pruning.
4. Conflict counting probes and releases the playbook lock before the real write;
   contention between probe and acquisition can be missed, and prompt contention
   uses a different lock. Thus `conflicts_prevented` is a heuristic count, not a
   complete measurement of conflicts prevented.
5. `context_efficiency_gain` is writes minus reads, clamped at zero. It does not
   measure a real no-memory baseline or actual model context savings.

## Actual promotion and prompt evidence

| Version | Origin | Accepted | Opt composite | Holdout composite | Decision |
|---|---|---:|---:|---:|---|
| 0 | baseline reference | yes | 12.57 | 16.00 | CONTINUE |
| 1 | curated fallback v1 HSV/IQR/order | yes | 30.59 | 32.31 | CONTINUE |
| 2 | curated fallback v2 yaw tracker | no | 27.42 | 43.06 | ROLLBACK_DRIFT |
| 3 | LLM Nemotron-3-Super-120B-A12B | no | 30.57 | 32.28 | ROLLBACK_DRIFT |
| 4 | curated fallback v3 force guard/place | yes | 82.02 | 85.54 | STOP_DONE |

Accepted nonbaseline promotions are monotonic on both composites. The baseline
is explicitly a reference and can violate safety; its accepted flag is not a
claim that it passes the improvement gate. Rejected candidates remain recorded
and are not promoted. All five recorded source hashes matched the files at audit.
These numbers are recorded ledger evidence, not newly rerun benchmark results.

Four agents each have evolved prompt versions `[1,2]` in the actual JSON file.
**Critical gap:** these records are written on acceptance, but are never read
back into future architect/specialist/verifier model messages. This demonstrates
versioned prompt storage, not operational prompt self-improvement. Rejected
attempts also do not append prompt evolution deltas.

## Completion assessment

The sensor-only excavation and genuine monotonic measured improvement have
concrete implementation and ledger evidence. The formal-decision workflow,
shared playbook and versioned prompt records are present, and 32 selected CPU
checks pass. All three tracks cannot yet be described as fully satisfying the
stricter mandate: frozen/mandatory DoD enforcement, per-agent bounded memory
lifecycle, production SQLite pruning, and consumption of evolved prompts remain
open. The new fleet operation must remain clearly separated from sensor-only
benchmark autonomy and LLM-driven accepted improvement claims.


## Production memory remediation (after initial audit)

The production store now bounds `read_context` to the requested estimated token
budget and an absolute maximum of **1499** (`ceil(characters/4)`); zero budgets
return empty context. Returned-token count, source-token count, effective budget
and truncation are recorded. This closes the initial unbounded-read finding.
`atomic_update` logs a completed `agent_operation` transaction even for empty
deltas; its existing `{merged, contended}` return contract is preserved.

Ephemeral SQLite transactions are retained to the newest **2000** rows, pruned
at initialization and after logged operations. Ledger JSON, version/diff rows
and per-seed evaluation rows are never removed by retention. Persistent SQLite
counters retain lifetime context read counts, returned read tokens, deposited
write tokens, conflict counts and pruned-row counts across retention/reopening.
Explicit user-requested `reset` still clears evidence and resets these counters,
as before; retention itself never calls reset. Row-count retention bounds
operational history count, not arbitrary caller-provided detail payload bytes.

The existing shared-brain unit suite now imports production memory_store.
Added `tests/test_production_memory.py` verifies externally oversized context,
small/zero budgets, retention across reopening with preserved evaluation
records/lifetime counts, and completion logging for empty deltas.

Final focused CPU command:
`.venv/bin/python -m pytest -q tests/test_shared_brain_unit.py tests/test_production_memory.py tests/test_agents_unit.py tests/test_tracks_unit.py`

Result: **34 passed in 0.68 s**. The initial static/integrity test also passed
independently earlier. No production ledger, core files or physics rollouts were
changed/run during this remediation. Orchestrator lifecycle, evolved-prompt
consumption and DoD remediation are assigned separately to root and require
its final evidence; the initial findings about those are historical until that
work is independently confirmed.

## Orchestrator remediation (after production memory fixes)

`agents/contract_orchestrator.py` now uses a finally-protected canonical per-agent
operation context. Goal compilation, architect, CV/control proposals, fallback
role actions, verification/evaluation, fixed gate checks, decisions and
accept/reject operations read bounded context before operating and log atomic
updates plus completion/error receipts afterwards. The latest stored agent
guidance is consumed in architect/specialist messages; verifier receipts record
its version/hash while the acceptance gate remains fixed. Playbook reads use
1100 estimated tokens; guidance is capped to 1200 characters, with the combined
context hard capped below 1500 estimated tokens. Existing historical playbook
holdout composite fragments are filtered from mutation context.

Mission compilation now normalizes exact mandatory clauses through
`agents.mission_contract`, reuses the existing mission when a ledger exists,
and pins a FrozenMission snapshot/file. The public `dod` getter is a defensive
copy, and different assignments after freezing fail. File/snapshot integrity is
checked before evaluation, the fixed gate and goal satisfaction. Existing
production DoD/ledger/prompt files were not manually rewritten.

Added tests verify actual guidance in architect/specialist messages, canonical
per-agent receipts and token bounds, exception-finally updates, defensive mission
copies, disk tamper detection, mission reuse, and preservation of the exact
mandatory threshold despite weaker LLM suggestions.

Focused CPU suite (tracks, agents, mission contract, production shared memory):
**46 passed**. Root is separately collecting fresh real model operation receipts;
these unit tests use offline stubs and do not establish a newly accepted LLM
controller. The actual historical accepted improvements remain curated fallbacks.

Remaining interpretive limits: context token counts are character/4 estimates;
conflict counters are still probes, not comprehensive contention measurement;
context-efficiency gain is an accounting proxy, not an experimentally measured
no-memory baseline. PING_HUMAN remains an enumerated decision with no current
execution branch. Fleet autonomy remains separate from the benchmark boundary.
