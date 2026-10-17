# Independent mission-gate security review (peer B)

HACP contract `c-9c8980772a2649919da5441826cc2f38`, revision 1.
Scope: `agents/contract_orchestrator.py`, `agents/mission_contract.py`, `shared_brain/memory_store.py`,
`scripts/finalize_verification.py`. To check the gate semantics I also read
`core/verifier_harness.py:200-256` (`_aggregate`, `check_contract`, `evaluate`) and `core/integrity.py:32-40`.
I only read code. I ran nothing: no tests, no simulations.

## Summary

The mandatory DoD floor holds. I found no way to drop or loosen the four `MANDATORY_ASSERTIONS`.
- `normalize_goal_assertions` (`mission_contract.py:91-117`) always appends the exact mandatory clauses.
- `FrozenMission` rejects documents that are missing a mandatory clause (`:140-143`), and documents that hold
  invalid, contradictory or duplicate clauses (`:136-138`, `:144-145`).

The confirmed weaknesses are in how the mission and the ledger are bound to each other across runs, and in
how much the final gate trusts ledger fields it never checks.

## Confirmed findings

### M1 (Medium): Mission DoD is frozen per process only and is not bound to the ledger, so extra goal clauses can be replaced between runs
- **Where:** `agents/contract_orchestrator.py:476-478`, `:479-500`, `:248-256`; `scripts/finalize_verification.py:109-111`
- **Evidence:**
  - `FrozenMission` pins the file only for the lifetime of one orchestrator.
  - `_entry` (`:248-256`) stores no mission digest in ledger entries.
  - When the ledger is non-empty, `compile_goal` loads whatever `generated_dod.json` is on disk (`:476-478`).
  - If that file is missing, it silently compiles a *new* mission, possibly from LLM output, and writes it (`:479-500`).
  - `finalize_verification.py:109-111` freezes whatever file exists at finalize time. It reports `mission.digest`
    but never compares it with any earlier recorded digest.
- **Impact:** Stricter goal clauses taken from the user's goal (for example `vision_3d_rmse_mm < 5`) can be dropped
  or replaced after generations have run. One way is to delete the file and re-run `compile_goal`. Another is to
  edit it to a different valid document that still contains the mandatory clauses. In both cases STOP_DONE is then
  judged against a weaker goal than the one the campaign was frozen on. The mandatory clauses are the lower bound.
- **Fix:**
  - Store `mission_sha256` on the first frozen mission, in every ledger entry or in a pinned digest file written
    under the ledger lock.
  - Make `compile_goal` refuse to recompile when the ledger is non-empty and the mission file is missing.
  - Make finalize require `mission.digest` to equal the digest recorded on the accepted entry.

### M2 (Medium): The ledger tamper check is self-referential, and the final Law 5 baseline and the accepted-version choice come from unverified ledger fields
- **Where:** `scripts/finalize_verification.py:29-40`, `:93-108`, `:133-135`; `shared_brain/memory_store.py:151-163`, `:305-317`
- **Evidence:**
  - `validate_ledger` only compares each file's hash with the `sha256` stored in the same `ledger.json`.
    Anyone who can write the ledger can make every row match.
  - The independent copy in `brain.db` (`versions` table, `record_version`, `memory_store.py:155-158`)
    is never checked against the ledger.
  - Finalize takes `accepted` flags and `best` straight from the ledger (`:93-96`). It takes the Law 5 baseline
    from the *historical* `previous['opt'/'holdout']['composite_score']` (`:106-108`, `:133-135`), which it never
    re-evaluates.
- **Impact:**
  - Lowering `previous[...]['composite_score']` in `ledger.json` makes the Law 5 non-regression clauses always
    pass (`verifier_harness.py:220-223`).
  - Marking a rejected, higher-version entry `accepted: true` makes finalize certify that candidate instead.
  - The fresh evaluation still enforces the absolute gates (force, latency, shatter), the `aggregate.passed`
    checks and the mandatory DoD. A regressed version can therefore reach `STOP_DONE`, but a version that fails
    the absolute gates cannot.
  - This requires write access to `shared_brain/`.
- **Fix:**
  - Cross-check every ledger entry (`version`, `sha256`, `accepted`, `opt`, `holdout`) against `brain.db`
    `versions` rows, or chain entry hashes or HMAC them.
  - Better: re-evaluate the previous accepted source fresh in finalize instead of trusting its recorded composites.

### L1 (Low–Medium): Version numbers are allocated outside the ledger lock, so concurrent writers can collide and overwrite evidence
- **Where:** `agents/contract_orchestrator.py:431-433`, `:457`; `shared_brain/memory_store.py:337-339`, `:309-317`, `:155`
- **Evidence:** `next_version()` reads the ledger under a lock that is released before `append_ledger` takes it
  again. `_accept` writes `versions/v{N}.py` before appending. `record_version` uses `INSERT OR REPLACE`.
- **Impact:** If two orchestrators run against the same `shared_brain/` (the file locks suggest this is expected),
  the second one gets the same `N`:
  - It overwrites `v{N}.py`. The first entry's hash then no longer matches, and `validate_ledger` blocks
    finalization (`finalize_verification.py:38-39`).
  - It silently replaces the `brain.db` evidence row for that version.
  - It leaves duplicate `version` values, so the choice of `best` (`:96`) becomes ambiguous.
- **Fix:** Allocate the version and append the entry inside a single critical section on the ledger lock.
  Use plain `INSERT` in `record_version` so collisions fail loudly.

### L2 (Low): A stale or partial final report can survive a failed run
- **Where:** `scripts/finalize_verification.py:151-160`
- **Evidence:** The report is written only at the end of `finally`. If anything before that raises, the write never
  happens and the exception escapes. Examples: a corrupt `evolved_agent_prompts.json` (`:154`), or a missing or
  locked `brain.db` (`:156`). `out.write_text` is also not atomic.
- **Impact:** A previous `dev/final_verification.json` with `"passed": true, "decision": "STOP_DONE"` stays on disk
  after a failed re-run. Any consumer that reads the file instead of the exit code sees a stale pass.
- **Fix:**
  - At the start, unlink `out` or write a failing placeholder.
  - Wrap the receipt and prompt collection in its own try/except.
  - Write the report atomically.

### L3 (Low): TOCTOU between the source hash check and evaluation
- **Where:** `scripts/finalize_verification.py:117-122`, `:143`
- **Evidence:** The source hash is checked before evaluation and again after it. `evaluate()` loads the file from
  disk in between (`verifier_harness.py:241`).
- **Impact:** A local writer could swap the file during evaluation and restore it afterwards, and neither check
  would notice.
- **Fix:** Copy the accepted source into a private temporary file, hash the copy, and evaluate the copy.

## Concerns (not confirmed as vulnerabilities)
- **Baseline skips the gate.** `ensure_baseline` accepts v0 without calling the checker
  (`contract_orchestrator.py:267-276`), and `decide` can label it `STOP_DONE` (`:275`). `finalize` re-applies every
  gate to fresh results, so this does not bypass the final gate. It does mean the in-run `STOP_DONE` for v0 is not
  gated by Law 5.
- **Drift probe can promote v0.** `drift_probe` will promote v0 again if it passes the gate (`:581-583`). This
  looks intended, but it can record the baseline as the newest accepted version.
- **In-memory integrity check proves little.** `FrozenMission.assert_unchanged`'s in-memory check (`:175-176`)
  rehashes bytes stored in the same object. Only the file comparison gives real protection.

## Checked and found sound
- **Mandatory clauses are enforced.** Incompatible clauses are dropped and duplicates are refused
  (`mission_contract.py:66-81`, `:136-145`).
- **The final pass condition is strict.** It requires Law 5, every frozen DoD clause, and both aggregates passing
  (`finalize_verification.py:146-147`). Any exception keeps `passed=False`.
- **Holdout diagnostics are blocked.** See `finalize_verification.py:122-124` and `verifier_harness.py:239-240`.
- **The integrity check fails closed.** `verify_integrity` raises on any mismatch (`core/integrity.py:37-39`).

## Limits
- I ran nothing. Every finding comes from reading the code.
- I did not inspect `agents/crusoe_client.py` (`FallbackSynthesizer`), `agents/sandbox_exec.py`, or the
  `LOCKED_FILES` list.
- The contract `inputs` are relative paths that resolve against the HACP project directory, not the repo root.
  The reviewed files themselves are the repo files named above.
