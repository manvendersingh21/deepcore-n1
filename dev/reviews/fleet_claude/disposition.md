# Peer A disposition: mission-gate security review

Root reviewed the independent report and accepts its five confirmed findings for remediation. The mandatory DoD floor itself was found to hold; the report does not establish a way to weaken those mandatory assertions.

| Finding | Root disposition | Remediation status |
|---|---|---|
| M1: mission rules are not persistently bound across runs | Accepted. | Persist a campaign mission digest/anchor, refuse recompilation when a non-empty ledger has no mission file, and require finalization to match the anchor. Any legacy-ledger migration will be explicit and recorded; historical rows will not be rewritten. Pending implementation. |
| M2: finalization trusts ledger-local hashes, acceptance flags, and previous scores | Accepted. | Cross-check JSON ledger entries against SQLite evidence for version/hash/acceptance/metrics. This does not claim protection against an attacker who can rewrite both stores. Pending implementation. |
| L1: version reservation and append are not one atomic operation | Accepted. | Hold a campaign/ledger lock across version reservation, source write, and evidence append; make collisions fail loudly. Pending implementation. |
| L2: a prior passing report can survive a failed finalization | Accepted. | Make failure reporting atomic and fail-closed even when receipt collection fails. Pending implementation. |
| L3: accepted source can change during evaluation | Accepted. | Evaluate a private hashed copy of the accepted source. Pending implementation. |

The root agent owns these changes. None was implemented or verified in the reviewed snapshot; this disposition records remediation intent, not completion.
