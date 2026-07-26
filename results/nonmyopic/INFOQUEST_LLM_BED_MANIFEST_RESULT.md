# InfoQuest LLM-BED Manifest Result

## Verdict

The content-blind, zero-call source manifest **passes every frozen gate**. This
authorizes only a separately preregistered zero-cost audit of official cached
baseline trajectories on the opportunity split. It does not establish a
non-myopic opportunity or policy result.

## Result

| Quantity | Observed |
| --- | ---: |
| Source revision | exact match |
| Seed-message records | 500 |
| Hidden-setting records | 500 |
| Trait records | 500 |
| Mechanics | 2 |
| Opportunity | 80 |
| Development | 30 |
| Holdout | 388 |
| OpenRouter calls / cost | `0 / $0` |
| OatML jobs | `0` |

All three byte-level source hashes match the pinned Hugging Face release. IDs
are exactly `0..499` in order across all files. Every record matches the frozen
schema; each hidden setting has exactly five nonempty constraints and five
nonempty checklist items; and setting personas and seed messages match their
corresponding seed record.

The four splits are nonempty, disjoint, and cover all 500 IDs. Their combined
SHA-256 is
`1d8f5adbfd30677311ec1a150e2b7c1f7d0804c6ed1d9facf7d13d82b0957edf`.

Only mechanics IDs `0` and `1` were semantically disclosed during source-shape
inspection. The manifest contains IDs, hashes, counts, and gate booleans, but
no seed message, persona, setting, trait, constraint, solution, or checklist
text.

The opportunity, development, and holdout content therefore remain sealed at
this checkpoint. Cached released trajectories can next test whether delayed
information discovery and history-conditioned questioning are prevalent, but
they cannot by themselves establish causal policy efficacy.

## Trajectory-Access Amendment

The first subsequent cached-trajectory inspection revealed that baseline rows,
unlike the three manifest source files, are not ID-ordered. Row-position
selection exposed ID `4` from the original holdout. The immediately recorded
access amendment permanently quarantines ID `4`, leaving the opportunity and
development splits unchanged and 387 effective holdout records. See
`INFOQUEST_TRAJECTORY_ACCESS_AMENDMENT.md`.
