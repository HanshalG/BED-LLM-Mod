# tau-Knowledge Retrieval Opportunity V2 Result

## Decision

Both the serving smoke and six-task opportunity stage passed every frozen gate.
This establishes an externally grounded non-myopic retrieval opportunity and
authorizes a separately preregistered target-blind scorer study. It does not
yet establish that an LLM can select the non-myopic root.

## Serving smoke

The smoke completed exactly 14 calls with zero reasoning tokens, retries,
forced exits, or malformed responses and cost `$0.09364500`. The two tasks
produced 3 and 4 distinct first-query top-1 documents. Both gained one required
document from a second retrieval; one had a non-myopic gap of one document.

The generated openings passed the frozen zero-call leakage audit. One reproduced
the script's explicit opening. The other volunteered only identity and
occupation required in every message while withholding income, fee ceiling,
spend volume, subscription status, and later application behavior.

Public artifact SHA-256:
`5a2291898472842a541859e180875c56a3530eff4f3d7db353f14aa7b586d0bc`.

Private raw checkpoint SHA-256:
`3fd44b558f85a9c13b0c8b38808ce7212afb8cddb2179330a53c158137e9e590`.

## Opportunity

The opportunity stage completed exactly 42 calls with zero reasoning tokens,
retries, forced exits, or malformed responses and cost `$0.28575750`.
All frozen gates passed:

- mean first-query top-1 diversity: `4.0` documents;
- oracle pair retrieved required policy on `6/6` tasks;
- pair gain of at least one document on `5/6` tasks;
- mean pair gain: `1.3333` documents;
- oracle and oracle-strength greedy roots differed on `2/6` tasks;
- non-myopic gap of at least one document on `2/6` tasks; and
- mean non-myopic gap: `0.3333` documents.

The two gap cases were substantively interpretable. A referral-link task
benefited from a different referral-specific root, while a card-selection task
benefited from beginning with subscription-benefit eligibility instead of the
strongest immediate feature query.

Public artifact SHA-256:
`9d834868379f13045a2fa7d59fdf4b0ce904dcfbac0df8284d72daa85f182616`.

Private raw checkpoint SHA-256:
`c7bb45de7406201530c6811c8d3223f86a3873b547d214085e5c25761e9499fa`.

## Interpretation

This result closes the opportunity question for this interface: an LLM can
generate path-dependent semantic search beliefs whose best two-step plan is
sometimes inaccessible beneath even an oracle-strength myopic first root.
The remaining first-link question is selection fidelity. The 20 original V1
confirmation tasks remain sealed and may be used only after a target-blind
scorer protocol is preregistered and committed.

## Budget

Combined V2 cost was `$0.37940250`. The project ledger is `$71.10190466`
spent with `$34.28289803` headroom. The live account has `$59.55727054`
remaining, or `$34.55727054` above the protected `$25` Monday reserve.
OatML was not used.
