# KnowU Dynamic-Support Cross-Judge Result

Date: 2026-07-26

Status: **frozen confirmation gate failed**

## Result

The independent Gemma 4 26B A4B serving gate passed:

- exact requests / HTTP attempts: 10 / 10
- all ten strict judgment objects parsed
- reasoning tokens, retries, and forced exits: 0
- cost: $0.00145171, below the $0.05 cap

The six-world confirmation then failed closed at the frozen exact parser:

- exact requests / HTTP attempts: 6 / 6
- reasoning tokens, retries, and forced exits: 0
- cost: $0.00721724, below the $0.15 cap
- five of six otherwise complete JSON responses were enclosed in Markdown code
  fences
- no support generation, user simulation, repair, retry, or reissue occurred

The preregistration makes this a serving failure. It leaves the GPT-5.4
support-entry result same-model and development-only, and it does not
authorize candidate ranking or a depth sweep.

## Zero-Call Diagnostic

After recording the frozen failure, a non-confirmatory diagnostic removed
only the five outer Markdown fences from the already-paid responses. All six
objects then parsed, but the scientific gates still failed:

- binary truth-presence agreement with GPT-5.4: 23 / 30 = 0.7667, below 0.80
- cross-judge initially missing worlds: 1
- cross-judge worlds with truth entry: 1
- preregistered target `T1W3`: initially present, score 95
- `T1W3` operating-system/platform scores: 95 and 90
- maximum target gain: 0, below 8

The disagreement is substantive. Gemma considered `T1W3` fully covered
before any question, erasing the GPT-5.4 entry event. It instead scored
`T1W2` as 40 initially and 80--90 after every branch, whereas GPT-5.4 kept
that world's dual-machine preference absent at 48--60. Thus the two model
families disagree both about baseline truth coverage and about which world
exhibits answer-conditioned support entry.

## Interpretation

The free-text truth-coverage endpoint is not robust enough to support the
next causal link. The original mechanics result remains evidence that one
model can regenerate a support containing a semantically richer state after
an answer, but the effect depends on the judge family and threshold
interpretation. Treating this as non-myopic policy evidence would therefore
be circular.

No KnowU ranking-fidelity or policy run is authorized from this checkpoint.
A future KnowU attempt would need a separately preregistered, less subjective
endpoint rather than a parser repair or rerun of these worlds.

Public serving artifact SHA-256:
`3580fd562054f62e8df0fcbc34c780c37b2cc0c51ac163fd95ba82e79212f2c8`.
Public failure artifact SHA-256:
`02158aaef276c6f15cd325e7d1c9612524fff4dedfa400818ee5f869c607dea0`.
Private confirmation raw SHA-256:
`6c556c96787367eef8cc33b40064be42cb93618905cc73094e17be583af36380`.
