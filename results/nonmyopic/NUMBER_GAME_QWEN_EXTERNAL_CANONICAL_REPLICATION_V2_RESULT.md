# Number Game Fresh Qwen External-Canonical Replication V2 Result

Run:
`number-game-qwen-external-canonical-replication-v2-20260729T063429Z`

Status: **scientific primary replicated; composite gated null on transport
retry cap**.

## Primary

On 32 wholly fresh Qwen-generated path-dependent belief trees and all 33
exact canonical concepts:

- depth-three Brier: `0.1051233`;
- myopic-EIG Brier: `0.1202950`;
- relative reduction: **12.612%**;
- whole-tree bootstrap difference interval:
  `[-0.024465,-0.007040]`;
- tree wins: **25/32**;
- Hamming reduction: `12.609%`;
- exact truth-extension coverage difference: `+0.01515`.

All three frozen scientific primary gates pass: at least 8% Brier gain,
interval strictly below zero, and at least 20 tree wins.

## Boundary And Controls

- Cross-fitted depth two: `0.1055680`; depth three improves only `0.421%`,
  interval `[-0.004309,+0.003908]`, 11/32 wins. The deeper-step diagnostic
  is null.
- Positive-test strategy: depth three improves `6.665%`,
  interval `[-0.012812,-0.002403]`, 22/32 wins.
- Uniform random candidate root: depth three improves `8.190%`,
  interval `[-0.012930,-0.005467]`, 27/32 wins.
- Fixed-support depth three: directional `3.843%` improvement, but its
  interval crosses zero.
- Exact-bank rank correlation is `0.323` for depth three versus `0.286` for
  depth two.

This independently reproduces the first fresh Qwen cohort's non-myopic versus
myopic effect (11.37%, interval below zero, 25/32 wins), while showing that
monotonic depth is not robust across cohorts.

## Mechanics

- Exactly 32 fresh trees, 33 canonical targets, and 1,856 accepted calls.
- 1,865 HTTP attempts and 9 provider-error retries.
- Exact attempt accounting, zero reasoning tokens, zero forced exits.
- Every initial, validation, retained first, and retained second support gate
  passed.
- Cost: `$2.36537776`, below the `$3.50` cap.

The frozen retry cap was eight. Nine transparent provider retries therefore
make the registered composite status `gated_null` by one retry even though
all accepted outputs, all support mechanics, and all scientific primary gates
pass. No rerun or reclassification is performed.

Public hashes:

- `RESULT.json`:
  `a03c5a6f6e01af403ce27ef7984e29176bf0aa5f40caeae2bb6d213ce8c5dc83`
- `TREES.json`:
  `f1ccd0b9a5f0e40786e8673a30400492626ae7a3f26de4dea42ac4316dc3d38c`
- `TARGETS.json`:
  `6b63cf269baa9580066b3b3a10fe16d76826c585f951e8f082d0f4b7c1442936`

Private raw-response SHA-256:
`de522c82d5b97149086d2e49848113f79e0d249779a7943256c6b01ea6b8403a`.
