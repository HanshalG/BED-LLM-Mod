# Number Game External Canonical-Target Replay Preregistration

Date frozen: 2026-07-29, after identifying the external source and before
running this replay or inspecting any canonical-target policy outcome.

## Question

Does the frozen Qwen-planner cross-fitted depth-three policy retain its
proper-score advantage over depth two and the existing controls on an
externally specified target bank?

The target bank is the complete 33-concept Number Game hypothesis space
listed by Tenenbaum and Griffiths, *Generalization, similarity, and Bayesian
inference*, DOI `10.1017/S0140525X01000061`: even, odd, prime, square, and
cube numbers; multiples 3--10; powers 2--10; final digits 1--9; repeated
two-digit numbers; and numbers below 100.

## Frozen Inputs

- Qwen 31-tree replay `RESULT.json` SHA-256:
  `a645c92126c9b5fd69e28532fb621028781dcf19d8ec890cd7758e6d1df710f2`.
- Qwen 31-tree replay `TREES.json` SHA-256:
  `7abae080ec286b1991a941bf1de0ceb4ea0d3997a648a43ae4bd6ceeaa6c151a`.
- All 31 complete trees are included. The incomplete 32nd formal tree remains
  excluded.
- The 33 target predicates are encoded in
  `scripts/number_game_external_canonical_replay.py` before execution.

The paper uses integers 1--100 while the frozen policies query 0--100. Each
predicate receives its natural value at zero: zero is even, square, cube, and
a multiple; it is not prime or a positive integer power; digit-ending
categories remain 1--9; repeated two-digit numbers remain 11--99; and the
catch-all-below-100 rule includes zero. No concept is removed or reweighted.

## Endpoint

- Zero model calls and zero spend.
- Replay every frozen policy on every one of the 33 external concepts.
- Equal weight for each concept, then equal weight for each of the 31 trees.
- Primary endpoint: posterior-predictive Brier score over the complete
  0--100 extension after three selected queries.
- Secondary endpoints: best-support Hamming error, exact truth-extension
  coverage, and per-root endpoint Brier.
- Uncertainty: the existing frozen 20,000-draw whole-tree bootstrap.
- Controls: cross-fitted depth two, retained-risk depth three,
  predictive-risk depth two, myopic EIG, fixed-support depth three, uniform
  random candidate root, and seeded positive-test strategy.

## Gates

The result is `passed` only if every mechanical and scientific gate holds:

- exactly 31 hash-bound trees and 33 unique targets;
- depth three improves Brier over cross-fitted depth two by at least 1%, the
  whole-tree 95% interval for candidate minus baseline lies below zero, and
  depth three wins at least 12 trees;
- Hamming and truth-extension coverage do not regress;
- depth three beats myopic and fixed-support depth three by at least 5%, and
  positive-test strategy by at least 3%, with each interval below zero; and
- depth-three rank fidelity is at least 0.7 and exceeds depth two by at least
  0.15.

Failure of any gate is a frozen null. No target deletion, family reweighting,
domain masking, root repair, threshold change, or replacement endpoint is
allowed.

## Interpretation

Passing would show that the already-generated LLM belief trees support a
non-myopic proper-score gain on a literature-defined concept bank rather than
only on targets proposed by another LLM. It would not make the trees
prospective: their policy-selection machinery remains development-frozen, and
the external bank supplies a new endpoint rather than a fresh planner run.

The associated 2016 human-response release (DOI
`10.7910/DVN/A8ZWLF`) is provenance evidence only. It is not used as a
sequential oracle because it does not release the hidden generating-rule
mapping for each stimulus set.
