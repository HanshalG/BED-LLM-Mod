# ChemBench LLM Proposal-Atlas Source Gate Result

Date: 2026-08-15 (Europe/London)

## Result

The frozen source-only gate passes and independently reconstructs from the
pinned ActiveSciBench-Chem source. It opens only the exact DeepSeek proposal
semantics block. It does not open a policy endpoint, v5, or a non-myopic result.

- Implementation commit: `ecf8e119e818e42701c6174a0da5dd4291b11449`.
- Source tasks: 36 unique hidden generators in nine difficulty/history strata.
- Mechanism coverage: 12 core families.
- Atlas split: 27 development tasks and nine held-out tasks.
- Request plan: 63 residual-aware and 63 history-blind requests, paired by seed.
- Prompt lengths: 3,936 to 6,579 characters after adding source-accurate formulas.
- Model calls and cost: zero and `$0.00`.

All source conditions pass: four tasks per stratum, unique generators, family
coverage, triggered latest states, source-oracle top-four recoverability,
prompt privacy, exact request counts, paired seeds, and prompt-size bounds.
The independent verifier exactly reconstructs the public manifest and sealed
labels and confirms that no model call occurred.

## Scope

This gate tests the first residual-to-edit transition from the nine primitive
mechanisms. Even if the paid semantic block passes, a separate prospective
gate must test proposal fidelity after accepted edits and evidence pruning.
Depth-two or depth-three policy evaluation is not authorized directly by this
source result or by primitive-pool proposal semantics alone.

## Bound Artifacts

- Public manifest SHA256:
  `0e00c44d02a56a9662738c1bda4f65127a04cc8122fede773f04aee471219661`.
- Sealed labels SHA256:
  `51a862fe4f04068158b16f296fcae10282abd07dae76ca32e7f9588bca56a108`.
- Source summary SHA256:
  `d2791361d87ac405dcda779e95358714f7f33d8f3b252e6ca33a8c9a53646a62`.
- Independent verification SHA256:
  `bf06c0aeda1170c0d70d5b3f9e8cdce13340ec25e37626acf7547777eea7886b`.
