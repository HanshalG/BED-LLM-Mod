# HotpotQA Shared Comparative V2 Preregistration

## Purpose

Test non-myopic first-link selection over LLM-generated, path-dependent belief
states on exact two-document HotpotQA directional unlocks.

The prior confirmation never reached scoring because one JSON refresh was
truncated. V2 is scientifically distinct as well as transport-distinct:

- it uses strict short line grammars;
- it replaces uncalibrated 0--100 future-only maximization with one shared
  comparative ranking of complete root-plus-continuation paths;
- myopic and non-myopic use the same aligned receding continuation selector,
  isolating the first root choice; and
- fixed-belief and shuffled-belief policies test whether correct
  path-dependent belief alignment is load-bearing.

## Frozen Source

Official Hugging Face converted Parquet revision `refs/convert/parquet`,
configuration `distractor`, split `train`.

- shard 0 SHA-256:
  `76d3bb3048a7cc73c1958107c0c5872a00d7e7d00c105b81e92f6769e7822e68`
- shard 1 SHA-256:
  `713661628434fbb19fff7392e2e321e4ed107e3c7c7784d0690946e5f722763f`
- development ordered-ID SHA-256:
  `01755657af1915c4c3171cbc2454f5c12ea6e48e18f8c5fae7cf7fbb68db3a7a`
- holdout ordered-ID SHA-256:
  `f1fc6d1119a8fdda01860055adf1d146f900433c76f1af508bc725ff4ec088d8`

The 100-row development split was materialized before model use and contains
exactly five rows satisfying the frozen structural qualification rule. Select
all five in split order:

```text
5adf874e5542995ec70e902a
5a7fc81955429969796c1b5f
5abd512655429924427fcfb4
5a8b1dd65542996c9b8d5fa6
5ac49ff65542996feb3fe91f
```

They comprise two easy and three medium questions. The holdout remains sealed.

## Model And Calls

Model: `openai/gpt-5.4`, temperature zero, explicit non-reasoning.

Each task uses exactly ten logical calls:

1. one eight-hypothesis initial belief call;
2. four root-paragraph belief refreshes;
3. one myopic root-ranking call from the initial belief;
4. one shared aligned two-step plan-ranking call;
5. one fixed-initial-belief plan-ranking control;
6. one cyclic-shuffled-belief plan-ranking control; and
7. one final answer from the non-myopic selected documents.

All outputs use strict bounded line grammars. Bounded adapter transport retries
are logged separately. Semantic repair or reissue is forbidden.

## Policies And Endpoint

- **Non-myopic:** top root and its continuation from the aligned shared plan.
- **Myopic receding:** top immediate root, then that root's continuation from
  the same aligned plan.
- **Fixed:** root and continuation from initial beliefs repeated at every root.
- **Shuffled:** root and continuation from cyclically wrong branch beliefs.
- **Random receding:** seeded random root, then its aligned continuation.

The model never sees answers, supporting facts, support roles, or endpoint
values. All policy choices and the final answer freeze before exact support
coverage is computed. The primary endpoint is annotated support-document
coverage after two selected articles. Final-answer token F1 is secondary.

## Serving Gate

Use only the already-open validation smoke record. Require exact ten logical
calls, HTTP attempts equal calls plus bounded retries, complete strict parsing,
zero reasoning tokens and forced exits, all four refreshes changed and
distinct, and cost at most `$0.25`.

## Development Gate

Run all five frozen development tasks only if serving passes. Every condition
is conjunctive:

- all 20 refreshes change and all five tasks have four distinct states;
- aligned root order differs from fixed and shuffled on at least `3/5` tasks;
- non-myopic changes at least `4/5` myopic roots;
- non-myopic selects the structurally enabling root on at least `4/5`;
- non-myopic covers at least `9/10` support documents;
- versus myopic: gain at least `3`, at least four wins, zero losses, and exact
  one-sided sign-flip `p <= .0625`;
- gain at least `2` versus each fixed, shuffled, and random control;
- aligned root pairwise accuracy at least `.65` and at least `+.10` over
  myopic; and
- mean final-answer token F1 at least `.50`.

Pass authorizes a separately frozen, powered test on untouched holdout rows.
Failure closes exact Hotpot V2 without task, threshold, parser, or prompt
repair.

## Budget

- Serving projected/cap: `$0.15 / $0.25`.
- Development projected/cap: `$0.75 / $1.50`.
- Authenticated pre-gate balance: approximately `$29.601082094`.
- No fixed reserve.
- OpenRouter only; no OatML or Slurm.
