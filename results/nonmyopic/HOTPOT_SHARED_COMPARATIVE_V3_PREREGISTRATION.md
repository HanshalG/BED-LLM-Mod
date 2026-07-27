# HotpotQA Shared Comparative V3 Preregistration

## Purpose

Test the same LLM-native, path-dependent non-myopic first-link policy as V2
with a scientifically unchanged but transport-distinct rank-per-root codec.

V2 stopped at its first aligned plan because the model returned a
space-delimited `ORDER` line. V3 does not accept, repair, or rerun that
response. It removes the separate order line and requires one row per root:

```text
R1|T3|2
R2|T1|1
R3|T7|4
R4|T2|3
```

The second field is the selected follow-up and the third is a unique rank from
one through four. Every other policy, control, source, endpoint, and
development threshold remains the same as frozen V2.

## Source And Split Integrity

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

Serving uses confirmation row `5ae67c685542996d980e7b84`. It is not a fresh
scientific example: this row was one of ten whose question, titles, and all
four root paragraphs were already sent to the model during the closed
future-uplift confirmation. The exact ordered list of those ten exposed IDs
must reproduce before serving.

If serving passes, development uses the same five V2-frozen rows:

```text
5adf874e5542995ec70e902a
5a7fc81955429969796c1b5f
5abd512655429924427fcfb4
5a8b1dd65542996c9b8d5fa6
5ac49ff65542996feb3fe91f
```

The holdout remains sealed.

## Model And Estimator

Model: `openai/gpt-5.4`, temperature zero, explicit non-reasoning.

Each task uses exactly ten logical calls:

1. one eight-hypothesis initial open-world belief;
2. four root-paragraph belief refreshes;
3. one myopic root rank from the initial belief;
4. one aligned complete-path rank using branch-matched refreshed beliefs;
5. one fixed-initial-belief complete-path rank;
6. one cyclic-shuffled-belief complete-path rank; and
7. one final answer from the non-myopic selected documents.

The aligned, fixed, and shuffled calls use the new four-row rank codec.
Bounded adapter transport retries are logged separately. Semantic repair,
extraction, delimiter substitution, or response reissue is forbidden.

Policies are unchanged:

- non-myopic selects the aligned rank-one root and its aligned follow-up;
- myopic selects its immediate rank-one root and uses that root's same aligned
  follow-up;
- fixed and shuffled use their own complete-path ranks and follow-ups;
- random uses a seeded random root and its aligned follow-up.

All choices and final answers freeze before supporting facts or endpoint
values are loaded. Primary endpoint is exact support-document coverage after
two selected articles. Final-answer token F1 is secondary.

## Gates

Serving uses the one previously exposed confirmation row and requires:

- exact ten logical requests;
- HTTP attempts equal requests plus logged transport retries;
- complete strict parsing with no repair;
- zero reasoning tokens and forced exits;
- all four refreshes changed and mutually distinct; and
- cost at most `$0.25`.

Only a complete serving pass authorizes development. Development runs all five
frozen rows and retains every V2 conjunctive gate:

- all 20 refreshes change and every task has four distinct states;
- aligned root order differs from fixed and shuffled on at least `3/5`;
- non-myopic changes at least `4/5` myopic roots;
- non-myopic selects the enabling root on at least `4/5`;
- non-myopic covers at least `9/10` supports;
- versus myopic: total gain at least `3`, at least four wins, zero losses, and
  exact one-sided sign-flip `p <= .0625`;
- gain at least `2` versus each fixed, shuffled, and random control;
- aligned root pairwise accuracy at least `.65` and at least `+.10` over
  myopic; and
- mean final-answer token F1 at least `.50`.

A pass authorizes only a separately frozen powered holdout test. Any serving
or development failure closes exact V3 with no same-row repair or rerun.

## Verification And Budget

Both official train shards reproduce. A real-file fake-adapter rehearsal on
the exposed serving row completes exactly ten calls, all parsers, frozen
choices, and delayed endpoint scoring. Fourteen focused V2/V3 tests pass.

- Serving projected/cap: `$0.15 / $0.25`.
- Development projected/cap: `$0.75 / $1.50`.
- Authenticated pre-V3 balance: `$29.591827094`.
- No fixed reserve.
- OpenRouter only; no OatML or Slurm.
