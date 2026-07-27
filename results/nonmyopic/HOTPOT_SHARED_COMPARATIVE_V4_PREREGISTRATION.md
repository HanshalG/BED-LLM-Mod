# HotpotQA Shared Comparative V4 Preregistration

## Purpose

Run the same frozen Hotpot non-myopic first-link test with numeric-root rows for
every ranking call.

V2 failed on the aligned standalone `ORDER` line. V3 converted the three
complete-path rankers but mistakenly retained the standalone myopic `ORDER`
line, which then failed before the new codec was reached. V4 changes only that
remaining transport surface:

```text
R1|2
R2|1
R3|4
R4|3
```

Myopic emits one unique rank per numeric root. Aligned, fixed, and shuffled
continue to emit `Rn|Tm|rank`. No standalone `ORDER` response remains.

This is the last Hotpot transport variant in this cycle. Failure closes the
route. No V2/V3 response is accepted, transformed, or rerun.

## Frozen Source And Tasks

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

Serving uses previously exposed confirmation row
`5adf89c05542993344016cdd`, the second row from the old failed confirmation.
It is distinct from the V2 and V3 serving rows and from every development and
holdout row.

Development remains exactly:

```text
5adf874e5542995ec70e902a
5a7fc81955429969796c1b5f
5abd512655429924427fcfb4
5a8b1dd65542996c9b8d5fa6
5ac49ff65542996feb3fe91f
```

The holdout remains sealed.

## Unchanged Scientific Protocol

Model: `openai/gpt-5.4`, temperature zero, explicit non-reasoning.

Each task uses exactly ten logical calls:

1. one eight-hypothesis initial open-world belief;
2. four root-paragraph belief refreshes;
3. one immediate myopic root rank;
4. aligned, fixed-belief, and cyclic-shuffled-belief complete-path ranks; and
5. one final answer from the selected non-myopic documents.

Non-myopic ranks complete root-plus-continuation paths using correctly aligned
post-root beliefs. Myopic selects only from the initial belief and then uses
the same aligned continuation for its selected root, isolating the first-link
choice. Fixed and shuffled test whether path-dependent belief alignment is
load-bearing. Seeded random is the final control.

All choices and final answers freeze before supporting facts and endpoint
values are loaded. Primary endpoint is exact two-document support coverage;
final-answer token F1 is secondary. Bounded logged transport retries are
allowed. Semantic repair, extraction, delimiter substitution, and response
reissue are forbidden.

## Gates

Serving requires exact ten requests, matching HTTP attempts plus logged
retries, all strict parses, zero reasoning and forced exits, four changed and
distinct branch beliefs, and cost at most `$0.25`.

Only a complete serving pass authorizes all five development tasks.
Development retains the V2/V3 conjunction:

- all 20 refreshes changed and branch-distinct;
- aligned differs from fixed and shuffled on at least `3/5`;
- at least four root changes from myopic and at least four enabling roots;
- non-myopic support coverage at least `9/10`;
- versus myopic: gain at least `3`, at least four wins, zero losses, and exact
  one-sided sign-flip `p <= .0625`;
- gain at least `2` versus fixed, shuffled, and random individually;
- aligned root pairwise accuracy at least `.65` and at least `+.10` over
  myopic; and
- mean final-answer token F1 at least `.50`.

Pass authorizes only a separately frozen powered holdout. Failure closes exact
V4 without a same-row or parser successor.

## Verification And Budget

Both source shards and the ten previously exposed confirmation IDs reproduce.
A real-file fake-adapter rehearsal on the exact V4 serving row completes all
ten requests, every parser, policy freezing, and delayed endpoint scoring.
Twenty-one focused V2/V3/V4 tests pass.

- Serving projected/cap: `$0.15 / $0.25`.
- Development projected/cap: `$0.75 / $1.50`.
- Authenticated pre-V4 balance: `$29.549457094`.
- No fixed reserve.
- OpenRouter only; no OatML or Slurm.
