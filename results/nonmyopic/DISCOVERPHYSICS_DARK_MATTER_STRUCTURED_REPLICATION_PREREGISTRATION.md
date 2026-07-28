# DiscoverPhysics Structured Fresh-Tree Replication

## Status

Frozen before any new model response or simulator endpoint. This is a
scientifically distinct V2 replication of the retained-support architecture.
It does not repair, rescore, or reverse the failed exact V1 replication.

## Motivation

The first frozen GPT-5.4 support tree passed every retained-support gate on 384
fresh physical maps:

- non-myopic center B reduced MSE by `22.6%` versus myopic northeast D;
- B reduced MSE by `2.41%` versus the same-root fixed-support B control, with
  paired interval `[.0341, .1100]`; and
- branch-conditioned support improved nearest-map coverage by `22.1%`.

The sole fresh-tree replication made all nine intended requests but failed
before endpoints because two JSON responses contained Devanagari text inside
numeric literals (`0. सात1` and `0. सात4`). Seven of nine responses were exact
valid objects. The V1 protocol prohibited repair and remains a failed serving
attempt.

## Prospective V2 Change

V2 changes only the response transport:

- model: `openai/gpt-5.4`;
- reasoning disabled;
- temperature `0`;
- same initial and branch-conditioned semantic prompts;
- same eight-hypothesis support and deterministic compiler;
- same exact official-simulator likelihoods;
- provider-enforced strict JSON Schema for initial and refresh outputs; and
- no clipping, coercion, semantic retry, partial support, manual edit, or reuse
  of any V1 output.

OpenRouter currently reports that the exact GPT-5.4 endpoint supports
`response_format` and `structured_outputs`. One discarded initial-support
preflight tests that exact route. Its semantic content is not inspected or
used. If it does not return a strictly valid compilable support, V2 closes
before the fresh tree.

## Phase A: Fresh Tree

After the discarded preflight, request exactly:

1. one fresh initial eight-map support; and
2. eight fresh branch-conditioned support/continuation objects, two for each
   of roots A--D.

Expected accepted model requests are therefore exactly `10`: one discarded
preflight plus nine scientific tree requests. Transport retries are allowed
under the existing adapter and logged separately; they do not create a new
semantic response. No forced-final continuation is allowed.

Phase A passes only if:

- exactly 10 accepted responses are charged;
- `http_attempts == requests + retry_count`;
- reasoning tokens and forced exits/finals are zero;
- all ten responses pass the frozen strict schema and semantic compiler;
- all eight scientific refreshes differ from the scientific initial support;
- at least three roots have branch-distinct refresh supports;
- center B's two branches choose distinct continuations;
- every refresh retains at least two regions;
- total adapter cost is at most `$0.25`;
- immediate EIG selects root D;
- retained-support lookahead selects root B; and
- B reduces internal retained-support trajectory risk by at least `10%`
  relative to D.

The retained-support internal objective uses `.95` mass on the initial support,
`.05` mass on the routed regenerated support, and exact full-history
likelihoods. These are unchanged from the positive confirmation.

Only after every Phase-A gate passes, write and hash `MODEL_FROZEN.json` and
`POLICY.json`. Phase-A failure leaves all fresh endpoint seeds unopened.

## Phase B: Untouched Physical Endpoint

Only after the frozen files exist and their hashes are recorded, generate:

- map seeds `24700--24715`, inclusive;
- 24 maps per seed, 384 maps total, 96 per region;
- asymmetric NE/NW/SW/SE prior `.4/.3/.2/.1`;
- observation-noise seed `24716`;
- bootstrap seed `24717`;
- 10,000 region-stratified bootstrap samples;
- 8 root observations and 4 continuation observations per map; and
- retained non-myopic B, retained myopic D, retained random A, and same-root
  fixed-support B.

The quarantined V1 seeds `24630--24645`, noise seed `24646`, and bootstrap seed
`24647` remain unopened. No old endpoint, alternate mixture mass, subset, or
additional fresh tree may be evaluated.

## Frozen Scientific Gates

All must pass:

- B reduces fresh-map trajectory MSE by at least `10%` versus D;
- the paired 95% bootstrap lower bound for `MSE(D)-MSE(B)` is positive;
- B reduces MSE by at least `5%` versus random A;
- retained B reduces MSE by at least `1%` versus same-root fixed B;
- the paired 95% bootstrap lower bound for
  `MSE(fixed B)-MSE(retained B)` is positive; and
- the retained union reduces nearest-support trajectory risk by at least `5%`.

Any failed gate is a replication null. No threshold, prompt, schema, seed,
support weight, action, or task subset may be changed after the result.

## Claim Scope

A pass would show that the small path-dependent support gain survives both a
new GPT-5.4 support tree and a new official-simulator endpoint under a robust
serialization interface. It would establish LLM-native support construction,
not LLM-estimated likelihoods, and would remain one physical domain with a
deterministic semantic-to-executable compiler.

## Accounting

- Projected OpenRouter cost: `$0.14`
- Hard run cap: `$0.25`
- Authenticated balance before implementation: `$8.990154844`
- Reserve: none
- OatML, Slurm, SSH, and cluster use: prohibited
