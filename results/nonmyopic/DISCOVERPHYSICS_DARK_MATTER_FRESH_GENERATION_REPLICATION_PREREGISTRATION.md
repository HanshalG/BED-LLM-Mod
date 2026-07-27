# DiscoverPhysics Fresh-Generation Retained-Support Replication

## Status

Frozen before any new model response. This is the sole fresh-generation
replication of the positive retained-support architecture.

## Fixed Motivation

The first frozen GPT-5.4 support tree passed every gate on 384 fresh physical
maps:

- retained-support B versus myopic D: `22.6%` MSE reduction;
- retained-support B versus same-root fixed B: `2.41%` reduction, paired
  interval `[.0341, .1100]`; and
- nearest-support risk reduction: `22.1%`.

Its main unresolved limitation is that physical maps and noise were fresh,
but LLM generation was not. This replication changes the LLM outputs once
while keeping the successful architecture fixed.

## Phase A: Fresh Support Tree

Run the existing simulator-grounded generator unchanged:

- script SHA-256:
  `f8f201d0991ca3c615144b8e638beb66428c3133bcf8589005389519e426f5ff`;
- model: `openai/gpt-5.4`;
- reasoning: disabled;
- temperature: `0`;
- one initial executable eight-map support call;
- eight independent branch-conditioned support/continuation calls;
- exact total: `9` requests;
- projected cost: `$0.14`;
- hard run cap: `$0.25`;
- no semantic repair, normalization, reparse, reissue, or manual retry.

Phase A passes only if:

- requests and HTTP attempts are exactly `9`;
- retries, reasoning tokens, and forced exits are all `0`;
- all nine outputs pass the frozen strict parser and compiler;
- all eight refreshed supports differ from the initial support;
- at least three roots have branch-distinct supports;
- center B branches choose distinct continuations;
- every refreshed support retains at least two regions; and
- total adapter cost is at most `$0.25`.

The generator also computes its old replacement-policy endpoint on already-
open seeds `24520--24525`. Those scientific metrics are ignored regardless of
sign and cannot authorize or reject Phase B.

After Phase A, commit and hash `MODEL_FROZEN.json` and `POLICY.json` before
generating a replication endpoint. A Phase-A failure closes the replication.

## Phase B: Fresh Retained-Support Endpoint

Only after a clean Phase A, evaluate:

- initial-support component mass `.95`;
- regenerated-support component mass `.05`;
- exact full-history official-simulator likelihoods;
- map seeds `24630--24645`, inclusive;
- `24` maps per seed, `384` total, `96` per region;
- asymmetric region prior NE/NW/SW/SE = `.4/.3/.2/.1`;
- observation-noise seed `24646`;
- bootstrap seed `24647`, `10,000` stratified samples;
- `8` root observations per map and `4` continuations per root observation;
- same-root fixed B, retained-support myopic, and random-root controls.

No other mixture mass, support combination, model output, or map subset may
be evaluated on these seeds.

## Frozen Scientific Gates

All must pass:

- fresh-support myopic root is D;
- fresh-support retained lookahead root is B;
- at least `10%` internal risk reduction for B versus D;
- at least `10%` fresh-map MSE reduction for B versus D;
- positive paired-bootstrap lower bound for `MSE(D)-MSE(B)`;
- at least `5%` MSE reduction for B versus random A;
- at least `1%` MSE reduction for retained B versus same-root fixed B;
- positive paired-bootstrap lower bound for
  `MSE(fixed B)-MSE(retained B)`; and
- at least `5%` retained-union nearest-support risk reduction.

Any failed gate is a replication null and closes this exact architecture. No
additional generation, alternate weight, second map block, subset, or rerun
is permitted.

## Claim Scope

A pass would show that the small path-dependent support gain survives both a
fresh GPT-5.4 support tree and a fresh official-simulator endpoint. It would
still rely on one model family, one prompt/compiler, and exact physical
likelihoods rather than LLM-estimated likelihoods.

OatML use is prohibited. The maximum new OpenRouter spend is `$0.25`, leaving
at least `$0.50432155` of the strict research allowance and the protected
`$25` account reserve.
