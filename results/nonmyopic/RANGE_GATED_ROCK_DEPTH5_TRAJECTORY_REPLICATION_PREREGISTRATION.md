# Focused Range-Gated Rock H5 Trajectory Replication Preregistration

Frozen before any live response, truth, observation, or endpoint from the
replication seeds below.

## Claim And Unchanged Protocol

The first audited trajectory run at seed `24245` showed that cached hierarchical
h5 exactly matched exhaustive d5 and gained `.283217` entropy-AUC nats over the
same compiled plans scored at h4 and `.204126` over matched-random h5 targets.
This package tests whether that result is robust to fresh paired trajectory seeds
and fresh provider calls. The original run is not included in the primary pooled
endpoint.

The environment, focused prior, eight-round budget, 50 paired trials per seed,
Gemma-4-26B-A4B thinking model, prompt, target compiler, exact verifier, cache,
serving limits, controls, and per-run gates remain byte-for-byte unchanged from
`RANGE_GATED_ROCK_DEPTH5_TRAJECTORY_PREREGISTRATION.md`. No new S0 is needed
because this is a direct replication of the already passed serving interface.

## Frozen Seeds

| Replication | Trajectory seed | Independent audit seed |
| --- | ---: | ---: |
| R1 | 24250 | 24251 |
| R2 | 24252 | 24253 |
| R3 | 24254 | 24255 |

The producer pooled bootstrap seed is `24256`; the independent pooled audit seed
is `24257`. Each bootstrap uses 10,000 replicates and samples 50 paired cases
with replacement within each trajectory seed before averaging all 150 cases.

## Frozen Gates

The replication package passes only if:

- all three unchanged producer runs pass every original mechanical and endpoint
  gate;
- all three independent trajectory audits pass;
- each of the six per-run paired mean gains is positive in every replication:
  entropy and truth-log AUC versus shared h4, matched-random h5, and exhaustive
  d4;
- all six pooled producer and independent-audit 95% lower bounds are positive;
- pooled exact-d5 gap recovery is at least `.60`;
- pooled registered-route and on-site-by-round-five rates are at least `.75`;
- the expected seeds, 150 paired cases, 1,200 logical decisions, per-run cache
  caps, usage, and no-LLM-inside-scoring invariants all validate.

No failed or missing replication may be replaced. There is no alternate seed,
prompt, model, token budget, projection, or threshold. A partial result is
reported as partial and does not become the registered pooled confirmation.

## Spend

- OpenRouter project ceiling: `$110`.
- Spend before replication responses: `$40.398566282459846`.
- Remaining: `$69.60143371754015`.
- Per-replication cap: `$0.50`.
- Package cap: `$1.50`.
