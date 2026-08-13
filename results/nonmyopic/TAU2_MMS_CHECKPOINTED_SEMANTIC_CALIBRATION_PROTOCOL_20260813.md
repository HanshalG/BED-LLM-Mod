# Tau2 MMS Checkpointed Semantic Calibration Protocol

Date frozen: 2026-08-13

## Predecessor Boundary

The first MMS array calibration is terminally closed after five clean responses
and one zero-cost provider error. Its batch API returned no response set, no
official observation was loaded, and no semantic or planning score was
computed. Its cohort and seeds are not reused.

This successor isolates only that serving failure. It retains the array codec,
semantic forward-model role, likelihood construction, Bayes updates, depth-two
planner, and every scientific threshold.

## Fresh Cohort And Interface

Select reserve positions three through five, inclusive, within each of
`mms_abroad` and `mms_home`. The six hashes are frozen in the public manifest;
each episode has four worlds and none appeared in either predecessor mechanics
cohort. Task IDs, source faults, selected tool responses, repair actions, and
task-success endpoints remain unserialized.

Use exact `deepseek/deepseek-v4-flash-0731` nonreasoning with the unchanged
strict ordered world/action/field array schema and closed string values. Code
alone constructs likelihoods, performs Bayes updates, and computes greedy and
depth-two information values. Prompts contain no selected task ID, source fault
ID, raw tool response, hidden truth, repair action, reward, or endpoint.

## Frozen Serving Transaction

- Six new seeds `202608130300` through `202608130305` in episode order.
- Temperature `0.0`, 6,000 maximum output tokens, zero retries.
- At most two concurrent requests; exactly six total HTTP attempts.
- Each completed response is atomically checkpointed by index and seed before
  another completion is processed.
- A complete ordered raw bank is created only after all six responses succeed.
- Selected official observations remain unopened until that complete bank is
  parsed successfully.
- Stage cap `$0.10`, per-request reservation `$0.01`, frozen Aug-13 account
  boundary `$220.134128880`, and predecessor reconciled spend retained.

Any provider error, missing response, malformed partial bank, parser failure,
binding mismatch, budget race, or accounting inconsistency fails closed. There
is no retry, resume, seed reuse, replacement episode, or partial scoring.

## Conjunctive Gates

The exact predecessor scientific gates remain:

1. Six accepted requests and HTTP attempts; zero retries, provider-error
   retries, reasoning tokens, or forced exits; cost within `$0.10`.
2. Exactly 216 cells; typed accuracy at least `0.90`; overall Brier at most
   `0.18`; each MMS-family Brier at most `0.25`.
3. Across 24 native follow-up cells: at least 23 exact signatures, at least 23
   true worlds top-ranked, mean truth posterior at least `0.65`, and posterior
   Brier at most `0.18`.
4. Equivalent-pair mean/max TV at most `0.03`/`0.10`.
5. Greedy avoids and depth two selects `installed_apps` in all six episodes;
   every gain is at least `0.10` nats and mean gain at least `0.50`.
6. Semantic-versus-source two-step Spearman is at least `0.80` over 48 root
   values.
7. The independent replay exactly validates partial/full banks, privacy,
   ordering, metrics, gates, serving usage, bindings, and unopened endpoints.

Passage authorizes only a separately frozen paired development protocol.
Failure closes this interface, cohort, and seeds and authorizes nothing.
