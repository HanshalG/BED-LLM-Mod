# Tau2 MMS Split-Partition Calibration Protocol

Date frozen: 2026-08-13

## Predecessor Boundary

The nine-action partition interface is terminally closed. Four of six responses
obeyed its codec, with 35/36 exact action partitions, but one response switched
schema and one exhausted 2,500 tokens while exposing malformed JSON. Native
permission separation was exact in only three of four valid responses. No
registered semantic score or endpoint opened.

This successor changes both cohort and interface before any new response. It
does not relax any predecessor gate or reuse any seed.

## Fresh Cohort

Select reserve positions nine through eleven within each of `mms_abroad` and
`mms_home`: six untouched four-world episodes, hash-bound in the public
manifest. They are disjoint from all three previous Tau2 semantic cohorts.
Task IDs, source faults, official observations, repair actions, and endpoints
remain unserialized.

## Split LLM Interface

Use exact `deepseek/deepseek-v4-flash-0731` nonreasoning. Each episode has two
independent seeded requests:

1. **Root equivalence:** for each of the eight legal root reads, return the
   exact action ID, a boolean `all_worlds_same`, and confidence. This is a small
   fixed array and cannot report the unlocked action.
2. **Native partition:** return only `messaging_permissions`, a canonical
   first-occurrence four-integer `world_groups` array, and confidence.

The native request explicitly states that the action is available only after
`installed_apps`, but asks only what visible permission observation each world
would yield. The model sees candidate-world descriptions and action semantics,
but no selected task ID, official response, hidden truth, repair, reward, EIG,
policy, or endpoint.

Code maps root booleans to constant or singleton partitions, converts both
interfaces' confidence to normalized likelihoods, performs Bayes updates, and
computes greedy and depth-two information values. Root and native responses are
never concatenated into a single model-generated plan.

## Frozen Execution

- Twelve calls: root seeds `202608130500`--`505`, native seeds
  `202608130600`--`605`, paired by episode.
- Temperature zero; maximum output 900 root tokens and 300 native tokens.
- Zero retries; concurrency two; exactly twelve maximum HTTP attempts.
- Atomic per-response checkpoints with request kind, episode index, and seed.
- A complete ordered bank is required before any official observation opens.
- Stage cap `$0.06`; per-request reservation `$0.004`.
- Europe/London daily cap `$5.00`, frozen Aug-13 cumulative usage boundary
  `$220.134128880`, chained through the partition predecessor's ledger.

Any malformed response, schema switch, truncation, provider error, missing
pair, binding mismatch, budget race, or verifier disagreement fails closed.
There is no retry, resume, replacement, or partial scoring.

## Conjunctive Gates

1. Exactly twelve accepted requests and HTTP attempts; zero retries,
   provider-error retries, reasoning tokens, or forced exits; cost at most
   `$0.06`.
2. Exactly 48 root decisions; at least 47 exact and root Brier at most `0.08`.
3. All six native partitions exact. Across their 36 pair relations, accuracy
   is `1.0`; native partition Brier is at most `0.08`.
4. Across 24 realized native answers, all 24 true worlds top-rank; mean truth
   posterior is at least `0.65`, and posterior Brier at most `0.18`.
5. Equivalent-world mean/max predicted TV is at most `0.03`/`0.10`.
6. Greedy avoids and depth two selects `installed_apps` in all six episodes;
   every semantic horizon gain is at least `0.50` nats.
7. Semantic-versus-source depth-two root-value Spearman is at least `0.90`
   over all 48 root values.
8. Independent producer-free replay, privacy, ordering, budget, and unopened
   downstream checks all pass.

Passage authorizes only a separately frozen paired policy-development protocol
with compute-matched myopic and random controls and sealed task-success
endpoints. Failure closes this interface, cohort, and seeds.
