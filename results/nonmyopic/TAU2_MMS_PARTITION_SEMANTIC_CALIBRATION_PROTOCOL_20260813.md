# Tau2 MMS Partition Semantic Calibration Protocol

Date frozen: 2026-08-13

## Motivation And Boundary

The checkpointed absolute-value interface is terminally null. Its 36 field
errors are world-invariant within nine episode/action/field groups, while its
candidate partitions, native answer obedience, and source-value ranking are
strong. The null is not reinterpreted or reopened.

This new interface asks only the decision-relevant semantic question: which
candidate worlds would yield distinguishable visible observations for each
action? It uses six untouched reserve episodes, positions six through eight in
each MMS family, and new seeds. No prior response, selected official response,
or endpoint is reused.

## Interface

Use exact `deepseek/deepseek-v4-flash-0731` nonreasoning. For each of the nine
actions, return an ordered four-integer `world_groups` array plus confidence.
Group labels are canonical by first occurrence: the first world is group zero;
each later world reuses an existing label exactly when its visible observation
would be identical, otherwise it uses the next integer. Thus arbitrary label
permutations cannot mask errors.

The model sees complete candidate-world descriptions, action IDs, and the
native unlock relation. It sees no selected task ID, source fault ID, official
tool output, hidden true world, repair action, reward, EIG, policy, or endpoint.
Code converts groups and confidence into normalized likelihood tables, performs
Bayes updates, and computes greedy and exact depth-two information values.

## Frozen Execution

- Six requests, seeds `202608130400` through `202608130405`.
- Temperature zero, 2,500 maximum output tokens, zero retries.
- Concurrency two; exactly six maximum HTTP attempts.
- Atomic response-level checkpoints; complete bank before official observations.
- Stage cap `$0.06`, per-request reservation `$0.006`.
- Europe/London daily cap `$5.00`, frozen Aug-13 cumulative boundary
  `$220.134128880`, chained through all prior reconciled spend.

Any serving, schema, canonical-label, binding, budget, or replay error fails
closed. There is no retry, resume, replacement, or partial scoring.

## Conjunctive Gates

1. Exactly six clean calls; zero retries, provider-error retries, reasoning, or
   forced exits; cost at most `$0.06`.
2. Exactly 54 action partitions and 324 world-pair relations. At least 52/54
   partitions are exact and pairwise relation accuracy is at least `0.98`.
3. Overall/family mean multiclass Brier is at most `0.08`/`0.10`.
4. All six unlocked `messaging_permissions` partitions are exact. Across 24
   realized native answers, all 24 true worlds top-rank; mean truth posterior is
   at least `0.65` and mean posterior Brier at most `0.18`.
5. For source-equivalent pairs, mean/max predicted TV is at most `0.03`/`0.10`.
6. Greedy avoids and depth two selects `installed_apps` in all six episodes;
   every semantic horizon gain is at least `0.50` nats.
7. Semantic-versus-source depth-two root-value Spearman is at least `0.90` over
   all 48 root values.
8. Independent producer-free replay, privacy, ordering, budget, and unopened
   downstream checks all pass.

Passage authorizes only a separately frozen paired policy-development protocol.
Failure closes this interface, cohort, and seeds and authorizes nothing.
