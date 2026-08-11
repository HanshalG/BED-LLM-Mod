# RegretBench Horizon-Opportunity Audit Protocol

Date frozen: 2026-08-11.

This is a retrospective, zero-call source audit motivated by the terminal failure of the typed-action calibration interface. It asks whether RegretBench's official hidden intent/facet structure contains an environment-grounded depth-two first-action opportunity worth pursuing with a new executable-value interface.

## Population

Audit every official `OpenDomainQA/test` CIG that passes the existing RegretBench LLM-native eligibility rule:

- 3--6 hidden intents;
- 2--4 semantic ask facets;
- every intent has a nonempty answer alias and a nonempty value for every ask facet;
- every ask facet has at least two distinct normalized values.

Verify the pinned RegretBench commit, all published test-file checksums, and the expected eligible count before reporting the result.

## Exact Opportunity Metric

Use a uniform prior over the official hidden intents. Each semantic ask facet deterministically partitions intents by its normalized official slot value.

For each task:

1. The greedy first action maximizes one-step mutual information.
2. The depth-two first action maximizes total expected information after an optimal second, different facet is chosen separately in each first-answer branch.
3. The depth-two first-action gain is the optimal depth-two information minus the depth-two information obtained when the first action is forced to the greedy action.

Report the count and maximum of strictly positive gains using tolerance `1e-12`, plus counts stratified by numbers of facets and intents.

Also report the same opportunity count for untouched tasks with exactly four executable canonical actions after excluding the four already frozen 132-task cohorts: original, factorized-v2, option-ID, and typed-action.

## Interpretation Boundary

This audit uses private source annotations only to characterize benchmark structure. It makes no model call, opens no policy endpoint, and examines no saved LLM response or endpoint outcome.

If the gain is zero, that means the benchmark's official fixed-support CIG provides no depth-two reason to prefer a different first action from greedy. It does not mathematically rule out path-dependent differences induced by an LLM's regenerated beliefs. However, such differences would lack an independent environment-grounded horizon opportunity and could therefore be regeneration noise. A zero result is sufficient to close RegretBench as the primary non-myopic headline route; it cannot be used as evidence that non-myopic planning generally fails.
