# Animals Multi-Sample Branch-Ranker Holdout Preregistration

Status: **frozen before any response involving the new 60-target set**.

## Protocol

- Fresh producer seed `24281`, bootstrap `24282`, audit bootstrap `24283`.
- Sixty distinct animal targets disjoint from every prior Animals development
  and holdout list.
- One state per target, three ordinary production candidates.
- Four independent non-thinking Gemma 4 26B hypothesis-list calls per branch at
  temperature `.7`, merged before unchanged validation and history filtering.
- Exact target-blind branch-content scorer from the passing development gate.
- No intermediate endpoint is written or inspected.

## Primary Endpoint And Gates

Primary paired endpoint: expected hidden-truth coverage of the ranker's selected
candidate minus immediate EIG's selected candidate.

Pass requires all of:

1. 60 states and 60 distinct targets;
2. at least 20 targets recovered in a six-branch union;
3. at least 20 states with nonzero candidate coverage spread;
4. ranker candidate Spearman is positive and exceeds immediate EIG;
5. producer and independent 95% paired-bootstrap lower bounds are positive;
6. ranker wins exceed losses;
7. ranker active-state regret is below immediate EIG;
8. all payload, response, summary, union-count, generation-count, and gate
   replays pass independent audit.

Any serving, mechanics, producer, or audit failure stops this exact line
without target, seed, prompt, model, threshold, or replacement repair.

Projected coverage cost is `$2`, hard cap `$4`; ranker cap is `$0.10`.
