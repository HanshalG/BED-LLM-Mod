# RegretBench Root Common-Random-Number Amendment

Date: 2026-08-07

**Status: prospective, implemented, unopened. Model calls: 0. Cost: $0.**

## Problem Found

The unopened dynamic depth-two protocol paired conditioned and history-blind
draws with the same seed, but assigned different generator seeds to different
candidate roots. It also assigned different first-refresh and final-refresh
seeds to different realized roots within a task.

That leaves two avoidable variance channels:

- estimated root risks can differ because each root received different model
  randomness, even if the root prompts had identical value; and
- paired realized policy outcomes can differ because policies selected roots
  with different endpoint-generation seeds, not only because their histories
  differed.

No model response, hidden truth, policy endpoint, or scientific result had been
opened when this issue was found.

## Prospective Fix

Simulated planning now uses one seed for each
`(task, simulated hypothesis, draw)` tuple and reuses it across all four roots:

```text
202608100000 + task_index*16 + hypothesis_index*2 + draw
```

The conditioned and history-blind request for each root remain adjacent and
same-seed. Therefore each seed is shared by eight requests: four roots times
two arms. Every different task/hypothesis/draw group retains a distinct seed.

Realized primary execution now uses:

```text
first refresh: 202608110000 + task_index
final refresh: 202608120000 + task_index
```

All distinct roots selected within one task share these stage-specific seeds;
different tasks remain independent. Prompt histories and official environment
replies still differ by selected root, so this couples only requested model
randomness rather than responses.

## New Replay Gates

Primary mechanics now require:

- every conditioned/blind pair has the exact manifest seed;
- every four-root simulated group shares one seed;
- every simulated, first-refresh, and final-refresh row matches its frozen
  formula exactly;
- all `64*8*2 = 1,024` simulated groups have distinct seeds;
- every realized task's selected roots share one first-refresh seed;
- every realized task's selected roots share one final-refresh seed; and
- first/final seeds are distinct across all 64 tasks.

These are executable result gates, not prose-only intentions.

## Adversarial Check

A seed-only candidate scorer with no root effect was evaluated under the old
and new schedules. The old schedule creates a `0.135796` root-risk spread from
seed luck alone. The new schedule gives all four roots exactly equal scores.
The exact-scale synthetic runner also passes every new manifest gate.

Request counts, concurrency, prompts, histories, task splits, endpoints,
policies, thresholds, draw counts, and budget caps are unchanged. This is a
pre-response variance-control repair, not scientific evidence.
