# Pi-Bench Dynamic-Support Serving Result

Date: 2026-07-28

**Status: the frozen mechanics serving gate passed; development is authorized.**

## Run

- run: `pi-bench-first-link-serving-v9-20260728T011000Z`;
- interface: `pi_bench_dynamic_support_v9`;
- implementation commit: `78c85bf`;
- tasks: the five preregistered one-per-persona mechanics tasks;
- replay source SHA256:
  `1b71f2766ca9eed25d0ae9b3b7a7c04f39c8310a3c60cc8bf0d28011c3345345`.

The run resumed exact request-message hashes from the v8 private checkpoint after a
multilingual surface validator repair. It replayed 69 nonreasoning and 5 naive
requests, then made 75 and 5 new requests respectively. Prompts, generated responses,
scores, and selection rules were unchanged.

## Integrity Gates

All frozen serving gates passed:

- all five tasks completed;
- exactly eight initial worlds and six root questions per task;
- path-sensitive refreshed supports on `5/5` tasks;
- myopic and depth-two roots differed on `3/5` tasks;
- semantic padding normalization: `1.7857%`, below the 10% ceiling;
- BED/judge reasoning tokens: `0`;
- naive reasoning tokens: `1,547`;
- forced exits: `0`;
- finite endpoints for every policy/task.

## Mechanics Signal

The serving cohort does not show a positive planning effect:

| Policy | Mean turn-1 coverage | Mean turn-2 coverage | Mean turn-2 increment |
|---|---:|---:|---:|
| Myopic | 0.2952 | 0.5905 | 0.2952 |
| Depth 2 | 0.2952 | 0.5905 | 0.2952 |
| Random | 0.2952 | 0.5905 | 0.2952 |
| Naive thinking | 0.2952 | 0.6190 | 0.3238 |

Depth two minus myopic after two turns was exactly zero on every task: 0 wins, 5
ties, and 0 losses. Among the three changed-root tasks, refreshed true-intent recall
was lower by `0.3333` on average. The largest mechanism miss was
`researcher_task_001`, where myopic refreshed recall was 1.0 and depth-two recall was
0.0, while both policies still covered two thirds of the three hidden intents after
two turns.

This is consistent with the source-audit caveat. Pi-Bench reveals at least one hidden
intent after every question, so single-intent questions often tie regardless of root.
A gain requires a selected history to improve the second question enough to target
multiple remaining intents. The five-task serving cohort did not exhibit that link.

Serving was preregistered as an integrity gate rather than an efficacy gate. Its pass
therefore authorizes the unchanged 25-task development run, which is the fixed
decision point for this variant.

## Usage

Combined original-plus-resumed serving usage:

- physical requests: `154`;
- total cost: `$3.212356`;
- BED reasoning tokens: `0`;
- naive reasoning tokens: `1,547`;
- forced exits: `0`.

Authenticated OpenRouter balance after serving: `$22.785762594`.

Public artifact:
`results/nonmyopic/pi_bench_first_link_serving/pi-bench-first-link-serving-v9-20260728T011000Z/SERVING_SMOKE.json`.
Raw prompts, responses, questions, replies, and hidden-intent text remain untracked.
