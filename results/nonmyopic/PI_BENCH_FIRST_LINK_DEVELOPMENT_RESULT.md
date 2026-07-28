# Pi-Bench Dynamic-Support Development Result

Date: 2026-07-28

**Status: the preregistered development gate failed. This variant is closed and
no confirmation run is authorized.**

## Run

- run: `pi-bench-first-link-development-v13-20260728T034000Z`;
- interface: `pi_bench_dynamic_support_v13`;
- source commit/tree:
  `383910b1698758a198b86037c63a111c8edc32ad` /
  `a90d1c76b05c1c3651cb3156bc0f80b21490d751`;
- release manifest SHA256:
  `ccdf9211016d6c77eefc6cb3aae4e0324640c252b9d3aa17551ad158fc61594e`;
- replay source SHA256:
  `d1b948b976e56242b99be5752ef1799f07a449a822362f0e23046f9003026421`;
- cohort: 25 of 30 frozen development tasks, including every task with at
  least three initially hidden intents.

The five excluded tasks had hidden-intent counts `2, 1, 1, 2, 2`. The run used
exact same-stage, hash-bound checkpoint replay after format-only validator
repairs. Prompts, responses, scores, policy selection, and endpoints were not
changed by replay. No development aggregate was inspected until the cohort was
complete.

## Integrity

All preregistered integrity gates passed:

- all 25 tasks completed with finite endpoints;
- exactly eight initial worlds and six root questions per task;
- path-sensitive refreshed supports on `25/25` tasks;
- myopic and depth-two roots differed on `14/25` tasks (`56%`);
- semantic padding normalization was `4.092%`, below the 10% ceiling;
- BED/judge reasoning tokens: `0`;
- naive-baseline reasoning tokens: `9,821`;
- forced exits: `0`.

## Policy Results

| Policy | Mean turn-1 coverage | Mean turn-2 coverage | Mean turn-2 increment | Turn-2 SD |
|---|---:|---:|---:|---:|
| Myopic | 0.2254 | 0.4423 | 0.2168 | 0.1808 |
| Depth 2 | 0.2226 | 0.4394 | 0.2168 | 0.1850 |
| Random | 0.2168 | 0.4337 | 0.2168 | 0.1876 |
| Naive thinking | 0.2197 | 0.4365 | 0.2168 | 0.1835 |

Depth two minus myopic final coverage was `-0.00286` (SD `0.01429`;
90% and 95% bootstrap intervals both `[-0.00857, 0]`), with `0` wins, `24`
ties, and `1` loss. The one-sided exact permutation p-value was `1.0`.

The mechanism gates also failed:

- changed-root refreshed true-intent recall difference: `-0.07710`
  (required at least `+0.05`);
- overall final-coverage difference: `-0.00286`
  (required at least `+0.03`);
- wins minus losses: `-1` (required at least `+2`);
- predicted-versus-realized advantage Spearman correlation: `-0.0151`.

The recorded `at_least_half_gain_through_turn2` flag is vacuous here because the
total coverage gain is negative; it is not evidence for the policy.

## Interpretation

This is a clean first-link failure, not a transport or support-generation
failure. The LLM produced genuinely path-dependent beliefs and changed the
selected root on more than half of tasks. Those changes did not preserve the
true hidden intent better, and the simulated advantage did not predict realized
advantage.

Pi-Bench's released user agent explains the flat endpoint. A question that does
not target a hidden intent still receives the first unmet intent as a fallback,
so almost every policy resolves one intent per turn. Every policy therefore had
the same mean second-turn increment, `0.21684`. Non-myopic selection could only
win if the induced second question covered multiple intents, which did not occur
in this cohort. Dynamic LLM support is necessary for the intended claim, but it
does not by itself create an enabling-information gap.

The preregistered decision rule closes this Pi-Bench variant. The 30-task
confirmation and 20-task retained cohorts remain sealed.

## Usage

Combined same-stage replay plus live development usage:

- physical requests: `756`;
- total cost: `$13.2424675`;
- BED/judge reasoning tokens: `0`;
- naive-baseline reasoning tokens: `9,821`;
- forced exits: `0`.

Authenticated OpenRouter balance after final accounting on 2026-07-28:
`$9.350490094`. The full balance is available for subsequent experiments; there
is no fixed reserve.

Public artifact:
`results/nonmyopic/pi_bench_first_link_development/pi-bench-first-link-development-v13-20260728T034000Z/DEVELOPMENT.json`.
Raw prompts, responses, questions, replies, and hidden-intent text remain
untracked.
