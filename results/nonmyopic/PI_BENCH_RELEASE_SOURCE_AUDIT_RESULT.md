# Pi-Bench Release Source Audit Result

Date: 2026-07-28

**Status: passed; a separate first-link experiment may be preregistered.**

## Pinned Source And Split

- official repository: `https://github.com/Simplified-Reasoning/Pi-Bench`;
- commit: `383910b1698758a198b86037c63a111c8edc32ad`;
- Git tree: `a90d1c76b05c1c3651cb3156bc0f80b21490d751`;
- tracked files: `15,581`;
- tracked-tree manifest SHA256:
  `920d167432637298840a035b28572291a0ed744cf8705579e75739a687b2866e`;
- checkout state: clean;
- license: Apache-2.0.

All five released persona directories contain exactly twenty ordered task files. The
path-only manifest froze:

- mechanics, sessions 1-4/persona: `20`;
- development, sessions 5-10/persona: `30`;
- confirmation, sessions 11-16/persona: `30`;
- retained, sessions 17-20/persona: `20`.

Manifest:
`results/nonmyopic/pi_bench_release_source_manifest.json`, SHA256
`ccdf9211016d6c77eefc6cb3aae4e0324640c252b9d3aa17551ad158fc61594e`.
No task value was opened before this artifact was materialized.

## Mechanics Cohort

Only the twenty frozen mechanics task values were opened:

- total hidden intents: `86`;
- hidden intents per task: minimum `1`, maximum `10`;
- tasks with at least three hidden intents: `16/20`;
- initial statuses: `86/86 not_provided`, `0/86 provided`;
- explicit cross-task dependencies in this early cohort: `1/20`;
- task types: 12 long-term, 5 short-term independent, 2 short-term contextual,
  and 1 release-labeled validation task.

The hidden-intent status is an executable privacy boundary. The target agent receives
the task's natural initial input but not the `not_provided` intent contents. Every
mechanics task therefore contains at least one requirement that must be inferred or
elicited rather than directly read from the task message.

## Counterfactual Interaction

The released user agent provides the required semantic transition:

1. it maintains an ordered private set of runtime hidden intents and statuses;
2. an LLM judges whether the latest free-form assistant response already satisfies
   each remaining intent;
3. an LLM separately judges whether it asks a targeted question about each remaining
   intent;
4. matched intents are returned and marked `provided`;
5. accurately inferred intents are marked `inferred`;
6. if no intent is targeted, the first remaining intent is revealed;
7. the released proactiveness evaluator scores the union of matched and inferred intent
   indexes, including declared earlier-task dependencies.

The question space is unrestricted natural language. There is no released menu or
complete question-by-intent table. Semantic matching from a question to private intent
descriptions is therefore load-bearing LLM work.

Fresh `UserAgent` instances provide forkable state. A zero-call fake-judge dry run
started two copies of `Financier_task_001` with the same visible initial message and
six identical private `not_provided` statuses. Different questions produced isolated
next states:

```text
fork 1: [provided, not_provided, not_provided, not_provided, not_provided, not_provided]
fork 2: [not_provided, provided, not_provided, not_provided, not_provided, not_provided]
```

The LLM client is OpenAI-compatible, supports concurrent requests and arbitrary extra
payload fields, and uses temperature zero for user-intent judgments. A fixed provider
seed can therefore be requested and every accepted response can be recorded for
paired replay.

## External Endpoint

Intent resolution is not a self-reported planner score. The user-agent log records
matched and inferred private intent indexes at each assistant turn. The released
evaluator validates indexes and computes covered-intent count divided by the task's
private hidden-intent count. Full-task completeness is separately evaluated from
checklists, tool traces, and artifacts.

The clarification-only first-link experiment can use the intent endpoint without
running unrelated tool workflows or exposing hidden intent text to the policy.

## Caveat And Opportunity

The simulator is deliberately controlled:

- replies are the matched hidden-intent contents rather than independent free-form
  generations;
- an untargeted question still reveals the first unmet intent;
- semantic matching considers the latest assistant response and remaining intent
  state, not the full dialogue wording;
- state is path-dependent because earlier inferred/provided intents leave the candidate
  set, while reply prose itself is not a stochastic latent process.

This design is excellent for attribution but may favor greedy policies: every turn
reveals at least one requirement, and a compound question may match several at once.
The only defensible non-myopic opportunity is whether a first question induces a
better regenerated semantic belief over the remaining requirements, enabling a better
second question. The next gate must measure that exact link and stop if there is no
candidate spread or matched-compute gain.

## Validation

- local manifest tests: `2 passed`;
- official Pi-Bench tests after isolated dependency installation: `25 passed`;
- fork dry run: passed;
- OpenRouter calls: `0`;
- OpenRouter spend: `$0`.

Development, confirmation, and retained task values remain unopened.
