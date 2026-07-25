# pi-Bench Dependency Observability Result

## Decision

The released pi-Bench dependency graph does not provide a canonical,
target-blind observation suitable for the planned exact non-myopic BED
wrapper. Close this construction before opening opportunity, development, or
holdout values and before any LLM call.

The dependency semantics are genuine: final tasks inherit preferences and
requirements from earlier sessions. However, those inherited preferences are
stored in the earlier tasks' hidden-intent annotations, not in an externally
released prior transcript or memory snapshot. Revealing those annotations as
an "inspect memory" observation would expose latent endpoint information to the
policy.

## Inspected Mechanics

Only the five preregistered mechanics tasks and their declared predecessor
tasks were opened:

| Persona | Final task | Dependencies |
| --- | --- | --- |
| Financier | `Financier_task_006` | `Financier_task_003` |
| Law trainee | `law_trainee_task_007` | tasks `001`, `004` |
| Marketer | `marketer_task_006` | task `005` |
| Pharmacist | `pharmacist_task_005` | tasks `002`, `004` |
| Researcher | `researcher_task_006` | tasks `004`, `001` |

These cases contain 38 final hidden intents and 34 predecessor hidden intents.
The inheritance is visible in examples such as:

- conclusion-first, tabular comparison, and Meta Review preferences in the
  Financier sequence;
- party-profile and evidence-collection structure in the law sequence;
- product-philosophy and risk analysis in the marketer sequence;
- evidence hierarchy and decisive-panel reading order in the pharmacist
  sequence;
- paper summaries, OpenReview/GitHub links, and follow-up suitability in the
  researcher sequence.

As a descriptive lexical check, 11/38 final intents have token-Jaccard
similarity of at least `.25` to a predecessor hidden intent. No final intent
reaches `.25` against the predecessor's observable initial request, title, or
description; the largest per-case observable similarity is below `.14`.
This threshold is descriptive, not a prospective efficacy gate.

## Runtime Finding

The official runner resets the communication channel after every task. The
external agent may maintain its own workspace or memory, but pi-Bench does not
freeze what that memory contains.

The `depends_on` graph is consumed by completeness/proactivity aggregation. It
does not cause the runtime or user simulator to expose a predecessor
observation. The official repository and Hugging Face release provide task
sources and aggregate leaderboard results, but no canonical agent trajectories,
prior-session transcripts, or reference memory snapshots.

Therefore:

1. Using predecessor hidden intents as observations is endpoint leakage.
2. Using predecessor requests/assets alone omits the inherited preference
   signal that makes the dependency valuable.
3. Generating a prior memory with our own agent creates a policy-dependent
   starting state, not an exact released environment transition.

## Consequence

pi-Bench remains strong evidence that semantic memory matters in long-horizon
agents, and it could support a separate end-to-end memory experiment. It does
not support the current clean first-link BED test without inventing or
privileging a memory state.

The frozen opportunity 10, development 5, and holdout 10 task values remain
unopened. No proxy answerer, hidden-intent-as-memory interface, paid smoke, or
threshold repair is authorized.

OpenRouter calls/cost: `0 / $0`. OatML jobs: `0`.
