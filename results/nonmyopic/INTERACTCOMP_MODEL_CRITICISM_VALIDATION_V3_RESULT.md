# InteractComp Model-Criticism Validation V3 Result

Date: 2026-07-25

## Outcome

**V3 completed cleanly but failed the frozen opportunity and comparative gates.
No InteractComp depth-two run is authorized, and this exact route is closed.**

All 656 requests completed with valid supports, semantic filtering,
classifications, true responses, and refreshed populations. The failure is
scientific: only one of six tasks recovered the target after a realized root, so
five tasks had zero endpoint variation and could not rank acquisition scores.

## Frozen Aggregate Results

| Metric | Result | Gate |
|---|---:|---:|
| Initial target omissions | 6/6 | at least 4 |
| Target recovered after any root | 1/6 | at least 4 |
| Endpoint range at least `1/8` | 1/6 | at least 4 |
| Rankable tasks | 1/6 | at least 4 |
| Tasks with at least two non-unknown responses | 4/6 | 6/6 |
| Mean selected endpoint, model criticism | `.02083` | descriptive |
| Mean selected endpoint, current EIG | `.00000` | model gain at least `.02` |
| Mean selected endpoint, augmented EIG | `.02083` | model gain at least `.02` |
| Mean selected endpoint, random | `.00000` | model gain at least `.02` |
| Model-criticism wins/losses versus current EIG | 1/0 | at least 2 / at most 1 |

The primary method exceeded current EIG and random by `.02083`, passing those
mean-gain gates exactly, but tied the compute-matched augmented-support EIG. It
beat current EIG on only one task.

Only task 193 was rankable. On that task:

- model criticism and augmented EIG selected a root with target mass `1/8`;
- current EIG and random selected a zero-target root;
- score-endpoint Spearman was `+.77460` for model criticism and augmented EIG;
  and
- current-EIG Spearman was `-1.0`.

The reported mean correlations equal this single task and therefore do not
establish aggregate ranking fidelity. Oracle-selection counts on zero-range
tasks are likewise tie artifacts and not evidence.

## Representation And Transition Diagnosis

Every initial enrolled support omitted the exact target. The target-blind
outside-support generator recovered the target before interaction on two tasks:

| Task | Target mass in retained auxiliary support | Maximum target mass after realized refresh |
|---:|---:|---:|
| 87 | `3/8` | `0/8` |
| 193 | `7/8` | `1/8` |
| Other four | `0/8` | `0/8` |

Task 87 demonstrates path-dependent forgetting: the auxiliary LLM population
found the missing answer three times, but all four fresh outcome-conditioned
regenerations dropped it. Task 193 retained one target particle on three roots,
creating the only useful endpoint variation.

Thus the limiting mechanism is not merely noisy root scoring. Target-blind
hypothesis discovery has insufficient prevalence (`2/6`), and replacing support
with a fresh branch population further reduces truth coverage (`1/6`). A
depth-two planner over these transitions would be optimizing mostly zero
endpoints.

A future method may preserve and Bayesian-filter compatible current and
auxiliary particles while adding refreshed particles, rather than replacing the
support. That is a distinct belief-update design and may be analyzed on these
now-open tasks, but any claim would require a new environment or fully fresh
prospective task block. No InteractComp V4, threshold relaxation, favorable
subset, wider proposal rerun, or depth-two efficacy run is authorized here.

## Descriptive Support-Preservation Replay

A zero-call post-result replay tested that proposed update on the frozen V3
artifacts:

1. retain current and auxiliary particles whose predicted Y/N/U label equals
   the realized response;
2. add the eight refreshed particles; and
3. measure exact target mass in the merged population.

This raises positive target coverage from one to two tasks and preserves the
target that fresh regeneration forgot on task 87:

| Mean merged-support endpoint | Value |
|---|---:|
| Model criticism | `.09848` |
| Compute-matched augmented EIG | `.09848` |
| Current EIG | `.00926` |
| Random | `.00926` |
| Oracle | `.09848` |

Task 87's four merged endpoints range only from `.0526` to `.0909`; task 193
ranges from `0` to `.50`; the other four remain zero everywhere. Model
criticism selects the oracle on both positive tasks, but augmented EIG selects
the same roots and achieves the identical endpoint. Support preservation
therefore repairs one forgetting failure but does not solve low target coverage
or establish a non-myopic advantage. This replay is descriptive and does not
alter the preregistered null.

## Integrity And Cost

- Preregistered commit: `59ec451`.
- Run ID: `interactcomp-model-criticism-validation-v3-20260725T121000Z`.
- Requests/attempts: `656/656` (`632` Mini, `24` GPT-5.4 responder).
- Retries/reasoning tokens/forced exits: `0/0/0`.
- Cost: `$0.36137155`.
- Public artifact SHA-256:
  `f3b5550ec057abf4519b966db7e1243794d97254a55130f09d7c75ed008b6cf5`.
- Private raw SHA-256:
  `ec509c45c6b33389f12773e7ab5ed4be5af252ba8f3305f3d9c8dec6f09d4ed7`.
- Project-ledger spend after V3: `$86.83705421920735`.
- Monday local allowance remaining: `$14.30625060000007`.
- Authenticated OpenRouter remaining: `$43.547748484`, or `$18.547748484`
  above the protected `$25` reserve.
- OatML resources used: none.
