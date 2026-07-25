# Active Task Disambiguation Code First-Link Audit Result

Date: 2026-07-25

## Outcome

**The aggregate gate failed. No APPS smoke is authorized.**

All 47 frozen HumanEval tasks were executable and usable, but the released
program populations were too often already correct or uniformly incorrect for
candidate tests to create a meaningful decision problem. Only 12 tasks met the
frozen unsaturated-support criterion and only four had at least `.10`
query-dependent hidden-test pass-fraction range, versus 25 required for each.

The audit used zero OpenRouter calls and no OatML resources.

## Frozen Results

| Metric | Result | Gate |
|---|---:|---:|
| Usable tasks | 47/47 | at least 35 |
| Mean initial hidden-test pass fraction | `.7340` | descriptive |
| Unsaturated tasks | 12 | at least 25 |
| Dynamic-range tasks | 4 | at least 25 |
| Mean selected gain over initial | `+.0310` | at least `+.05` |
| Mean selected gain over candidate mean | `-.0062` | at least `+.02` |
| Tasks with finite within-task rho | 8 | descriptive |
| Mean finite EIG/pass Spearman rho | `.8502` | at least `.10` |
| Positive-rho tasks | 7/8 | descriptive |
| Mean top-1 regret | `.0213` | at most `.15` |

The usable-task, correlation, and regret gates passed. The two prevalence and
two mean-gain gates failed, so the conjunction failed.

## Diagnostic Decomposition

The four dynamic-range tasks were HumanEval `17`, `91`, `147`, and `154`.
Tasks `91`, `147`, and `154` had EIG/pass correlations of `.913`, `1.0`, and
`1.0`, respectively, and the selected query reached their best observed
posterior pass fraction. Task `17` was the exception: its initial population
was 90% correct, the selected query conditioned on a zero-correctness outcome,
and its EIG/pass correlation was `-.111`.

This is consistent with a healthy first link on the few tasks that expose
useful correctness variation, but not with a sufficiently ambiguous
environment. Across all tasks, the EIG-selected query improved correctness by
only `.0310` over the initial population and was `.0062` worse than the mean
candidate query. The low aggregate regret is largely a consequence of 43 tasks
having less than `.10` endpoint range, not evidence of a broad planning gain.

## Interpretation

Executable LLM-generated programs remain a promising belief representation:
their exact output partitions strongly rank external correctness where the
released particles and candidate tests create dynamic range. The public
HumanEval traces, however, are mostly saturated and were not designed around
underspecified intent. They cannot support the planned non-myopic claim without
selecting a favorable post-outcome subset.

Per the frozen rule, this closes the exact released HumanEval route. The sealed
APPS continuation will not be run as a workaround. A scientifically distinct
code route must begin with tasks deliberately constructed to admit multiple
plausible executable specifications and must pass the same target-blind
opportunity screen before paid path-dependent generation.

## Integrity And Cost

- Source repository:
  `https://github.com/kasia-kobalczyk/active-task-disambiguation`.
- Source commit: `4c8ecb4d4ffdbffcc611366743fc1e2461037772`.
- HumanEval JSONL SHA-256:
  `882c3d56432b2b5b9e568398d7ebdf54f2c84fdb05fef3b833a3d935ad71861c`.
- Population: all 47 complete released GPT-4o-mini `active-reasoning`
  `iter_0` traces.
- Candidate EIG partitions were frozen before any official hidden tests ran.
- OpenRouter calls: 0.
- OpenRouter cost: `$0`.
- Audit artifact:
  `results/nonmyopic/atd_code_first_link_audit/AUDIT.json`.
- Artifact SHA-256:
  `a41c07aed2768f995274940fd6a0363e650cbb2909e14de4430c93d044873693`.
