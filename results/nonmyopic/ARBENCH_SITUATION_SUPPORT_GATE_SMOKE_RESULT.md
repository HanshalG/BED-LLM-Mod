# AR-Bench Situation-Puzzle Support Gate: Serving Result

Date: 2026-07-24

Preregistration commit: `3e69c0e`

## Result

The serving smoke passed every frozen interface condition:

- exactly 2 tasks and 2 one-question branches completed;
- exactly 10 physical requests;
- zero reasoning tokens, retries, forced exits, or runtime failures;
- every support, question, answer, and semantic-coverage response parsed.

The run cost `$0.00364724` in the project ledger.

Both smoke cases were already covered by their initial eight-explanation supports. The
two public puzzles both concerned a person appearing alive after a funeral or wake,
and Gemma independently proposed medical misdiagnosis or suspended animation. Their
initial semantic scores were `0.95` and `0.90`; the single observed branch left both
scores unchanged. Thus the smoke shows no support-recovery opportunity and provides an
early saturation warning, but saturation was not a frozen serving-failure condition.

## Decision

The preregistered 12-case formal mechanism gate remains authorized without changing
the sample, support size, prompts, models, or thresholds. It will fail closed unless at
least 6/12 initial supports omit the hidden explanation, at least 3 omissions are
recovered, mean oracle gain is at least `0.10`, and at least 4/12 cases have branch
score spread at least `0.15`.

Artifacts:

- `results/nonmyopic/arbench_situation_support_gate/serving_smoke_20260724/SERVING_SMOKE.json`
- `results/nonmyopic/arbench_situation_support_gate/serving_smoke_20260724/RAW_RESPONSES.json`
- `results/nonmyopic/arbench_situation_support_gate/serving_smoke_20260724/run.log`
