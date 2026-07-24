# AR-Bench Situation-Puzzle Support Gate: Formal Result

Date: 2026-07-24

Preregistration commit: `3e69c0e`

Serving-result commit: `d02e6a7`

## Frozen Result

The formal gate failed:

| Endpoint | Result | Threshold |
|---|---:|---:|
| Cases completed | 12/12 | 12/12 |
| Branches completed | 48/48 | 48/48 |
| Physical requests | 132 | exactly 132 |
| Reasoning tokens | 0 | 0 |
| Initial omissions | 3/12 | at least 6/12 |
| Omitted truths recovered | 1/3 | at least 3 |
| Mean oracle best-match gain | +0.0575 | at least +0.10 |
| Cases with branch spread at least .15 | 9/12 | at least 4/12 |

The run cost `$0.05350195` in the project ledger. Together with the serving
smoke, the complete AR-Bench support-gate line cost `$0.05714919`.

## Audit

The apparent path dependence did not represent reliable truth recovery:

- Nine initial supports already covered the hidden mechanism, leaving insufficient
  headroom for regenerated hypotheses to be load-bearing.
- The one nominal recovery concerned Emily surviving her funeral. The initial
  explanation and recovered explanation both used the same
  coma/suspended-animation misdiagnosis mechanism and both omitted the grave escape,
  yet the evaluator changed its score from `0.70` to `0.90`. This is not persuasive
  evidence of a newly recovered causal hypothesis.
- Across all 48 branches, the mean score change from the initial support was
  `-0.1154`: 10 branches improved, 18 tied, and 20 worsened. Ten branches dropped by
  at least `0.15`.
- The strongest genuine-looking change was on the imaginary-wife puzzle, where one
  answer moved the best score from `0.00` to `0.70`, but it still failed the frozen
  coverage threshold.
- The answer distribution was balanced enough to exercise the interface
  (`20 Yes`, `24 No`, `4 Unknown`), so the null is not caused by a single constant
  oracle label.

The branch-spread condition passed because refreshed supports often discarded a good
initial explanation on some branches. That is unstable support replacement, not the
positive path-dependent recovery mechanism needed for non-myopic BED.

## Decision

Close this AR-Bench Situation-Puzzle apparatus. Do not add semantic likelihood,
ranking fidelity, depth comparison, policy evaluation, support-size tuning, or
post-hoc case selection from these responses. A distinct next route must make initial
semantic support genuinely incomplete while preserving useful hypotheses during
evidence-conditioned updates.

Artifacts:

- `results/nonmyopic/arbench_situation_support_gate/formal_seed24301_20260724/GATE.json`
- `results/nonmyopic/arbench_situation_support_gate/formal_seed24301_20260724/RAW_RESPONSES.json`
- `results/nonmyopic/arbench_situation_support_gate/formal_seed24301_20260724/run.log`
