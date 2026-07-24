# ClinDiag Multisample Atomic Opportunity Result

Date: 2026-07-24

Run: `clindiag-multisample-atomic-opportunity-20260724T223154Z`

Status: **failed the preregistered opportunity conjunction; no planner or
holdout is authorized.**

## Result

All four cases, 38 belief states per case, and three independent supports per
state completed. The representation was stable but had no useful non-myopic
coverage gap:

| Endpoint | Required | Observed |
|---|---:|---:|
| Initial mean coverage | at most .50 | .25 |
| Cases with one-step coverage spread >= 1/3 | at least 2 | 1 |
| Cases with pair coverage gain >= 1/3 | at least 2 | 0 |
| Cases with non-myopic soft gap >= .10 | at least 1 | 0 |
| Mean non-myopic soft gap | at least .03 | .0125 |
| Mean pair gain over best one step | at least .05 | .0450 |
| Cases where oracle first differs | at least 1 | 2 |
| Mean replay soft gap | at most .10 | .0133 |
| Maximum replay soft gap | at most .20 | .0533 |

One case was already perfectly recalled after a physical-examination action
and another after an examination action in all three samples. The two omitted
diagnoses did not achieve empirical coverage after any pair. They showed only
small soft-score movements: `.05` for alpha-methylacyl-CoA racemase deficiency
and `.13` pair gain for Langerhans cell histiocytosis.

## Interpretation

Three-sample de-anchored refresh resolves the earlier single-list stability
problem. The exact-prompt replay gaps are small, so the null is not explained
by winner noise. The remaining issue is structural: these stored atomic
observations either make recall myopically easy or still fail to introduce the
target after two steps. Pair order changes the selected oracle first action on
two cases, but not enough to create truth coverage.

The exact multisample fixed-six-action ClinDiag route is closed. There is no
threshold repair, case replacement, larger sample count, planner, or use of the
sealed 60-case holdout.

## Integrity and Cost

- exactly 608 physical requests: 456 full GPT-5.4 support generations and 152
  GPT-5.4 Mini semantic measurements;
- zero reasoning tokens, structured retries, forced exits, or runtime failures;
- cost `$1.39884075`;
- private raw SHA-256
  `4bb3bdc148906ccfc6edd994e618377776692f088da1d63ef79abfa1a17f7e98`;
- project spend `$66.52709534`, leaving `$38.85770735` before the Monday reserve
  ceiling.
