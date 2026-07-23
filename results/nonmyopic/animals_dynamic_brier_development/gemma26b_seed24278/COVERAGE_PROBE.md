# 20 Questions Coverage-Dynamics Probe

Exploratory diagnostic only. The target is measurement-only after each counterfactual belief update; it is not supplied to questioner scoring or regeneration.

- States: `20`
- Candidate rows: `60`
- Mean within-state coverage spread: `0.10612083333333333`
- Median within-state coverage spread: `0.0`
- Max within-state coverage spread: `0.9166666666666666`
- Mean immediate-EIG coverage regret: `0.02717083333333333`
- Mean dynamic-Brier paired coverage gain: `-0.020583333333333335`
- Dynamic-Brier versus immediate-EIG W/T/L: `[0, 19, 1]`
- Spearman(immediate EIG, expected truth coverage): `0.12326715879201236`
- Spearman(support retention, expected truth coverage): `0.1910619828460985`
- Spearman(surviving MAP mass, expected truth coverage): `0.12596681530641335`
- Spearman(dynamic Brier gain, expected truth coverage): `-0.16459088198724164`

Per-state candidate/branch values are in `COVERAGE_PROBE.json`.
