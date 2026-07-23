# 20 Questions Coverage-Dynamics Probe

Exploratory diagnostic only. The target is measurement-only after each counterfactual belief update; it is not supplied to questioner scoring or regeneration.

- States: `20`
- Candidate rows: `60`
- Mean within-state coverage spread: `0.05579166666666666`
- Median within-state coverage spread: `0.0`
- Max within-state coverage spread: `0.5099999999999999`
- Mean immediate-EIG coverage regret: `0.05579166666666666`
- Mean dynamic-Brier paired coverage gain: `0.0`
- Dynamic-Brier versus immediate-EIG W/T/L: `[0, 20, 0]`
- Spearman(immediate EIG, expected truth coverage): `-0.10054263626240854`
- Spearman(support retention, expected truth coverage): `0.05724141265108599`
- Spearman(surviving MAP mass, expected truth coverage): `-0.20102743652189262`
- Spearman(dynamic Brier gain, expected truth coverage): `0.1344758527003114`

Per-state candidate/branch values are in `COVERAGE_PROBE.json`.
