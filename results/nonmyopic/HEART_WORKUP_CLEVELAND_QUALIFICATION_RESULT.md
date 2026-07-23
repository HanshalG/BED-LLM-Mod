# Cleveland Heart Workup Qualification Result

The preregistered exact depth qualification **passed every gate** on the independent
processed Cleveland cohort. This authorizes a separately preregistered compact
non-thinking Gemma 4 26B serving smoke and proposal-quality gate; it does not yet
establish that an LLM policy can recover the exact depth-two advantage.

| Endpoint (d2 - d1) | Mean gain | Paired 95% CI | W/T/L |
| --- | ---: | ---: | ---: |
| Entropy AUC | +0.068747 | [+0.057311, +0.080303] | 171/87/39 |
| Truth-log AUC | +0.068747 | [+0.047312, +0.090220] | 153/87/57 |
| Final entropy | +0.030001 | [+0.016066, +0.048131] | 13/284/0 |
| Earlier workup rounds | +2.222222 | [+2.020202, +2.420875] | 210/87/0 |

Depth one ordered the workup in rounds 4/5/6 for 63/71/163 patients. Depth two
ordered it in rounds 2/3/4/5/6 for 142/48/45/55/7 patients and never ordered it
later on a paired case. Both policies first queried chest-pain type. The depth-two
mean entropy trace becomes lower from round three onward and reaches zero by round
eight; depth one's final mean entropy is 0.030001.

An independent audit rebuilt every posterior from the raw 297-row cohort and used a
separate expected-cumulative-entropy recursion to re-score every recorded action.
All actions were independently optimal at their stated horizons, all stored costs
and metrics matched within `1e-12`, and fresh-bootstrap lower bounds remained
positive: +0.056895 for entropy AUC and +0.047452 for truth-log AUC.

All qualification and audit computations were exact and made zero LLM calls.

Artifacts:

- `results/nonmyopic/HEART_WORKUP_CLEVELAND_PREREGISTRATION.md`
- `results/nonmyopic/heart_workup_cleveland_qualification_20260723/REPORT.json`
- `results/nonmyopic/heart_workup_cleveland_qualification_20260723/REPORT.md`
- `results/nonmyopic/heart_workup_cleveland_qualification_audit_20260723/AUDIT.json`
- `results/nonmyopic/heart_workup_cleveland_qualification_audit_20260723/AUDIT.md`

Data sources: UCI Heart Disease, https://doi.org/10.24432/C52P4X; UCI Statlog Heart,
https://doi.org/10.24432/C57303. Both UCI repository pages report CC BY 4.0.
